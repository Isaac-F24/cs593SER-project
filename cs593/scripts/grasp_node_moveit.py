#!/usr/bin/env python3
"""
Top-down grasp node for FR3 dual-arm setup.
Reads block poses from /gz_world_poses, solves IK, and executes grasp via
joint_trajectory_controller.
"""
import threading
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import builtin_interfaces.msg
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from moveit_controllers import LeftArmMoveitController, RightArmMoveitController
from gripper_control import GripperController


# ---------------------------------------------------------------------------
# FR3 kinematics
# ---------------------------------------------------------------------------
_JOINT_ORIGINS = [
    (0,       0,      0.333,  0,           0, 0),
    (0,       0,      0,     -np.pi/2,     0, 0),
    (0,      -0.316,  0,      np.pi/2,     0, 0),
    (0.0825,  0,      0,      np.pi/2,     0, 0),
    (-0.0825, 0.384,  0,     -np.pi/2,     0, 0),
    (0,       0,      0,      np.pi/2,     0, 0),
    (0.088,   0,      0,      np.pi/2,     0, 0),
]
_FLANGE = (0, 0, 0.107, 0, 0, 0)
_EE_ROT = (0, 0, 0, 0, 0, -np.pi / 4)
_TCP    = (0, 0, 0.1034, 0, 0, 0)

_JOINT_LIMITS = np.array([
    (-2.9007,  2.9007),
    (-1.8361,  1.8361),
    (-2.9007,  2.9007),
    (-3.0770, -0.1169),
    (-2.8763,  2.8763),
    ( 0.4398,  4.6216),
    (-3.0508,  3.0508),
])

# Safe home configuration
_HOME_Q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

# Top-down: TCP z-axis points straight into world -z (gripper faces down)
R_TOP_DOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

# Side grasp from -y side: TCP +z = world +y (gripper aims at block from -y),
# TCP +y = world +x (fingers close along world x → grip the block's x-sides),
# TCP +x = world +z (right-hand frame). Used by the right arm to pinch the
# block at its centre while another arm grasps from above.
R_SIDE_FROM_NEG_Y = np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]], dtype=float)

# Block half-extent (matches world file 0.055^3 cubes)
_BLOCK_HALF_H = 0.0275


def _handoff_R(base_y):
    """
    Handoff orientation: gripper rotated parallel to the table so it doesn't
    block the side-grasping partner. Wrist sits on the holding arm's own y
    side and TCP +z points back toward that side; fingers close along
    world +z, gripping the block's top/bottom faces while the partner takes
    the (free) ±x faces. World +x stays in the gripper plane so the wrist
    rotation from a top-down lift is a clean ~90° about world x.
    """
    sign = 1.0 if base_y >= 0 else -1.0
    return np.array([[sign,  0.0,    0.0 ],
                     [0.0,   0.0,  -sign ],
                     [0.0,   1.0,    0.0 ]], dtype=float)


def _rpy_matrix(roll, pitch, yaw):
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _tf(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    T[:3, :3] = _rpy_matrix(roll, pitch, yaw)
    return T


def fk(q, base_xyz):
    T = _tf(*base_xyz, 0, 0, 0)
    for i, origin in enumerate(_JOINT_ORIGINS):
        T = T @ _tf(*origin) @ _tf(0, 0, 0, 0, 0, q[i])
    T = T @ _tf(*_FLANGE) @ _tf(*_EE_ROT) @ _tf(*_TCP)
    return T


def ik(target_pos, target_R, base_xyz, q_seed=None, attempts=8):
    """
    Numerical IK via SLSQP with random restarts and early exit.
    Returns joint angles or None.
    """
    target_pos = np.asarray(target_pos)

    def cost(q):
        T = fk(q, base_xyz)
        dp = T[:3, 3] - target_pos
        R_err = T[:3, :3].T @ target_R
        cos_a = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        return float(np.dot(dp, dp) + 0.3 * np.arccos(cos_a) ** 2)

    bounds = list(map(tuple, _JOINT_LIMITS))
    best_q, best_cost = None, float('inf')

    for i in range(attempts):
        q0 = (q_seed.copy() if (i == 0 and q_seed is not None)
              else np.random.uniform(_JOINT_LIMITS[:, 0], _JOINT_LIMITS[:, 1]))
        res = minimize(cost, q0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-9})
        if res.fun < best_cost:
            best_cost, best_q = res.fun, res.x
            if best_cost < 1e-5:
                break  # tight solution found, no need for more restarts

    return best_q if best_cost < 5e-3 else None



def _handoff_R(base_y):
    """
    Handoff orientation: gripper rotated parallel to the table so it doesn't
    block the side-grasping partner. Wrist sits on the holding arm's own y
    side and TCP +z points back toward that side; fingers close along
    world +z, gripping the block's top/bottom faces while the partner takes
    the (free) ±x faces. World +x stays in the gripper plane so the wrist
    rotation from a top-down lift is a clean ~90° about world x.
    """
    sign = 1.0 if base_y >= 0 else -1.0
    return np.array([[sign,  0.0,    0.0 ],
                     [0.0,   0.0,  -sign ],
                     [0.0,   1.0,    0.0 ]], dtype=float)

def rotation_matrix_to_quaternion(rotation_matrix: np.ndarray):
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    return quaternion

# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class GraspNode(Node):
    def __init__(self, executor):
        super().__init__('grasp_node')

        self.declare_parameter('arm', 'left')
        self.declare_parameter('target_block', 'blue_box')
        self.declare_parameter('grasp_x_offset', 0.0)
        self.declare_parameter('grasp_y_offset', 0.0)
        self.declare_parameter('grasp_mode', 'top')  # 'top' or 'side'
        # Handoff pose (top-grasp only): TCP target to move to after the lift,
        # placing the block where a side-grasping partner can reach it.
        self.declare_parameter('handoff_x', 0.45)
        self.declare_parameter('handoff_y', 0.0)
        self.declare_parameter('handoff_z', 0.30)

        arm = self.get_parameter('arm').get_parameter_value().string_value
        self.target_block = self.get_parameter('target_block').get_parameter_value().string_value
        self.grasp_x_offset = self.get_parameter('grasp_x_offset').get_parameter_value().double_value
        self.grasp_y_offset = self.get_parameter('grasp_y_offset').get_parameter_value().double_value
        self.grasp_mode = self.get_parameter('grasp_mode').get_parameter_value().string_value
        self.handoff_xyz = np.array([
            self.get_parameter('handoff_x').get_parameter_value().double_value,
            self.get_parameter('handoff_y').get_parameter_value().double_value,
            self.get_parameter('handoff_z').get_parameter_value().double_value,
        ])
        if self.grasp_mode not in ('top', 'side'):
            self.get_logger().warn(
                f"Unknown grasp_mode '{self.grasp_mode}', defaulting to 'top'")
            self.grasp_mode = 'top'
        self.grasp_R = R_SIDE_FROM_NEG_Y if self.grasp_mode == 'side' else R_TOP_DOWN

        self.arm = arm
        self.base_xyz = (0.0, 0.5, 0.0) if arm == 'left' else (0.0, -0.5, 0.0)
        self.arm_joints   = [f'{arm}_fr3_joint{i+1}' for i in range(7)]
        self.gripper_joints = [
            f'{arm}_fr3_finger_joint1',
            f'{arm}_fr3_finger_joint2',
        ]

        self.left_arm_client = ActionClient(
            self, FollowJointTrajectory,
            f'/left/left_fr3_arm_controller/follow_joint_trajectory'
        )
        self.right_arm_client = ActionClient(
            self, FollowJointTrajectory,
            f'/right/right_fr3_arm_controller/follow_joint_trajectory'
        )

        self.block_poses = {}
        self._finger_pos = None

        self.create_subscription(TFMessage, '/gz_world_poses', self._on_poses, 10)
        self.create_subscription(JointState, f'/{arm}/joint_states', self._on_joint_states, 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.1, self.update_pose)
        self.hand_pose = None

        self._marker_pub = self.create_publisher(MarkerArray, '/grasp_markers', 10)
        self._marker_id  = 0

        self._done = False
        self.create_timer(2.0, self._try_grasp)
        self.get_logger().info(
            f'Grasp node ready — arm={arm}, target={self.target_block}, '
            f'mode={self.grasp_mode}')
        
        self.left_arm_controller = LeftArmMoveitController(executor)
        self.right_arm_controller = RightArmMoveitController(executor)
        
        self.gripper_controller = GripperController(ros_node=self)

    # ------------------------------------------------------------------
    def _on_poses(self, msg: TFMessage):
        for tf in msg.transforms:
            fid = tf.child_frame_id
            for name in ('blue_box', 'red_box', 'green_box'):
                if name in fid:
                    self.block_poses[name] = (
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z,
                    )

    def _on_joint_states(self, msg: JointState):
        joint_name = f'{self.arm}_fr3_finger_joint1'
        try:
            idx = msg.name.index(joint_name)
            self._finger_pos = msg.position[idx]
        except ValueError:
            pass

    def update_pose(self):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform("world", "left_fr3_hand", now)
        except Exception:
            return

        self.hand_pose = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
        

    def _publish_markers(self, waypoints: dict):
        colors = {
            'approach': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9),
            'grasp':    ColorRGBA(r=1.0, g=0.2, b=0.0, a=0.9),
            'lift':     ColorRGBA(r=0.0, g=0.4, b=1.0, a=0.9),
        }
        ma = MarkerArray()
        for name, pos in waypoints.items():
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id = f'{self.arm}_wp', self._marker_id; self._marker_id += 1
            m.type, m.action = Marker.SPHERE, Marker.ADD
            m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.025
            m.color = colors.get(name, ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8))
            m.lifetime = builtin_interfaces.msg.Duration(sec=120)
            ma.markers.append(m)
            lbl = Marker()
            lbl.header = m.header
            lbl.ns, lbl.id = f'{self.arm}_lbl', self._marker_id; self._marker_id += 1
            lbl.type, lbl.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            lbl.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]) + 0.04)
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.022
            lbl.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lbl.text = name
            lbl.lifetime = builtin_interfaces.msg.Duration(sec=120)
            ma.markers.append(lbl)
        self._marker_pub.publish(ma)

    # ------------------------------------------------------------------
    def _cartesian_line(self, pos_start, pos_end, q_seed, n_steps=12,
                        target_R=None):
        """
        Solve IK at n_steps+1 equally-spaced points along a straight
        Cartesian line, each seeded from the previous solution. target_R
        defaults to self.grasp_R; pass an explicit rotation to plan a
        line in a different end-effector frame (e.g. the horizontal
        handoff frame during retreat).
        Returns list of joint arrays, or None if any step fails.
        """
        if target_R is None:
            target_R = self.grasp_R
        configs, q = [], q_seed.copy()
        for i in range(n_steps + 1):
            alpha = i / n_steps
            pos = pos_start + alpha * (pos_end - pos_start)
            q = ik(pos, target_R, self.base_xyz, q_seed=q, attempts=5)
            if q is None:
                self.get_logger().error(
                    f'Cartesian IK failed at step {i}/{n_steps} '
                    f'(pos={pos.round(3)})')
                return None
            configs.append(q.copy())
        return configs

    def _send_trajectory(self, traj) -> bool:
        """Send a JointTrajectory goal and block the grasp thread until done."""

        if self.arm == "left":
            self._arm_client = self.left_arm_client
        else:
            self._arm_client = self.right_arm_client

        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Arm action server unavailable')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        done, success = threading.Event(), [False]

        def _on_result(fut):
            code = fut.result().result.error_code
            if code != 0:
                self.get_logger().warn(f'Trajectory error_code={code}')
            success[0] = (code == 0)
            done.set()

        def _on_goal(fut):
            handle = fut.result()
            if not handle.accepted:
                self.get_logger().error('Trajectory goal rejected')
                done.set()
                return
            handle.get_result_async().add_done_callback(_on_result)

        self._arm_client.send_goal_async(goal).add_done_callback(_on_goal)
        done.wait()
        return success[0]

    def _build_trajectory(self, configs, total_sec):
        """
        Pack joint configs into a timestamped JointTrajectory.
        Only the first and last points are pinned to zero velocity; intermediates
        are left unspecified so JTC's cubic spline carries momentum smoothly
        through them instead of decelerating to a halt at every waypoint.
        """
        traj = JointTrajectory()
        traj.joint_names = self.arm_joints
        n = len(configs)
        for i, q in enumerate(configs):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            if i == 0 or i == n - 1:
                pt.velocities = [0.0] * 7
            t = total_sec * i / max(n - 1, 1)
            pt.time_from_start = builtin_interfaces.msg.Duration(
                sec=int(t), nanosec=int((t % 1) * 1e9))
            traj.points.append(pt)
        return traj


    # --------------------------------------------

    def _try_grasp(self):
        if self._done:
            return
        if self.target_block not in self.block_poses:
            self.get_logger().info(
                f'Waiting for {self.target_block} pose on /gz_world_poses ...')
            return
        self._done = True
        threading.Thread(target=self._execute_grasp, daemon=True).start()

    def _execute_grasp(self):
        bx, by, bz = self.block_poses[self.target_block]
        self.get_logger().info(
            f'Block {self.target_block} at ({bx:.3f}, {by:.3f}, {bz:.3f})')

        # Apply user-tunable offset to centre the TCP over the block
        tx = bx + self.grasp_x_offset
        ty = by + self.grasp_y_offset
        if self.grasp_x_offset or self.grasp_y_offset:
            self.get_logger().info(
                f'Grasp offset applied: dx={self.grasp_x_offset:+.3f} '
                f'dy={self.grasp_y_offset:+.3f} → target ({tx:.3f}, {ty:.3f})')

        # Grasp geometry
        if self.grasp_mode == 'side':
            # Side grasp: TCP at block centre height; approach from the arm-base
            # side along world y by 20 cm; lift = retract 5 cm + raise 8 cm so
            # the block clears its neighbours without colliding the gripper
            # against a top-grasping arm working the same block.
            approach_dy = -0.20 if self.base_xyz[1] < ty else 0.20

            # adjust 
            if self.arm == "left":
                ty += _BLOCK_HALF_H + 0.065
            else:
                ty -= _BLOCK_HALF_H + 0.065

            grasp_pos    = np.array([tx, ty, bz])
            approach_pos = np.array([tx, ty + approach_dy, bz])
            lift_pos     = np.array([tx, ty + 0.25 * approach_dy, bz + 0.08])
        else:
            # Top-down: approach above, grip at block centre, lift back up.
            # Cubes are short enough that the original "8 cm below the top"
            # rule would put the TCP under the table — centre is the right
            # target now.
            grasp_z    = bz + _BLOCK_HALF_H + 0.065                         # TCP at block centre
            approach_z = bz + _BLOCK_HALF_H + 0.20   # 20 cm above block top
            lift_z     = approach_z
            approach_pos = np.array([tx, ty, approach_z])
            grasp_pos    = np.array([tx, ty, grasp_z])
            lift_pos     = np.array([tx, ty, lift_z])

        self._publish_markers({
            'approach': approach_pos,
            'grasp':    grasp_pos,
            'lift':     lift_pos,
        })


        self.get_logger().info('Executing grasp sequence')

        # 1. Move to approach (gripper closed — avoids flailing during swing).
        #    Joint-space-densify the path so the controller gets a fine-grained
        #    command stream instead of a single big cubic, and stretch duration
        #    so the gz velocity-motor has time to track without overshoot.
        self.get_logger().info('Step 1/6  Moving to approach position')

        if self.arm == "left":
            self.left_arm_controller.go_to_pose(approach_pos, rotation_matrix_to_quaternion(self.grasp_R))
        else:
            self.right_arm_controller.go_to_pose(approach_pos, rotation_matrix_to_quaternion(self.grasp_R))

        time.sleep(1.0)  # settle before opening gripper

        # 2. Hold above target so position can be verified before opening
        self.get_logger().info('Step 2/6  Holding above target for 2 s ...')
        time.sleep(2.0)

        #   side: ... → close → release-partner (6) → lift (7)
        #   top:  ... → close → lift (6) → move-to-handoff (7) → retreat (8)
        # Side does the handoff release itself; top finishes by carrying the
        # block to a central reachable pose so a side-grasping partner can pick
        # it up next, then retreats clear of the block once the partner has
        # taken over.
        n_steps = 8 if self.grasp_mode == 'top' else 7

        # 3. Open gripper — both fingers actively position-tracked, so this can
        #    block on completion. Width capped at 0.04 m (FR3 finger hard limit).
        self.get_logger().info(f'Step 3/{n_steps}  Opening gripper')

        self.gripper_controller.open_gripper(left=(True if self.arm == "left" else False))

        # 4. Straight-line Cartesian descent / inward approach to grasp point.
        self.get_logger().info(f'Step 4/{n_steps}  Moving to grasp')
        
        if self.arm == "left":
            self.left_arm_controller.go_to_pose(grasp_pos, rotation_matrix_to_quaternion(self.grasp_R))
        else:
            self.right_arm_controller.go_to_pose(grasp_pos, rotation_matrix_to_quaternion(self.grasp_R))

        time.sleep(0.5)  # let oscillations settle


        # 5. Close gripper on the block — blocking so the lift doesn't start
        #    until the fingers have actually moved into contact.
        self.get_logger().info(f'Step 5/{n_steps}  Closing gripper')

        self.gripper_controller.close_gripper(left=(True if self.arm == "left" else False), close_goal=0.005) #Note- close goal should be higher for the block-change this

        # 6. (side-grasp only) Release the partner arm so the handoff completes
        #    before this arm lifts. Done before the lift so the partner isn't
        #    fighting this arm's pull.
        if self.grasp_mode == 'side':
            self.get_logger().info(f'Step 6/{n_steps}  Releasing other gripper (handoff)')
            time.sleep(0.3)  # let this arm's grip settle on the block

            self.gripper_controller.open_gripper(left=(False if self.arm == "left" else True))  # opposite arm

        # Lift / retract straight back to the approach pose. Step 6 in top mode
        # (release-partner is step 6 in side mode and lift becomes step 7 there).
        lift_step = 7 if self.grasp_mode == 'side' else 6
        self.get_logger().info(f'Step {lift_step}/{n_steps}  Lifting')

        if self.arm == "left":
            self.left_arm_controller.go_to_pose(lift_pos, rotation_matrix_to_quaternion(self.grasp_R))
        else:
            self.right_arm_controller.go_to_pose(lift_pos, rotation_matrix_to_quaternion(self.grasp_R))


        # 7. (top-grasp only) Carry the block to the handoff pose so a side-
        #    grasping partner can reach it. The wrist rotates so the gripper
        #    lies parallel to the table (R_handoff): gripper body sits on
        #    this arm's own y side, fingers grip the block top/bottom, and
        #    the ±x faces are left clear for the side arm coming in along
        #    world y. Seeded from the lift config so the IK finds a nearby
        #    branch — joint-space-interpolating across the ~90° wrist roll.
        if self.grasp_mode == 'top':
            self.get_logger().info(
                f'Step 7/{n_steps}  Moving to handoff '
                f'({self.handoff_xyz[0]:.2f}, {self.handoff_xyz[1]:.2f}, '
                f'{self.handoff_xyz[2]:.2f})')
            
            handoff_rotation_matrix = _handoff_R(self.base_xyz[1])
            
            if self.arm == "left":
                self.left_arm_controller.go_to_pose(self.handoff_xyz, rotation_matrix_to_quaternion(_handoff_R(self.base_xyz[1])))
            else:
                self.right_arm_controller.go_to_pose(self.handoff_xyz, rotation_matrix_to_quaternion(_handoff_R(self.base_xyz[1])))

            # 8. Wait for the side-grasping partner to take over and release
            #    this gripper, then retreat away from the block. With the
            #    gripper horizontal, "up" no longer pulls the fingers off the
            #    block — instead retract along world y on this arm's own side
            #    (away from the side-grasp arm) so the open fingers slide
            #    clear without crossing the partner's approach corridor.
            self.get_logger().info(
                f'Step 8/{n_steps}  Waiting for partner to release gripper ...')
            deadline = time.time() + 60.0
            released = False
            while time.time() < deadline:
                if self._finger_pos is not None and self._finger_pos > 0.02:
                    released = True
                    break
                time.sleep(0.1)
            if not released:
                self.get_logger().warn(
                    'Partner release not detected — retreating anyway')

            handoff_pos = self.handoff_xyz
            retreat_dy = 0.15 if self.base_xyz[1] >= 0 else -0.15
            retreat_pos = handoff_pos + np.array([0.0, retreat_dy, 0.10])

            self.get_logger().info(
                f'Step 8/{n_steps}  Retreating to '
                f'({retreat_pos[0]:.2f}, {retreat_pos[1]:.2f}, {retreat_pos[2]:.2f})')
        
            if self.arm == "left":
                self.left_arm_controller.go_to_pose(retreat_pos, rotation_matrix_to_quaternion(_handoff_R(self.base_xyz[1])))
            else:
                self.right_arm_controller.go_to_pose(retreat_pos, rotation_matrix_to_quaternion(_handoff_R(self.base_xyz[1])))

        self.get_logger().info('Grasp sequence complete')


def main():
    rclpy.init()
    executor = SingleThreadedExecutor()
    graspNode = GraspNode(executor)
    executor.add_node(graspNode)
    executor.spin()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
