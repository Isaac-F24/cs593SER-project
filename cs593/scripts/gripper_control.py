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
from rclpy.executors import SingleThreadedExecutor



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


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class GripperController():
    def __init__(self, ros_node: Node = None):

        if ros_node is None:
            self.node = Node("gripper_node")
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.executor.spin)
            self.thread.start()
            self.using_executor = True
        else:
            self.node = ros_node
        

        self.left_gripper_joints = [
            f'left_fr3_finger_joint1',
            f'left_fr3_finger_joint2',
        ]

        self.right_gripper_joints = [
            f'right_fr3_finger_joint1',
            f'right_fr3_finger_joint2',
        ]

        self.node.create_subscription(JointState, f'/left/joint_states', self._on_left_joint_states, 10)
        self.node.create_subscription(JointState, f'/right/joint_states', self._on_right_joint_states, 10)

        self.left_gripper_client = ActionClient(self.node, FollowJointTrajectory,f'/left/left_fr3_gripper_controller/follow_joint_trajectory')
        self.right_gripper_client = ActionClient(self.node, FollowJointTrajectory,f'/right/right_fr3_gripper_controller/follow_joint_trajectory')

        self._left_finger_pos = None
        self._right_finger_pos = None

    def cleanup(self):
        if (self.using_executor):
            self.executor.shutdown()
            if self.thread.is_alive():
                self.thread.join()
            self.node.destroy_node()

    # ------------------------------------------------------------------

    def _on_left_joint_states(self, msg: JointState):
        joint_name = f'left_fr3_finger_joint1'
        try:
            idx = msg.name.index(joint_name)
            self._left_finger_pos = msg.position[idx]
        except ValueError:
            pass

    def _on_right_joint_states(self, msg: JointState):
        joint_name = f'right_fr3_finger_joint1'
        try:
            idx = msg.name.index(joint_name)
            self._right_finger_pos = msg.position[idx]
        except ValueError:
            pass

    def _wait_gripper(self, target: float, tol: float = 0.003, timeout: float = 3.0, left: bool=True) -> bool:
        """
        Block until finger_joint1 reaches target ± tol, or timeout.
        left- true = left gripper, false=right gripper
        returns false if it times out, else true
        """

        deadline = time.time() + timeout
        while time.time() < deadline:
            if left:
                pos = self._left_finger_pos
            else:
                pos = self._right_finger_pos

            if pos is not None and abs(pos - target) <= tol:
                return True
            
            time.sleep(0.05)

        if left:
            pos = self._left_finger_pos
        else:
            pos = self._right_finger_pos
        self.node.get_logger().warn(
            f'Gripper did not reach {target:.3f} m within {timeout:.1f} s '
            f'(last read: {pos})')
        
        return False

    def _send_gripper(self, width: float, duration_sec: float = 1.5,
                      blocking: bool = True, left: bool=True) -> bool:
        """
        Send a gripper position goal.
        blocking=True  — wait until the trajectory completes (use for close).
        blocking=False — fire-and-forget with a long duration so the controller
                         stays in active tracking mode rather than passive hold
                         (use for open-and-keep-open during arm motion).

        left- true = left gripper, false=right gripper
        returns whether it was able to send a gripper command
        """

        if left:
            client = self.left_gripper_client
            joint_names = self.left_gripper_joints
        else:
            client = self.right_gripper_client
            joint_names = self.right_gripper_joints

        if not client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error('Gripper action server unavailable')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(width)] * len(joint_names)
        pt.velocities = [0.0] * len(joint_names)
        pt.time_from_start = builtin_interfaces.msg.Duration(sec=int(duration_sec))
        goal.trajectory.points = [pt]

        done = threading.Event()

        def _on_result(fut): 
            done.set()

        def _on_goal(fut):
            handle = fut.result()
            if not handle.accepted:
                self.node.get_logger().error('Gripper goal rejected')
                done.set()
                return
            handle.get_result_async().add_done_callback(_on_result)

        client.send_goal_async(goal).add_done_callback(_on_goal)
        if blocking:
            done.wait()
        return True

    # ------------------------------------------------------------------
    def open_gripper(self, left: bool=True):
        """
        Open the gripper, and block until it is open
        left- true = left gripper, false=right gripper
        """

        success = self._send_gripper(0.034, duration_sec=2.0, blocking=True, left=left)
        if not success:
            return False
        
        return self._wait_gripper(0.034, tol=0.003, timeout=3.0, left=left)
    
    def open_gripper_async(self, left: bool=True):
        """
        Send the command to open the gripper
        left- true = left gripper, false=right gripper
        """

        success = self._send_gripper(0.034, duration_sec=2.0, blocking=False, left=left)

        return success

    def close_gripper(self, left: bool=True, close_goal: float=0.005):
        """
        Close the gripper, and block until it is closed
        left- true = left gripper, false=right gripper
        close goal- what the gripper should close to (this value will be higher if it's around a block)
        """
        success = self._send_gripper(0.005, duration_sec=2.5, blocking=True, left=left)
        if not success:
            return False
        
        return self._wait_gripper(0.005, tol=0.003, timeout=3.0, left=left)



def main():
    # example usage
    rclpy.init()

    controller = GripperController()

    success = controller.open_gripper(left=True)
    if not success: print("Failed!")

    success = controller.open_gripper(left=False)
    if not success: print("Failed!")

    success = controller.close_gripper(left=False)
    if not success: print("Failed!")

    success = controller.close_gripper(left=True)
    if not success: print("Failed!")

    controller.cleanup()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
