#!/usr/bin/env python3
"""
Mirror Gazebo block poses into MoveIt's planning scene as collision objects so
both move_group instances plan around the cubes. Subscribes to /gz_world_poses
and publishes PlanningScene diffs to /left/planning_scene and
/right/planning_scene.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneWorld

_BLOCK_NAMES = ('blue_box', 'red_box', 'green_box')
_BLOCK_SIDE = 0.055  # matches lab_scene.world cube extent


class PlanningSceneBlocks(Node):
    def __init__(self):
        super().__init__('planning_scene_blocks')

        self.declare_parameter('publish_period_sec', 0.5)
        self.declare_parameter('planning_frame', 'world')
        self.declare_parameter('namespaces', ['left', 'right'])
        self.period = self.get_parameter('publish_period_sec').value
        self.frame  = self.get_parameter('planning_frame').value
        namespaces  = self.get_parameter('namespaces').value

        # Latched-style QoS so move_group sees the latest scene even if it
        # subscribes after we publish.
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self._scene_pubs = [
            self.create_publisher(PlanningScene, f'/{ns}/planning_scene', qos)
            for ns in namespaces
        ]

        self.block_poses: dict[str, Pose] = {}
        self.create_subscription(TFMessage, '/gz_world_poses', self._on_poses, 10)
        self.create_timer(self.period, self._publish_scene)

        self.get_logger().info(
            f'planning_scene_blocks ready — frame={self.frame}, '
            f'targets={list(namespaces)}')

    def _on_poses(self, msg: TFMessage):
        for tf in msg.transforms:
            for name in _BLOCK_NAMES:
                if name in tf.child_frame_id:
                    p = Pose()
                    p.position.x = tf.transform.translation.x
                    p.position.y = tf.transform.translation.y
                    p.position.z = tf.transform.translation.z
                    p.orientation = tf.transform.rotation
                    self.block_poses[name] = p

    def _make_collision_object(self, name: str, pose: Pose) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = self.frame
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = name

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [_BLOCK_SIDE, _BLOCK_SIDE, _BLOCK_SIDE]

        co.primitives = [prim]
        co.primitive_poses = [pose]
        co.operation = CollisionObject.ADD
        return co

    def _publish_scene(self):
        if not self.block_poses:
            return

        scene = PlanningScene()
        scene.is_diff = True
        scene.world = PlanningSceneWorld()
        scene.world.collision_objects = [
            self._make_collision_object(name, pose)
            for name, pose in self.block_poses.items()
        ]

        for pub in self._scene_pubs:
            pub.publish(scene)


def main():
    rclpy.init()
    rclpy.spin(PlanningSceneBlocks())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
