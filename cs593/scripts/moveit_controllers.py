#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
import threading

def generateJointNames(prefix: str = "") -> list[str]:
    return [
        prefix + "fr3_joint1",
        prefix + "fr3_joint2",
        prefix + "fr3_joint3",
        prefix + "fr3_joint4",
        prefix + "fr3_joint5",
        prefix + "fr3_joint6",
        prefix + "fr3_joint7",
    ]

class LeftArmMoveitController():
    def __init__(self):
        self.node = Node("left_arm_moveit_node", namespace="left")

        joint_names = generateJointNames("left_")

        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=joint_names,
            base_link_name="left_link0",      
            end_effector_name="left_fr3_hand",
            group_name="left_fr3_arm",
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin)
        self.thread.start()

    def go_to_pose(self, pose, orientation):
        """
        Move to pose and wait for execution to finish

        args:
        position- target position, list of [x,y,z] (in world frame)
        orientation- target orientation, quaternion list of [x,y,z,w]

        Returns
        success- boolean of whether move was successful or not
        """
        self.moveit2.move_to_pose(
            position=pose,
            quat_xyzw=orientation, 
            frame_id="world"
        )

        success = self.moveit2.wait_until_executed()

        return success
    
    def go_to_pose_async(self, pose, orientation):
        """
        Start a move to pose, don't wait for results

        args:
        position- target position, list of [x,y,z] (in world frame)
        orientation- target orientation, quaternion list of [x,y,z,w]
        """
        self.moveit2.move_to_pose(
            position=pose,
            quat_xyzw=orientation, 
            frame_id="world"
        )

    def cleanup(self):
        self.executor.shutdown()
        if self.thread.is_alive():
            self.thread.join()
        self.node.destroy_node()
        
class RightArmMoveitController():
    def __init__(self):
        self.node = Node("right_arm_moveit_node", namespace="right")

        joint_names = generateJointNames("right_")

        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=joint_names,
            base_link_name="right_link0",      
            end_effector_name="right_fr3_hand",
            group_name="right_fr3_arm",
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin)
        self.thread.start()

    def go_to_pose(self, pose, orientation):
        """
        Move to pose and wait for execution to finish

        args:
        position- target position, list of [x,y,z] (in world frame)
        orientation- target orientation, quaternion list of [x,y,z,w]

        Returns
        success- boolean of whether move was successful or not
        """
        self.moveit2.move_to_pose(
            position=pose,
            quat_xyzw=orientation, 
            frame_id="world"
        )

        success = self.moveit2.wait_until_executed()

        return success
    
    def go_to_pose_async(self, pose, orientation):
        """
        Start a move to pose, don't wait for results

        args:
        position- target position, list of [x,y,z] (in world frame)
        orientation- target orientation, quaternion list of [x,y,z,w]
        """
        self.moveit2.move_to_pose(
            position=pose,
            quat_xyzw=orientation, 
            frame_id="world"
        )

    def cleanup(self):
        self.executor.shutdown()
        if self.thread.is_alive():
            self.thread.join()
        self.node.destroy_node()


if __name__ == "__main__":
    # example usage
    
    rclpy.init()

    leftArmController = LeftArmMoveitController()

    rightArmController = RightArmMoveitController()

    success = leftArmController.go_to_pose([0.3,0.5,0.5],[0.5,0.5,0,0])
    if (not success):
        print("Failure!")
#
    leftArmController.cleanup()
    rightArmController.cleanup()

    rclpy.shutdown()