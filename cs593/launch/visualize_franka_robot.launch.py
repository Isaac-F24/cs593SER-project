# Copyright (c) 2024 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch import LaunchContext, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import  LaunchConfiguration
from launch_ros.actions import Node

def generate_robot_state_publisher(context: LaunchContext, namespace, should_load_gripper, franka_hand, start_coords: tuple[float]):
    load_gripper_str = context.perform_substitution(should_load_gripper)
    franka_hand_str = context.perform_substitution(franka_hand)
    namespace = context.perform_substitution(namespace)

    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        "fr3",
        'fr3.urdf.xacro'
    )

    robot_description_config = xacro.process_file(
        franka_xacro_file,
        mappings={
            'hand': load_gripper_str,
            'ros2_control': 'true',
            'gazebo': 'true',
            'ee_id': franka_hand_str,
            "arm_prefix": namespace,
            "xyz": f"{start_coords[0]} {start_coords[1]} {start_coords[2]}"
        }
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=namespace,
        output='both',
        parameters=[
            {'robot_description': robot_description_config.toxml()},
        ]
    )

    return [robot_state_publisher]


def generate_launch_description():
    load_gripper_name = 'load_gripper'
    franka_hand_name = 'franka_hand'

    # Launch args/configurations
    should_load_gripper = LaunchConfiguration(load_gripper_name)
    franka_hand = LaunchConfiguration(franka_hand_name)
    left_namespace = LaunchConfiguration("left_namespace")
    right_namespace = LaunchConfiguration("right_namespace")

    load_gripper_launch_argument = DeclareLaunchArgument(
        load_gripper_name,
        default_value='true',
        description='true/false for activating the gripper'
    )
    franka_hand_launch_argument = DeclareLaunchArgument(
        franka_hand_name,
        default_value='franka_hand',
        description='Default value: franka_hand'
    )
    namespace_launch_argument = DeclareLaunchArgument(
        "left_namespace",
        default_value='left',
        description='Namespace for the robot. If not set, the robot will be launched in the root namespace.'
    )
    namespace2_launch_argument = DeclareLaunchArgument(
        "right_namespace",
        default_value="right",
        description="Namespace for da other robot"
    )

    # Gazebo Sim
    os.environ['GZ_SIM_RESOURCE_PATH'] = os.path.dirname(get_package_share_directory('franka_description'))
    gazebo_package_dir = get_package_share_directory('ros_gz_sim')
    world_file = os.path.join(get_package_share_directory("cs593"), "worlds", "lab_scene.world")
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_package_dir, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'{world_file} -r', }.items(),
    )

    # RVIZ
    rviz_file = os.path.join(get_package_share_directory('cs593'), 'config', 'lab_scene.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        namespace="",
        arguments=['--display-config', rviz_file, '-f', 'world'],
    )

    # left arm
    left_robot_state_publisher = OpaqueFunction(
        function=generate_robot_state_publisher,
        args=[left_namespace, should_load_gripper, franka_hand, (0, 0.5, 0)]
    )

    left_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=left_namespace,
        parameters=[
            {'source_list': ['/joint_states'],
             'rate': 30}
        ]
    )

    left_spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=left_namespace,
        arguments=['--topic', 'robot_description',
                   '--name', f'{left_namespace}'
        ],
        output='screen',
    )

    left_joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", 
                   "--controller-manager", [left_namespace, "/controller_manager"],
        ],
        output="screen",
    )

    # right arm
    right_robot_state_publisher = OpaqueFunction(
        function=generate_robot_state_publisher,
        args=[right_namespace, should_load_gripper, franka_hand, (0, -0.5, 0)]
    )

    right_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        namespace=right_namespace,
        parameters=[
            {'source_list': ['/joint_states'],
             'rate': 30}
        ]
    )

    right_spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        namespace=right_namespace,
        arguments=['--topic', 'robot_description',
                   '--name', f'{right_namespace}'
        ],
        output='screen',
    )

    right_joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", 
                   "--controller-manager", [right_namespace, "/controller_manager"],
        ],
        output="screen",
    )


    return LaunchDescription([
        load_gripper_launch_argument,
        franka_hand_launch_argument,
        namespace_launch_argument,
        namespace2_launch_argument,

        gazebo_launch,
        rviz_node,

        left_robot_state_publisher,
        left_joint_state_publisher,
        left_spawn_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=left_spawn_node,
                on_exit=[left_joint_state_broadcaster_spawner],
            )
        ),

        right_robot_state_publisher,
        right_joint_state_publisher,
        right_spawn_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=right_spawn_node,
                on_exit=[right_joint_state_broadcaster_spawner],
            )
        ),

    ])
