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
import yaml

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch import LaunchContext, LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import  LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node

def generate_robot_state_publisher(context: LaunchContext, namespace, start_coords: tuple[float]):
    load_gripper_str = 'true'
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
            'ee_id': "franka_hand",
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
            {'robot_description': robot_description_config.toxml(),
             'use_sim_time': True},
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

    # xacro files 
    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots',
        "fr3",
        'fr3.urdf.xacro'
    )

    franka_semantic_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        "robots", 
        "fr3", 
        "fr3.srdf.xacro"
    )

    # moveIt planning configuration
    kinematics_yaml_content = load_yaml(
        'franka_fr3_moveit_config', 'config/kinematics.yaml'
    )

    moveit_simple_controllers_yaml = load_yaml(
        'franka_fr3_moveit_config', 'config/fr3_controllers.yaml'
    )
    moveit_controllers_config = {
        'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
        'moveit_controller_manager': 'moveit_simple_controller_manager'
                                     '/MoveItSimpleControllerManager',
    }

    trajectory_execution_config = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    ompl_planning_pipeline_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_planning_yaml = load_yaml(
        'franka_fr3_moveit_config', 'config/ompl_planning.yaml'
    )
    ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)

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

    # Gazebo <-> ROS bridge: sim clock + per-model world poses (consumed by
    # grasp_node on /gz_world_poses).
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/default/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        remappings=[('/world/default/pose/info', '/gz_world_poses')],
        output='screen',
    )

    # Controller config files
    grasp_cfg_dir = os.path.join(
        get_package_share_directory('cs593'), 'config', 'grasp')
    left_arm_ctrl_yaml     = os.path.join(grasp_cfg_dir, 'left_arm_controller.yaml')
    left_gripper_ctrl_yaml = os.path.join(grasp_cfg_dir, 'left_gripper_controller.yaml')
    right_arm_ctrl_yaml     = os.path.join(grasp_cfg_dir, 'right_arm_controller.yaml')
    right_gripper_ctrl_yaml = os.path.join(grasp_cfg_dir, 'right_gripper_controller.yaml')

    # RVIZ
    rviz_file = os.path.join(get_package_share_directory('cs593'), 'config', 'lab_scene.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        namespace="",
        arguments=['--display-config', rviz_file, '-f', 'world'],
        parameters=[{'use_sim_time': True}],
    )

    # left arm
    left_robot_state_publisher = OpaqueFunction(
        function=generate_robot_state_publisher,
        args=[left_namespace, (0, 0.5, 0)]
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

    left_xacro_file_content = ParameterValue(
        Command([
            "xacro ", franka_xacro_file,
            " hand:=true",
            " ros2_control:=true",
            " gazebo:=true",
            " ee_id:=franka_hand",
            " arm_prefix:=", left_namespace,
            " xyz:=", f"\"{0} {0.5} {0}\""
        ]),
        value_type=str
    )

    left_srdf_xacro_content = ParameterValue(
        Command([
            "xacro ", franka_semantic_xacro_file,
            " hand:=true",
            " ee_id:=franka_hand"
            " arm_prefix:=", left_namespace, "_"
        ]),
        value_type=str
    )

    left_arm_controller_spawner_node = Node(
        package="controller_manager",
        executable="spawner",
        namespace=left_namespace,
        arguments=[
            "left_fr3_arm_controller", 
            "-c", "controller_manager" 
        ],
    )

    left_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        namespace=left_namespace,
        output='screen',
        parameters=[
            {
                "robot_description": left_xacro_file_content
            },
            {
                "robot_description_semantic": left_srdf_xacro_content
            },
            kinematics_yaml_content,
            trajectory_execution_config,
            moveit_controllers_config,
            planning_scene_monitor_parameters,
            ompl_planning_pipeline_config,
            {'use_sim_time': True}
        ],
    )

    left_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_fr3_arm_controller",
                   "--controller-manager", [left_namespace, "/controller_manager"],
                   "--param-file", left_arm_ctrl_yaml,
        ],
        output="screen",
    )

    left_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_fr3_gripper_controller",
                   "--controller-manager", [left_namespace, "/controller_manager"],
                   "--param-file", left_gripper_ctrl_yaml,
        ],
        output="screen",
    )

    # right arm
    right_robot_state_publisher = OpaqueFunction(
        function=generate_robot_state_publisher,
        args=[right_namespace, (0, -0.5, 0)]
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

    right_xacro_file_content = ParameterValue(
        Command([
            "xacro ", franka_xacro_file,
            " hand:=true",
            " ros2_control:=true",
            " gazebo:=true",
            " ee_id:=franka_hand",
            " arm_prefix:=", right_namespace,
            " xyz:=", f"\"{0} {-0.5} {0}\""
        ]),
        value_type=str
    )

    right_srdf_xacro_content = ParameterValue(
        Command([
            "xacro ", franka_semantic_xacro_file,
            " hand:=true",
            " ee_id:=franka_hand"
            " arm_prefix:=", right_namespace, "_",
        ]),
        value_type=str
    )

    right_arm_controller_spawner_node = Node(
        package="controller_manager",
        executable="spawner",
        namespace=right_namespace,
        arguments=[
            "right_fr3_arm_controller", 
            "-c", "controller_manager" 
        ],
    )

    right_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        namespace=right_namespace,
        output='screen',
        parameters=[
            {
                "robot_description": right_xacro_file_content
            },
            {
                "robot_description_semantic": right_srdf_xacro_content
            },
            kinematics_yaml_content,
            trajectory_execution_config,
            moveit_controllers_config,
            planning_scene_monitor_parameters,
            ompl_planning_pipeline_config,
            {'use_sim_time': True}
        ],
    )
    right_arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_fr3_arm_controller",
                   "--controller-manager", [right_namespace, "/controller_manager"],
                   "--param-file", right_arm_ctrl_yaml,
        ],
        output="screen",
    )

    right_gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_fr3_gripper_controller",
                   "--controller-manager", [right_namespace, "/controller_manager"],
                   "--param-file", right_gripper_ctrl_yaml,
        ],
        output="screen",
    )


    return LaunchDescription([
        load_gripper_launch_argument,
        franka_hand_launch_argument,

        namespace_launch_argument,
        namespace2_launch_argument,

        gazebo_launch,
        gz_bridge,
        rviz_node,

        left_robot_state_publisher,
        left_spawn_node,
        left_move_group_node,
        left_arm_controller_spawner_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=left_spawn_node,
                on_exit=[left_joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=left_joint_state_broadcaster_spawner,
                on_exit=[left_arm_controller_spawner, left_gripper_controller_spawner],
            )
        ),

        right_robot_state_publisher,
        right_spawn_node,
        right_move_group_node,
        right_arm_controller_spawner_node,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=right_spawn_node,
                on_exit=[right_joint_state_broadcaster_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=right_joint_state_broadcaster_spawner,
                on_exit=[right_arm_controller_spawner, right_gripper_controller_spawner],
            )
        ),

    ])

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError: 
        return None