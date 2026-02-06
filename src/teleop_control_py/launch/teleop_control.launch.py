#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("teleop_control_py")
    default_params = os.path.join(pkg_share, "config", "teleop_params.yaml")
    default_python = os.path.expanduser("~/clds/bin/python3")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to teleop parameter file",
    )

    python_exec_arg = DeclareLaunchArgument(
        "python_executable",
        default_value=default_python,
        description="Python interpreter to run teleop_control_node",
    )

    return LaunchDescription(
        [
            params_file_arg,
            python_exec_arg,
            Node(
                package="teleop_control_py",
                executable=LaunchConfiguration("python_executable"),
                arguments=["-m", "teleop_control_py.teleop_control_node"],
                name="teleop_control_node",
                output="screen",
                parameters=[LaunchConfiguration("params_file")],
            ),
        ]
    )
