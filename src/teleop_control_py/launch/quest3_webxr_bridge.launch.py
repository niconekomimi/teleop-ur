#!/usr/bin/env python3
"""Launch the Quest 3 WebXR bridge."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_bridge(context, *args, **kwargs):
    parameters = [LaunchConfiguration("params_file")]
    advertised_host = LaunchConfiguration("advertised_host").perform(context).strip()
    if advertised_host:
        parameters.append({"advertised_host": advertised_host})

    return [
        Node(
            package="teleop_control_py",
            executable="quest3_webxr_bridge_node",
            name="quest3_webxr_bridge_node",
            output="screen",
            parameters=parameters,
        )
    ]


def generate_launch_description() -> LaunchDescription:
    teleop_share = get_package_share_directory("teleop_control_py")
    default_params = os.path.join(teleop_share, "config", "quest3_webxr_bridge_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to quest3 WebXR bridge parameter file",
    )
    advertised_host_arg = DeclareLaunchArgument(
        "advertised_host",
        default_value="",
        description="Optional LAN host/IP override printed for Quest direct access.",
    )

    return LaunchDescription([params_file_arg, advertised_host_arg, OpaqueFunction(function=_launch_bridge)])
