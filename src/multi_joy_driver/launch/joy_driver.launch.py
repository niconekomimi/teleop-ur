from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    default_params = os.path.join(
        get_package_share_directory("multi_joy_driver"),
        "config",
        "joy_driver_params.yaml",
    )

    profile_arg = DeclareLaunchArgument("profile", default_value="auto")
    device_path_arg = DeclareLaunchArgument("device_path", default_value="")
    device_name_arg = DeclareLaunchArgument("device_name", default_value="")
    params_file_arg = DeclareLaunchArgument("params_file", default_value=default_params)

    venv = os.environ.get("VIRTUAL_ENV", "").strip()
    default_python = os.path.join(venv, "bin", "python3") if venv else "/usr/bin/python3"
    python_exec_arg = DeclareLaunchArgument(
        "python_executable",
        default_value=default_python,
        description="Python executable path, e.g. /home/user/clds/bin/python3",
    )

    node = Node(
        package="multi_joy_driver",
        executable="joy_driver_node",
        name="joy_driver_node",
        output="screen",
        prefix=[LaunchConfiguration("python_executable")],
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "profile": LaunchConfiguration("profile"),
                "device_path": LaunchConfiguration("device_path"),
                "device_name": LaunchConfiguration("device_name"),
            },
        ],
    )

    return LaunchDescription([
        profile_arg,
        device_path_arg,
        device_name_arg,
        params_file_arg,
        python_exec_arg,
        node,
    ])
