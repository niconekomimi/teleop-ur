# #!/usr/bin/env python3
# from launch import LaunchDescription
# from launch import conditions
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import Node
# import os
# from ament_index_python.packages import get_package_share_directory


# def generate_launch_description():
#     # 声明所有可配置的参数
#     ur_type_arg = DeclareLaunchArgument(
#         "ur_type",
#         default_value="ur5",
#         description="UR robot type (ur5, ur10, etc.)",
#     )
    
#     robot_ip_arg = DeclareLaunchArgument(
#         "robot_ip",
#         default_value="192.168.1.211",
#         description="UR robot IP address",
#     )
    
#     reverse_ip_arg = DeclareLaunchArgument(
#         "reverse_ip",
#         default_value="192.168.1.10",
#         description="Reverse connection IP address",
#     )
    
#     launch_rviz_arg = DeclareLaunchArgument(
#         "launch_rviz",
#         default_value="false",
#         description="Launch RViz for visualization",
#     )

#     enable_camera_arg = DeclareLaunchArgument(
#         "enable_camera",
#         default_value="true",
#         description="Enable RealSense camera (set to false if USB issues persist)",
#     )

#     # 1. SoftHand (夹爪) 节点
#     softhand_node = Node(
#         package="qbsofthand_control",
#         executable="qbsofthand_control_node",
#         name="qbsofthand_control_node",
#         output="screen",
#     )

#     # 2. UR 机械臂驱动 Launch
#     ur_driver_share = get_package_share_directory("ur_robot_driver")
#     ur_launch = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(ur_driver_share, "launch", "ur_control.launch.py")
#         ),
#         launch_arguments={
#             "ur_type": LaunchConfiguration("ur_type"),
#             "robot_ip": LaunchConfiguration("robot_ip"),
#             "reverse_ip": LaunchConfiguration("reverse_ip"),
#             "launch_rviz": LaunchConfiguration("launch_rviz"),
#         }.items(),
#     )

#     # 3. RealSense 相机 Launch（失败时不阻止其他节点启动）
#     # 可以通过 enable_camera:=false 禁用相机
#     realsense_share = get_package_share_directory("realsense2_camera")
#     realsense_launch = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(realsense_share, "launch", "rs_launch.py")
#         ),
#         launch_arguments={
#             "initial_reset": "true",  # 启动时重置相机
#             "enable_color": "true",
#             "enable_depth": "true",
#             "enable_infra1": "false",
#             "enable_infra2": "false",
#             "depth_module.depth_profile": "848x480x30",
#             "rgb_camera.color_profile": "640x480x30",
#             "depth_module.depth_format_preference": "[Z16]",
#         }.items(),
#         # 可选：当 enable_camera=false 时不启动
#         condition=launch.conditions.IfCondition(LaunchConfiguration("enable_camera")),
#     )

#     # 4. Teleop 控制节点 Launch
#     teleop_share = get_package_share_directory("teleop_control_py")
#     teleop_launch = IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             os.path.join(teleop_share, "launch", "teleop_control.launch.py")
#         ),
#     )

#     return LaunchDescription(
#         [
#             ur_type_arg,
#             robot_ip_arg,
#             reverse_ip_arg,
#             launch_rviz_arg,
#             enable_camera_arg,
#             softhand_node,
#             ur_launch,
#             realsense_launch,
#             teleop_launch,
#         ]
#     )
