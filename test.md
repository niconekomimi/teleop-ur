# realsence相机
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true filters:=spatial,temporal

# 手
ros2 run qbsofthand_control qbsofthand_control_node

# ur5
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.211 launch_rviz:=false reverse_ip:=192.168.1.10


# moveit + servo节点（ur_moveit.launch.py 里默认 launch_servo:=true，会启动 moveit_servo/servo_node_main）
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_rviz:=false launch_servo:=true

添加命令激活
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger "{}"

# 重要：让 Servo 真正驱动机械臂，需要切到 forward_position_controller
# （默认通常是 scaled_joint_trajectory_controller 在 active，Servo 的输出话题是 /forward_position_controller/commands）
# 先看当前控制器状态：
# ros2 control list_controllers
# 切换控制器（注意 STRICT 下不要尝试 deactivate 已经是 inactive 的控制器，否则可能直接失败）：
# ros2 control switch_controllers --activate forward_position_controller --deactivate scaled_joint_trajectory_controller --strict
# 如果还是失败，换 best-effort：
# ros2 control switch_controllers --activate forward_position_controller --deactivate scaled_joint_trajectory_controller --best-effort

ros2 run teleop_control_py servo_pose_follower --ros-args --params-file /home/rvl/collect_datasets_ws/src/teleop_control_py/config/teleop_params.yaml

# 控制包（teleop节点本身会自动加载其参数文件）
ros2 launch teleop_control_py teleop_control.launch.py

# 诊断（机械臂不动时先看这些）
# 1) Servo 是否在跑/是否在发输出
# ros2 topic echo /servo_node/status -n 1
# ros2 topic hz /servo_node/delta_twist_cmds
# ros2 topic info /forward_position_controller/commands
# 2) 是否有 /target_pose 在更新
# ros2 topic hz /target_pose
# 3) base_link <-> base 是否存在固定变换（你说的180°翻转）
# ros2 run tf2_ros tf2_echo base_link base

# 测试相机

