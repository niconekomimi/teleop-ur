# 遥操作（Teleop）

## 启动

```bash
# RealSense（用于手部追踪 可选：用 disparity/spatial/temporal 滤波）
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true filters:=disparity,spatial,temporal

# UR5 驱动
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.211 launch_rviz:=false reverse_ip:=192.168.1.10

# MoveIt Servo（teleop 依赖 servo）
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_rviz:=false launch_servo:=true

# 软手（如果你用 qbsofthand）
ros2 run qbsofthand_control qbsofthand_control_node

# 手柄控制（如用）
ros2 launch multi_joy_driver joy_driver.launch.py

# teleop（手势模式）
ros2 launch teleop_control_py teleop_control.launch.py

# teleop（Xbox 模式，默认不需要死人键）
ros2 launch teleop_control_py teleop_control.launch.py start_joy_node:=true
# 并在 src/teleop_control_py/config/teleop_params.yaml 里设置：
# control_mode: "xbox"
# xbox_require_deadman: false
# 默认映射：左摇杆前后左右，右摇杆旋转，LT上升，RT下降，A/B控制夹爪
```


# 录制（DataCollector / LIBERO HDF5）

## 1) 编译 + source

```bash
cd ~/collect_datasets_ws
# 注意：用当前 venv 的 python 驱动 colcon，保证 ros2 可执行脚本使用 clds 解释器
python3 -m colcon build --packages-select teleop_control_py --symlink-install
source install/setup.bash
```

## 2) 启动相机 + 机器人（按需）

```bash
# RealSense
ros2 launch realsense2_camera rs_launch.py

# OAK-D
ros2 launch depthai_examples rgb_stereo_node.launch.py

# UR5 驱动（如果还没启动）
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=192.168.1.211 launch_rviz:=false reverse_ip:=192.168.1.10
```

## 3) 运行 DataCollectorNode（用 YAML 参数）

先配置：`src/teleop_control_py/config/data_collector_params.yaml`
（相机映射、夹爪类型、home_joint_positions、输出路径）

```bash
ros2 run teleop_control_py data_collector_node --ros-args --params-file src/teleop_control_py/config/data_collector_params.yaml
```

## 4) 开始/停止录制 + go_home

```bash
# 开始
ros2 service call /data_collector/start std_srvs/srv/Trigger {}
# 停止
ros2 service call /data_collector/stop  std_srvs/srv/Trigger {}
# 回home点
ros2 service call /data_collector/go_home std_srvs/srv/Trigger {}

```

## 5) 校验 HDF5 结构

```bash
python3 - <<'PY'
import h5py

path = '/home/rvl/collect_datasets_ws/data/libero_demos.hdf5'
with h5py.File(path, 'r') as f:
    assert 'data' in f, 'missing /data'
    demos = sorted(f['data'].keys())
    print('demos:', demos)
    if not demos:
        raise SystemExit('no demos recorded')

    g = f['data'][demos[0]]
    print('num_samples:', g.attrs.get('num_samples', None))
    print('actions', g['actions'].shape, g['actions'].dtype)

    obs = g['obs']
    for k in ['agentview_rgb','eye_in_hand_rgb','robot0_joint_pos','robot0_eef_pos','robot0_eef_quat']:
        print(k, obs[k].shape, obs[k].dtype)

    n = g['actions'].shape[0]
    assert obs['agentview_rgb'].shape[0] == n
    assert obs['eye_in_hand_rgb'].shape[0] == n
    assert obs['robot0_joint_pos'].shape[0] == n
    assert obs['robot0_eef_pos'].shape[0] == n
    assert obs['robot0_eef_quat'].shape[0] == n
    print('OK, N =', n)
PY
```

## 6) 最少排错

- `ModuleNotFoundError: No module named 'h5py'`：`python3 -m pip install h5py`
- 一直 N=0：先看 DataCollectorNode 的 stats 日志（会提示 `no_pose / no_gripper / joint_map_fail` 等）
    - pose：`ros2 topic hz /tcp_pose_broadcaster/pose`（或你 YAML 里配置的 `tool_pose_topic`）
    - joint：`ros2 topic echo /joint_states --once | sed -n 's/^name: //p'`（确认包含 YAML 的 `joint_names`）
    - gripper：如果是 qbsofthand 且没有持续状态话题，建议在 YAML 里 `require_gripper: false`
- `queue full; dropping sample`：调大 YAML 里的 `queue_maxsize` 或 `writer_batch_size`，或把 `image_compression` 改成 `none`
