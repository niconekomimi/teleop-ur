# ROS 2 基础指令说明

## 环境设置
在使用 ROS 2 之前，需要确保已经正确设置了工作空间环境。
```bash
source /opt/ros/<ros2_distro>/setup.bash
source ~/ros2_ws/install/setup.bash
```
将 `<ros2_distro>` 替换为你的 ROS 2 发行版名称，例如 `humble`。

## 常用指令

### 1. 检查节点
列出当前运行的所有 ROS 2 节点：
```bash
ros2 node list
```

### 2. 检查话题
列出当前可用的话题：
```bash
ros2 topic list
```
查看某个话题的消息类型：
```bash
ros2 topic info <topic_name>
```
将 `<topic_name>` 替换为具体的话题名称。

### 3. 发布消息
向某个话题发布消息：
```bash
ros2 topic pub <topic_name> <message_type> "{<message_content>}"
```
例如：
```bash
ros2 topic pub /example_topic std_msgs/String "{data: 'Hello, ROS 2!'}"
```

### 4. 检查服务
列出当前可用的服务：
```bash
ros2 service list
```
查看某个服务的类型：
```bash
ros2 service type <service_name>
```
调用服务：
```bash
ros2 service call <service_name> <service_type> "{<service_request>}"
```
例如：
```bash
ros2 service call /example_service example_interfaces/srv/AddTwoInts "{a: 2, b: 3}"
```

### 5. 检查动作
列出当前可用的动作：
```bash
ros2 action list
```
查看某个动作的类型：
```bash
ros2 action info <action_name>
```

### 6. 启动Launch文件
运行一个 launch 文件：
```bash
ros2 launch <package_name> <launch_file_name>
```
例如：
```bash
ros2 launch my_package example_launch.py
```

### 7. 编译工作空间
在工作空间根目录下运行以下命令：
```bash
colcon build
```
清理之前的编译：
```bash
colcon build --clean
```

### 8. 检查日志
查看 ROS 2 日志：
```bash
ros2 bag info <bag_file>
```

## 包管理

### 1. 创建包
使用以下命令创建一个新的 ROS 2 包：
```bash
ros2 pkg create <package_name> --build-type <build_type> --dependencies <dependency1> <dependency2>
```
- `<package_name>`：包的名称。
- `<build_type>`：构建类型，例如 `ament_cmake` 或 `ament_python`。
- `<dependency>`：包的依赖项，可以指定多个。

例如：
```bash
ros2 pkg create my_package --build-type ament_cmake --dependencies rclcpp std_msgs
```

### 2. 删除包
删除包时，只需从 `src` 目录中移除对应的包文件夹：
```bash
rm -rf ~/ros2_ws/src/<package_name>
```
然后清理工作空间并重新编译：
```bash
colcon build
```

## 常见问题排查

1. **节点无法启动**
   - 确保已经正确设置了环境变量。
   - 检查节点依赖是否已安装。

2. **无法找到话题/服务**
   - 确保相关节点已启动。
   - 使用 `ros2 node list` 检查节点是否在运行。

3. **编译失败**
   - 检查 `colcon` 输出的错误信息。
   - 确保所有依赖已安装。

## 参考
- [ROS 2 官方文档](https://docs.ros.org/en/)
- [ROS 2 教程](https://index.ros.org/doc/ros2/Tutorials/)