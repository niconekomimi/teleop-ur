from glob import glob
from setuptools import find_packages, setup

package_name = "teleop_control_py"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/teleop_control_py"]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "launch/teleop_control.launch.py",
            "launch/control_system.launch.py",
        ]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rvl",
    maintainer_email="rvl@example.com",
    description="MediaPipe hand teleoperation for UR5 and SoftHand.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "teleop_control_node = teleop_control_py.nodes.teleop_control_node:main",
            "robot_commander_node = teleop_control_py.nodes.robot_commander_node:main",
            "servo_pose_follower = teleop_control_py.hardware.servo_pose_follower:main",
            "data_collector_node = teleop_control_py.nodes.data_collector_node:main",
            "teleop_gui = teleop_control_py.gui.app:main",
        ],
    },
)
