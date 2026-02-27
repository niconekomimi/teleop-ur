from setuptools import setup

package_name = "multi_joy_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/joy_driver_params.yaml"]),
        ("share/" + package_name + "/launch", ["launch/joy_driver.launch.py"]),
    ],
    install_requires=["setuptools", "evdev"],
    zip_safe=True,
    maintainer="rvl",
    maintainer_email="rvl@example.com",
    description="Extensible ROS2 joystick driver publishing /joy for Xbox/PS5 and more.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "joy_driver_node = multi_joy_driver.joy_driver_node:main",
        ],
    },
)
