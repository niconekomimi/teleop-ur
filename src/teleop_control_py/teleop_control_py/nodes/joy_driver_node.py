#!/usr/bin/env python3
"""Joystick driver node publishing /joy (sensor_msgs/Joy) for Xbox, PS5, generic gamepads."""

from __future__ import annotations

import sys
import threading
import time
from typing import List, Optional, Tuple

import rclpy
from evdev import InputDevice, categorize, ecodes, list_devices
from rclpy.node import Node
from sensor_msgs.msg import Joy

from ..hardware.input.joy_device_profiles import (
    JoyProfile,
    build_profiles,
    choose_profile_and_device,
    normalize_axis,
)


class JoyDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("joy_driver_node")

        self.declare_parameter("profile", "auto")
        self.declare_parameter("device_path", "")
        self.declare_parameter("device_name", "")
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("deadzone", 0.05)
        self.declare_parameter("autoreconnect", True)
        self.declare_parameter("scan_interval_sec", 1.0)

        self.requested_profile = self.get_parameter("profile").value
        self.requested_device_path = self.get_parameter("device_path").value
        self.requested_device_name = self.get_parameter("device_name").value
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.deadzone = float(self.get_parameter("deadzone").value)
        self.autoreconnect = bool(self.get_parameter("autoreconnect").value)
        self.scan_interval_sec = float(self.get_parameter("scan_interval_sec").value)

        self.profiles = build_profiles(self.deadzone)
        self.profile: JoyProfile = self.profiles["generic"]

        self.device: Optional[InputDevice] = None
        self.axes: List[float] = [0.0] * self.profile.axis_count
        self.buttons: List[int] = [0] * self.profile.button_count

        self.joy_pub = self.create_publisher(Joy, "/joy", 20)

        timer_period = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0.0 else 0.01
        self.pub_timer = self.create_timer(timer_period, self._publish_joy)

        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        self.get_logger().info(f"Python executable: {sys.executable}")
        self.get_logger().info("joy_driver_node started, publishing to /joy")

    def _list_available_devices(self) -> List[Tuple[str, str]]:
        devices: List[Tuple[str, str]] = []
        for path in list_devices():
            try:
                dev = InputDevice(path)
                devices.append((path, dev.name))
                dev.close()
            except Exception:
                continue
        return devices

    def _try_connect(self) -> bool:
        candidates = self._list_available_devices()

        chosen_path = None
        chosen_name = None
        chosen_profile = None

        if self.requested_device_path:
            try:
                dev = InputDevice(self.requested_device_path)
                chosen_path = self.requested_device_path
                chosen_name = dev.name
                key = self.requested_profile if self.requested_profile != "auto" else "generic"
                chosen_profile = self.profiles.get(key, self.profiles["generic"])
                dev.close()
            except Exception as exc:
                self.get_logger().warning(
                    f"Configured device_path not available: {self.requested_device_path} ({exc})"
                )
                return False
        else:
            chosen_path, chosen_name, chosen_profile = choose_profile_and_device(
                available=candidates,
                requested_profile=self.requested_profile,
                requested_name=self.requested_device_name,
                profiles=self.profiles,
            )

        if not chosen_path or not chosen_profile:
            return False

        try:
            device = InputDevice(chosen_path)
            device.grab()

            with self.state_lock:
                self.device = device
                self.profile = chosen_profile
                self.axes = [0.0] * self.profile.axis_count
                self.buttons = [0] * self.profile.button_count

            self.get_logger().info(
                f"Connected: '{chosen_name}' ({chosen_path}), profile={self.profile.name}"
            )
            return True
        except Exception as exc:
            self.get_logger().warning(f"Failed to open {chosen_path}: {exc}")
            return False

    def _disconnect(self) -> None:
        with self.state_lock:
            if self.device is not None:
                try:
                    self.device.ungrab()
                except Exception:
                    pass
                try:
                    self.device.close()
                except Exception:
                    pass
            self.device = None

    def _reader_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.device is None:
                connected = self._try_connect()
                if not connected:
                    time.sleep(max(0.2, self.scan_interval_sec))
                    continue

            current = self.device
            if current is None:
                time.sleep(0.1)
                continue

            try:
                for event in current.read_loop():
                    if self.stop_event.is_set():
                        return
                    if event.type == ecodes.EV_ABS:
                        abs_event = categorize(event)
                        self._handle_axis(event.code, abs_event.event.value)
                    elif event.type == ecodes.EV_KEY:
                        self._handle_button(event.code, event.value)
            except OSError:
                self.get_logger().warning("Joystick disconnected, waiting for reconnect...")
                self._disconnect()
                if not self.autoreconnect:
                    return
                time.sleep(max(0.2, self.scan_interval_sec))
            except Exception as exc:
                self.get_logger().error(f"Reader loop error: {exc}")
                self._disconnect()
                time.sleep(max(0.2, self.scan_interval_sec))

    def _handle_axis(self, code: int, value: int) -> None:
        spec = self.profile.axis_specs.get(code)
        if spec is None:
            return

        normalized = normalize_axis(value, spec)
        with self.state_lock:
            if spec.index >= len(self.axes):
                return
            self.axes[spec.index] = normalized

    def _handle_button(self, code: int, value: int) -> None:
        btn_index = self.profile.button_indices.get(code)
        if btn_index is None:
            return

        with self.state_lock:
            if btn_index >= len(self.buttons):
                return
            self.buttons[btn_index] = 1 if value else 0

    def _publish_joy(self) -> None:
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()

        with self.state_lock:
            msg.axes = list(self.axes)
            msg.buttons = list(self.buttons)

        self.joy_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.stop_event.set()
        self._disconnect()
        if self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JoyDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass  # Context may already be shut down by launch


if __name__ == "__main__":
    main()
