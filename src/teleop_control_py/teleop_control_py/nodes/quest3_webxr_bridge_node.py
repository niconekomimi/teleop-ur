#!/usr/bin/env python3
"""Quest 3 WebXR bridge based on Vuer motion controller streaming."""

from __future__ import annotations

import asyncio
import ipaddress
from pathlib import Path
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32MultiArray

from ..utils.transform_utils import quat_normalize_xyzw

try:
    from vuer import Vuer, VuerSession
    from vuer.schemas import MotionControllers
except Exception:
    Vuer = None
    VuerSession = Any
    MotionControllers = None


def _rotation_matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    m00, m01, m02 = float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])
    m10, m11, m12 = float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])
    m20, m21, m22 = float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])

    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif (m00 > m11) and (m00 > m22):
        scale = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale

    return quat_normalize_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _matrix4_from_column_major(values: Any) -> Optional[np.ndarray]:
    try:
        matrix = np.asarray(values, dtype=np.float64).reshape(4, 4, order="F")
    except Exception:
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _dict_value(data: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data:
            return data[key]
    return default


@dataclass(frozen=True)
class ControllerSnapshot:
    matrix_col_major: np.ndarray
    state: dict[str, Any]


@dataclass(frozen=True)
class Quest3Snapshot:
    stamp_monotonic: float
    left: Optional[ControllerSnapshot]
    right: Optional[ControllerSnapshot]


class Quest3WebXRBridgeNode(Node):
    """Expose Quest 3 controller matrices and button states as ROS topics."""

    def __init__(self) -> None:
        super().__init__("quest3_webxr_bridge_node")

        self.declare_parameter("vuer_host", "0.0.0.0")
        self.declare_parameter("vuer_port", 8012)
        self.declare_parameter("vuer_client_domain", "https://vuer.ai")
        self.declare_parameter("public_wss_url", "")
        self.declare_parameter("advertised_host", "")
        self.declare_parameter("auto_generate_self_signed_cert", True)
        self.declare_parameter("tls_workdir", "/tmp/teleop_control_py/quest3_webxr_tls")
        self.declare_parameter("ssl_cert_file", "")
        self.declare_parameter("ssl_key_file", "")
        self.declare_parameter("stream_key", "quest3-controllers")
        self.declare_parameter("stream_left", True)
        self.declare_parameter("stream_right", True)
        self.declare_parameter("publish_rate_hz", 90.0)
        self.declare_parameter("stale_timeout_sec", 0.5)
        self.declare_parameter("frame_id", "quest3_world")
        self.declare_parameter("left_pose_topic", "/quest3/left_controller/pose")
        self.declare_parameter("right_pose_topic", "/quest3/right_controller/pose")
        self.declare_parameter("left_matrix_topic", "/quest3/left_controller/matrix")
        self.declare_parameter("right_matrix_topic", "/quest3/right_controller/matrix")
        self.declare_parameter("joy_topic", "/quest3/input/joy")
        self.declare_parameter("connected_topic", "/quest3/connected")
        self.declare_parameter("debug_log_all_vuer_events", True)

        if Vuer is None or MotionControllers is None:
            raise RuntimeError(
                "Quest3 WebXR bridge requires the 'vuer' package. "
                "Install it first, for example: pip install 'vuer>=0.1.4,<0.2'"
            )

        self._vuer_host = str(self.get_parameter("vuer_host").value).strip() or "0.0.0.0"
        self._vuer_port = int(self.get_parameter("vuer_port").value)
        self._vuer_client_domain = str(self.get_parameter("vuer_client_domain").value).strip() or "https://vuer.ai"
        self._public_wss_url = str(self.get_parameter("public_wss_url").value).strip()
        self._advertised_host = str(self.get_parameter("advertised_host").value).strip() or self._detect_local_ip()
        self._auto_generate_self_signed_cert = bool(self.get_parameter("auto_generate_self_signed_cert").value)
        self._tls_workdir = Path(str(self.get_parameter("tls_workdir").value).strip() or "/tmp/teleop_control_py/quest3_webxr_tls")
        self._ssl_cert_file = str(self.get_parameter("ssl_cert_file").value).strip()
        self._ssl_key_file = str(self.get_parameter("ssl_key_file").value).strip()
        if (not self._ssl_cert_file and not self._ssl_key_file) and self._auto_generate_self_signed_cert:
            self._ssl_cert_file, self._ssl_key_file = self._ensure_local_self_signed_tls()
        if bool(self._ssl_cert_file) != bool(self._ssl_key_file):
            raise RuntimeError("ssl_cert_file and ssl_key_file must be set together for secure WebSocket serving.")
        self._stream_key = str(self.get_parameter("stream_key").value).strip() or "quest3-controllers"
        self._stream_left = bool(self.get_parameter("stream_left").value)
        self._stream_right = bool(self.get_parameter("stream_right").value)
        self._publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._stale_timeout_sec = max(0.05, float(self.get_parameter("stale_timeout_sec").value))
        self._frame_id = str(self.get_parameter("frame_id").value).strip() or "quest3_world"
        self._debug_log_all_vuer_events = bool(self.get_parameter("debug_log_all_vuer_events").value)

        self._left_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("left_pose_topic").value).strip() or "/quest3/left_controller/pose",
            qos_profile_sensor_data,
        )
        self._right_pose_pub = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("right_pose_topic").value).strip() or "/quest3/right_controller/pose",
            qos_profile_sensor_data,
        )
        self._left_matrix_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("left_matrix_topic").value).strip() or "/quest3/left_controller/matrix",
            qos_profile_sensor_data,
        )
        self._right_matrix_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("right_matrix_topic").value).strip() or "/quest3/right_controller/matrix",
            qos_profile_sensor_data,
        )
        self._joy_pub = self.create_publisher(
            Joy,
            str(self.get_parameter("joy_topic").value).strip() or "/quest3/input/joy",
            qos_profile_sensor_data,
        )
        self._connected_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("connected_topic").value).strip() or "/quest3/connected",
            10,
        )

        self._lock = threading.Lock()
        self._latest_snapshot: Optional[Quest3Snapshot] = None
        self._connected = False
        self._received_packets = 0
        self._received_vuer_events = 0
        self._last_vuer_event_name = ""
        self._last_vuer_event_log_ns = 0
        self._stop_event = threading.Event()
        self._server_thread = threading.Thread(target=self._run_vuer_server, daemon=True)

        self._timer = self.create_timer(1.0 / self._publish_rate_hz, self._publish_latest_snapshot)
        self._status_timer = self.create_timer(0.25, self._publish_connection_status)

        self._server_thread.start()
        self._log_startup_hints()

    def _detect_local_ip(self) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip:
                return str(ip)
        except Exception:
            pass
        finally:
            sock.close()

        try:
            hostname_ip = socket.gethostbyname(socket.gethostname())
            if hostname_ip and hostname_ip != "127.0.0.1":
                return str(hostname_ip)
        except Exception:
            pass
        return "127.0.0.1"

    def _ensure_local_self_signed_tls(self) -> tuple[str, str]:
        self._tls_workdir.mkdir(parents=True, exist_ok=True)
        cert_path = self._tls_workdir / "quest3_bridge_cert.pem"
        key_path = self._tls_workdir / "quest3_bridge_key.pem"
        config_path = self._tls_workdir / "quest3_bridge_openssl.cnf"

        if cert_path.exists() and key_path.exists():
            return str(cert_path), str(key_path)

        try:
            ipaddress.ip_address(self._advertised_host)
            alt_name_block = (
                f"IP.1 = {self._advertised_host}\n"
                "DNS.1 = localhost\n"
            )
        except ValueError:
            alt_name_block = (
                f"DNS.1 = {self._advertised_host}\n"
                "DNS.2 = localhost\n"
            )

        openssl_config = (
            "[req]\n"
            "default_bits = 2048\n"
            "prompt = no\n"
            "default_md = sha256\n"
            "x509_extensions = req_ext\n"
            "distinguished_name = dn\n"
            "\n"
            "[dn]\n"
            f"CN = {self._advertised_host}\n"
            "\n"
            "[req_ext]\n"
            "subjectAltName = @alt_names\n"
            "\n"
            "[alt_names]\n"
            f"{alt_name_block}"
        )
        config_path.write_text(openssl_config, encoding="utf-8")

        command = [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "3650",
            "-nodes",
            "-keyout",
            str(key_path),
            "-out",
            str(cert_path),
            "-config",
            str(config_path),
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to auto-generate self-signed TLS cert: {result.stderr.strip()}")
        return str(cert_path), str(key_path)

    def _log_startup_hints(self) -> None:
        ws_scheme = "wss" if self._ssl_cert_file and self._ssl_key_file else "ws"
        local_https_url = f"https://{self._advertised_host}:{self._vuer_port}"
        local_wss_url = f"wss://{self._advertised_host}:{self._vuer_port}"
        self.get_logger().info(
            "Quest 3 WebXR bridge starting: "
            f"server={ws_scheme}://{self._vuer_host}:{self._vuer_port}, "
            f"client_domain={self._vuer_client_domain}"
        )
        if ws_scheme == "wss":
            self.get_logger().info(f"Quest direct URL: {local_https_url}")
            self.get_logger().info(
                "Quest alternate URL: "
                f"{self._vuer_client_domain}?ws={local_wss_url}"
            )
            self.get_logger().info(
                "First connection on Quest usually needs one manual certificate acceptance."
            )
        elif self._public_wss_url:
            self.get_logger().info(
                "Quest browser URL: "
                f"{self._vuer_client_domain}?ws={self._public_wss_url}"
            )
        else:
            self.get_logger().info(
                "Set 'public_wss_url' to a reachable wss:// endpoint. "
                "Per Vuer docs, Quest browser access should use a secure WebSocket URL."
            )

    def _create_vuer_app(self) -> Vuer:
        kwargs: dict[str, Any] = {
            "host": self._vuer_host,
            "port": self._vuer_port,
            "domain": self._vuer_client_domain,
            "queries": {"reconnect": "true"},
        }
        if self._ssl_cert_file:
            kwargs["cert"] = self._ssl_cert_file
        if self._ssl_key_file:
            kwargs["key"] = self._ssl_key_file
        app = Vuer(**kwargs)

        @app.add_handler("CONTROLLER_MOVE")
        async def _on_controller_move(event, _session: VuerSession) -> None:
            self._log_vuer_event("CONTROLLER_MOVE", getattr(event, "value", None))
            self._handle_controller_move(getattr(event, "value", None))

        @app.add_handler("CAMERA_MOVE")
        async def _on_camera_move(event, _session: VuerSession) -> None:
            self._log_vuer_event("CAMERA_MOVE", getattr(event, "value", None))

        @app.add_handler("HAND_MOVE")
        async def _on_hand_move(event, _session: VuerSession) -> None:
            self._log_vuer_event("HAND_MOVE", getattr(event, "value", None))

        @app.add_handler("INPUT")
        async def _on_input(event, _session: VuerSession) -> None:
            self._log_vuer_event("INPUT", getattr(event, "value", None))

        @app.spawn(start=True)
        async def _main(session: VuerSession) -> None:
            session.upsert @ MotionControllers(
                key=self._stream_key,
                stream=True,
                left=self._stream_left,
                right=self._stream_right,
            )
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)

        return app

    def _run_vuer_server(self) -> None:
        try:
            self._vuer_app = self._create_vuer_app()
            self._vuer_app.run()
        except Exception as exc:
            self.get_logger().error(f"Quest 3 Vuer server exited with error: {exc}")

    def _log_vuer_event(self, event_name: str, payload: Any) -> None:
        self._received_vuer_events += 1
        self._last_vuer_event_name = str(event_name)
        if not self._debug_log_all_vuer_events:
            return

        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_vuer_event_log_ns) < int(0.2 * 1e9):
            return
        self._last_vuer_event_log_ns = now_ns

        payload_type = type(payload).__name__
        payload_keys = sorted(str(k) for k in payload.keys()) if isinstance(payload, dict) else []
        self.get_logger().info(
            "Vuer event received: "
            f"name={event_name}, "
            f"is_controller_move={str(event_name).upper() == 'CONTROLLER_MOVE'}, "
            f"payload_type={payload_type}, payload_keys={payload_keys}"
        )

    def _parse_controller(self, payload: Any, matrix_key: str, state_key: str) -> Optional[ControllerSnapshot]:
        matrix = _matrix4_from_column_major(_dict_value(payload, matrix_key))
        if matrix is None:
            return None
        raw_state = _dict_value(payload, state_key, default={})
        state = raw_state if isinstance(raw_state, dict) else {}
        return ControllerSnapshot(matrix_col_major=matrix.reshape(-1, order="F"), state=state)

    def _handle_controller_move(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        snapshot = Quest3Snapshot(
            stamp_monotonic=time.monotonic(),
            left=self._parse_controller(payload, "left", "leftState"),
            right=self._parse_controller(payload, "right", "rightState"),
        )
        with self._lock:
            self._latest_snapshot = snapshot
            self._received_packets += 1
        if not self._connected:
            self._connected = True
            self.get_logger().info("Quest 3 controller stream connected.")

    def _snapshot_to_pose(self, matrix_values: np.ndarray) -> PoseStamped:
        matrix = np.asarray(matrix_values, dtype=np.float64).reshape(4, 4, order="F")
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self._frame_id
        pose.pose.position.x = float(matrix[0, 3])
        pose.pose.position.y = float(matrix[1, 3])
        pose.pose.position.z = float(matrix[2, 3])
        quat = _rotation_matrix_to_quat_xyzw(matrix[:3, :3])
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        return pose

    def _matrix_msg(self, matrix_values: np.ndarray) -> Float32MultiArray:
        msg = Float32MultiArray()
        msg.data = [float(v) for v in np.asarray(matrix_values, dtype=np.float32).reshape(-1)]
        return msg

    def _state_axis(self, state: dict[str, Any], *keys: str) -> float:
        return _as_float(_dict_value(state, *keys, default=0.0), default=0.0)

    def _state_button(self, state: dict[str, Any], *keys: str) -> int:
        return 1 if _as_bool(_dict_value(state, *keys, default=False), default=False) else 0

    def _build_joy(self, snapshot: Quest3Snapshot) -> Joy:
        left_state = snapshot.left.state if snapshot.left is not None else {}
        right_state = snapshot.right.state if snapshot.right is not None else {}

        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [
            self._state_axis(left_state, "triggerValue", "trigger_value"),
            self._state_axis(left_state, "squeezeValue", "squeeze_value", "gripValue"),
            self._state_axis(left_state, "touchpadX", "touchpad_x"),
            self._state_axis(left_state, "touchpadY", "touchpad_y"),
            self._state_axis(left_state, "thumbstickX", "thumbstick_x"),
            self._state_axis(left_state, "thumbstickY", "thumbstick_y"),
            self._state_axis(right_state, "triggerValue", "trigger_value"),
            self._state_axis(right_state, "squeezeValue", "squeeze_value", "gripValue"),
            self._state_axis(right_state, "touchpadX", "touchpad_x"),
            self._state_axis(right_state, "touchpadY", "touchpad_y"),
            self._state_axis(right_state, "thumbstickX", "thumbstick_x"),
            self._state_axis(right_state, "thumbstickY", "thumbstick_y"),
        ]
        msg.buttons = [
            self._state_button(left_state, "trigger"),
            self._state_button(left_state, "squeeze", "grip"),
            self._state_button(left_state, "touchpad"),
            self._state_button(left_state, "thumbstick"),
            self._state_button(left_state, "xButton", "aButton", "primaryButton"),
            self._state_button(left_state, "yButton", "bButton", "secondaryButton"),
            self._state_button(right_state, "trigger"),
            self._state_button(right_state, "squeeze", "grip"),
            self._state_button(right_state, "touchpad"),
            self._state_button(right_state, "thumbstick"),
            self._state_button(right_state, "aButton", "xButton", "primaryButton"),
            self._state_button(right_state, "bButton", "yButton", "secondaryButton"),
        ]
        return msg

    def _publish_latest_snapshot(self) -> None:
        with self._lock:
            snapshot = self._latest_snapshot
        if snapshot is None:
            return
        if (time.monotonic() - snapshot.stamp_monotonic) > self._stale_timeout_sec:
            return

        if snapshot.left is not None:
            self._left_pose_pub.publish(self._snapshot_to_pose(snapshot.left.matrix_col_major))
            self._left_matrix_pub.publish(self._matrix_msg(snapshot.left.matrix_col_major))
        if snapshot.right is not None:
            self._right_pose_pub.publish(self._snapshot_to_pose(snapshot.right.matrix_col_major))
            self._right_matrix_pub.publish(self._matrix_msg(snapshot.right.matrix_col_major))
        self._joy_pub.publish(self._build_joy(snapshot))

    def _publish_connection_status(self) -> None:
        with self._lock:
            snapshot = self._latest_snapshot
        connected = snapshot is not None and (time.monotonic() - snapshot.stamp_monotonic) <= self._stale_timeout_sec
        if self._connected != connected:
            self._connected = connected
            if connected:
                self.get_logger().info("Quest 3 stream active.")
            else:
                self.get_logger().warn("Quest 3 stream timed out.")
        msg = Bool()
        msg.data = bool(connected)
        self._connected_pub.publish(msg)

    def destroy_node(self) -> bool:  # type: ignore[override]
        self._stop_event.set()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Quest3WebXRBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
