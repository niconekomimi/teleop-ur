"""Resource availability checking and hardware conflict detection for the core layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .shm_registry import DaemonEntry, SHMRegistry


@dataclass(frozen=True)
class CameraRuntimeContext:
    collector_running: bool = False
    collector_global_source: Optional[str] = None
    collector_wrist_source: Optional[str] = None
    collector_global_serial: Optional[str] = None
    collector_wrist_serial: Optional[str] = None
    inference_running: bool = False
    inference_global_source: Optional[str] = None
    inference_wrist_source: Optional[str] = None
    inference_global_serial: Optional[str] = None
    inference_wrist_serial: Optional[str] = None
    active_camera_drivers: tuple[str, ...] = ()
    active_camera_driver_devices: tuple[tuple[str, str], ...] = ()


class HardwareConflictError(RuntimeError):
    """Raised when a camera resource is already occupied by another module."""

    def __init__(self, conflicts: Sequence[str]) -> None:
        self.conflicts = [str(item) for item in conflicts if str(item).strip()]
        super().__init__("；".join(self.conflicts) if self.conflicts else "硬件占用冲突")


class ResourceManager:
    """Validates resource availability and coordinates DeviceManager startup/teardown."""

    CAMERA_NAMES = ("realsense", "oakd")

    # ------------------------------------------------------------------
    # Camera source normalisation (unchanged from HardwareManager)
    # ------------------------------------------------------------------

    def normalize_camera_source(self, source: Optional[str]) -> str:
        value = (source or "").strip().lower()
        if value == "oakd":
            return "oakd"
        if value in {"realsense", "rs"}:
            return "realsense"
        return value

    # ------------------------------------------------------------------
    # Process-level camera driver queries (requires ProcessManager duck-type)
    # ------------------------------------------------------------------

    def camera_driver_running(self, process_manager, camera_name: str) -> bool:
        normalized = self.normalize_camera_source(camera_name)
        return normalized in self.CAMERA_NAMES and process_manager.is_running(
            f"camera_driver_{normalized}"
        )

    def active_camera_drivers(self, process_manager) -> list[str]:
        return [
            camera_name
            for camera_name in self.CAMERA_NAMES
            if self.camera_driver_running(process_manager, camera_name)
        ]

    # ------------------------------------------------------------------
    # Conflict detection (unchanged logic from HardwareManager)
    # ------------------------------------------------------------------

    def collector_camera_occupancy(self, context: CameraRuntimeContext) -> dict[str, bool]:
        if not context.collector_running:
            return self._empty_camera_occupancy()
        return self._camera_occupancy(context.collector_global_source, context.collector_wrist_source)

    def inference_camera_occupancy(self, context: CameraRuntimeContext) -> dict[str, bool]:
        if not context.inference_running:
            return self._empty_camera_occupancy()
        return self._camera_occupancy(context.inference_global_source, context.inference_wrist_source)

    def check_camera_availability(
        self,
        requester: str,
        requested_sources: Sequence[str],
        context: CameraRuntimeContext,
        requested_serial_numbers: Optional[Sequence[str]] = None,
        require_distinct_views: bool = False,
    ) -> None:
        requested_devices = self._normalize_requested_devices(requested_sources, requested_serial_numbers)
        conflicts: list[str] = []

        if require_distinct_views:
            conflicts.extend(self._distinct_view_conflicts(requested_devices))

        if requester in {"collector", "inference"}:
            active_devices = self._active_camera_driver_devices(context)
            for requested_source, requested_serial in requested_devices:
                for active_source, active_serial in active_devices:
                    if requested_source != active_source:
                        continue
                    if requested_serial and active_serial and requested_serial != active_serial:
                        continue
                    conflicts.append(
                        self._driver_conflict_message(
                            source=requested_source,
                            requested_serial=requested_serial,
                            active_serial=active_serial,
                        )
                    )
                    break

        if conflicts:
            unique_conflicts: list[str] = []
            seen: set[str] = set()
            for conflict in conflicts:
                text = str(conflict).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                unique_conflicts.append(text)
            raise HardwareConflictError(unique_conflicts)

    def camera_status(self, camera_name: str, context: CameraRuntimeContext) -> tuple[str, str]:
        normalized = self.normalize_camera_source(camera_name)
        collector_usage = self.collector_camera_occupancy(context)
        inference_usage = self.inference_camera_occupancy(context)
        active_drivers = {
            self.normalize_camera_source(source)
            for source in context.active_camera_drivers
        }

        if inference_usage.get(normalized, False) and collector_usage.get(normalized, False):
            return "采集/推理共享", "#e67700"
        if inference_usage.get(normalized, False):
            return "推理模块占用", "#e67700"
        if collector_usage.get(normalized, False):
            return "采集节点占用", "#e67700"
        if normalized in active_drivers:
            return "ROS2 驱动占用", "#2b8a3e"
        return "空闲", "#6c757d"

    def joystick_status(
        self,
        joystick_devices: Sequence[str],
        teleop_running: bool,
        input_type: str,
    ) -> tuple[str, str]:
        if joystick_devices:
            text = f"已接入: {self._compact_joystick_name(joystick_devices[0])}"
            if len(joystick_devices) > 1:
                text = f"{text} (+{len(joystick_devices) - 1})"
            if teleop_running and input_type == "joy":
                text = f"手柄占用 | {text}"
            return text, "#2b8a3e"
        return "未检测到", "#c92a2a"

    # ------------------------------------------------------------------
    # SHM layer queries
    # ------------------------------------------------------------------

    def shm_camera_alive(self, spec_identifier: str) -> bool:
        return SHMRegistry.is_alive(spec_identifier)

    def list_shm_cameras(self) -> list[DaemonEntry]:
        return SHMRegistry.list_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _camera_occupancy(self, *sources: Optional[str]) -> dict[str, bool]:
        normalized_sources = {
            self.normalize_camera_source(source)
            for source in sources
            if self.normalize_camera_source(source) in self.CAMERA_NAMES
        }
        return {
            "realsense": "realsense" in normalized_sources,
            "oakd": "oakd" in normalized_sources,
        }

    def _empty_camera_occupancy(self) -> dict[str, bool]:
        return {camera_name: False for camera_name in self.CAMERA_NAMES}

    def _normalize_requested_devices(
        self,
        requested_sources: Sequence[str],
        requested_serial_numbers: Optional[Sequence[str]],
    ) -> list[tuple[str, str]]:
        serial_values = list(requested_serial_numbers or [])
        devices: list[tuple[str, str]] = []
        for index, source in enumerate(requested_sources):
            normalized_source = self.normalize_camera_source(source)
            if normalized_source not in self.CAMERA_NAMES:
                continue
            serial = serial_values[index] if index < len(serial_values) else ""
            normalized_serial = str(serial).strip()
            devices.append((normalized_source, normalized_serial))
        return devices

    def _distinct_view_conflicts(self, requested_devices: Sequence[tuple[str, str]]) -> list[str]:
        by_source: dict[str, list[str]] = {}
        for source, serial in requested_devices:
            by_source.setdefault(source, []).append(str(serial).strip())

        conflicts: list[str] = []
        for source, serials in by_source.items():
            if len(serials) < 2:
                continue
            if any(not serial for serial in serials):
                conflicts.append(
                    f"当前推理需要两路不同相机视角：{source} 至少一路未指定序列号，无法确认是不同设备。"
                )
                continue
            if len(set(serials)) < len(serials):
                conflicts.append(f"当前推理需要两路不同相机视角：{source} 序列号重复。")
        return conflicts

    def _active_camera_driver_devices(self, context: CameraRuntimeContext) -> list[tuple[str, str]]:
        devices: list[tuple[str, str]] = []
        for source, serial in context.active_camera_driver_devices:
            normalized_source = self.normalize_camera_source(source)
            if normalized_source not in self.CAMERA_NAMES:
                continue
            devices.append((normalized_source, str(serial).strip()))

        if not devices:
            for source in context.active_camera_drivers:
                normalized_source = self.normalize_camera_source(source)
                if normalized_source not in self.CAMERA_NAMES:
                    continue
                devices.append((normalized_source, ""))
            return devices

        seen = set()
        deduped: list[tuple[str, str]] = []
        for source, serial in devices:
            key = (source, serial)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((source, serial))

        for source in context.active_camera_drivers:
            normalized_source = self.normalize_camera_source(source)
            if normalized_source not in self.CAMERA_NAMES:
                continue
            key = (normalized_source, "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)

        return deduped

    def _driver_conflict_message(self, *, source: str, requested_serial: str, active_serial: str) -> str:
        req = str(requested_serial).strip()
        active = str(active_serial).strip()
        if req and active:
            return f"ROS2 相机驱动正在占用 {source}（SN={active}）。"
        if req and not active:
            return (
                f"ROS2 相机驱动正在占用 {source}（驱动 SN 未知，无法确认与请求 SN={req} 是否不同）。"
            )
        if not req and active:
            return f"ROS2 相机驱动正在占用 {source}（SN={active}，当前请求未指定 SN）。"
        return f"ROS2 相机驱动正在占用 {source}。"

    @staticmethod
    def _compact_joystick_name(raw_name: str) -> str:
        text = str(raw_name or "").strip()
        if not text:
            return "手柄"
        if "/" in text:
            text = text.rsplit("/", 1)[-1]
        if text.startswith("usb-"):
            text = text[4:]
        cleaned = text.replace("_", " ").replace("-", " ")
        parts = [part for part in cleaned.split() if part]
        if not parts:
            return "手柄"
        if len(parts) > 3:
            parts = parts[-3:]
        result = " ".join(parts)
        if len(result) > 24:
            result = result[:24].rstrip()
        return result or "手柄"
