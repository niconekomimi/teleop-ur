"""Hardware occupancy and conflict detection helpers for the teleop GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .process_manager import ProcessManager


@dataclass(frozen=True)
class CameraRuntimeContext:
    collector_running: bool = False
    collector_global_source: Optional[str] = None
    collector_wrist_source: Optional[str] = None
    inference_running: bool = False
    inference_global_source: Optional[str] = None
    inference_wrist_source: Optional[str] = None
    active_camera_drivers: tuple[str, ...] = ()


class HardwareConflictError(RuntimeError):
    """Raised when a camera resource is already occupied by another module."""

    def __init__(self, conflicts: Sequence[str]) -> None:
        self.conflicts = [str(item) for item in conflicts if str(item).strip()]
        super().__init__("；".join(self.conflicts) if self.conflicts else "硬件占用冲突")


class HardwareManager:
    CAMERA_NAMES = ("realsense", "oakd")

    def normalize_camera_source(self, source: Optional[str]) -> str:
        value = (source or "").strip().lower()
        if value == "oakd":
            return "oakd"
        if value in {"realsense", "rs"}:
            return "realsense"
        return value

    def camera_driver_running(self, process_manager: ProcessManager, camera_name: str) -> bool:
        normalized = self.normalize_camera_source(camera_name)
        return normalized in self.CAMERA_NAMES and process_manager.is_running(f"camera_driver_{normalized}")

    def active_camera_drivers(self, process_manager: ProcessManager) -> list[str]:
        return [
            camera_name
            for camera_name in self.CAMERA_NAMES
            if self.camera_driver_running(process_manager, camera_name)
        ]

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
        require_distinct_views: bool = False,
    ) -> None:
        requested = [
            self.normalize_camera_source(source)
            for source in requested_sources
            if self.normalize_camera_source(source) in self.CAMERA_NAMES
        ]
        requested_set = set(requested)
        conflicts: list[str] = []

        if require_distinct_views and len(requested) >= 2 and len(requested_set) < len(requested):
            conflicts.append("当前推理需要两路不同相机视角，请不要把全局相机和手部相机设置成同一源。")

        collector_usage = self.collector_camera_occupancy(context)
        inference_usage = self.inference_camera_occupancy(context)
        active_camera_drivers = {
            self.normalize_camera_source(source)
            for source in context.active_camera_drivers
            if self.normalize_camera_source(source) in self.CAMERA_NAMES
        }

        # Collector and inference now share cameras through the shm microservice,
        # so only direct ROS2 drivers remain exclusive camera owners.
        if requester in {"collector", "inference"}:
            for source in sorted(requested_set):
                if source in active_camera_drivers:
                    conflicts.append(f"ROS2 相机驱动正在占用 {source}")

        if conflicts:
            raise HardwareConflictError(conflicts)

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
            text = joystick_devices[0]
            if teleop_running and input_type == "joy":
                text = f"被遥操作占用: {text}"
            return text, "#2b8a3e"
        return "未检测到", "#c92a2a"

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
