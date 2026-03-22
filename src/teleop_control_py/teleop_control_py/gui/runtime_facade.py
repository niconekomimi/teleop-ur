"""Runtime facade for GUI-facing process and hardware status queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from teleop_control_py.core.resource_manager import CameraRuntimeContext, ResourceManager
from teleop_control_py.gui.runtime import ProcessManager
from teleop_control_py.gui.support import build_camera_driver_command, detect_joystick_devices, get_local_ip


@dataclass(frozen=True)
class RuntimeSnapshot:
    local_ip: str
    teleop_running: bool
    robot_driver_standalone_running: bool
    robot_driver_running: bool
    collector_running: bool
    preview_running: bool
    inference_running: bool
    camera_context: CameraRuntimeContext
    active_camera_drivers: tuple[str, ...]
    selected_camera_driver: str
    selected_camera_driver_running: bool
    joystick_status_text: str
    joystick_status_color: str
    realsense_status_text: str
    realsense_status_color: str
    oakd_status_text: str
    oakd_status_color: str


class GuiRuntimeFacade:
    """Aggregates runtime state so the main window does not query managers directly."""

    def __init__(
        self,
        parent=None,
        *,
        camera_enable_depth: Optional[Mapping[str, bool]] = None,
    ) -> None:
        self._process_manager = ProcessManager(parent)
        self._resource_manager = ResourceManager()
        raw_enable_depth = dict(camera_enable_depth or {})
        self._camera_enable_depth = {
            "realsense": bool(raw_enable_depth.get("realsense", False)),
            "oakd": bool(raw_enable_depth.get("oakd", False)),
        }

    @property
    def process_manager(self) -> ProcessManager:
        return self._process_manager

    @property
    def log_signal(self):
        return self._process_manager.log_signal

    @property
    def process_exited(self):
        return self._process_manager.process_exited

    def stop_all_processes(self) -> None:
        self._process_manager.stop_all()

    def is_process_running(self, key: str) -> bool:
        return self._process_manager.is_running(key)

    def list_ros_image_topics(self, *, log_errors: bool = False) -> list[str]:
        return self._process_manager.list_ros_image_topics(log_errors=log_errors)

    def build_camera_context(
        self,
        *,
        collector_global_source: Optional[str] = None,
        collector_wrist_source: Optional[str] = None,
        collector_global_serial: Optional[str] = None,
        collector_wrist_serial: Optional[str] = None,
        inference_global_source: Optional[str] = None,
        inference_wrist_source: Optional[str] = None,
        inference_global_serial: Optional[str] = None,
        inference_wrist_serial: Optional[str] = None,
        inference_running: bool = False,
        active_camera_driver_devices: Optional[Sequence[tuple[str, str]]] = None,
    ) -> CameraRuntimeContext:
        active_camera_drivers = tuple(self._resource_manager.active_camera_drivers(self._process_manager))
        active_devices: list[tuple[str, str]] = []
        for source, serial in list(active_camera_driver_devices or []):
            normalized_source = self._resource_manager.normalize_camera_source(source)
            if normalized_source not in self._resource_manager.CAMERA_NAMES:
                continue
            active_devices.append((normalized_source, str(serial).strip()))

        return CameraRuntimeContext(
            collector_running=self.is_process_running('data_collector'),
            collector_global_source=collector_global_source,
            collector_wrist_source=collector_wrist_source,
            collector_global_serial=str(collector_global_serial or "").strip(),
            collector_wrist_serial=str(collector_wrist_serial or "").strip(),
            inference_running=bool(inference_running),
            inference_global_source=inference_global_source,
            inference_wrist_source=inference_wrist_source,
            inference_global_serial=str(inference_global_serial or "").strip(),
            inference_wrist_serial=str(inference_wrist_serial or "").strip(),
            active_camera_drivers=active_camera_drivers,
            active_camera_driver_devices=tuple(active_devices),
        )

    def runtime_snapshot(
        self,
        *,
        collector_global_source: Optional[str] = None,
        collector_wrist_source: Optional[str] = None,
        collector_global_serial: Optional[str] = None,
        collector_wrist_serial: Optional[str] = None,
        inference_global_source: Optional[str] = None,
        inference_wrist_source: Optional[str] = None,
        inference_global_serial: Optional[str] = None,
        inference_wrist_serial: Optional[str] = None,
        inference_running: bool = False,
        input_type: str = "joy",
        selected_camera_driver: str = "realsense",
        preview_running: bool = False,
        active_camera_driver_devices: Optional[Sequence[tuple[str, str]]] = None,
    ) -> RuntimeSnapshot:
        teleop_running = self.is_process_running('teleop')
        robot_driver_standalone_running = self.is_process_running('robot_driver')
        collector_running = self.is_process_running('data_collector')
        camera_context = self.build_camera_context(
            collector_global_source=collector_global_source,
            collector_wrist_source=collector_wrist_source,
            collector_global_serial=collector_global_serial,
            collector_wrist_serial=collector_wrist_serial,
            inference_global_source=inference_global_source,
            inference_wrist_source=inference_wrist_source,
            inference_global_serial=inference_global_serial,
            inference_wrist_serial=inference_wrist_serial,
            inference_running=inference_running,
            active_camera_driver_devices=active_camera_driver_devices,
        )
        active_camera_drivers = tuple(camera_context.active_camera_drivers)
        joystick_text, joystick_color = self._resource_manager.joystick_status(
            detect_joystick_devices(),
            teleop_running=teleop_running,
            input_type=input_type,
        )
        realsense_text, realsense_color = self._resource_manager.camera_status('realsense', camera_context)
        oakd_text, oakd_color = self._resource_manager.camera_status('oakd', camera_context)
        return RuntimeSnapshot(
            local_ip=get_local_ip(),
            teleop_running=teleop_running,
            robot_driver_standalone_running=robot_driver_standalone_running,
            robot_driver_running=teleop_running or robot_driver_standalone_running,
            collector_running=collector_running,
            preview_running=bool(preview_running),
            inference_running=bool(inference_running),
            camera_context=camera_context,
            active_camera_drivers=active_camera_drivers,
            selected_camera_driver=selected_camera_driver,
            selected_camera_driver_running=self.camera_driver_running(selected_camera_driver),
            joystick_status_text=joystick_text,
            joystick_status_color=joystick_color,
            realsense_status_text=realsense_text,
            realsense_status_color=realsense_color,
            oakd_status_text=oakd_text,
            oakd_status_color=oakd_color,
        )

    def ensure_camera_availability(
        self,
        *,
        requester: str,
        requested_sources: Sequence[str],
        requested_serial_numbers: Optional[Sequence[str]] = None,
        context: CameraRuntimeContext,
        require_distinct_views: bool = False,
    ) -> None:
        self._resource_manager.check_camera_availability(
            requester=requester,
            requested_sources=requested_sources,
            requested_serial_numbers=requested_serial_numbers,
            context=context,
            require_distinct_views=require_distinct_views,
        )

    def camera_driver_running(self, camera_name: str) -> bool:
        return self._resource_manager.camera_driver_running(self._process_manager, camera_name)

    def start_camera_driver(self, camera_name: str) -> bool:
        normalized = self._resource_manager.normalize_camera_source(camera_name)
        if not normalized:
            return False
        return self._process_manager.run_subprocess(
            f'camera_driver_{normalized}',
            build_camera_driver_command(
                normalized,
                enable_depth=self._camera_enable_depth.get(normalized, False),
            ),
        )

    def stop_camera_driver(self, camera_name: str) -> None:
        normalized = self._resource_manager.normalize_camera_source(camera_name)
        if not normalized:
            return
        self._process_manager.kill_subprocess(f'camera_driver_{normalized}')
