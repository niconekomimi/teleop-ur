"""Core-side inference worker lifecycle service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QThread

from .inference_worker import InferenceWorker


@dataclass(frozen=True)
class InferenceWorkerCallbacks:
    action: Callable[[object], None]
    preview: Callable[[object, object], None]
    status: Callable[[str], None]
    log: Callable[[str], None]
    error: Callable[[str], None]
    finished: Callable[[], None]


@dataclass(frozen=True)
class InferenceLaunchConfig:
    checkpoint_dir: str
    task_name: str
    task_embedding_path: str
    global_camera_source: str
    wrist_camera_source: str
    loop_hz: float
    device: str | None = None
    state_provider: Optional[Callable[[], object]] = None
    global_camera_serial_number: str = ""
    wrist_camera_serial_number: str = ""
    global_camera_enable_depth: bool = False
    wrist_camera_enable_depth: bool = False
    backend: str = "real_il"
    prompt_provider: Optional[Callable[[], str]] = None
    openpi_host: str = "127.0.0.1"
    openpi_port: int = 18000
    openpi_api_key: str = ""


class InferenceService:
    """Owns the inference worker lifecycle independently of any concrete UI."""

    def __init__(self) -> None:
        self._worker: Optional[QThread] = None

    @property
    def worker(self) -> Optional[QThread]:
        return self._worker

    def is_running(self) -> bool:
        worker = self._worker
        return worker is not None and worker.isRunning()

    def start_inference(
        self,
        config: InferenceLaunchConfig,
        callbacks: InferenceWorkerCallbacks,
        *,
        parent=None,
    ) -> QThread:
        worker = self._worker
        if worker is not None and worker.isRunning():
            return worker

        backend = str(config.backend).strip().lower() or "real_il"
        if backend == "openpi_remote":
            from .openpi_remote_worker import OpenPiRemoteWorker

            worker = OpenPiRemoteWorker(
                openpi_host=config.openpi_host,
                openpi_port=int(config.openpi_port),
                openpi_api_key=config.openpi_api_key,
                remote_prompt=config.task_name,
                global_camera_source=config.global_camera_source,
                wrist_camera_source=config.wrist_camera_source,
                loop_hz=float(config.loop_hz),
                state_provider=config.state_provider,
                prompt_provider=config.prompt_provider,
                global_camera_serial_number=config.global_camera_serial_number,
                wrist_camera_serial_number=config.wrist_camera_serial_number,
                global_camera_enable_depth=bool(config.global_camera_enable_depth),
                wrist_camera_enable_depth=bool(config.wrist_camera_enable_depth),
                parent=parent,
            )
        else:
            worker = InferenceWorker(
                checkpoint_dir=config.checkpoint_dir,
                task_name=config.task_name,
                task_embedding_path=config.task_embedding_path,
                global_camera_source=config.global_camera_source,
                wrist_camera_source=config.wrist_camera_source,
                loop_hz=float(config.loop_hz),
                device=config.device,
                state_provider=config.state_provider,
                global_camera_serial_number=config.global_camera_serial_number,
                wrist_camera_serial_number=config.wrist_camera_serial_number,
                global_camera_enable_depth=bool(config.global_camera_enable_depth),
                wrist_camera_enable_depth=bool(config.wrist_camera_enable_depth),
                parent=parent,
            )
        worker.action_signal.connect(callbacks.action)
        worker.preview_signal.connect(callbacks.preview)
        worker.status_signal.connect(callbacks.status)
        worker.log_signal.connect(callbacks.log)
        worker.error_signal.connect(callbacks.error)

        def _clear_worker_reference() -> None:
            if self._worker is worker:
                self._worker = None

        worker.finished.connect(_clear_worker_reference)
        worker.finished.connect(callbacks.finished)
        self._worker = worker
        worker.start()
        return worker

    def stop_inference(self, timeout_ms: int = 4000) -> bool:
        worker = self._worker
        self._worker = None
        if worker is None:
            return True
        worker.stop()
        return bool(worker.wait(timeout_ms))

    def request_task_update(self, task_name: str, task_embedding_path: str) -> bool:
        worker = self._worker
        if worker is None or not worker.isRunning() or not hasattr(worker, "request_task_update"):
            return False
        worker.request_task_update(task_name=task_name, task_embedding_path=task_embedding_path)
        return True

    def set_preview_streaming(self, enabled: bool) -> bool:
        worker = self._worker
        if worker is None or not worker.isRunning() or not hasattr(worker, "set_preview_streaming"):
            return False
        worker.set_preview_streaming(bool(enabled))
        return True
