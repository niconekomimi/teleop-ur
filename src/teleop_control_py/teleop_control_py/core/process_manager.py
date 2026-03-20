"""Subprocess lifecycle management for the teleop GUI."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

from PySide6.QtCore import QObject, QTimer, Signal


class ProcessManager(QObject):
    """Owns long-lived subprocesses and reports unexpected exits back to the GUI."""

    log_signal = Signal(str)
    process_exited = Signal(str, int)

    def __init__(self, parent: Optional[QObject] = None, poll_interval_ms: int = 1000) -> None:
        super().__init__(parent)
        self._processes: Dict[str, subprocess.Popen] = {}
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._poll_subprocesses)
        self._watch_timer.start(poll_interval_ms)

    def keys(self) -> list[str]:
        return list(self._processes.keys())

    def is_running(self, key: str) -> bool:
        proc = self._processes.get(key)
        return proc is not None and proc.poll() is None

    def run_subprocess(self, key: str, cmd_list: Sequence[str]) -> bool:
        cmd = [str(item) for item in cmd_list]
        existing = self._processes.get(key)
        if existing is not None and existing.poll() is None:
            self.log_signal.emit(f"{key} 已在运行。")
            return True

        self._processes.pop(key, None)
        self.log_signal.emit(f"执行指令: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        except Exception as exc:
            self.log_signal.emit(f"启动 {key} 失败: {exc}")
            return False

        self._processes[key] = proc
        return True

    def kill_subprocess(self, key: str) -> None:
        proc = self._processes.get(key)
        if proc is None:
            return

        try:
            if proc.poll() is None:
                self.log_signal.emit(f"正在终止 {key} (SIGINT)...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                    proc.wait(timeout=3)
                    self.log_signal.emit(f"{key} 已正常关闭。")
                except subprocess.TimeoutExpired:
                    self.log_signal.emit(f"超时！正在强制终止 {key} (SIGKILL)...")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        proc.wait(timeout=2)
                    except Exception as exc:
                        self.log_signal.emit(f"强制终止失败: {exc}")
                except Exception as exc:
                    self.log_signal.emit(f"终止进程发生异常: {exc}")
        finally:
            self._processes.pop(key, None)

    def stop_all(self) -> None:
        for key in list(self._processes.keys()):
            self.kill_subprocess(key)

    def list_ros_image_topics(self, timeout_sec: float = 2.0, log_errors: bool = False) -> list[str]:
        try:
            result = subprocess.run(
                ["ros2", "topic", "list", "-t"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except Exception as exc:
            if log_errors:
                self.log_signal.emit(f"刷新手势识别输入话题失败: {exc}")
            return []

        if result.returncode != 0:
            if log_errors:
                detail = result.stderr.strip() or f"返回码 {result.returncode}"
                self.log_signal.emit(f"刷新手势识别输入话题失败: {detail}")
            return []

        topics: list[str] = []
        for line in result.stdout.splitlines():
            entry = line.strip()
            if not entry or "sensor_msgs/msg/Image" not in entry:
                continue
            topics.append(entry.split()[0])
        return topics

    def find_package_share_file(
        self,
        package_name: str,
        relative_path: str,
        timeout_sec: float = 2.0,
    ) -> Optional[Path]:
        try:
            result = subprocess.run(
                ["ros2", "pkg", "prefix", package_name],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        prefix = result.stdout.strip()
        if not prefix:
            return None

        candidate = Path(prefix) / "share" / package_name / relative_path
        return candidate if candidate.exists() else None

    def _poll_subprocesses(self) -> None:
        for key, proc in list(self._processes.items()):
            returncode = proc.poll()
            if returncode is None:
                continue
            self._handle_process_exit(key, int(returncode))

    def _handle_process_exit(self, key: str, returncode: int) -> None:
        self._processes.pop(key, None)
        self.log_signal.emit(f"进程 {key} 已退出，返回码: {returncode}")
        self.process_exited.emit(key, returncode)
