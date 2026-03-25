"""Subprocess lifecycle management for the teleop GUI."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

from PySide6.QtCore import QObject, QTimer, Signal


class ProcessManager(QObject):
    """Owns long-lived subprocesses and reports unexpected exits back to the GUI."""

    _REGISTRY_PATH = Path("/tmp/teleop_control_py_gui_processes.json")

    log_signal = Signal(str)
    process_exited = Signal(str, int)

    def __init__(self, parent: Optional[QObject] = None, poll_interval_ms: int = 1000) -> None:
        super().__init__(parent)
        self._owner_pid = os.getpid()
        self._processes: Dict[str, subprocess.Popen] = {}
        self._watch_timer = QTimer(self)
        self._watch_timer.timeout.connect(self._poll_subprocesses)
        self._watch_timer.start(poll_interval_ms)
        atexit.register(self._atexit_stop_all)
        self._reap_stale_processes()

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
        self._register_process(key, proc, cmd)
        return True

    def kill_subprocess(self, key: str, *, emit_logs: bool = True) -> None:
        proc = self._processes.get(key)
        if proc is None:
            self._remove_registry_entry(key, owner_pid=self._owner_pid)
            return

        try:
            self._terminate_process_tree(
                key=key,
                root_pid=proc.pid,
                root_pgid=self._safe_getpgid(proc.pid),
                emit_logs=emit_logs,
            )
        finally:
            self._processes.pop(key, None)
            self._remove_registry_entry(key, owner_pid=self._owner_pid)

    def stop_all(self, *, emit_logs: bool = True) -> None:
        for key in list(self._processes.keys()):
            self.kill_subprocess(key, emit_logs=emit_logs)

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
        self._remove_registry_entry(key, owner_pid=self._owner_pid)
        self.log_signal.emit(f"进程 {key} 已退出，返回码: {returncode}")
        self.process_exited.emit(key, returncode)

    def _atexit_stop_all(self) -> None:
        try:
            self.stop_all(emit_logs=False)
        except Exception:
            pass

    def _pid_exists(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        status_path = Path("/proc") / str(pid) / "status"
        try:
            content = status_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return True
        for line in content.splitlines():
            if not line.startswith("State:"):
                continue
            state_text = line.split(":", 1)[1].strip().upper()
            if state_text.startswith("Z"):
                return False
            break
        return True

    def _safe_getpgid(self, pid: int) -> Optional[int]:
        if pid <= 0:
            return None
        try:
            return os.getpgid(pid)
        except ProcessLookupError:
            return None
        except PermissionError:
            return None

    def _load_registry(self) -> list[dict]:
        try:
            payload = json.loads(self._REGISTRY_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except Exception:
            return []
        return payload if isinstance(payload, list) else []

    def _write_registry(self, entries: list[dict]) -> None:
        if not entries:
            try:
                self._REGISTRY_PATH.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            return

        tmp_path = self._REGISTRY_PATH.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(entries, ensure_ascii=True, indent=2), encoding="utf-8")
            tmp_path.replace(self._REGISTRY_PATH)
        except Exception:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    def _register_process(self, key: str, proc: subprocess.Popen, cmd: Sequence[str]) -> None:
        entries = self._load_registry()
        entries = [
            entry for entry in entries
            if not (
                int(entry.get("owner_pid", -1)) == self._owner_pid
                and str(entry.get("key", "")) == key
            )
        ]
        entries.append(
            {
                "key": key,
                "pid": int(proc.pid),
                "pgid": self._safe_getpgid(proc.pid) or int(proc.pid),
                "owner_pid": self._owner_pid,
                "cmd": list(cmd),
                "registered_at": time.time(),
            }
        )
        self._write_registry(entries)

    def _remove_registry_entry(self, key: str, *, owner_pid: Optional[int] = None) -> None:
        entries = self._load_registry()
        kept = []
        for entry in entries:
            if str(entry.get("key", "")) != key:
                kept.append(entry)
                continue
            if owner_pid is not None and int(entry.get("owner_pid", -1)) != owner_pid:
                kept.append(entry)
        self._write_registry(kept)

    def _reap_stale_processes(self) -> None:
        entries = self._load_registry()
        kept: list[dict] = []

        for entry in entries:
            key = str(entry.get("key", "")).strip()
            pid = int(entry.get("pid", 0) or 0)
            pgid = int(entry.get("pgid", 0) or 0)
            owner_pid = int(entry.get("owner_pid", 0) or 0)
            if not key or pid <= 0:
                continue

            process_alive = self._pid_exists(pid)
            owner_alive = self._pid_exists(owner_pid)
            if process_alive and owner_pid > 0 and not owner_alive:
                self._terminate_process_tree(
                    key=key,
                    root_pid=pid,
                    root_pgid=pgid or None,
                    emit_logs=False,
                )
                continue

            if process_alive:
                kept.append(entry)

        self._write_registry(kept)

    def _collect_descendant_pids(self, root_pid: int) -> set[int]:
        children: Dict[int, list[int]] = {}
        proc_root = Path("/proc")
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            status_path = entry / "status"
            try:
                content = status_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            pid: Optional[int] = None
            ppid: Optional[int] = None
            for line in content.splitlines():
                if line.startswith("Pid:"):
                    try:
                        pid = int(line.split(":", 1)[1].strip())
                    except Exception:
                        pid = None
                elif line.startswith("PPid:"):
                    try:
                        ppid = int(line.split(":", 1)[1].strip())
                    except Exception:
                        ppid = None
                if pid is not None and ppid is not None:
                    break

            if pid is None or ppid is None:
                continue
            children.setdefault(ppid, []).append(pid)

        descendants: set[int] = set()
        stack = [root_pid]
        while stack:
            parent = stack.pop()
            for child in children.get(parent, []):
                if child in descendants:
                    continue
                descendants.add(child)
                stack.append(child)
        descendants.discard(root_pid)
        return descendants

    def _collect_process_groups(self, pids: set[int]) -> set[int]:
        groups: set[int] = set()
        for pid in pids:
            pgid = self._safe_getpgid(pid)
            if pgid is not None:
                groups.add(pgid)
        return groups

    def _send_signal_to_groups(self, pgids: set[int], sig: signal.Signals) -> None:
        for pgid in pgids:
            if pgid <= 0:
                continue
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                continue
            except Exception:
                continue

    def _send_signal_to_pids(self, pids: set[int], sig: signal.Signals) -> None:
        for pid in pids:
            if pid <= 0:
                continue
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
            except Exception:
                continue

    def _wait_for_pids_to_exit(self, pids: set[int], timeout_sec: float) -> set[int]:
        deadline = time.monotonic() + max(0.0, timeout_sec)
        remaining = {pid for pid in pids if self._pid_exists(pid)}
        while remaining and time.monotonic() < deadline:
            time.sleep(0.1)
            remaining = {pid for pid in remaining if self._pid_exists(pid)}
        return remaining

    def _terminate_process_tree(
        self,
        *,
        key: str,
        root_pid: int,
        root_pgid: Optional[int],
        emit_logs: bool,
    ) -> None:
        tracked_pids = {root_pid}
        tracked_pids.update(self._collect_descendant_pids(root_pid))
        tracked_pids = {pid for pid in tracked_pids if pid > 0}
        tracked_pgids = self._collect_process_groups(tracked_pids)
        if root_pgid is not None and root_pgid > 0:
            tracked_pgids.add(root_pgid)

        alive_before = {pid for pid in tracked_pids if self._pid_exists(pid)}
        if not alive_before:
            return

        if emit_logs:
            self.log_signal.emit(f"正在终止 {key} (SIGINT)...")
        self._send_signal_to_groups(tracked_pgids, signal.SIGINT)
        self._send_signal_to_pids(alive_before, signal.SIGINT)

        remaining = self._wait_for_pids_to_exit(alive_before, timeout_sec=3.0)
        if not remaining:
            if emit_logs:
                self.log_signal.emit(f"{key} 已正常关闭。")
            return

        if emit_logs:
            self.log_signal.emit(f"{key} 存在残留子进程，正在强制终止 (SIGKILL)...")
        remaining_pgids = self._collect_process_groups(remaining)
        if root_pgid is not None and root_pgid > 0:
            remaining_pgids.add(root_pgid)
        self._send_signal_to_groups(remaining_pgids, signal.SIGKILL)
        self._send_signal_to_pids(remaining, signal.SIGKILL)

        survivors = self._wait_for_pids_to_exit(remaining, timeout_sec=2.0)
        if emit_logs:
            if survivors:
                survivor_text = ", ".join(str(pid) for pid in sorted(survivors))
                self.log_signal.emit(f"强制终止后仍有残留 PID: {survivor_text}")
            else:
                self.log_signal.emit(f"{key} 已强制关闭。")
