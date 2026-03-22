"""Unified health monitor for subprocesses, camera daemons, ROS topics, and Qt threads."""

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Protocol, runtime_checkable

from PySide6.QtCore import QObject, QTimer, Signal

from .shm_registry import SHMRegistry


# ---------------------------------------------------------------------------
# WatchdogTarget protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class WatchdogTarget(Protocol):
    key: str

    def check(self) -> bool:
        """Return True if healthy."""
        ...

    def on_failure(self) -> None:
        """Called once when the target transitions to unhealthy."""
        ...


# ---------------------------------------------------------------------------
# Concrete target implementations
# ---------------------------------------------------------------------------

class ProcessWatchdogTarget:
    """Monitors a subprocess.Popen; calls on_failure_cb on unexpected exit."""

    def __init__(
        self,
        key: str,
        process,
        on_failure_cb: Optional[Callable[[], None]] = None,
    ) -> None:
        self.key = key
        self._process = process
        self._on_failure_cb = on_failure_cb

    def check(self) -> bool:
        return self._process.poll() is None

    def on_failure(self) -> None:
        if self._on_failure_cb:
            self._on_failure_cb()


class DaemonWatchdogTarget:
    """Monitors a CameraDaemon via SHMRegistry; cleans up dead entries on failure."""

    def __init__(
        self,
        key: str,
        spec_identifier: str,
        on_failure_cb: Optional[Callable[[], None]] = None,
    ) -> None:
        self.key = key
        self._spec_identifier = spec_identifier
        self._on_failure_cb = on_failure_cb

    def check(self) -> bool:
        return SHMRegistry.is_alive(self._spec_identifier)

    def on_failure(self) -> None:
        SHMRegistry.cleanup_dead()
        if self._on_failure_cb:
            self._on_failure_cb()


class TopicWatchdogTarget:
    """
    ROS topic heartbeat: call touch() on each message receipt.
    check() returns False if no message arrived within timeout_sec.
    """

    def __init__(
        self,
        key: str,
        timeout_sec: float,
        on_failure_cb: Optional[Callable[[], None]] = None,
    ) -> None:
        self.key = key
        self.timeout_sec = timeout_sec
        self._last_seen: float = time.monotonic()
        self._on_failure_cb = on_failure_cb

    def touch(self) -> None:
        self._last_seen = time.monotonic()

    def check(self) -> bool:
        return (time.monotonic() - self._last_seen) < self.timeout_sec

    def on_failure(self) -> None:
        if self._on_failure_cb:
            self._on_failure_cb()


class ThreadWatchdogTarget:
    """
    Monitors a QThread (e.g. InferenceWorker).
    call touch() inside the thread loop each iteration.
    check() combines isRunning() with an activity timestamp.
    """

    def __init__(
        self,
        key: str,
        thread,
        activity_timeout_sec: float = 10.0,
        on_failure_cb: Optional[Callable[[], None]] = None,
    ) -> None:
        self.key = key
        self._thread = thread
        self._activity_timeout_sec = activity_timeout_sec
        self._last_active: float = time.monotonic()
        self._on_failure_cb = on_failure_cb

    def touch(self) -> None:
        self._last_active = time.monotonic()

    def check(self) -> bool:
        if not self._thread.isRunning():
            return False
        return (time.monotonic() - self._last_active) < self._activity_timeout_sec

    def on_failure(self) -> None:
        if self._on_failure_cb:
            self._on_failure_cb()


# ---------------------------------------------------------------------------
# Watchdog orchestrator
# ---------------------------------------------------------------------------

class Watchdog(QObject):
    """Polls all registered WatchdogTargets and notifies on failure."""

    # (key, reason)
    failure_signal = Signal(str, str)

    def __init__(
        self,
        orchestrator_notify: Optional[Callable[[str], None]] = None,
        poll_interval_ms: int = 1000,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._orchestrator_notify = orchestrator_notify
        self._targets: dict[str, WatchdogTarget] = {}
        self._failed: set[str] = set()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(poll_interval_ms)

    def add_target(self, target: WatchdogTarget) -> None:
        self._targets[target.key] = target
        self._failed.discard(target.key)

    def remove_target(self, key: str) -> None:
        self._targets.pop(key, None)
        self._failed.discard(key)

    def _poll(self) -> None:
        for key, target in list(self._targets.items()):
            healthy = target.check()
            if healthy:
                self._failed.discard(key)
            elif key not in self._failed:
                self._failed.add(key)
                try:
                    target.on_failure()
                except Exception:
                    pass
                if self._orchestrator_notify is not None:
                    try:
                        self._orchestrator_notify(key)
                    except Exception:
                        pass
                self.failure_signal.emit(key, "unhealthy")
