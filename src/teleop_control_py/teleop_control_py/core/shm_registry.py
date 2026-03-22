"""Process-level registry for CameraDaemon processes backed by shared memory."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from teleop_control_py.hardware.cameras.shm_camera import CameraDaemon


@dataclass
class DaemonEntry:
    spec_identifier: str
    pid: int
    process: "CameraDaemon"
    registered_at: float  # time.monotonic()


class SHMRegistry:
    """Process-level singleton that tracks all CameraDaemon processes started by this process."""

    _entries: Dict[str, DaemonEntry] = {}

    @classmethod
    def register(cls, spec_identifier: str, daemon: "CameraDaemon") -> None:
        cls._entries[spec_identifier] = DaemonEntry(
            spec_identifier=spec_identifier,
            pid=daemon.pid or 0,
            process=daemon,
            registered_at=time.monotonic(),
        )

    @classmethod
    def release(cls, spec_identifier: str) -> None:
        """Remove the entry without terminating the process."""
        cls._entries.pop(spec_identifier, None)

    @classmethod
    def is_alive(cls, spec_identifier: str) -> bool:
        entry = cls._entries.get(spec_identifier)
        if entry is None:
            return False
        if entry.pid <= 0:
            return False
        try:
            os.kill(entry.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def list_all(cls) -> list[DaemonEntry]:
        return list(cls._entries.values())

    @classmethod
    def cleanup_dead(cls) -> list[str]:
        """Remove entries for dead processes; return list of cleaned identifiers."""
        dead: list[str] = []
        for identifier in list(cls._entries.keys()):
            if not cls.is_alive(identifier):
                cls._entries.pop(identifier, None)
                dead.append(identifier)
        return dead
