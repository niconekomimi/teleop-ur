"""Utilities for logging high-level inference actions to CSV for later analysis."""

from __future__ import annotations

import csv
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def _sanitize_fragment(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    normalized = normalized.strip("._-")
    return normalized or fallback


def _vector(values, size: int) -> list[float]:
    if values is None:
        return [float("nan")] * size
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return [float("nan")] * size
    if array.size < size:
        padded = np.full(size, np.nan, dtype=np.float64)
        padded[: array.size] = array
        array = padded
    return [float(value) for value in array[:size]]


def _timing_float(timing: Optional[dict[str, object]], key: str) -> float:
    if not timing:
        return float("nan")
    try:
        value = timing.get(key, float("nan"))
        return float(value)
    except Exception:
        return float("nan")


def _timing_int(timing: Optional[dict[str, object]], key: str, default: int) -> int:
    if not timing:
        return int(default)
    try:
        return int(timing.get(key, default))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class InferenceActionLogSession:
    session_dir: Path
    csv_path: Path
    metadata_path: Path


class InferenceActionLogger:
    """Records one inference session as a CSV plus a lightweight metadata JSON."""

    def __init__(self, output_root: Path, *, flush_every_n: int = 10) -> None:
        self._output_root = Path(output_root).expanduser().resolve()
        self._flush_every_n = max(1, int(flush_every_n))
        self._session: Optional[InferenceActionLogSession] = None
        self._csv_handle = None
        self._writer: Optional[csv.DictWriter] = None
        self._start_monotonic = 0.0
        self._step_idx = 0
        self._metadata: dict[str, object] = {}

    @property
    def active(self) -> bool:
        return self._session is not None and self._writer is not None and self._csv_handle is not None

    @property
    def current_csv_path(self) -> Optional[Path]:
        if self._session is None:
            return None
        return self._session.csv_path

    def start(
        self,
        *,
        checkpoint_dir: str,
        task_name: str,
        task_embedding_path: str,
        loop_hz: float,
        global_camera_source: str,
        wrist_camera_source: str,
        device: str,
        control_hz: float,
    ) -> InferenceActionLogSession:
        self.close()

        timestamp = datetime.now().astimezone()
        session_name = (
            f"{timestamp:%Y%m%d_%H%M%S}_"
            f"{_sanitize_fragment(task_name, 'task')}_"
            f"{_sanitize_fragment(Path(checkpoint_dir).name, 'model')}"
        )
        session_dir = self._output_root / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        session = InferenceActionLogSession(
            session_dir=session_dir,
            csv_path=session_dir / "actions.csv",
            metadata_path=session_dir / "metadata.json",
        )

        fieldnames = [
            "step_idx",
            "wall_time_iso",
            "elapsed_sec",
            "execution_enabled",
            "cycle_compute_ms",
            "camera_fetch_ms",
            "preprocess_ms",
            "robot_state_ms",
            "policy_call_ms",
            "is_replan_step",
            "plan_step_idx",
            "replan_every",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
            "gripper",
            "eef_x",
            "eef_y",
            "eef_z",
            "eef_qx",
            "eef_qy",
            "eef_qz",
            "eef_qw",
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
        ]
        csv_handle = session.csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()

        self._metadata = {
            "created_at_iso": timestamp.isoformat(timespec="seconds"),
            "checkpoint_dir": str(Path(checkpoint_dir).expanduser().resolve()),
            "task_name": str(task_name),
            "task_embedding_path": str(Path(task_embedding_path).expanduser().resolve()),
            "loop_hz": float(loop_hz),
            "control_hz": float(control_hz),
            "global_camera_source": str(global_camera_source),
            "wrist_camera_source": str(wrist_camera_source),
            "device": str(device or "auto"),
            "steps_recorded": 0,
        }
        session.metadata_path.write_text(json.dumps(self._metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        self._session = session
        self._csv_handle = csv_handle
        self._writer = writer
        self._start_monotonic = time.perf_counter()
        self._step_idx = 0
        return session

    def append(
        self,
        action,
        *,
        execution_enabled: bool,
        robot_state: Optional[dict] = None,
        timing: Optional[dict[str, object]] = None,
    ) -> bool:
        if not self.active:
            return False

        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size < 7:
            return False

        pose = _vector((robot_state or {}).get("pose"), 3)
        quat = _vector((robot_state or {}).get("quat"), 4)
        joints = _vector((robot_state or {}).get("joints"), 6)
        elapsed_sec = max(0.0, time.perf_counter() - self._start_monotonic)
        row = {
            "step_idx": int(self._step_idx),
            "wall_time_iso": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "elapsed_sec": float(elapsed_sec),
            "execution_enabled": 1 if execution_enabled else 0,
            "cycle_compute_ms": _timing_float(timing, "cycle_compute_ms"),
            "camera_fetch_ms": _timing_float(timing, "camera_fetch_ms"),
            "preprocess_ms": _timing_float(timing, "preprocess_ms"),
            "robot_state_ms": _timing_float(timing, "robot_state_ms"),
            "policy_call_ms": _timing_float(timing, "policy_call_ms"),
            "is_replan_step": _timing_int(timing, "is_replan_step", 0),
            "plan_step_idx": _timing_int(timing, "plan_step_idx", -1),
            "replan_every": _timing_int(timing, "replan_every", -1),
            "vx": float(action_array[0]),
            "vy": float(action_array[1]),
            "vz": float(action_array[2]),
            "wx": float(action_array[3]),
            "wy": float(action_array[4]),
            "wz": float(action_array[5]),
            "gripper": float(action_array[6]),
            "eef_x": pose[0],
            "eef_y": pose[1],
            "eef_z": pose[2],
            "eef_qx": quat[0],
            "eef_qy": quat[1],
            "eef_qz": quat[2],
            "eef_qw": quat[3],
            "joint_0": joints[0],
            "joint_1": joints[1],
            "joint_2": joints[2],
            "joint_3": joints[3],
            "joint_4": joints[4],
            "joint_5": joints[5],
        }
        assert self._writer is not None
        self._writer.writerow(row)
        self._step_idx += 1

        if self._csv_handle is not None and (self._step_idx % self._flush_every_n == 0):
            self._csv_handle.flush()

        return True

    def close(self) -> Optional[Path]:
        session = self._session
        if session is None:
            return None

        csv_path = session.csv_path
        try:
            self._metadata["closed_at_iso"] = datetime.now().astimezone().isoformat(timespec="seconds")
            self._metadata["steps_recorded"] = int(self._step_idx)
            self._metadata["duration_sec"] = float(max(0.0, time.perf_counter() - self._start_monotonic))
            session.metadata_path.write_text(
                json.dumps(self._metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        try:
            if self._csv_handle is not None:
                self._csv_handle.flush()
                self._csv_handle.close()
        finally:
            self._csv_handle = None
            self._writer = None
            self._session = None
            self._metadata = {}
            self._start_monotonic = 0.0
            self._step_idx = 0

        return csv_path

    @staticmethod
    def annotate_result(
        csv_path: Path | str,
        *,
        outcome: str,
        stop_reason: str = "",
    ) -> Optional[Path]:
        try:
            resolved_csv = Path(csv_path).expanduser().resolve()
        except Exception:
            return None

        metadata_path = resolved_csv.parent / "metadata.json"
        if not metadata_path.is_file():
            return None

        normalized_outcome = str(outcome).strip().lower()
        if normalized_outcome not in {"success", "failure", "unknown"}:
            normalized_outcome = "unknown"

        success_value: Optional[bool]
        if normalized_outcome == "success":
            success_value = True
        elif normalized_outcome == "failure":
            success_value = False
        else:
            success_value = None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

        metadata["task_outcome"] = normalized_outcome
        metadata["task_success"] = success_value
        metadata["result_annotated_at_iso"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if stop_reason:
            metadata["stop_reason"] = str(stop_reason)

        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata_path

    @staticmethod
    def discard_session(csv_path: Path | str) -> Optional[Path]:
        try:
            resolved_csv = Path(csv_path).expanduser().resolve()
        except Exception:
            return None

        session_dir = resolved_csv.parent
        if not session_dir.exists():
            return None

        shutil.rmtree(session_dir)
        return session_dir
