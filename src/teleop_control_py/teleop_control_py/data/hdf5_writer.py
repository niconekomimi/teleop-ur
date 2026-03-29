#!/usr/bin/env python3
"""HDF5 持久化线程与采样数据结构。"""

from __future__ import annotations

import os
import queue
import sys
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class Sample:
    """单帧采样结果，字段布局与 robomimic/LIBERO 对齐。"""

    demo_name: str
    agentview_rgb: np.ndarray
    eye_in_hand_rgb: np.ndarray
    robot0_joint_pos: np.ndarray
    robot0_gripper_qpos: np.ndarray
    robot0_eef_pos: np.ndarray
    robot0_eef_quat: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class Command:
    """控制写线程的简单命令对象。"""

    kind: str
    demo_name: Optional[str] = None


class HDF5WriterThread(threading.Thread):
    """后台写线程，负责将样本批量追加到 HDF5。"""

    def __init__(
        self,
        output_path: str,
        item_queue: "queue.Queue[object]",
        compression: Optional[str] = "lzf",
        batch_size: int = 32,
        flush_every_n: int = 200,
        logger: Optional[object] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._output_path = output_path
        self._queue = item_queue
        self._compression = compression
        self._batch_size = max(1, int(batch_size))
        self._flush_every_n = max(1, int(flush_every_n))
        self._logger = logger

        self._stop_evt = threading.Event()
        self._h5: Optional[h5py.File] = None
        self._data_group: Optional[h5py.Group] = None
        self._current_demo: Optional[str] = None
        self._demo_handles: Dict[str, Dict[str, object]] = {}
        self._demo_counts: Dict[str, int] = {}

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self._queue.put_nowait(Command(kind="close"))
        except queue.Full:
            pass

    def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            log_fn = getattr(self._logger, level, None)
            if callable(log_fn):
                try:
                    log_fn(msg)
                    return
                except Exception:
                    pass
        print(f"[HDF5WriterThread][{level.upper()}] {msg}", file=sys.stderr, flush=True)

    def _ensure_file_open(self) -> None:
        if self._h5 is not None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        self._h5 = h5py.File(self._output_path, "a")
        self._data_group = self._h5.require_group("data")

    def _create_demo_if_needed(self, demo_name: str) -> None:
        assert self._data_group is not None
        if demo_name in self._demo_handles:
            return

        demo_group = self._data_group.require_group(demo_name)
        obs_group = demo_group.require_group("obs")

        def _dataset(
            group: h5py.Group,
            name: str,
            shape_tail: Tuple[int, ...],
            dtype: np.dtype,
            compression: Optional[str],
        ) -> h5py.Dataset:
            if name not in group:
                return group.create_dataset(
                    name,
                    shape=(0,) + shape_tail,
                    maxshape=(None,) + shape_tail,
                    chunks=(1,) + shape_tail,
                    dtype=dtype,
                    compression=compression,
                )

            dataset_obj = group[name]
            if not isinstance(dataset_obj, h5py.Dataset):
                raise TypeError(f"Expected dataset at {group.name}/{name}, got {type(dataset_obj)!r}")

            dataset = dataset_obj
            expected_rank = 1 + len(shape_tail)
            if len(dataset.shape) != expected_rank or tuple(dataset.shape[1:]) != shape_tail:
                raise ValueError(
                    f"Dataset {dataset.name} shape mismatch: existing={dataset.shape}, expected=(N, {shape_tail})"
                )
            if dataset.dtype != np.dtype(dtype):
                raise ValueError(
                    f"Dataset {dataset.name} dtype mismatch: existing={dataset.dtype}, expected={np.dtype(dtype)}"
                )

            is_resizable = dataset.chunks is not None and dataset.maxshape is not None and dataset.maxshape[0] is None
            if is_resizable:
                return dataset

            existing = dataset[...]
            del group[name]
            migrated = group.create_dataset(
                name,
                shape=existing.shape,
                maxshape=(None,) + shape_tail,
                chunks=(1,) + shape_tail,
                dtype=np.dtype(dtype),
                compression=compression,
            )
            if existing.shape[0] > 0:
                migrated[...] = existing
            self._log("warn", f"HDF5: migrated legacy non-resizable dataset {migrated.name}")
            return migrated

        existing_count = 0
        if "actions" in demo_group and isinstance(demo_group["actions"], h5py.Dataset):
            existing_count = int(demo_group["actions"].shape[0])
        existing_count = max(existing_count, int(demo_group.attrs.get("num_samples", 0)))

        handles: Dict[str, object] = {
            "demo_group": demo_group,
            "actions": _dataset(demo_group, "actions", (7,), np.float32, None),
            "agentview_rgb": _dataset(obs_group, "agentview_rgb", (224, 224, 3), np.uint8, self._compression),
            "eye_in_hand_rgb": _dataset(obs_group, "eye_in_hand_rgb", (224, 224, 3), np.uint8, self._compression),
            "robot0_joint_pos": _dataset(obs_group, "robot0_joint_pos", (6,), np.float32, None),
            "robot0_gripper_qpos": _dataset(obs_group, "robot0_gripper_qpos", (1,), np.float32, None),
            "robot0_eef_pos": _dataset(obs_group, "robot0_eef_pos", (3,), np.float32, None),
            "robot0_eef_quat": _dataset(obs_group, "robot0_eef_quat", (4,), np.float32, None),
        }

        demo_group.attrs["num_samples"] = existing_count
        self._demo_handles[demo_name] = handles
        self._demo_counts[demo_name] = existing_count
        if existing_count > 0:
            self._log("warn", f"HDF5: reusing existing group data/{demo_name}, appending after {existing_count} samples")
        else:
            self._log("info", f"HDF5: created group data/{demo_name}")

    def _finalize_demo(self, demo_name: str) -> None:
        handles = self._demo_handles.get(demo_name)
        if not handles:
            return
        demo_group: h5py.Group = handles["demo_group"]  # type: ignore[assignment]
        demo_group.attrs["num_samples"] = int(self._demo_counts.get(demo_name, 0))

    def _discard_demo(self, demo_name: str) -> bool:
        self._ensure_file_open()
        assert self._h5 is not None
        assert self._data_group is not None

        if demo_name == self._current_demo:
            self._current_demo = None

        self._demo_handles.pop(demo_name, None)
        self._demo_counts.pop(demo_name, None)

        if demo_name not in self._data_group:
            self._log("warn", f"HDF5: cannot discard missing group data/{demo_name}")
            return False

        del self._data_group[demo_name]
        self._h5.flush()
        self._log("info", f"HDF5: discarded group data/{demo_name}")
        return True

    def _append_batch(self, demo_name: str, batch: "list[Sample]") -> None:
        self._ensure_file_open()
        assert self._h5 is not None
        assert self._data_group is not None
        self._create_demo_if_needed(demo_name)
        handles = self._demo_handles[demo_name]

        start_index = int(self._demo_counts[demo_name])
        batch_size = len(batch)
        end_index = start_index + batch_size

        def _resize_and_write(dataset: h5py.Dataset, data: np.ndarray) -> None:
            dataset.resize((end_index,) + dataset.shape[1:])
            dataset[start_index:end_index] = data

        _resize_and_write(handles["actions"], np.stack([sample.actions for sample in batch], axis=0))  # type: ignore[arg-type]
        _resize_and_write(
            handles["agentview_rgb"],
            np.stack([sample.agentview_rgb for sample in batch], axis=0),
        )  # type: ignore[arg-type]
        _resize_and_write(
            handles["eye_in_hand_rgb"],
            np.stack([sample.eye_in_hand_rgb for sample in batch], axis=0),
        )  # type: ignore[arg-type]
        _resize_and_write(
            handles["robot0_joint_pos"],
            np.stack([sample.robot0_joint_pos for sample in batch], axis=0),
        )  # type: ignore[arg-type]
        _resize_and_write(
            handles["robot0_gripper_qpos"],
            np.stack([sample.robot0_gripper_qpos for sample in batch], axis=0),
        )  # type: ignore[arg-type]
        _resize_and_write(
            handles["robot0_eef_pos"],
            np.stack([sample.robot0_eef_pos for sample in batch], axis=0),
        )  # type: ignore[arg-type]
        _resize_and_write(
            handles["robot0_eef_quat"],
            np.stack([sample.robot0_eef_quat for sample in batch], axis=0),
        )  # type: ignore[arg-type]

        self._demo_counts[demo_name] = end_index
        if (end_index % self._flush_every_n) == 0:
            self._h5.flush()

    def run(self) -> None:
        pending: "list[Sample]" = []
        pending_demo: Optional[str] = None

        try:
            while not self._stop_evt.is_set():
                try:
                    item = self._queue.get(timeout=0.25)
                except queue.Empty:
                    item = None

                if item is None:
                    if pending_demo is not None and pending:
                        self._append_batch(pending_demo, pending)
                        pending = []
                    continue

                if isinstance(item, Command):
                    if item.kind == "start_demo":
                        if item.demo_name is None:
                            continue
                        if pending_demo is not None and pending:
                            self._append_batch(pending_demo, pending)
                            pending = []
                        pending_demo = item.demo_name
                        self._ensure_file_open()
                        self._create_demo_if_needed(item.demo_name)
                        self._current_demo = item.demo_name
                    elif item.kind == "stop_demo":
                        if pending_demo is not None and pending:
                            self._append_batch(pending_demo, pending)
                            pending = []
                        if item.demo_name is not None:
                            self._finalize_demo(item.demo_name)
                        self._current_demo = None
                        pending_demo = None
                        if self._h5 is not None:
                            self._h5.flush()
                    elif item.kind == "discard_demo":
                        if pending_demo is not None and pending:
                            self._append_batch(pending_demo, pending)
                            pending = []
                        if item.demo_name is not None:
                            self._discard_demo(item.demo_name)
                            if pending_demo == item.demo_name:
                                pending_demo = None
                        if self._h5 is not None:
                            self._h5.flush()
                    elif item.kind == "close":
                        break
                    continue

                if isinstance(item, Sample):
                    if pending_demo is None:
                        pending_demo = item.demo_name
                    if item.demo_name != pending_demo:
                        if pending:
                            self._append_batch(pending_demo, pending)
                            pending = []
                        pending_demo = item.demo_name
                    pending.append(item)
                    if len(pending) >= self._batch_size:
                        assert pending_demo is not None
                        self._append_batch(pending_demo, pending)
                        pending = []

            if pending_demo is not None and pending:
                self._append_batch(pending_demo, pending)

            if self._current_demo is not None:
                self._finalize_demo(self._current_demo)

        except Exception as exc:  # noqa: BLE001
            self._log("error", f"HDF5 writer thread crashed: {exc!r}")
        finally:
            try:
                if self._h5 is not None:
                    for demo_name in list(self._demo_handles.keys()):
                        self._finalize_demo(demo_name)
                    self._h5.flush()
                    self._h5.close()
            except Exception as exc:  # noqa: BLE001
                self._log("error", f"HDF5 writer thread close failed: {exc!r}")