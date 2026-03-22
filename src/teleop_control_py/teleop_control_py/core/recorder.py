"""Recorder service wrapper around the HDF5 writer thread."""

from __future__ import annotations

import os
import queue
import threading
from typing import Optional

import h5py

from ..data.hdf5_writer import Command, HDF5WriterThread, Sample


class RecorderService:
    def __init__(
        self,
        output_path: str,
        *,
        compression: Optional[str],
        queue_maxsize: int,
        batch_size: int,
        flush_every_n: int,
        logger: Optional[object] = None,
    ) -> None:
        self.output_path = str(output_path)
        self._logger = logger
        self._lock = threading.Lock()
        self._queue: 'queue.Queue[object]' = queue.Queue(maxsize=max(1, int(queue_maxsize)))
        self._writer = HDF5WriterThread(
            output_path=self.output_path,
            item_queue=self._queue,
            compression=compression,
            batch_size=int(batch_size),
            flush_every_n=int(flush_every_n),
            logger=logger,
        )
        self._writer.start()
        self._recording = False
        self._current_demo_name: Optional[str] = None
        self._demo_index = self._discover_next_demo_index(self.output_path)

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def current_demo_name(self) -> Optional[str]:
        with self._lock:
            return self._current_demo_name

    @property
    def next_demo_index(self) -> int:
        with self._lock:
            return self._demo_index

    def queue_size(self) -> int:
        try:
            return int(self._queue.qsize())
        except Exception:
            return -1

    def start_demo(self) -> tuple[bool, str, Optional[str]]:
        with self._lock:
            if self._recording:
                return False, 'Already recording', self._current_demo_name
            demo_name = f'demo_{self._demo_index}'
            try:
                self._queue.put_nowait(Command(kind='start_demo', demo_name=demo_name))
            except queue.Full:
                return False, 'Queue full; cannot start demo', None
            self._recording = True
            self._current_demo_name = demo_name
            self._demo_index += 1
            return True, f'Started recording {demo_name}', demo_name

    def stop_demo(self) -> tuple[bool, str, Optional[str]]:
        with self._lock:
            if not self._recording or self._current_demo_name is None:
                return False, 'Not recording', None
            demo_name = self._current_demo_name
            self._recording = False
            self._current_demo_name = None
        try:
            self._queue.put_nowait(Command(kind='stop_demo', demo_name=demo_name))
        except queue.Full:
            self._log('warn', 'Recorder queue full while stopping demo; metadata will be finalized on close.')
        return True, f'Stopped recording {demo_name}', demo_name

    def discard_last_demo(self) -> tuple[bool, str, Optional[str]]:
        with self._lock:
            if self._recording:
                return False, 'Cannot discard demo while recording', None
            if self._demo_index <= 0:
                return False, 'No demo to discard', None
            demo_name = f'demo_{self._demo_index - 1}'
            try:
                self._queue.put_nowait(Command(kind='discard_demo', demo_name=demo_name))
            except queue.Full:
                return False, 'Queue full; cannot discard demo', None
            self._demo_index -= 1
            return True, f'Discarded {demo_name}', demo_name

    def enqueue_sample(self, sample: Sample) -> bool:
        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            return False

    def close(self) -> None:
        try:
            if self.recording:
                self.stop_demo()
        except Exception:
            pass
        try:
            self._writer.stop()
        except Exception:
            pass
        try:
            self._writer.join(timeout=2.0)
        except Exception:
            pass

    def _log(self, level: str, message: str) -> None:
        if self._logger is None:
            return
        log_fn = getattr(self._logger, level, None)
        if callable(log_fn):
            log_fn(message)

    def _discover_next_demo_index(self, output_path: str) -> int:
        if not output_path or not os.path.exists(output_path):
            return 0
        try:
            with h5py.File(output_path, 'r') as handle:
                data_group = handle.get('data')
                if not isinstance(data_group, h5py.Group):
                    return 0
                next_index = 0
                for name in data_group.keys():
                    if not name.startswith('demo_'):
                        continue
                    suffix = name[5:]
                    if suffix.isdigit():
                        next_index = max(next_index, int(suffix) + 1)
                return next_index
        except Exception as exc:  # noqa: BLE001
            self._log('warn', f'Failed to discover next demo index: {exc!r}')
            return 0
