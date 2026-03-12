#!/usr/bin/env python3
"""teleop GUI side helpers for calling Real_IL online inference."""

from __future__ import annotations

import pickle
import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
except Exception:  # noqa: BLE001
    PackageNotFoundError = Exception

    def get_package_share_directory(_package_name: str) -> str:
        raise PackageNotFoundError()

from teleop_control_py.transform_utils import center_crop_square_and_resize_rgb


def _discover_workspace_root() -> Path:
    candidate_roots: list[Path] = []
    current_file = Path(__file__).resolve()
    candidate_roots.extend([current_file.parent] + list(current_file.parents))
    cwd = Path.cwd().resolve()
    candidate_roots.extend([cwd] + list(cwd.parents))

    seen: set[Path] = set()
    for candidate in candidate_roots:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "Real_IL" / "real_robot" / "infer.py").is_file() and (candidate / "models").exists():
            return candidate

    try:
        share_dir = Path(get_package_share_directory("teleop_control_py")).resolve()
        for candidate in [share_dir.parents[2], share_dir.parents[3]]:
            if (candidate / "Real_IL" / "real_robot" / "infer.py").is_file() and (candidate / "models").exists():
                return candidate
    except (PackageNotFoundError, IndexError):
        pass

    return current_file.parent


WORKSPACE_ROOT = _discover_workspace_root()
REAL_IL_ROOT = WORKSPACE_ROOT / "Real_IL"
MODELS_ROOT = WORKSPACE_ROOT / "models"
DATA_ROOT = WORKSPACE_ROOT / "data"
TASK_EMBEDDINGS_ROOT = REAL_IL_ROOT / "task_embeddings"
ROOT_ENV_NAME = "root"


def _normalize_camera_source(source: str) -> str:
    value = str(source).strip().lower()
    if value in {"realsense", "rs"}:
        return "realsense"
    if value == "oakd":
        return "oakd"
    return value


def _normalize_env_name(name: str) -> str:
    return "_".join(part for part in str(name).strip().lower().replace("-", "_").split("_") if part)


def _env_aliases(name: str) -> set[str]:
    normalized = _normalize_env_name(name)
    if not normalized:
        return set()

    aliases = {normalized}
    if normalized.endswith("ies") and len(normalized) > 3:
        aliases.add(normalized[:-3] + "y")
    if normalized.endswith("s") and len(normalized) > 1:
        aliases.add(normalized[:-1])
    else:
        aliases.add(normalized + "s")
    return aliases


def _embedding_env_name(pkl_path: Path) -> str:
    stem = pkl_path.stem
    if stem.startswith("data_"):
        stem = stem[len("data_") :]
    return _normalize_env_name(stem)


def _load_real_robot_policy_class():
    if not (REAL_IL_ROOT / "real_robot" / "infer.py").is_file():
        raise FileNotFoundError(f"Real_IL not found under workspace root: {REAL_IL_ROOT}")

    if str(REAL_IL_ROOT) not in sys.path:
        sys.path.insert(0, str(REAL_IL_ROOT))

    for module_name, module in list(sys.modules.items()):
        if module_name not in {"agents", "real_robot"} and not module_name.startswith(("agents.", "real_robot.")):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            module_path = Path(module_file).resolve()
        except Exception:
            sys.modules.pop(module_name, None)
            continue
        if REAL_IL_ROOT not in [module_path, *module_path.parents]:
            sys.modules.pop(module_name, None)

    from real_robot.infer import RealRobotPolicy

    return RealRobotPolicy


def _load_camera_client_classes():
    from teleop_control_py.camera_client import OAKClient, RealSenseClient

    return RealSenseClient, OAKClient


def _is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "last_model.pth").is_file() and (path / ".hydra" / "config.yaml").is_file()


def discover_checkpoint_dirs(models_root: Path = MODELS_ROOT) -> list[Path]:
    if not models_root.exists():
        return []

    seen: set[Path] = set()
    discovered: list[Path] = []
    if _is_checkpoint_dir(models_root):
        resolved = models_root.resolve()
        seen.add(resolved)
        discovered.append(resolved)

    for weights_path in sorted(models_root.rglob("last_model.pth")):
        checkpoint_dir = weights_path.parent.resolve()
        if checkpoint_dir in seen or not _is_checkpoint_dir(checkpoint_dir):
            continue
        seen.add(checkpoint_dir)
        discovered.append(checkpoint_dir)

    return discovered


def _task_name_from_demo_path(demo_path: Path) -> str:
    stem = demo_path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem


def _env_name_from_demo_path(demo_path: Path, data_root: Path = DATA_ROOT) -> str:
    try:
        relative_parent = demo_path.resolve().parent.relative_to(data_root.resolve())
    except ValueError:
        return demo_path.parent.name or ROOT_ENV_NAME

    return relative_parent.as_posix() if relative_parent.parts else ROOT_ENV_NAME


def discover_task_inventory(data_root: Path = DATA_ROOT) -> Dict[str, list[Path]]:
    inventory: Dict[str, list[Path]] = {}
    if not data_root.exists():
        return inventory

    for demo_path in sorted(data_root.rglob("*.hdf5")):
        env_name = _env_name_from_demo_path(demo_path, data_root=data_root)
        inventory.setdefault(env_name, []).append(demo_path.resolve())

    return inventory


def discover_task_envs(data_root: Path = DATA_ROOT) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(path.name for path in data_root.iterdir() if path.is_dir())


def discover_task_names(env_name: str, embeddings_root: Path = TASK_EMBEDDINGS_ROOT) -> list[str]:
    if not env_name:
        return []
    embedding_path = guess_embedding_path(env_name=env_name, embeddings_root=embeddings_root)
    if embedding_path is None:
        return []
    return sorted(load_embedding_keys(embedding_path))


def discover_demo_path(env_name: str, task_name: str, data_root: Path = DATA_ROOT) -> Optional[Path]:
    inventory = discover_task_inventory(data_root=data_root)
    for demo_path in inventory.get(env_name, []):
        if _task_name_from_demo_path(demo_path) == task_name:
            return demo_path
    return None


@lru_cache(maxsize=64)
def _embedding_keys_cache(path_str: str, mtime_ns: int) -> frozenset[str]:
    path = Path(path_str)
    with open(path, "rb") as handle:
        embedding_map = pickle.load(handle)
    if not isinstance(embedding_map, dict):
        return frozenset()
    return frozenset(str(key) for key in embedding_map.keys())


def load_embedding_keys(pkl_path: Path) -> set[str]:
    if not pkl_path.is_file():
        return set()
    try:
        stat = pkl_path.stat()
        return set(_embedding_keys_cache(str(pkl_path.resolve()), int(stat.st_mtime_ns)))
    except Exception:
        return set()


def guess_embedding_path(
    env_name: str,
    task_name: str | None = None,
    embeddings_root: Path = TASK_EMBEDDINGS_ROOT,
) -> Optional[Path]:
    if not embeddings_root.exists():
        return None

    env_aliases = _env_aliases(env_name)
    if not env_aliases:
        return None

    candidates = sorted(embeddings_root.glob("*.pkl"))
    exact_matches: list[Path] = []
    partial_matches: list[Path] = []
    for candidate in candidates:
        candidate_env = _embedding_env_name(candidate)
        candidate_aliases = _env_aliases(candidate_env)
        if env_aliases & candidate_aliases:
            exact_matches.append(candidate.resolve())
            continue
        if any(alias in candidate_env or candidate_env in alias for alias in env_aliases):
            partial_matches.append(candidate.resolve())

    for candidate in [*exact_matches, *partial_matches]:
        if task_name is None or task_name in load_embedding_keys(candidate):
            return candidate
    return None


def format_action(action: np.ndarray) -> str:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    formatter = {"float_kind": lambda value: f"{value: .6f}"}
    return np.array2string(action, formatter=formatter)


def build_robot_state_vector(robot_state: Optional[dict]) -> Optional[np.ndarray]:
    if not robot_state:
        return None

    joints = robot_state.get("joints", [])
    gripper = robot_state.get("gripper", None)
    if len(joints) != 6 or gripper is None:
        return None

    return np.asarray([*joints, float(gripper)], dtype=np.float32)


class InferenceWorker(QThread):
    action_signal = Signal(object)
    preview_signal = Signal(object, object)
    status_signal = Signal(str)
    log_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(
        self,
        checkpoint_dir: str,
        task_name: str,
        task_embedding_path: str,
        global_camera_source: str,
        wrist_camera_source: str,
        loop_hz: float,
        device: str | None = None,
        state_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.checkpoint_dir = str(Path(checkpoint_dir).expanduser().resolve())
        self.task_name = str(task_name).strip()
        self.task_embedding_path = str(Path(task_embedding_path).expanduser().resolve())
        self.global_camera_source = _normalize_camera_source(global_camera_source)
        self.wrist_camera_source = _normalize_camera_source(wrist_camera_source)
        self.loop_hz = max(0.2, float(loop_hz))
        self.device = str(device).strip() if device else None
        self.state_provider = state_provider
        self._running = True

    def info(self, message: str) -> None:
        self.log_signal.emit(str(message))

    def warn(self, message: str) -> None:
        self.log_signal.emit(str(message))

    def stop(self) -> None:
        self._running = False

    def _build_camera_client(self, source: str):
        RealSenseClient, OAKClient = _load_camera_client_classes()
        if source == "realsense":
            return RealSenseClient(logger=self)
        if source == "oakd":
            return OAKClient(logger=self)
        raise ValueError(f"Unsupported camera source: {source}")

    @staticmethod
    def _resolve_image_size(policy, obs_key: str) -> int:
        shape_meta = getattr(policy.cfg, "shape_meta", None)
        if shape_meta is None:
            return 224

        obs_cfg = getattr(shape_meta, "obs", None)
        if obs_cfg is None:
            return 224

        key_cfg = getattr(obs_cfg, obs_key, None)
        if key_cfg is None:
            return 224

        shape = getattr(key_cfg, "shape", None)
        if shape is None or len(shape) < 2:
            return 224

        height = int(shape[-2])
        width = int(shape[-1])
        return max(height, width)

    def _read_robot_state(self) -> Optional[np.ndarray]:
        if self.state_provider is None:
            return None
        try:
            state = self.state_provider()
        except Exception as exc:
            self.log_signal.emit(f"读取机器人状态失败: {exc}")
            return None

        if state is None:
            return None

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        return state if state.size > 0 else None

    def _wait_for_initial_frames(self, global_camera, wrist_camera) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        global_bgr = None
        wrist_bgr = None

        while self._running:
            if global_bgr is None:
                global_bgr = global_camera.get_bgr_frame()
            if wrist_bgr is None:
                wrist_bgr = wrist_camera.get_bgr_frame()

            if global_bgr is not None and wrist_bgr is not None:
                return global_bgr, wrist_bgr

            self.status_signal.emit("等待首帧")
            time.sleep(0.01)

        return None, None

    def run(self) -> None:
        global_camera = None
        wrist_camera = None
        policy = None
        target_period = 1.0 / max(self.loop_hz, 1e-6)
        completed_cycles: deque[float] = deque(maxlen=max(5, int(round(self.loop_hz * 2))))

        try:
            self.status_signal.emit("加载模型中")
            policy_class = _load_real_robot_policy_class()
            policy = policy_class(
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                task_name=self.task_name,
                task_embedding_path=self.task_embedding_path,
            )
            agentview_size = self._resolve_image_size(policy, "agentview_image")
            wrist_size = self._resolve_image_size(policy, "eye_in_hand_image")

            self.status_signal.emit("打开相机中")
            global_camera = self._build_camera_client(self.global_camera_source)
            wrist_camera = self._build_camera_client(self.wrist_camera_source)
            global_bgr, wrist_bgr = self._wait_for_initial_frames(global_camera, wrist_camera)
            if not self._running:
                return
            if global_bgr is None or wrist_bgr is None:
                raise RuntimeError("相机已打开，但未收到首帧")
            self.preview_signal.emit(global_bgr, wrist_bgr)

            self.log_signal.emit(
                "推理已启动: "
                f"model={self.checkpoint_dir}, task={self.task_name}, "
                f"embedding={self.task_embedding_path}, cameras=({self.global_camera_source}, {self.wrist_camera_source}), "
                f"device={self.device or 'auto'}"
            )

            self.status_signal.emit("相机已就绪")
            next_cycle = time.perf_counter()
            while self._running:
                cycle_start = time.perf_counter()
                if completed_cycles:
                    global_bgr = global_camera.get_bgr_frame()
                    wrist_bgr = wrist_camera.get_bgr_frame()

                if global_bgr is None or wrist_bgr is None:
                    self.status_signal.emit("等待相机帧")
                    time.sleep(min(target_period, 0.05))
                    next_cycle = time.perf_counter()
                    continue

                self.preview_signal.emit(global_bgr, wrist_bgr)
                agentview_rgb = center_crop_square_and_resize_rgb(global_bgr, agentview_size)
                wrist_rgb = center_crop_square_and_resize_rgb(wrist_bgr, wrist_size)
                robot_state = self._read_robot_state()
                action = policy.predict_action(
                    agentview_image=agentview_rgb,
                    eye_in_hand_image=wrist_rgb,
                    robot_states=robot_state,
                )
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                self.action_signal.emit(action)

                now = time.perf_counter()
                completed_cycles.append(now)
                if len(completed_cycles) >= 2:
                    window_elapsed = max(completed_cycles[-1] - completed_cycles[0], 1e-6)
                    actual_hz = (len(completed_cycles) - 1) / window_elapsed
                else:
                    actual_hz = 0.0
                latency_ms = (now - cycle_start) * 1000.0
                self.status_signal.emit(f"运行中 | {actual_hz:.2f} Hz | {latency_ms:.1f} ms")

                next_cycle += target_period
                sleep_duration = next_cycle - time.perf_counter()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                else:
                    next_cycle = time.perf_counter()
        except Exception as exc:
            self.error_signal.emit(str(exc))
        finally:
            for camera in (global_camera, wrist_camera):
                if camera is None:
                    continue
                try:
                    camera.stop()
                except Exception:
                    pass

            self.status_signal.emit("未启动")
