"""HDF5 数据集结构重建工具。"""

from __future__ import annotations

import os
from typing import Iterable

import h5py
import numpy as np

VALID_COMPRESSION_VALUES = {"inherit", "gzip", "lzf"}


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key in src.keys():
        dst[key] = src[key]


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    names = [name for name in data_group.keys() if str(name).startswith("demo_")]

    def key_fn(name: str):
        try:
            return int(name.split("_")[-1])
        except Exception:
            return name

    return sorted(names, key=key_fn)


def quat_to_rotvec_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion last dim must be 4, got {quat.shape}")

    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    quat = quat / norm

    xyz = quat[..., :3]
    w = np.clip(quat[..., 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))

    axis = np.zeros_like(xyz, dtype=np.float64)
    valid = s > 1e-8
    if np.any(valid):
        axis[valid] = xyz[valid] / s[valid][..., None]

    rotvec = axis * angle[..., None]
    rotvec[~valid] = 0.0
    return rotvec.astype(np.float32)


def require_dataset(group: h5py.Group, path: str) -> h5py.Dataset:
    if path not in group:
        raise KeyError(f"Missing dataset: {group.name}/{path}")
    obj = group[path]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Expected dataset at {group.name}/{path}, got {type(obj)!r}")
    return obj


def infer_demo_length(demo_group: h5py.Group) -> int:
    actions = require_dataset(demo_group, "actions")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions shape (N, 7), got {actions.shape} at {actions.name}")
    return int(actions.shape[0])


def normalize_compression_value(compression: str | None) -> str | None:
    if compression is None:
        return None
    normalized = str(compression).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    if normalized not in VALID_COMPRESSION_VALUES:
        raise ValueError(
            f"Unsupported compression value: {compression!r}. "
            f"Expected one of: inherit, gzip, lzf, none"
        )
    return normalized


def validate_compression_args(compression: str | None, compression_opts: int | None) -> None:
    if compression_opts is None:
        return
    if compression in {None, "inherit", "lzf"}:
        raise ValueError("compression_opts is only supported when compression='gzip'")
    if compression == "gzip" and not 0 <= int(compression_opts) <= 9:
        raise ValueError(f"gzip compression level must be in [0, 9], got {compression_opts}")


def adjusted_chunks_like(src: h5py.Dataset | None, data: np.ndarray) -> tuple[int, ...] | None:
    if src is None or src.chunks is None:
        return None
    chunk_shape = list(src.chunks)
    data_shape = list(data.shape)
    if len(chunk_shape) != len(data_shape):
        return None
    if any(size <= 0 for size in data_shape):
        return None
    chunk_shape = [min(chunk_size, data_size) for chunk_size, data_size in zip(chunk_shape, data_shape)]
    if not all(size > 0 for size in chunk_shape):
        return None
    return tuple(chunk_shape)


def dataset_create_kwargs_like(src: h5py.Dataset | None, data: np.ndarray) -> dict[str, object]:
    kwargs: dict[str, object] = {"dtype": data.dtype}
    if src is None:
        return kwargs

    if src.compression is not None:
        kwargs["compression"] = src.compression
    if src.compression_opts is not None:
        kwargs["compression_opts"] = src.compression_opts
    if src.shuffle:
        kwargs["shuffle"] = src.shuffle
    if src.fletcher32:
        kwargs["fletcher32"] = src.fletcher32
    chunks = adjusted_chunks_like(src, data)
    if chunks is not None:
        kwargs["chunks"] = chunks
    return kwargs


def dataset_create_kwargs_uniform(
    src: h5py.Dataset | None,
    data: np.ndarray,
    compression: str | None,
    compression_opts: int | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {"dtype": data.dtype}
    chunks = adjusted_chunks_like(src, data)
    if chunks is not None:
        kwargs["chunks"] = chunks
    if compression is not None:
        kwargs["compression"] = compression
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
    return kwargs


def create_dataset(
    parent: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    like: h5py.Dataset | None = None,
    compression: str | None = "inherit",
    compression_opts: int | None = None,
) -> h5py.Dataset:
    compression = normalize_compression_value(compression)
    validate_compression_args(compression, compression_opts)
    if compression == "inherit":
        kwargs = dataset_create_kwargs_like(like, data)
    else:
        kwargs = dataset_create_kwargs_uniform(like, data, compression, compression_opts)
    return parent.create_dataset(name, data=data, **kwargs)


def load_gripper(demo_group: h5py.Group, num_samples: int) -> np.ndarray:
    obs_group = demo_group["obs"]
    if not isinstance(obs_group, h5py.Group):
        raise TypeError(f"Expected group at {demo_group.name}/obs")

    if "robot0_gripper_qpos" in obs_group:
        gripper = np.asarray(obs_group["robot0_gripper_qpos"], dtype=np.float32)
    else:
        actions = np.asarray(require_dataset(demo_group, "actions"), dtype=np.float32)
        gripper = actions[:, -1:].astype(np.float32)

    if gripper.shape != (num_samples, 1):
        raise ValueError(f"Expected gripper shape {(num_samples, 1)}, got {gripper.shape} at {demo_group.name}")
    return gripper


def validate_first_dim(name: str, array: np.ndarray, num_samples: int) -> None:
    if array.shape[0] != num_samples:
        raise ValueError(f"Dataset {name} first dim mismatch: expected {num_samples}, got {array.shape}")


def rebuild_demo(
    src_demo: h5py.Group,
    dst_demo: h5py.Group,
    include_states: bool,
    compression: str | None,
    compression_opts: int | None,
    progress_callback=None,
    progress_prefix: str = "",
) -> int:
    def report_progress(dataset_name: str) -> None:
        if progress_callback is None:
            return
        full_name = f"{progress_prefix}/{dataset_name}" if progress_prefix else dataset_name
        progress_callback(full_name)

    num_samples = infer_demo_length(src_demo)
    obs_group = src_demo.get("obs")
    if not isinstance(obs_group, h5py.Group):
        raise TypeError(f"Expected group at {src_demo.name}/obs")

    actions_src = require_dataset(src_demo, "actions")
    joint_states_src = require_dataset(obs_group, "robot0_joint_pos")
    ee_pos_src = require_dataset(obs_group, "robot0_eef_pos")
    ee_quat_src = require_dataset(obs_group, "robot0_eef_quat")
    agentview_rgb_src = require_dataset(obs_group, "agentview_rgb")
    eye_in_hand_rgb_src = require_dataset(obs_group, "eye_in_hand_rgb")

    actions = np.asarray(actions_src, dtype=np.float32)
    joint_states = np.asarray(joint_states_src, dtype=np.float32)
    gripper_states = load_gripper(src_demo, num_samples)
    ee_pos = np.asarray(ee_pos_src, dtype=np.float32)
    ee_quat = np.asarray(ee_quat_src, dtype=np.float32)
    agentview_rgb = np.asarray(agentview_rgb_src, dtype=np.uint8)
    eye_in_hand_rgb = np.asarray(eye_in_hand_rgb_src, dtype=np.uint8)
    gripper_src = obs_group.get("robot0_gripper_qpos")
    if gripper_src is not None and not isinstance(gripper_src, h5py.Dataset):
        raise TypeError(f"Expected dataset at {src_demo.name}/obs/robot0_gripper_qpos")

    for name, array in (
        ("actions", actions),
        ("robot0_joint_pos", joint_states),
        ("robot0_gripper_qpos", gripper_states),
        ("robot0_eef_pos", ee_pos),
        ("robot0_eef_quat", ee_quat),
        ("agentview_rgb", agentview_rgb),
        ("eye_in_hand_rgb", eye_in_hand_rgb),
    ):
        validate_first_dim(name, array, num_samples)

    if joint_states.shape[1:] != (6,):
        raise ValueError(f"Expected joint_states shape (N, 6), got {joint_states.shape} at {src_demo.name}")
    if ee_pos.shape[1:] != (3,):
        raise ValueError(f"Expected ee_pos shape (N, 3), got {ee_pos.shape} at {src_demo.name}")
    if ee_quat.shape[1:] != (4,):
        raise ValueError(f"Expected ee_quat shape (N, 4), got {ee_quat.shape} at {src_demo.name}")

    ee_ori = quat_to_rotvec_xyzw(ee_quat)
    ee_states = np.concatenate([ee_pos, ee_ori], axis=1).astype(np.float32)
    robot_states = np.concatenate([joint_states, gripper_states], axis=1).astype(np.float32)
    dones = np.zeros((num_samples,), dtype=np.uint8)
    if num_samples > 0:
        dones[-1] = 1
    rewards = np.zeros((num_samples,), dtype=np.float32)
    states = np.zeros((num_samples, 110), dtype=np.float32)

    copy_attrs(src_demo.attrs, dst_demo.attrs)
    report_progress("actions")
    create_dataset(dst_demo, "actions", actions, like=actions_src, compression=compression, compression_opts=compression_opts)
    report_progress("dones")
    create_dataset(dst_demo, "dones", dones, like=actions_src, compression=compression, compression_opts=compression_opts)
    report_progress("rewards")
    create_dataset(dst_demo, "rewards", rewards, like=actions_src, compression=compression, compression_opts=compression_opts)
    report_progress("robot_states")
    create_dataset(
        dst_demo,
        "robot_states",
        robot_states,
        like=joint_states_src,
        compression=compression,
        compression_opts=compression_opts,
    )
    if include_states:
        report_progress("states")
        create_dataset(
            dst_demo,
            "states",
            states,
            like=joint_states_src,
            compression=compression,
            compression_opts=compression_opts,
        )

    dst_obs = dst_demo.create_group("obs")
    report_progress("obs/agentview_rgb")
    create_dataset(
        dst_obs,
        "agentview_rgb",
        agentview_rgb,
        like=agentview_rgb_src,
        compression=compression,
        compression_opts=compression_opts,
    )
    report_progress("obs/eye_in_hand_rgb")
    create_dataset(
        dst_obs,
        "eye_in_hand_rgb",
        eye_in_hand_rgb,
        like=eye_in_hand_rgb_src,
        compression=compression,
        compression_opts=compression_opts,
    )
    report_progress("obs/ee_pos")
    create_dataset(dst_obs, "ee_pos", ee_pos, like=ee_pos_src, compression=compression, compression_opts=compression_opts)
    report_progress("obs/ee_ori")
    create_dataset(dst_obs, "ee_ori", ee_ori, like=ee_quat_src, compression=compression, compression_opts=compression_opts)
    report_progress("obs/ee_states")
    create_dataset(
        dst_obs,
        "ee_states",
        ee_states,
        like=ee_pos_src,
        compression=compression,
        compression_opts=compression_opts,
    )
    report_progress("obs/gripper_states")
    create_dataset(
        dst_obs,
        "gripper_states",
        gripper_states,
        like=gripper_src,
        compression=compression,
        compression_opts=compression_opts,
    )
    report_progress("obs/joint_states")
    create_dataset(
        dst_obs,
        "joint_states",
        joint_states,
        like=joint_states_src,
        compression=compression,
        compression_opts=compression_opts,
    )

    dst_demo.attrs["num_samples"] = num_samples
    return num_samples


def iter_demo_names(data_group: h5py.Group, renumber: bool) -> Iterable[tuple[str, str]]:
    names = sorted_demo_names(data_group)
    for index, source_name in enumerate(names):
        target_name = f"demo_{index}" if renumber else source_name
        yield source_name, target_name


def rebuild_file(
    input_path: str,
    output_path: str,
    include_states: bool = True,
    renumber: bool = True,
    compression: str | None = "inherit",
    compression_opts: int | None = None,
    progress_callback=None,
) -> list[tuple[str, str, int]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    compression = normalize_compression_value(compression)
    validate_compression_args(compression, compression_opts)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)

        src_data = src.get("data")
        if not isinstance(src_data, h5py.Group):
            raise RuntimeError("Input HDF5 has no /data group")

        dst_data = dst.create_group("data")
        copy_attrs(src_data.attrs, dst_data.attrs)

        demo_pairs = list(iter_demo_names(src_data, renumber=renumber))
        if not demo_pairs:
            raise RuntimeError("No demo_* groups found under /data")

        results: list[tuple[str, str, int]] = []
        for source_name, target_name in demo_pairs:
            src_demo = src_data[source_name]
            if not isinstance(src_demo, h5py.Group):
                continue

            dst_demo = dst_data.create_group(target_name)
            num_samples = rebuild_demo(
                src_demo,
                dst_demo,
                include_states=include_states,
                compression=compression,
                compression_opts=compression_opts,
                progress_callback=progress_callback,
                progress_prefix=target_name,
            )
            results.append((source_name, target_name, num_samples))
        return results
