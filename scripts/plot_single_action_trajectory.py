#!/usr/bin/env python3
"""Plot one demo's action trajectory and per-step jump magnitude.

Example:
  /home/rvl/clds/bin/python scripts/plot_single_action_trajectory.py \
    --input /home/rvl/collect_datasets_ws/data/libero_demos.hdf5 \
    --demo demo_0 \
    --freq 10 \
    --output /tmp/demo_0_action_trajectory.png
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_HZ = 10.0
ARM_DIM = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a single demo or inference-action CSV trajectory and jumps.")
    parser.add_argument("--input", required=True, help="Input HDF5 path, CSV path, or inference session directory.")
    parser.add_argument("--demo", default="", help="Demo name such as demo_0. If empty, use --demo-index.")
    parser.add_argument("--demo-index", type=int, default=0, help="Demo index used when --demo is not provided.")
    parser.add_argument("--freq", type=float, default=DEFAULT_INPUT_HZ, help="Sampling frequency in Hz.")
    parser.add_argument(
        "--trim-zero-threshold",
        type=float,
        default=1e-4,
        help="L2 threshold used to trim leading / trailing near-zero arm actions.",
    )
    parser.add_argument("--trim-context", type=int, default=2, help="Extra frames kept around active action region.")
    parser.add_argument(
        "--jump-percentile",
        type=float,
        default=95.0,
        help="Percentile used to highlight large jumps on the jump subplot.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="How many largest jumps to print and export.")
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Include the gripper subplot in the figure. Disabled by default to focus on arm vibration.",
    )
    parser.add_argument(
        "--include-idle",
        action="store_true",
        help="For inference CSV input, keep rows where execution_enabled == 0 instead of filtering them out.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/single_action_trajectory.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV path for jump statistics. Defaults to <output>_jumps.csv.",
    )
    return parser.parse_args()


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    names = [str(name) for name in data_group.keys() if str(name).startswith("demo_")]

    def key_fn(name: str):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return name

    return sorted(names, key=key_fn)


def resolve_demo_name(data_group: h5py.Group, demo_name: str, demo_index: int) -> str:
    if demo_name:
        if demo_name not in data_group:
            raise KeyError(f"Demo {demo_name} not found under /data.")
        return demo_name

    names = sorted_demo_names(data_group)
    if not names:
        raise RuntimeError("No demo_* groups found under /data.")
    if demo_index < 0 or demo_index >= len(names):
        raise IndexError(f"demo-index {demo_index} out of range [0, {len(names) - 1}]")
    return names[demo_index]


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        candidate = path / "actions.csv"
        if candidate.is_file():
            return candidate
    return path


def estimate_sample_hz(time_sec: np.ndarray) -> float:
    if time_sec.ndim != 1 or time_sec.size < 2:
        return math.nan
    diffs = np.diff(time_sec)
    finite = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if finite.size == 0:
        return math.nan
    return float(1.0 / np.median(finite))


def build_uniform_time_axis(num_samples: int, freq_hz: float) -> np.ndarray:
    if freq_hz <= 0.0:
        freq_hz = DEFAULT_INPUT_HZ
    return np.arange(num_samples, dtype=np.float64) / float(freq_hz)


def load_csv_actions(
    csv_path: Path,
    *,
    include_idle: bool,
    fallback_freq_hz: float,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float]:
    rows: list[list[float]] = []
    raw_indices: list[int] = []
    elapsed: list[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["vx", "vy", "vz", "wx", "wy", "wz"]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"CSV missing required columns: {', '.join(missing)}")

        for fallback_idx, row in enumerate(reader):
            if not include_idle and "execution_enabled" in row:
                try:
                    if int(round(float(row["execution_enabled"]))) <= 0:
                        continue
                except Exception:
                    pass

            rows.append(
                [
                    float(row["vx"]),
                    float(row["vy"]),
                    float(row["vz"]),
                    float(row["wx"]),
                    float(row["wy"]),
                    float(row["wz"]),
                    float(row.get("gripper", 0.0) or 0.0),
                ]
            )
            try:
                raw_indices.append(int(float(row.get("step_idx", fallback_idx))))
            except Exception:
                raw_indices.append(int(fallback_idx))
            try:
                elapsed.append(float(row.get("elapsed_sec", math.nan)))
            except Exception:
                elapsed.append(math.nan)

    if len(rows) < 2:
        raise RuntimeError(f"CSV has too few action rows after filtering: {csv_path}")

    actions_all = np.asarray(rows, dtype=np.float32)
    raw_index_array = np.asarray(raw_indices, dtype=np.int64)
    elapsed_sec = np.asarray(elapsed, dtype=np.float64)
    if not np.all(np.isfinite(elapsed_sec)):
        elapsed_sec = build_uniform_time_axis(actions_all.shape[0], fallback_freq_hz)
    else:
        elapsed_sec = elapsed_sec - float(elapsed_sec[0])

    sample_hz = estimate_sample_hz(elapsed_sec)
    if not math.isfinite(sample_hz):
        sample_hz = float(fallback_freq_hz if fallback_freq_hz > 0.0 else DEFAULT_INPUT_HZ)

    session_name = csv_path.parent.name if csv_path.stem == "actions" else csv_path.stem
    return session_name, actions_all, raw_index_array, elapsed_sec, sample_hz


def trim_active_range(actions: np.ndarray, threshold: float, context: int) -> tuple[int, int] | None:
    if actions.ndim != 2 or actions.shape[0] == 0:
        return None
    norms = np.linalg.norm(actions[:, :ARM_DIM], axis=1)
    active = np.flatnonzero(norms > threshold)
    if active.size == 0:
        return None
    start = max(0, int(active[0]) - max(0, int(context)))
    end = min(int(actions.shape[0]), int(active[-1]) + 1 + max(0, int(context)))
    if start >= end:
        return None
    return start, end


def per_step_jump(actions: np.ndarray) -> dict[str, np.ndarray]:
    diff = np.diff(actions, axis=0)
    return {
        "arm_l2": np.linalg.norm(diff[:, :ARM_DIM], axis=1),
        "linear_l2": np.linalg.norm(diff[:, :3], axis=1),
        "angular_l2": np.linalg.norm(diff[:, 3:6], axis=1),
        "gripper_abs": np.abs(diff[:, 6]) if diff.shape[1] >= 7 else np.zeros(diff.shape[0], dtype=np.float64),
        "full_diff": diff,
    }


def write_jump_csv(
    csv_path: Path,
    jump_time: np.ndarray,
    jump_stats: dict[str, np.ndarray],
    raw_frame_indices: np.ndarray,
) -> None:
    fieldnames = [
        "jump_idx",
        "frame_from",
        "frame_to",
        "time_sec",
        "arm_jump_l2",
        "linear_jump_l2",
        "angular_jump_l2",
        "gripper_jump_abs",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(int(jump_time.shape[0])):
            writer.writerow(
                {
                    "jump_idx": idx,
                    "frame_from": int(raw_frame_indices[idx]),
                    "frame_to": int(raw_frame_indices[idx + 1]),
                    "time_sec": float(jump_time[idx]),
                    "arm_jump_l2": float(jump_stats["arm_l2"][idx]),
                    "linear_jump_l2": float(jump_stats["linear_l2"][idx]),
                    "angular_jump_l2": float(jump_stats["angular_l2"][idx]),
                    "gripper_jump_abs": float(jump_stats["gripper_abs"][idx]),
                }
            )


def main() -> int:
    args = parse_args()

    input_path = resolve_input_path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.freq <= 0.0:
        raise ValueError("--freq must be > 0.")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else output_path.with_name(output_path.stem + "_jumps.csv")

    if input_path.suffix.lower() == ".csv":
        demo_name, actions_all, raw_indices_all, time_all_sec, effective_hz = load_csv_actions(
            input_path,
            include_idle=bool(args.include_idle),
            fallback_freq_hz=float(args.freq),
        )
    else:
        with h5py.File(input_path, "r") as handle:
            data_group = handle.get("data")
            if not isinstance(data_group, h5py.Group):
                raise RuntimeError(f"HDF5 has no /data group: {input_path}")

            demo_name = resolve_demo_name(data_group, str(args.demo).strip(), int(args.demo_index))
            demo_group = data_group[demo_name]
            if "actions" not in demo_group:
                raise RuntimeError(f"{demo_name} has no actions dataset.")

            actions_all = np.asarray(demo_group["actions"], dtype=np.float32)
            if actions_all.ndim != 2 or actions_all.shape[0] < 2:
                raise RuntimeError(f"{demo_name} actions shape is invalid: {actions_all.shape}")

            raw_indices_all = np.arange(actions_all.shape[0], dtype=np.int64)
            time_all_sec = build_uniform_time_axis(actions_all.shape[0], float(args.freq))
            effective_hz = float(args.freq)

    trim = trim_active_range(actions_all, threshold=float(args.trim_zero_threshold), context=int(args.trim_context))
    if trim is None:
        trim_start, trim_end = 0, int(actions_all.shape[0])
    else:
        trim_start, trim_end = trim

    actions = actions_all[trim_start:trim_end]
    raw_indices = raw_indices_all[trim_start:trim_end]
    time_sec = time_all_sec[trim_start:trim_end]
    time_sec = time_sec - float(time_sec[0])
    jump_stats = per_step_jump(actions)
    jump_time = time_sec[1:]

    percentile = float(np.clip(args.jump_percentile, 0.0, 100.0))
    threshold = float(np.percentile(jump_stats["arm_l2"], percentile)) if jump_stats["arm_l2"].size else 0.0
    spike_idx = np.flatnonzero(jump_stats["arm_l2"] >= threshold)

    order = np.argsort(jump_stats["arm_l2"])[::-1]
    top_k = max(1, int(args.top_k))
    top_idx = order[:top_k]

    include_gripper = bool(args.include_gripper)
    nrows = 4 if include_gripper else 3
    fig_height = 10 if include_gripper else 8.2
    fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_height), sharex=True, constrained_layout=True)

    linear_labels = ["vx", "vy", "vz"]
    angular_labels = ["wx", "wy", "wz"]
    linear_colors = ["#006d77", "#0a9396", "#94d2bd"]
    angular_colors = ["#bb3e03", "#ca6702", "#ee9b00"]

    for dim in range(3):
        axes[0].plot(time_sec, actions[:, dim], label=linear_labels[dim], color=linear_colors[dim], linewidth=1.6)
        axes[1].plot(time_sec, actions[:, 3 + dim], label=angular_labels[dim], color=angular_colors[dim], linewidth=1.6)

    axes[0].set_ylabel("linear cmd")
    axes[1].set_ylabel("angular cmd")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[1].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    jump_ax = axes[3] if include_gripper else axes[2]
    if include_gripper:
        axes[2].plot(time_sec, actions[:, 6], color="#6a4c93", linewidth=1.6, label="gripper")
        axes[2].set_ylabel("gripper")
        axes[2].legend(loc="upper right", fontsize=9)
        axes[2].grid(True, alpha=0.3)

    jump_ax.plot(jump_time, jump_stats["arm_l2"], color="#d00000", linewidth=1.6, label="arm jump ||a_t-a_(t-1)||")
    jump_ax.plot(jump_time, jump_stats["linear_l2"], color="#1d3557", linewidth=1.1, alpha=0.8, label="linear jump")
    jump_ax.plot(jump_time, jump_stats["angular_l2"], color="#457b9d", linewidth=1.1, alpha=0.8, label="angular jump")
    if spike_idx.size:
        jump_ax.scatter(
            jump_time[spike_idx],
            jump_stats["arm_l2"][spike_idx],
            color="#d00000",
            s=28,
            zorder=5,
            label=f">= P{percentile:g}",
        )
    jump_ax.axhline(threshold, color="#d00000", linestyle="--", linewidth=1.0, alpha=0.6)
    jump_ax.set_ylabel("jump mag")
    jump_ax.set_xlabel("time (s)")
    jump_ax.legend(loc="upper right", fontsize=9)
    jump_ax.grid(True, alpha=0.3)

    title = (
        f"Action Trajectory: {demo_name} | ~{effective_hz:.2f} Hz | "
        f"frames {trim_start}:{trim_end} / {actions_all.shape[0]}"
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    write_jump_csv(csv_path, jump_time, jump_stats, raw_indices)

    print(f"Saved figure: {output_path}")
    print(f"Saved jump csv: {csv_path}")
    print(f"Demo: {demo_name}")
    print(f"Effective sample rate: {effective_hz:.3f} Hz")
    print(f"Trimmed range: [{trim_start}, {trim_end}) from total {actions_all.shape[0]} frames")
    print(f"Highlighted jump threshold (P{percentile:g}): {threshold:.6f}")
    print("Top jumps:")
    for rank, idx in enumerate(top_idx, start=1):
        print(
            f"  {rank}. frames {raw_indices[idx]}->{raw_indices[idx + 1]}"
            f" | t={jump_time[idx]:.3f}s"
            f" | arm={jump_stats['arm_l2'][idx]:.6f}"
            f" | linear={jump_stats['linear_l2'][idx]:.6f}"
            f" | angular={jump_stats['angular_l2'][idx]:.6f}"
            f" | gripper={jump_stats['gripper_abs'][idx]:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
