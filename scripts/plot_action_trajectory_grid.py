#!/usr/bin/env python3
"""Plot multiple action trajectories in one comparison grid.

Example:
  /home/rvl/clds/bin/python scripts/plot_action_trajectory_grid.py \
    --input data/inference_action_logs/run_10hz --label 10Hz \
    --input data/inference_action_logs/run_15hz --label 15Hz \
    --input data/inference_action_logs/run_20hz --label 20Hz \
    --output /tmp/action_grid.png
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_HZ = 10.0
ARM_DIM = 6


@dataclass(frozen=True)
class SequenceData:
    label: str
    demo_name: str
    effective_hz: float
    actions: np.ndarray
    time_sec: np.ndarray
    raw_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multiple action trajectories as a comparison grid.")
    parser.add_argument("--input", action="append", required=True, help="Input HDF5 path, CSV path, or inference session directory.")
    parser.add_argument("--label", action="append", default=None, help="Display label for each input.")
    parser.add_argument(
        "--freq",
        action="append",
        type=float,
        default=None,
        help=f"Fallback sampling frequency in Hz for each input. Defaults to {DEFAULT_INPUT_HZ:g} Hz.",
    )
    parser.add_argument("--output", default="/tmp/action_trajectory_grid.png", help="Output PNG path.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument("--demo", default="", help="Demo name for HDF5 input such as demo_0.")
    parser.add_argument("--demo-index", type=int, default=0, help="Demo index used when --demo is not provided.")
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
    parser.add_argument(
        "--include-idle",
        action="store_true",
        help="For inference CSV input, keep rows where execution_enabled == 0 instead of filtering them out.",
    )
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Include the gripper subplot in the figure.",
    )
    return parser.parse_args()


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        candidate = path / "actions.csv"
        if candidate.is_file():
            return candidate
    return path


def build_uniform_time_axis(num_samples: int, freq_hz: float) -> np.ndarray:
    hz = float(freq_hz if freq_hz > 0.0 else DEFAULT_INPUT_HZ)
    return np.arange(num_samples, dtype=np.float64) / hz


def estimate_sample_hz(time_sec: np.ndarray) -> float:
    if time_sec.ndim != 1 or time_sec.size < 2:
        return math.nan
    diffs = np.diff(time_sec)
    finite = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if finite.size == 0:
        return math.nan
    return float(1.0 / np.median(finite))


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
    }


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


def load_sequence(
    path: Path,
    *,
    label: str,
    demo_name: str,
    demo_index: int,
    fallback_freq_hz: float,
    include_idle: bool,
    trim_zero_threshold: float,
    trim_context: int,
) -> SequenceData:
    if path.suffix.lower() == ".csv":
        resolved_name, actions_all, raw_indices_all, time_all_sec, effective_hz = load_csv_actions(
            path,
            include_idle=include_idle,
            fallback_freq_hz=fallback_freq_hz,
        )
    else:
        with h5py.File(path, "r") as handle:
            data_group = handle.get("data")
            if not isinstance(data_group, h5py.Group):
                raise RuntimeError(f"HDF5 has no /data group: {path}")

            resolved_name = resolve_demo_name(data_group, demo_name, demo_index)
            demo_group = data_group[resolved_name]
            if "actions" not in demo_group:
                raise RuntimeError(f"{resolved_name} has no actions dataset.")

            actions_all = np.asarray(demo_group["actions"], dtype=np.float32)
            if actions_all.ndim != 2 or actions_all.shape[0] < 2:
                raise RuntimeError(f"{resolved_name} actions shape is invalid: {actions_all.shape}")

            raw_indices_all = np.arange(actions_all.shape[0], dtype=np.int64)
            time_all_sec = build_uniform_time_axis(actions_all.shape[0], fallback_freq_hz)
            effective_hz = float(fallback_freq_hz if fallback_freq_hz > 0.0 else DEFAULT_INPUT_HZ)

    trim = trim_active_range(actions_all, threshold=trim_zero_threshold, context=trim_context)
    if trim is None:
        trim_start, trim_end = 0, int(actions_all.shape[0])
    else:
        trim_start, trim_end = trim

    actions = actions_all[trim_start:trim_end]
    raw_indices = raw_indices_all[trim_start:trim_end]
    time_sec = time_all_sec[trim_start:trim_end]
    time_sec = time_sec - float(time_sec[0])

    return SequenceData(
        label=label,
        demo_name=resolved_name,
        effective_hz=float(effective_hz),
        actions=actions,
        time_sec=time_sec,
        raw_indices=raw_indices,
    )


def add_margin(ymin: float, ymax: float, *, symmetric: bool = False) -> tuple[float, float]:
    if symmetric:
        limit = max(abs(ymin), abs(ymax))
        margin = max(0.05, 0.08 * limit if limit > 0.0 else 0.1)
        return -limit - margin, limit + margin

    span = ymax - ymin
    margin = 0.08 * span if span > 1e-9 else 0.1
    return ymin - margin, ymax + margin


def build_sequences(args: argparse.Namespace) -> list[SequenceData]:
    inputs = [resolve_input_path(path) for path in args.input]
    labels = list(args.label or [])
    freqs = list(args.freq or [])

    if labels and len(labels) != len(inputs):
        raise ValueError(f"--label count ({len(labels)}) must match --input count ({len(inputs)}).")
    if freqs and len(freqs) != len(inputs):
        raise ValueError(f"--freq count ({len(freqs)}) must match --input count ({len(inputs)}).")
    if not labels:
        labels = [path.stem for path in inputs]
    if not freqs:
        freqs = [DEFAULT_INPUT_HZ for _ in inputs]

    sequences: list[SequenceData] = []
    for path, label, freq in zip(inputs, labels, freqs):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        sequences.append(
            load_sequence(
                path,
                label=str(label),
                demo_name=str(args.demo).strip(),
                demo_index=int(args.demo_index),
                fallback_freq_hz=float(freq),
                include_idle=bool(args.include_idle),
                trim_zero_threshold=float(args.trim_zero_threshold),
                trim_context=int(args.trim_context),
            )
        )
    return sequences


def main() -> int:
    args = parse_args()
    sequences = build_sequences(args)
    include_gripper = bool(args.include_gripper)
    ncols = len(sequences)
    nrows = 4 if include_gripper else 3

    linear_stack = np.concatenate([seq.actions[:, :3].reshape(-1) for seq in sequences])
    angular_stack = np.concatenate([seq.actions[:, 3:6].reshape(-1) for seq in sequences])
    linear_ylim = add_margin(float(np.min(linear_stack)), float(np.max(linear_stack)), symmetric=True)
    angular_ylim = add_margin(float(np.min(angular_stack)), float(np.max(angular_stack)), symmetric=True)

    jump_data: list[dict[str, np.ndarray]] = [per_step_jump(seq.actions) for seq in sequences]
    jump_max = max(
        float(np.max(stats["arm_l2"])) if stats["arm_l2"].size else 0.0
        for stats in jump_data
    )
    jump_ylim = (0.0, jump_max * 1.12 if jump_max > 0.0 else 1.0)

    if include_gripper:
        gripper_stack = np.concatenate([seq.actions[:, 6].reshape(-1) for seq in sequences])
        gripper_ylim = add_margin(float(np.min(gripper_stack)), float(np.max(gripper_stack)), symmetric=True)

    fig_width = max(15.0, 3.8 * ncols)
    fig_height = 10.0 if include_gripper else 8.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=True)

    linear_labels = ["vx", "vy", "vz"]
    angular_labels = ["wx", "wy", "wz"]
    linear_colors = ["#006d77", "#0a9396", "#94d2bd"]
    angular_colors = ["#bb3e03", "#ca6702", "#ee9b00"]

    percentile = float(np.clip(args.jump_percentile, 0.0, 100.0))
    for col, (seq, stats) in enumerate(zip(sequences, jump_data)):
        jump_time = seq.time_sec[1:]
        threshold = float(np.percentile(stats["arm_l2"], percentile)) if stats["arm_l2"].size else 0.0
        spike_idx = np.flatnonzero(stats["arm_l2"] >= threshold)

        linear_ax = axes[0, col]
        angular_ax = axes[1, col]
        jump_ax = axes[3, col] if include_gripper else axes[2, col]

        for dim in range(3):
            linear_ax.plot(seq.time_sec, seq.actions[:, dim], color=linear_colors[dim], linewidth=1.4)
            angular_ax.plot(seq.time_sec, seq.actions[:, 3 + dim], color=angular_colors[dim], linewidth=1.4)

        linear_ax.set_ylim(*linear_ylim)
        angular_ax.set_ylim(*angular_ylim)
        linear_ax.grid(True, alpha=0.28)
        angular_ax.grid(True, alpha=0.28)

        if include_gripper:
            gripper_ax = axes[2, col]
            gripper_ax.plot(seq.time_sec, seq.actions[:, 6], color="#6a4c93", linewidth=1.4)
            gripper_ax.set_ylim(*gripper_ylim)
            gripper_ax.grid(True, alpha=0.28)

        jump_ax.plot(jump_time, stats["arm_l2"], color="#d00000", linewidth=1.5, label="arm jump")
        jump_ax.plot(jump_time, stats["linear_l2"], color="#1d3557", linewidth=1.0, alpha=0.85, label="linear")
        jump_ax.plot(jump_time, stats["angular_l2"], color="#457b9d", linewidth=1.0, alpha=0.85, label="angular")
        if spike_idx.size:
            jump_ax.scatter(
                jump_time[spike_idx],
                stats["arm_l2"][spike_idx],
                color="#d00000",
                s=18,
                zorder=5,
            )
        jump_ax.axhline(threshold, color="#d00000", linestyle="--", linewidth=0.9, alpha=0.6)
        jump_ax.set_ylim(*jump_ylim)
        jump_ax.grid(True, alpha=0.28)
        jump_ax.set_xlabel("time (s)")

        linear_ax.set_title(f"{seq.label}\n~{seq.effective_hz:.2f} Hz", fontsize=11)

        if col == 0:
            linear_ax.set_ylabel("linear cmd")
            angular_ax.set_ylabel("angular cmd")
            if include_gripper:
                axes[2, col].set_ylabel("gripper")
            jump_ax.set_ylabel("jump mag")

            linear_handles = [plt.Line2D([], [], color=linear_colors[idx], linewidth=1.8, label=linear_labels[idx]) for idx in range(3)]
            angular_handles = [plt.Line2D([], [], color=angular_colors[idx], linewidth=1.8, label=angular_labels[idx]) for idx in range(3)]
            linear_ax.legend(handles=linear_handles, loc="upper right", ncol=3, fontsize=8)
            angular_ax.legend(handles=angular_handles, loc="upper right", ncol=3, fontsize=8)
            jump_ax.legend(loc="upper right", fontsize=8)

    title = str(args.title).strip() or "Action Trajectory Comparison"
    fig.suptitle(title, fontsize=14)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved figure: {output_path}")
    for seq in sequences:
        print(
            f"  {seq.label}: {seq.demo_name}"
            f" | ~{seq.effective_hz:.3f} Hz"
            f" | frames={seq.actions.shape[0]}"
            f" | duration={seq.time_sec[-1]:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
