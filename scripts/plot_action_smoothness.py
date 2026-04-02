#!/usr/bin/env python3
"""Plot action smoothness metrics from one or more teleop HDF5 files or inference CSV logs.

Example:
  python scripts/plot_action_smoothness.py \
    --input data/run_10hz.hdf5 --label 10Hz --freq 10 \
    --input data/run_15hz.hdf5 --label 15Hz --freq 15 \
    --input data/run_20hz.hdf5 --label 20Hz --freq 20 \
    --output-dir /tmp/action_smoothness

Outputs:
  - per_demo_metrics.csv
  - summary_metrics.csv
  - smoothness_summary.png

Notes:
  - Lower values are better for every smoothness metric in this script.
  - By default the gripper channel is excluded, because binary open/close events
    can dominate the derivatives and mask arm vibration.
  - Leading / trailing near-zero action ranges are trimmed by default so that
    long stationary windows do not dilute the metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_HZ = 10.0
DEFAULT_HF_CUTOFF_HZ = 2.0
ARM_DIM = 6


@dataclass(frozen=True)
class ExperimentSpec:
    path: Path
    label: str
    freq_hz: float


@dataclass(frozen=True)
class DemoMetrics:
    label: str
    input_path: str
    freq_hz: float
    demo_name: str
    raw_samples: int
    used_samples: int
    trim_start: int
    trim_end: int
    command_accel_rms: float
    command_jerk_rms: float
    command_total_variation: float
    hf_energy_ratio: float
    eef_linear_accel_rms: float
    eef_linear_jerk_rms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot action smoothness metrics from HDF5 demos or inference CSV logs.")
    parser.add_argument("--input", action="append", required=True, help="Input HDF5 path, CSV path, or inference session directory.")
    parser.add_argument("--label", action="append", default=None, help="Legend label for each input. Defaults to filename stem.")
    parser.add_argument(
        "--freq",
        action="append",
        type=float,
        default=None,
        help=f"Sampling frequency in Hz for each input. Defaults to {DEFAULT_INPUT_HZ:g} Hz.",
    )
    parser.add_argument("--output-dir", default="/tmp/action_smoothness", help="Directory to store CSVs and plots.")
    parser.add_argument(
        "--trim-zero-threshold",
        type=float,
        default=1e-4,
        help="L2-norm threshold used to trim leading / trailing near-zero actions.",
    )
    parser.add_argument(
        "--trim-context",
        type=int,
        default=2,
        help="Extra frames kept around the detected non-zero action window.",
    )
    parser.add_argument(
        "--hf-cutoff-hz",
        type=float,
        default=DEFAULT_HF_CUTOFF_HZ,
        help="Cutoff frequency in Hz for the high-frequency energy ratio metric.",
    )
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Include the 7th gripper channel in smoothness metrics.",
    )
    parser.add_argument(
        "--include-idle",
        action="store_true",
        help="For inference CSV input, keep rows where execution_enabled == 0 instead of filtering them out.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=8,
        help="Minimum number of trimmed samples required for a demo to contribute to the summary.",
    )
    parser.add_argument("--title", default="", help="Optional title shown on the summary figure.")
    return parser.parse_args()


def build_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
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

    experiments: list[ExperimentSpec] = []
    for path, label, freq in zip(inputs, labels, freqs):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        if freq <= 0.0:
            raise ValueError(f"Frequency must be > 0, got {freq} for {path}")
        experiments.append(ExperimentSpec(path=path, label=str(label), freq_hz=float(freq)))
    return experiments


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        candidate = path / "actions.csv"
        if candidate.is_file():
            return candidate
    return path


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    names = [str(name) for name in data_group.keys() if str(name).startswith("demo_")]

    def key_fn(name: str):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return name

    return sorted(names, key=key_fn)


def trim_active_range(
    actions: np.ndarray,
    threshold: float,
    context: int,
) -> tuple[int, int] | None:
    if actions.ndim != 2 or actions.shape[0] == 0:
        return None

    norms = np.linalg.norm(actions, axis=1)
    active = np.flatnonzero(norms > threshold)
    if active.size == 0:
        return None

    start = max(0, int(active[0]) - max(0, int(context)))
    end = min(int(actions.shape[0]), int(active[-1]) + 1 + max(0, int(context)))
    if start >= end:
        return None
    return start, end


def estimate_sample_hz(time_sec: np.ndarray) -> float:
    if time_sec.ndim != 1 or time_sec.size < 2:
        return math.nan
    diffs = np.diff(time_sec)
    finite = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if finite.size == 0:
        return math.nan
    return float(1.0 / np.median(finite))


def rms(array: np.ndarray) -> float:
    if array.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64), dtype=np.float64)))


def finite_difference(signal: np.ndarray, dt: float, order: int) -> np.ndarray:
    if signal.shape[0] <= order:
        return np.empty((0,) + signal.shape[1:], dtype=np.float64)
    result = np.asarray(signal, dtype=np.float64)
    for _ in range(order):
        result = np.diff(result, axis=0) / dt
    return result


def high_frequency_energy_ratio(signal: np.ndarray, sample_hz: float, cutoff_hz: float) -> float:
    if signal.ndim != 2 or signal.shape[0] < 4:
        return math.nan

    centered = np.asarray(signal, dtype=np.float64) - np.mean(signal, axis=0, keepdims=True, dtype=np.float64)
    spectrum = np.fft.rfft(centered, axis=0)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(centered.shape[0], d=1.0 / float(sample_hz))

    usable = freqs > 0.0
    total = float(np.sum(power[usable]))
    if total <= 0.0:
        return 0.0

    high = freqs >= float(cutoff_hz)
    return float(np.sum(power[high]) / total)


def load_csv_session(
    path: Path,
    *,
    include_idle: bool,
    fallback_freq_hz: float,
) -> tuple[str, np.ndarray, np.ndarray | None, float]:
    action_rows: list[list[float]] = []
    eef_rows: list[list[float]] = []
    elapsed_rows: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["vx", "vy", "vz", "wx", "wy", "wz"]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"CSV missing required columns: {', '.join(missing)}")

        has_eef = all(name in (reader.fieldnames or []) for name in ["eef_x", "eef_y", "eef_z"])
        for row in reader:
            if not include_idle and "execution_enabled" in row:
                try:
                    if int(round(float(row["execution_enabled"]))) <= 0:
                        continue
                except Exception:
                    pass

            action_rows.append(
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
            if has_eef:
                eef_rows.append(
                    [
                        float(row.get("eef_x", math.nan)),
                        float(row.get("eef_y", math.nan)),
                        float(row.get("eef_z", math.nan)),
                    ]
                )
            try:
                elapsed_rows.append(float(row.get("elapsed_sec", math.nan)))
            except Exception:
                elapsed_rows.append(math.nan)

    if not action_rows:
        raise RuntimeError(f"CSV contains no usable action rows: {path}")

    actions_all = np.asarray(action_rows, dtype=np.float32)
    eef_all = np.asarray(eef_rows, dtype=np.float32) if eef_rows else None
    elapsed_sec = np.asarray(elapsed_rows, dtype=np.float64)
    if np.all(np.isfinite(elapsed_sec)) and elapsed_sec.size >= 2:
        elapsed_sec = elapsed_sec - float(elapsed_sec[0])
        sample_hz = estimate_sample_hz(elapsed_sec)
    else:
        sample_hz = math.nan

    if not math.isfinite(sample_hz):
        sample_hz = float(fallback_freq_hz if fallback_freq_hz > 0.0 else DEFAULT_INPUT_HZ)

    session_name = path.parent.name if path.stem == "actions" else path.stem
    return session_name, actions_all, eef_all, sample_hz


def compute_sequence_metrics(
    spec: ExperimentSpec,
    demo_name: str,
    actions_all: np.ndarray,
    *,
    include_gripper: bool,
    trim_zero_threshold: float,
    trim_context: int,
    hf_cutoff_hz: float,
    min_samples: int,
    eef_pos_all: np.ndarray | None = None,
    sample_hz: float | None = None,
) -> DemoMetrics | None:
    if actions_all.ndim != 2 or actions_all.shape[0] == 0:
        return None

    resolved_sample_hz = float(sample_hz if sample_hz is not None else spec.freq_hz)
    if not math.isfinite(resolved_sample_hz) or resolved_sample_hz <= 0.0:
        resolved_sample_hz = DEFAULT_INPUT_HZ

    action_channels = actions_all if include_gripper else actions_all[:, :ARM_DIM]
    trim = trim_active_range(action_channels, threshold=trim_zero_threshold, context=trim_context)
    if trim is None:
        return None

    trim_start, trim_end = trim
    actions = action_channels[trim_start:trim_end]
    used_samples = int(actions.shape[0])
    if used_samples < int(min_samples):
        return None

    dt = 1.0 / resolved_sample_hz
    command_accel = finite_difference(actions, dt=dt, order=1)
    command_jerk = finite_difference(actions, dt=dt, order=2)
    command_total_variation = finite_difference(actions, dt=dt, order=1)

    hf_ratio = high_frequency_energy_ratio(actions, sample_hz=resolved_sample_hz, cutoff_hz=hf_cutoff_hz)

    eef_linear_accel_rms = math.nan
    eef_linear_jerk_rms = math.nan
    if eef_pos_all is not None and eef_pos_all.ndim == 2 and eef_pos_all.shape[0] >= trim_end:
        eef_pos = eef_pos_all[trim_start:trim_end]
        if np.all(np.isfinite(eef_pos)):
            eef_linear_accel = finite_difference(eef_pos, dt=dt, order=2)
            eef_linear_jerk = finite_difference(eef_pos, dt=dt, order=3)
            eef_linear_accel_rms = rms(eef_linear_accel)
            eef_linear_jerk_rms = rms(eef_linear_jerk)

    return DemoMetrics(
        label=spec.label,
        input_path=str(spec.path),
        freq_hz=float(resolved_sample_hz),
        demo_name=demo_name,
        raw_samples=int(actions_all.shape[0]),
        used_samples=used_samples,
        trim_start=int(trim_start),
        trim_end=int(trim_end),
        command_accel_rms=rms(command_accel),
        command_jerk_rms=rms(command_jerk),
        command_total_variation=rms(command_total_variation),
        hf_energy_ratio=hf_ratio,
        eef_linear_accel_rms=eef_linear_accel_rms,
        eef_linear_jerk_rms=eef_linear_jerk_rms,
    )


def compute_demo_metrics(
    spec: ExperimentSpec,
    demo_name: str,
    demo_group: h5py.Group,
    *,
    include_gripper: bool,
    trim_zero_threshold: float,
    trim_context: int,
    hf_cutoff_hz: float,
    min_samples: int,
) -> DemoMetrics | None:
    if "actions" not in demo_group:
        return None

    actions_all = np.asarray(demo_group["actions"], dtype=np.float32)
    eef_pos_all = None
    obs_group = demo_group.get("obs")
    if isinstance(obs_group, h5py.Group) and "robot0_eef_pos" in obs_group:
        eef_pos_all = np.asarray(obs_group["robot0_eef_pos"], dtype=np.float32)

    return compute_sequence_metrics(
        spec,
        demo_name,
        actions_all,
        include_gripper=include_gripper,
        trim_zero_threshold=trim_zero_threshold,
        trim_context=trim_context,
        hf_cutoff_hz=hf_cutoff_hz,
        min_samples=min_samples,
        eef_pos_all=eef_pos_all,
        sample_hz=float(spec.freq_hz),
    )


def analyze_experiment(
    spec: ExperimentSpec,
    *,
    include_gripper: bool,
    trim_zero_threshold: float,
    trim_context: int,
    hf_cutoff_hz: float,
    min_samples: int,
    include_idle: bool,
) -> list[DemoMetrics]:
    metrics: list[DemoMetrics] = []
    if spec.path.suffix.lower() == ".csv":
        session_name, actions_all, eef_pos_all, sample_hz = load_csv_session(
            spec.path,
            include_idle=include_idle,
            fallback_freq_hz=float(spec.freq_hz),
        )
        metric = compute_sequence_metrics(
            spec,
            session_name,
            actions_all,
            include_gripper=include_gripper,
            trim_zero_threshold=trim_zero_threshold,
            trim_context=trim_context,
            hf_cutoff_hz=hf_cutoff_hz,
            min_samples=min_samples,
            eef_pos_all=eef_pos_all,
            sample_hz=sample_hz,
        )
        return [metric] if metric is not None else []

    with h5py.File(spec.path, "r") as handle:
        data_group = handle.get("data")
        if not isinstance(data_group, h5py.Group):
            raise RuntimeError(f"HDF5 has no /data group: {spec.path}")

        for demo_name in sorted_demo_names(data_group):
            demo_group = data_group[demo_name]
            if not isinstance(demo_group, h5py.Group):
                continue
            demo_metrics = compute_demo_metrics(
                spec,
                demo_name,
                demo_group,
                include_gripper=include_gripper,
                trim_zero_threshold=trim_zero_threshold,
                trim_context=trim_context,
                hf_cutoff_hz=hf_cutoff_hz,
                min_samples=min_samples,
            )
            if demo_metrics is not None:
                metrics.append(demo_metrics)
    return metrics


def summarize_metrics(rows: list[DemoMetrics]) -> list[dict[str, float | str | int]]:
    groups: dict[str, list[DemoMetrics]] = {}
    for row in rows:
        groups.setdefault(row.label, []).append(row)

    metric_names = [
        "command_accel_rms",
        "command_jerk_rms",
        "command_total_variation",
        "hf_energy_ratio",
        "eef_linear_accel_rms",
        "eef_linear_jerk_rms",
    ]

    summaries: list[dict[str, float | str | int]] = []
    for label, group_rows in groups.items():
        group_rows = sorted(group_rows, key=lambda item: item.demo_name)
        summary: dict[str, float | str | int] = {
            "label": label,
            "input_path": group_rows[0].input_path,
            "freq_hz": group_rows[0].freq_hz,
            "num_demos": len(group_rows),
            "raw_samples_total": int(sum(row.raw_samples for row in group_rows)),
            "used_samples_total": int(sum(row.used_samples for row in group_rows)),
        }
        for metric_name in metric_names:
            values = np.asarray([getattr(row, metric_name) for row in group_rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"{metric_name}_mean"] = float(np.mean(finite)) if finite.size else math.nan
            summary[f"{metric_name}_std"] = float(np.std(finite)) if finite.size else math.nan
        summaries.append(summary)

    summaries.sort(key=lambda item: (float(item["freq_hz"]), str(item["label"])))
    add_relative_smoothness_score(summaries)
    return summaries


def add_relative_smoothness_score(summaries: list[dict[str, float | str | int]]) -> None:
    score_metrics = [
        "command_jerk_rms_mean",
        "eef_linear_jerk_rms_mean",
        "hf_energy_ratio_mean",
    ]

    minima: dict[str, float] = {}
    for metric_name in score_metrics:
        values = [
            float(summary[metric_name])
            for summary in summaries
            if math.isfinite(float(summary[metric_name])) and float(summary[metric_name]) > 0.0
        ]
        minima[metric_name] = min(values) if values else math.nan

    for summary in summaries:
        ratios: list[float] = []
        for metric_name in score_metrics:
            value = float(summary[metric_name])
            reference = minima[metric_name]
            if math.isfinite(value) and math.isfinite(reference) and reference > 0.0:
                ratios.append(value / reference)
        summary["relative_smoothness_score"] = float(np.mean(ratios)) if ratios else math.nan


def write_per_demo_csv(rows: list[DemoMetrics], output_path: Path) -> None:
    fieldnames = [
        "label",
        "input_path",
        "freq_hz",
        "demo_name",
        "raw_samples",
        "used_samples",
        "trim_start",
        "trim_end",
        "command_accel_rms",
        "command_jerk_rms",
        "command_total_variation",
        "hf_energy_ratio",
        "eef_linear_accel_rms",
        "eef_linear_jerk_rms",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def write_summary_csv(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_metric(
    ax: plt.Axes,
    summaries: list[dict[str, float | str | int]],
    mean_key: str,
    std_key: str,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(summaries), dtype=np.int64)
    y = np.asarray([float(summary[mean_key]) for summary in summaries], dtype=np.float64)
    yerr = np.asarray([float(summary[std_key]) for summary in summaries], dtype=np.float64)
    labels = [f"{summary['label']}\n{float(summary['freq_hz']):g} Hz" for summary in summaries]

    finite = np.isfinite(y)
    if np.any(finite):
        ax.errorbar(
            x[finite],
            y[finite],
            yerr=np.where(np.isfinite(yerr[finite]), yerr[finite], 0.0),
            fmt="-o",
            capsize=4,
            linewidth=2.0,
            markersize=6.0,
        )
        best_idx = int(np.nanargmin(y))
        ax.scatter([x[best_idx]], [y[best_idx]], s=90, marker="*", zorder=5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, alpha=0.3)


def plot_summary_figure(
    summaries: list[dict[str, float | str | int]],
    output_path: Path,
    *,
    title: str,
    hf_cutoff_hz: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    _plot_metric(
        axes[0, 0],
        summaries,
        "command_accel_rms_mean",
        "command_accel_rms_std",
        "Command Accel RMS",
        "lower is smoother",
    )
    _plot_metric(
        axes[0, 1],
        summaries,
        "command_jerk_rms_mean",
        "command_jerk_rms_std",
        "Command Jerk RMS",
        "lower is smoother",
    )
    _plot_metric(
        axes[1, 0],
        summaries,
        "eef_linear_jerk_rms_mean",
        "eef_linear_jerk_rms_std",
        "EEF Linear Jerk RMS",
        "lower is smoother",
    )
    _plot_metric(
        axes[1, 1],
        summaries,
        "hf_energy_ratio_mean",
        "hf_energy_ratio_std",
        f"HF Energy Ratio (>= {hf_cutoff_hz:g} Hz)",
        "lower is smoother",
    )

    final_title = title.strip() or "Action Smoothness Summary"
    fig.suptitle(final_title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_ranking(summaries: Iterable[dict[str, float | str | int]]) -> None:
    ordered = sorted(
        summaries,
        key=lambda item: (
            math.inf
            if not math.isfinite(float(item["relative_smoothness_score"]))
            else float(item["relative_smoothness_score"])
        ),
    )
    print("\nSmoothness ranking (lower score is smoother):")
    for idx, summary in enumerate(ordered, start=1):
        score = float(summary["relative_smoothness_score"])
        score_text = f"{score:.4f}" if math.isfinite(score) else "nan"
        print(
            f"  {idx}. {summary['label']} ({float(summary['freq_hz']):g} Hz)"
            f" | score={score_text}"
            f" | cmd_jerk={float(summary['command_jerk_rms_mean']):.6f}"
            f" | eef_jerk={float(summary['eef_linear_jerk_rms_mean']):.6f}"
            f" | hf_ratio={float(summary['hf_energy_ratio_mean']):.6f}"
        )


def main() -> int:
    args = parse_args()
    experiments = build_experiments(args)

    all_rows: list[DemoMetrics] = []
    for spec in experiments:
        rows = analyze_experiment(
            spec,
            include_gripper=bool(args.include_gripper),
            trim_zero_threshold=float(args.trim_zero_threshold),
            trim_context=int(args.trim_context),
            hf_cutoff_hz=float(args.hf_cutoff_hz),
            min_samples=int(args.min_samples),
            include_idle=bool(args.include_idle),
        )
        if not rows:
            print(f"[warn] no usable demos found for {spec.label} ({spec.path})")
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No usable demos found in the provided inputs.")

    summaries = summarize_metrics(all_rows)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_demo_csv = output_dir / "per_demo_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
    summary_png = output_dir / "smoothness_summary.png"

    write_per_demo_csv(all_rows, per_demo_csv)
    write_summary_csv(summaries, summary_csv)
    plot_summary_figure(
        summaries,
        summary_png,
        title=str(args.title),
        hf_cutoff_hz=float(args.hf_cutoff_hz),
    )

    print(f"Saved per-demo metrics: {per_demo_csv}")
    print(f"Saved summary metrics: {summary_csv}")
    print(f"Saved summary figure: {summary_png}")
    print_ranking(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
