#!/usr/bin/env python3
"""将当前采集得到的 HDF5 重建为指定的数据集结构。"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "teleop_control_py" / "teleop_control_py" / "dataset_rebuilder.py"

if not MODULE_PATH.exists():
    raise FileNotFoundError(f"Cannot find dataset rebuilder module: {MODULE_PATH}")

_SPEC = importlib.util.spec_from_file_location("teleop_control_py.dataset_rebuilder", MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Failed to load module spec from: {MODULE_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
rebuild_file = _MODULE.rebuild_file
sorted_demo_names = _MODULE.sorted_demo_names


def count_rebuild_steps(input_path: str, include_states: bool) -> tuple[int, int]:
    with h5py.File(input_path, "r") as source_h5:
        source_data = source_h5.get("data")
        if not isinstance(source_data, h5py.Group):
            raise RuntimeError("Input HDF5 has no /data group")

        demo_names = [name for name in sorted_demo_names(source_data) if isinstance(source_data.get(name), h5py.Group)]
        if not demo_names:
            raise RuntimeError("No demo_* groups found under /data")

    datasets_per_demo = 12 if include_states else 11
    return len(demo_names), len(demo_names) * datasets_per_demo + 1


def render_progress(completed: int, total: int, label: str) -> None:
    total = max(1, int(total))
    completed = max(0, min(int(completed), total))
    bar_width = 32
    filled = int(bar_width * completed / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    message = f"\r[{bar}] {completed}/{total} {label}"
    sys.stdout.write(message)
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild teleop HDF5 to a new dataset schema.")
    parser.add_argument("--input", required=True, help="Input HDF5 path")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument(
        "--compression",
        default="inherit",
        choices=["inherit", "none", "lzf", "gzip"],
        help="Compression policy for output datasets: inherit source settings, disable compression, or force lzf/gzip",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="gzip compression level (0-9), only valid when --compression gzip",
    )
    parser.add_argument(
        "--omit-states",
        action="store_true",
        help="Do not write the top-level states dataset",
    )
    parser.add_argument(
        "--renumber",
        action="store_true",
        help="Rewrite demo names to continuous demo_0, demo_1, ...",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f"Output exists: {args.output} (use --force to overwrite)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    include_states = not args.omit_states
    demo_count, total_steps = count_rebuild_steps(args.input, include_states)

    output_path = Path(args.output)
    temp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    completed_steps = 0

    def update_progress(dataset_path: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        render_progress(completed_steps, total_steps, f"正在转换 {dataset_path}")

    render_progress(0, total_steps, f"准备转换 {demo_count} 个 demo")

    try:
        results = rebuild_file(
            input_path=args.input,
            output_path=str(temp_output_path),
            include_states=include_states,
            renumber=args.renumber,
            compression=args.compression,
            compression_opts=args.compression_level,
            progress_callback=update_progress,
        )
        render_progress(total_steps, total_steps, "正在写入输出文件")
        os.replace(temp_output_path, output_path)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        sys.stdout.write("\n")
        raise

    sys.stdout.write("\n")
    print(f"Found demos: {len(results)}")
    for source_name, target_name, num_samples in results:
        print(f"  {source_name} -> {target_name}: {num_samples} frames")
    print(f"Done. Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
