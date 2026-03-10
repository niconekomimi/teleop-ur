#!/usr/bin/env python3
"""将当前采集得到的 HDF5 重建为指定的数据集结构。"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

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
    results = rebuild_file(
        input_path=args.input,
        output_path=args.output,
        include_states=not args.omit_states,
        renumber=args.renumber,
        compression=args.compression,
        compression_opts=args.compression_level,
    )
    print(f"Found demos: {len(results)}")
    for source_name, target_name, num_samples in results:
        print(f"  {source_name} -> {target_name}: {num_samples} frames")
    print(f"Done. Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
