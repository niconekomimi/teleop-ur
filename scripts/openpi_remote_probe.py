#!/usr/bin/env python3
"""Probe an openpi websocket policy server, typically through an SSH tunnel.

Examples:
    python3 scripts/openpi_remote_probe.py
    python3 scripts/openpi_remote_probe.py --host 127.0.0.1 --port 18000 --infer --env libero
    python3 scripts/openpi_remote_probe.py --uri ws://127.0.0.1:18000 --infer --env droid
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe an openpi websocket policy server.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host when --uri is not provided.")
    parser.add_argument("--port", type=int, default=18000, help="Server port when --uri is not provided.")
    parser.add_argument("--uri", default="", help="Full websocket URI, for example ws://127.0.0.1:18000.")
    parser.add_argument(
        "--health-url",
        default="",
        help="Optional explicit health-check URL. Defaults to http://<host>:<port>/healthz.",
    )
    parser.add_argument("--api-key", default="", help="Optional Api-Key header value.")
    parser.add_argument("--connect-timeout", type=float, default=5.0, help="Websocket connect timeout in seconds.")
    parser.add_argument("--health-timeout", type=float, default=3.0, help="Health-check timeout in seconds.")
    parser.add_argument("--skip-healthz", action="store_true", help="Skip the HTTP /healthz probe.")
    parser.add_argument("--infer", action="store_true", help="Send one dummy inference request after connecting.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of dummy inference requests to send when --infer is enabled.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Optional sleep interval in seconds between repeated inference requests.",
    )
    parser.add_argument(
        "--env",
        choices=("libero", "droid", "aloha", "aloha_sim", "ur5"),
        default="libero",
        help="Dummy observation format to use with --infer.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Dummy image width/height.")
    parser.add_argument("--libero-state-dim", type=int, default=8, help="State dimension for --env libero.")
    parser.add_argument("--aloha-state-dim", type=int, default=14, help="State dimension for --env aloha*.")
    parser.add_argument("--droid-joint-dim", type=int, default=7, help="Joint state dimension for --env droid.")
    parser.add_argument("--ur5-joint-dim", type=int, default=6, help="Joint state dimension for --env ur5.")
    parser.add_argument("--prompt", default="do something", help="Prompt used for dummy inference.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to build deterministic dummy observations.",
    )
    return parser.parse_args()


def build_ws_uri(args: argparse.Namespace) -> str:
    if args.uri:
        return str(args.uri).strip()
    return f"ws://{args.host}:{int(args.port)}"


def build_health_url(args: argparse.Namespace) -> str:
    if args.health_url:
        return str(args.health_url).strip()
    return f"http://{args.host}:{int(args.port)}/healthz"


def probe_healthz(url: str, timeout_sec: float) -> tuple[bool, str]:
    request = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
            return True, f"HTTP {response.status} {body}".strip()
    except urllib.error.URLError as exc:
        return False, repr(exc)
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def build_dummy_observation(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    image_shape_hwc = (int(args.image_size), int(args.image_size), 3)
    image_shape_chw = (3, int(args.image_size), int(args.image_size))

    if args.env == "libero":
        return {
            "observation/state": rng.random(int(args.libero_state_dim), dtype=np.float32),
            "observation/image": rng.integers(0, 256, size=image_shape_hwc, dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, size=image_shape_hwc, dtype=np.uint8),
            "prompt": args.prompt,
        }

    if args.env == "droid":
        return {
            "observation/exterior_image_1_left": rng.integers(0, 256, size=image_shape_hwc, dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0, 256, size=image_shape_hwc, dtype=np.uint8),
            "observation/joint_position": rng.random(int(args.droid_joint_dim), dtype=np.float32),
            "observation/gripper_position": rng.random(1, dtype=np.float32),
            "prompt": args.prompt,
        }

    if args.env == "ur5":
        return {
            "observation/image": rng.integers(0, 256, size=image_shape_hwc, dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, size=image_shape_hwc, dtype=np.uint8),
            "observation/joint_position": rng.random(int(args.ur5_joint_dim), dtype=np.float32),
            "observation/gripper_position": rng.random(1, dtype=np.float32),
            "prompt": args.prompt,
        }

    return {
        "state": rng.random(int(args.aloha_state_dim), dtype=np.float32),
        "images": {
            "cam_high": rng.integers(0, 256, size=image_shape_chw, dtype=np.uint8),
            "cam_low": rng.integers(0, 256, size=image_shape_chw, dtype=np.uint8),
            "cam_left_wrist": rng.integers(0, 256, size=image_shape_chw, dtype=np.uint8),
            "cam_right_wrist": rng.integers(0, 256, size=image_shape_chw, dtype=np.uint8),
        },
        "prompt": args.prompt,
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            summary[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            continue
        if isinstance(value, dict):
            summary[key] = summarize_payload(value)
            continue
        summary[key] = to_jsonable(value)
    return summary


def summarize_action_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_payload(result)
    if "actions" in result:
        actions = np.asarray(result["actions"])
        summary["actions"] = {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "first_row": to_jsonable(actions[0]) if actions.ndim >= 2 and actions.shape[0] > 0 else None,
        }
    return summary


def format_ms_stats(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    return {
        "count": float(len(values_ms)),
        "mean_ms": float(statistics.mean(values_ms)),
        "min_ms": float(min(values_ms)),
        "max_ms": float(max(values_ms)),
        "median_ms": float(statistics.median(values_ms)),
        "stdev_ms": float(statistics.pstdev(values_ms)) if len(values_ms) > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()
    ws_uri = build_ws_uri(args)
    health_url = build_health_url(args)

    print(f"[probe] websocket uri: {ws_uri}")
    if not args.skip_healthz:
        print(f"[probe] healthz url: {health_url}")
        health_ok, health_message = probe_healthz(health_url, timeout_sec=float(args.health_timeout))
        status_text = "ok" if health_ok else "failed"
        print(f"[healthz] {status_text}: {health_message}")
    else:
        print("[healthz] skipped")

    headers = {"Authorization": f"Api-Key {args.api_key}"} if args.api_key else None
    packer = msgpack_numpy.Packer()

    try:
        connect_start = time.perf_counter()
        with websockets.sync.client.connect(
            ws_uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=float(args.connect_timeout),
        ) as websocket:
            connect_ms = (time.perf_counter() - connect_start) * 1000.0
            print(f"[ws] connected in {connect_ms:.1f} ms")

            metadata_frame = websocket.recv()
            if isinstance(metadata_frame, str):
                print("[ws] metadata error frame:")
                print(metadata_frame)
                return 2

            metadata = msgpack_numpy.unpackb(metadata_frame)
            print("[ws] server metadata:")
            print(json.dumps(to_jsonable(metadata), ensure_ascii=False, indent=2, sort_keys=True))

            if not args.infer:
                print("[infer] skipped")
                return 0

            observation = build_dummy_observation(args)
            print("[infer] dummy observation summary:")
            print(json.dumps(summarize_payload(observation), ensure_ascii=False, indent=2, sort_keys=True))

            repeat = max(1, int(args.repeat))
            interval = max(0.0, float(args.interval))
            roundtrip_ms_values: list[float] = []
            policy_infer_ms_values: list[float] = []
            server_infer_ms_values: list[float] = []
            last_result_summary: dict[str, Any] | None = None

            for index in range(repeat):
                infer_start = time.perf_counter()
                websocket.send(packer.pack(observation))
                response = websocket.recv()
                roundtrip_ms = (time.perf_counter() - infer_start) * 1000.0

                if isinstance(response, str):
                    print(f"[infer] server returned an error frame on iteration {index + 1}:")
                    print(response)
                    return 3

                result = msgpack_numpy.unpackb(response)
                roundtrip_ms_values.append(roundtrip_ms)

                policy_timing = result.get("policy_timing", {})
                server_timing = result.get("server_timing", {})
                policy_infer_ms = policy_timing.get("infer_ms")
                server_infer_ms = server_timing.get("infer_ms")
                if policy_infer_ms is not None and math.isfinite(float(policy_infer_ms)):
                    policy_infer_ms_values.append(float(policy_infer_ms))
                if server_infer_ms is not None and math.isfinite(float(server_infer_ms)):
                    server_infer_ms_values.append(float(server_infer_ms))

                actions = np.asarray(result.get("actions")) if "actions" in result else None
                action_shape = list(actions.shape) if actions is not None else None
                print(
                    "[infer] "
                    f"iter {index + 1}/{repeat} | round-trip {roundtrip_ms:.1f} ms"
                    + (f" | policy {float(policy_infer_ms):.1f} ms" if policy_infer_ms is not None else "")
                    + (f" | server {float(server_infer_ms):.1f} ms" if server_infer_ms is not None else "")
                    + (f" | actions {action_shape}" if action_shape is not None else "")
                )

                last_result_summary = summarize_action_result(result)
                if interval > 0.0 and index + 1 < repeat:
                    time.sleep(interval)

            print("[infer] timing summary:")
            summary = {
                "roundtrip": format_ms_stats(roundtrip_ms_values),
                "policy_infer": format_ms_stats(policy_infer_ms_values),
                "server_infer": format_ms_stats(server_infer_ms_values),
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

            if last_result_summary is not None:
                print("[infer] last response summary:")
                print(json.dumps(last_result_summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
