#!/usr/bin/env python3
import argparse
from typing import Dict, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


AXIS_LABELS: Dict[str, Dict[int, str]] = {
    "xbox": {
        0: "left_stick_x",
        1: "left_stick_y",
        2: "right_stick_x",
        3: "right_stick_y",
        4: "left_trigger",
        5: "right_trigger",
        6: "dpad_x",
        7: "dpad_y",
    },
    "ps5": {
        0: "left_stick_x",
        1: "left_stick_y",
        2: "right_stick_x",
        3: "right_stick_y",
        4: "left_trigger",
        5: "right_trigger",
        6: "dpad_x",
        7: "dpad_y",
    },
    "generic": {
        0: "axis_0",
        1: "axis_1",
        2: "axis_2",
        3: "axis_3",
        4: "axis_4",
        5: "axis_5",
        6: "axis_6",
        7: "axis_7",
    },
}

BUTTON_LABELS: Dict[str, Dict[int, str]] = {
    "xbox": {
        0: "A",
        1: "B",
        2: "X",
        3: "Y",
        4: "LB",
        5: "RB",
        6: "BACK",
        7: "START",
        8: "XBOX",
        9: "L3",
        10: "R3",
        11: "DPAD_UP",
        12: "DPAD_DOWN",
        13: "DPAD_LEFT",
        14: "DPAD_RIGHT",
    },
    "ps5": {
        0: "CROSS",
        1: "CIRCLE",
        2: "TRIANGLE",
        3: "SQUARE",
        4: "L1",
        5: "R1",
        6: "CREATE",
        7: "OPTIONS",
        8: "PS",
        9: "L3",
        10: "R3",
        11: "DPAD_UP",
        12: "DPAD_DOWN",
        13: "DPAD_LEFT",
        14: "DPAD_RIGHT",
    },
    "generic": {
        0: "button_0",
        1: "button_1",
        2: "button_2",
        3: "button_3",
        4: "button_4",
        5: "button_5",
        6: "button_6",
        7: "button_7",
        8: "button_8",
        9: "button_9",
        10: "button_10",
        11: "button_11",
        12: "button_12",
        13: "button_13",
        14: "button_14",
    },
}


class JoyMappingVerifier(Node):
    def __init__(self, topic: str, profile: str, axis_threshold: float):
        super().__init__("joy_mapping_verifier")
        self.topic = topic
        self.profile = profile if profile in AXIS_LABELS else "generic"
        self.axis_threshold = axis_threshold

        self.prev_axes: List[float] = []
        self.prev_buttons: List[int] = []

        self.axis_labels = AXIS_LABELS[self.profile]
        self.button_labels = BUTTON_LABELS[self.profile]

        self.create_subscription(Joy, self.topic, self._on_joy, 10)

        self.get_logger().info(f"Listening {self.topic}, profile={self.profile}")
        self._print_legend()

    def _print_legend(self) -> None:
        self.get_logger().info("=== Axis Mapping ===")
        for idx in sorted(self.axis_labels.keys()):
            self.get_logger().info(f"  axes[{idx}] -> {self.axis_labels[idx]}")
        self.get_logger().info("=== Button Mapping ===")
        for idx in sorted(self.button_labels.keys()):
            self.get_logger().info(f"  buttons[{idx}] -> {self.button_labels[idx]}")
        self.get_logger().info("====================")

    def _on_joy(self, msg: Joy) -> None:
        if not self.prev_axes:
            self.prev_axes = list(msg.axes)
        if not self.prev_buttons:
            self.prev_buttons = list(msg.buttons)

        for idx, val in enumerate(msg.buttons):
            prev = self.prev_buttons[idx] if idx < len(self.prev_buttons) else 0
            if prev != val:
                label = self.button_labels.get(idx, f"button_{idx}")
                state = "PRESSED" if val else "RELEASED"
                self.get_logger().info(f"[BUTTON] {state:<8} idx={idx:<2} name={label}")

        for idx, val in enumerate(msg.axes):
            prev = self.prev_axes[idx] if idx < len(self.prev_axes) else 0.0
            if abs(val - prev) >= self.axis_threshold:
                label = self.axis_labels.get(idx, f"axis_{idx}")
                self.get_logger().info(f"[AXIS]   MOVE      idx={idx:<2} name={label:<14} value={val:+.3f}")

        self.prev_buttons = list(msg.buttons)
        self.prev_axes = list(msg.axes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify /joy button and axis mapping")
    parser.add_argument("--topic", default="/joy", help="Joy topic name")
    parser.add_argument(
        "--profile",
        default="xbox",
        choices=["xbox", "ps5", "generic"],
        help="Mapping legend profile",
    )
    parser.add_argument(
        "--axis-threshold",
        type=float,
        default=0.15,
        help="Minimum axis delta to print movement",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = JoyMappingVerifier(
        topic=args.topic,
        profile=args.profile,
        axis_threshold=args.axis_threshold,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
