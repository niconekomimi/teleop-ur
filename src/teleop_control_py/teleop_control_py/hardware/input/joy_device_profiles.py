"""Joystick device profiles for evdev-based Joy driver (Xbox, PS5, generic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from evdev import ecodes


@dataclass(frozen=True)
class AxisSpec:
    index: int
    min_value: int
    max_value: int
    invert: bool = False
    deadzone: float = 0.05


@dataclass(frozen=True)
class JoyProfile:
    name: str
    aliases: List[str]
    axis_specs: Dict[int, AxisSpec]
    button_indices: Dict[int, int]
    axis_count: int
    button_count: int


def _common_button_map() -> Dict[int, int]:
    return {
        ecodes.BTN_SOUTH: 0,
        ecodes.BTN_EAST: 1,
        ecodes.BTN_NORTH: 2,
        ecodes.BTN_WEST: 3,
        ecodes.BTN_TL: 4,
        ecodes.BTN_TR: 5,
        ecodes.BTN_SELECT: 6,
        ecodes.BTN_START: 7,
        ecodes.BTN_MODE: 8,
        ecodes.BTN_THUMBL: 9,
        ecodes.BTN_THUMBR: 10,
        ecodes.BTN_DPAD_UP: 11,
        ecodes.BTN_DPAD_DOWN: 12,
        ecodes.BTN_DPAD_LEFT: 13,
        ecodes.BTN_DPAD_RIGHT: 14,
    }


def _common_axis_map(deadzone: float) -> Dict[int, AxisSpec]:
    return {
        ecodes.ABS_X: AxisSpec(index=0, min_value=-32768, max_value=32767, deadzone=deadzone),
        ecodes.ABS_Y: AxisSpec(index=1, min_value=-32768, max_value=32767, invert=True, deadzone=deadzone),
        ecodes.ABS_RX: AxisSpec(index=2, min_value=-32768, max_value=32767, deadzone=deadzone),
        ecodes.ABS_RY: AxisSpec(index=3, min_value=-32768, max_value=32767, invert=True, deadzone=deadzone),
        ecodes.ABS_Z: AxisSpec(index=4, min_value=0, max_value=255, deadzone=0.0),
        ecodes.ABS_RZ: AxisSpec(index=5, min_value=0, max_value=255, deadzone=0.0),
        ecodes.ABS_HAT0X: AxisSpec(index=6, min_value=-1, max_value=1, deadzone=0.0),
        ecodes.ABS_HAT0Y: AxisSpec(index=7, min_value=-1, max_value=1, invert=True, deadzone=0.0),
    }


def _profile_xbox(deadzone: float) -> JoyProfile:
    return JoyProfile(
        name="xbox",
        aliases=["xbox", "x-box", "xbox wireless", "microsoft"],
        axis_specs=_common_axis_map(deadzone),
        button_indices=_common_button_map(),
        axis_count=8,
        button_count=15,
    )


def _profile_ps5(deadzone: float) -> JoyProfile:
    button_map = _common_button_map().copy()
    button_map[ecodes.BTN_TRIGGER_HAPPY1] = 11
    button_map[ecodes.BTN_TRIGGER_HAPPY2] = 12
    button_map[ecodes.BTN_TRIGGER_HAPPY3] = 13
    button_map[ecodes.BTN_TRIGGER_HAPPY4] = 14

    return JoyProfile(
        name="ps5",
        aliases=["dualsense", "wireless controller", "ps5", "sony interactive entertainment"],
        axis_specs=_common_axis_map(deadzone),
        button_indices=button_map,
        axis_count=8,
        button_count=15,
    )


def _profile_generic(deadzone: float) -> JoyProfile:
    return JoyProfile(
        name="generic",
        aliases=["joystick", "gamepad", "controller"],
        axis_specs=_common_axis_map(deadzone),
        button_indices=_common_button_map(),
        axis_count=8,
        button_count=15,
    )


def build_profiles(deadzone: float) -> Dict[str, JoyProfile]:
    profiles = {
        "xbox": _profile_xbox(deadzone),
        "ps5": _profile_ps5(deadzone),
        "generic": _profile_generic(deadzone),
    }
    profiles["auto"] = profiles["generic"]
    return profiles


def normalize_axis(value: int, spec: AxisSpec) -> float:
    if spec.max_value == spec.min_value:
        return 0.0

    scaled = 2.0 * (value - spec.min_value) / float(spec.max_value - spec.min_value) - 1.0
    if spec.min_value >= 0:
        scaled = (value - spec.min_value) / float(spec.max_value - spec.min_value)

    if spec.invert:
        scaled = -scaled

    if abs(scaled) < spec.deadzone:
        return 0.0
    if scaled > 1.0:
        return 1.0
    if scaled < -1.0:
        return -1.0
    return float(scaled)


def infer_profile_key(device_name: str, requested: str) -> str:
    requested = (requested or "auto").lower().strip()
    if requested and requested != "auto":
        return requested

    lower_name = device_name.lower()
    if any(k in lower_name for k in ["xbox", "microsoft"]):
        return "xbox"
    if any(k in lower_name for k in ["dualsense", "ps5", "wireless controller", "sony"]):
        return "ps5"
    return "generic"


def score_device_name(device_name: str, profile: JoyProfile) -> int:
    lower_name = device_name.lower()
    best = 0
    for alias in profile.aliases:
        if alias in lower_name:
            best = max(best, len(alias))
    return best


def choose_profile_and_device(
    available: List[Tuple[str, str]],
    requested_profile: str,
    requested_name: str,
    profiles: Dict[str, JoyProfile],
) -> Tuple[Optional[str], Optional[str], Optional[JoyProfile]]:
    if not available:
        return None, None, None

    requested_name_lower = (requested_name or "").lower().strip()
    filtered = available
    if requested_name_lower:
        filtered = [(path, name) for path, name in available if requested_name_lower in name.lower()]
        if not filtered:
            return None, None, None

    best_candidate = None
    best_score = -1

    for path, name in filtered:
        key = infer_profile_key(name, requested_profile)
        profile = profiles.get(key, profiles["generic"])
        score = score_device_name(name, profile)
        if requested_profile and requested_profile != "auto" and key == requested_profile:
            score += 100
        if score > best_score:
            best_score = score
            best_candidate = (path, name, profile)

    if best_candidate is None:
        path, name = filtered[0]
        key = infer_profile_key(name, requested_profile)
        return path, name, profiles.get(key, profiles["generic"])

    return best_candidate
