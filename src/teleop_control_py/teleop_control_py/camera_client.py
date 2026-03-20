#!/usr/bin/env python3
"""Compatibility wrapper. Prefer ``teleop_control_py.hardware.camera_client``."""

from .hardware.camera_client import OAKClient, RealSenseClient

__all__ = ["OAKClient", "RealSenseClient"]
