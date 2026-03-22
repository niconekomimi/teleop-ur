#!/usr/bin/env python3
"""Hardware layer interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics for the color/rgb stream."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class CameraFrame:
    """Extended camera output: BGR, optional depth, intrinsics, disparity."""

    bgr: np.ndarray
    depth: Optional[np.ndarray] = None
    intrinsics: Optional[CameraIntrinsics] = None
    disparity: Optional[np.ndarray] = None

    def get_bgr(self) -> np.ndarray:
        return self.bgr


class BaseCamera(ABC):
    """Minimal camera contract for all hardware backends."""

    @abstractmethod
    def start(self) -> None:
        """Acquire resources and begin serving frames."""

    @abstractmethod
    def stop(self) -> None:
        """Release resources owned by this camera client."""

    @abstractmethod
    def get_bgr_frame(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame, or ``None`` when unavailable."""

    def get_frame(self) -> Optional[CameraFrame]:
        """Return extended frame (BGR + depth + intrinsics + disparity if enabled). Default uses get_bgr_frame()."""
        bgr = self.get_bgr_frame()
        if bgr is None:
            return None
        return CameraFrame(bgr=bgr)
