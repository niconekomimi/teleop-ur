#!/usr/bin/env python3
"""Hardware layer interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


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
