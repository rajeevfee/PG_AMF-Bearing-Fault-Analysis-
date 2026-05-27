"""
Utility helpers: configuration loading, reproducibility seeding,
device selection, and bearing characteristic-frequency computation.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def load_config(*yaml_paths: str | Path) -> Dict[str, Any]:
    """
    Load and merge one or more YAML config files (later files override earlier).

    Parameters
    ----------
    *yaml_paths:
        Paths to YAML files.  The first file provides the base; subsequent
        files are merged on top using a shallow update.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    merged: Dict[str, Any] = {}
    for p in yaml_paths:
        with open(p, "r") as fh:
            data = yaml.safe_load(fh) or {}
        _deep_update(merged, data)
    return merged


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Random seed set to %d", seed)


# =============================================================================
# Device selection
# =============================================================================

def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string to a :class:`torch.device`.

    Parameters
    ----------
    device_str:
        ``"auto"`` selects CUDA when available, otherwise CPU.
        Otherwise forwarded directly to ``torch.device``.
    """
    if device_str == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device_str)
    logger.info("Using device: %s", dev)
    return dev


# =============================================================================
# Bearing characteristic frequencies
# =============================================================================

def compute_char_freqs(
    n_balls: int,
    ball_diameter: float,
    pitch_diameter: float,
    shaft_freq: float,
) -> Dict[str, float]:
    """
    Compute bearing fault characteristic frequencies.

    Parameters
    ----------
    n_balls:
        Number of rolling elements (Nb).
    ball_diameter:
        Ball diameter in mm (Bd).
    pitch_diameter:
        Pitch circle diameter in mm (Pd).
    shaft_freq:
        Shaft rotation frequency in Hz (fr).

    Returns
    -------
    dict
        Keys: ``BPFO``, ``BPFI``, ``BSF``, ``FTF``, ``fr`` — all in Hz.
    """
    chi = ball_diameter / pitch_diameter
    bpfo = round((n_balls / 2) * shaft_freq * (1 - chi), 3)
    bpfi = round((n_balls / 2) * shaft_freq * (1 + chi), 3)
    bsf  = round((pitch_diameter / (2 * ball_diameter)) * shaft_freq * (1 - chi**2), 3)
    ftf  = round((shaft_freq / 2) * (1 - chi), 3)
    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf, "fr": shaft_freq}


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a timestamp format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
