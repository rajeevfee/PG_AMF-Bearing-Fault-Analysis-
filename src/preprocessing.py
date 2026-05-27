"""
Signal preprocessing, segmentation, and statistical feature extraction.

Implements:
  - Mean-removal preprocessing per segment
  - Sliding-window segmentation
  - 6 time-domain statistical features per channel (12 total for 2-channel input)
  - Feature normalization via StandardScaler
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature names — used for figure labels and table headers
# ---------------------------------------------------------------------------

STAT_FEAT_NAMES_1CH = ["RMS", "Kurtosis", "Crest Factor", "Skewness", "Variance", "Peak-to-Peak"]
STAT_FEAT_NAMES = [f + c for c in ["_Ch1", "_Ch2"] for f in STAT_FEAT_NAMES_1CH]


# =============================================================================
# Preprocessing helpers
# =============================================================================

def preprocess_segment(seg: np.ndarray, do_mean_removal: bool = True) -> np.ndarray:
    """
    Apply basic preprocessing to a single 1-D segment.

    Parameters
    ----------
    seg:
        Raw 1-D vibration segment.
    do_mean_removal:
        If ``True`` (default), subtract the segment mean to remove DC offset.

    Returns
    -------
    float32 array of the same length.
    """
    seg = np.asarray(seg, dtype=np.float64)
    if do_mean_removal:
        seg = seg - seg.mean()
    return seg.astype(np.float32)


def segment_signal(
    signal: np.ndarray,
    seg_len: int = 8192,
    hop: int = 4096,
) -> np.ndarray:
    """
    Slice a 1-D signal into overlapping fixed-length windows.

    Parameters
    ----------
    signal:
        1-D time series.
    seg_len:
        Window length in samples.
    hop:
        Step size between successive windows.

    Returns
    -------
    np.ndarray of shape ``(N_segments, seg_len)``.
    """
    n = len(signal)
    starts = np.arange(0, n - seg_len + 1, hop)
    return np.stack([signal[i : i + seg_len] for i in starts], axis=0)


# =============================================================================
# Dataset builder
# =============================================================================

def build_dataset(
    df_raw: pd.DataFrame,
    seg_len: int = 8192,
    hop: int = 4096,
    min_segs: int = 50,
    max_segs: int = 200,
    do_mean_removal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a (X, y) dataset from the raw DataFrame returned by
    :meth:`~src.dataset.BearingDataLoader.to_dataframe`.

    Both channels are stacked along axis 1 to produce shape
    ``(N_total, 2, seg_len)``.

    Parameters
    ----------
    df_raw:
        DataFrame with columns ``ch1``, ``ch2``, ``label``.
    seg_len, hop:
        Segmentation parameters (see :func:`segment_signal`).
    min_segs:
        Classes with fewer segments than this are skipped with a warning.
    max_segs:
        Maximum number of segments to retain per class.
    do_mean_removal:
        Forwarded to :func:`preprocess_segment`.

    Returns
    -------
    X : np.ndarray
        Shape ``(N, 2, seg_len)`` — float32.
    y : np.ndarray
        Shape ``(N,)`` — int64 class labels.
    """
    all_segs: List[np.ndarray] = []
    all_labels: List[int] = []

    for _, row in df_raw.iterrows():
        label = int(row["label"])
        segs1 = segment_signal(row["ch1"], seg_len, hop)
        segs2 = segment_signal(row["ch2"], seg_len, hop)
        n = min(len(segs1), len(segs2), max_segs)
        if n < min_segs:
            logger.warning(
                "Class '%s' has only %d segments (< min=%d) — skipped.",
                row.get("fault_name", label),
                n,
                min_segs,
            )
            continue
        proc1 = np.stack([preprocess_segment(s, do_mean_removal) for s in segs1[:n]])
        proc2 = np.stack([preprocess_segment(s, do_mean_removal) for s in segs2[:n]])
        X_cls = np.stack([proc1, proc2], axis=1)  # (n, 2, seg_len)
        all_segs.append(X_cls)
        all_labels.extend([label] * n)

    X = np.concatenate(all_segs, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = 0.20,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    Randomly shuffle and split ``(X, y)`` into train / val / test.

    Returns
    -------
    Tuple of eight arrays:
        ``X_train, y_train, X_val, y_val, X_test, y_test``
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    n = len(y)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    logger.info(
        "Split — train: %d | val: %d | test: %d | class dist: %s",
        n_train, n_val, n_test, np.bincount(y).tolist(),
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# Statistical feature extraction  (Branch a — baseline)
# =============================================================================

def stat_features_single(seg: np.ndarray) -> np.ndarray:
    """
    Compute 6 time-domain statistical features for a 1-D segment.

    Features: RMS, Kurtosis, Crest Factor, Skewness, Variance, Peak-to-Peak.

    Returns
    -------
    np.ndarray of shape ``(6,)`` — float32.
    """
    seg = seg.astype(np.float64)
    mu = np.mean(seg)
    sigma = np.std(seg)
    eps = 1e-10

    rms = np.sqrt(np.mean(seg**2))
    kurt = np.mean((seg - mu) ** 4) / (sigma**4 + eps)
    crest = np.max(np.abs(seg)) / (rms + eps)
    skew = np.mean((seg - mu) ** 3) / (sigma**3 + eps)
    var = sigma**2
    p2p = np.max(seg) - np.min(seg)

    return np.array([rms, kurt, crest, skew, var, p2p], dtype=np.float32)


def extract_stat_features(X: np.ndarray) -> np.ndarray:
    """
    Extract 12 statistical features from a 2-channel segment array.

    Parameters
    ----------
    X:
        Shape ``(N, 2, seg_len)``.

    Returns
    -------
    np.ndarray of shape ``(N, 12)`` — float32.
    """
    N = X.shape[0]
    f1 = np.stack([stat_features_single(X[i, 0]) for i in range(N)])
    f2 = np.stack([stat_features_single(X[i, 1]) for i in range(N)])
    return np.concatenate([f1, f2], axis=1).astype(np.float32)


def fit_scaler(F_train: np.ndarray) -> StandardScaler:
    """Fit a :class:`StandardScaler` on training features and return it."""
    scaler = StandardScaler()
    scaler.fit(F_train)
    return scaler
