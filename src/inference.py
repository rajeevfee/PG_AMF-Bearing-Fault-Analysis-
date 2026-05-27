"""
Inference utilities: load trained checkpoints and predict fault classes
for new raw vibration segments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.model import MLPBaseline, PGAMFClassifier
from src.preprocessing import (
    extract_stat_features,
    fit_scaler,
    preprocess_segment,
    segment_signal,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Checkpoint I/O
# =============================================================================

def load_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """
    Load saved weights into *model* (in-place).

    Parameters
    ----------
    model:
        Model instance with the same architecture as at save time.
    checkpoint_path:
        Path to the ``.pt`` file produced by the trainer.

    Returns
    -------
    The same model with weights loaded.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    logger.info("Checkpoint loaded: %s", path)
    return model


# =============================================================================
# PG-AMF inference pipeline
# =============================================================================

class PGAMFInference:
    """
    End-to-end inference wrapper for the PG-AMF classifier.

    Accepts raw 1-D vibration arrays from two channels, applies the same
    segmentation and preprocessing used during training, and returns
    predicted class labels with confidence scores.

    Parameters
    ----------
    model:
        Trained :class:`~src.model.PGAMFClassifier`.
    fault_names:
        List of human-readable class names in label order.
    device:
        Inference device.
    seg_len:
        Segment length used during training.
    hop:
        Hop size used during training.
    do_mean_removal:
        Whether to apply mean removal (must match training setting).
    """

    def __init__(
        self,
        model: PGAMFClassifier,
        fault_names: List[str],
        device: torch.device,
        seg_len: int = 8192,
        hop: int = 4096,
        do_mean_removal: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.fault_names = fault_names
        self.device = device
        self.seg_len = seg_len
        self.hop = hop
        self.do_mean_removal = do_mean_removal

    @torch.no_grad()
    def predict(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray,
        batch_size: int = 64,
    ) -> Dict:
        """
        Predict fault class for raw two-channel vibration data.

        Parameters
        ----------
        ch1, ch2:
            1-D float arrays of equal length (raw vibration samples).
        batch_size:
            Inference mini-batch size.

        Returns
        -------
        dict with keys:
            ``predictions``  — int array ``(N_segs,)``
            ``class_names``  — list of predicted class name strings
            ``probabilities``— float array ``(N_segs, n_classes)``
            ``majority_vote``— most frequent predicted class name
        """
        # Segment both channels
        segs1 = segment_signal(ch1, self.seg_len, self.hop)
        segs2 = segment_signal(ch2, self.seg_len, self.hop)
        n = min(len(segs1), len(segs2))

        proc1 = np.stack([preprocess_segment(s, self.do_mean_removal) for s in segs1[:n]])
        proc2 = np.stack([preprocess_segment(s, self.do_mean_removal) for s in segs2[:n]])
        X = np.stack([proc1, proc2], axis=1)  # (N, 2, seg_len)

        all_logits: List[torch.Tensor] = []
        for i in range(0, n, batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(self.device)
            all_logits.append(self.model(xb))

        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        majority = int(np.bincount(preds).argmax())
        return {
            "predictions": preds,
            "class_names": [self.fault_names[p] for p in preds],
            "probabilities": probs,
            "majority_vote": self.fault_names[majority],
        }


# =============================================================================
# Statistical baseline inference pipeline
# =============================================================================

class StatMLPInference:
    """
    End-to-end inference wrapper for the statistical MLP baseline.

    Parameters
    ----------
    model:
        Trained :class:`~src.model.MLPBaseline`.
    scaler:
        Fitted :class:`~sklearn.preprocessing.StandardScaler`.
    fault_names:
        Human-readable class names in label order.
    device:
        Inference device.
    seg_len, hop, do_mean_removal:
        Must match training-time settings.
    """

    def __init__(
        self,
        model: MLPBaseline,
        scaler,
        fault_names: List[str],
        device: torch.device,
        seg_len: int = 8192,
        hop: int = 4096,
        do_mean_removal: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.scaler = scaler
        self.fault_names = fault_names
        self.device = device
        self.seg_len = seg_len
        self.hop = hop
        self.do_mean_removal = do_mean_removal

    @torch.no_grad()
    def predict(self, ch1: np.ndarray, ch2: np.ndarray) -> Dict:
        """
        Predict fault class from raw two-channel vibration data.

        Returns the same dict structure as :meth:`PGAMFInference.predict`.
        """
        segs1 = segment_signal(ch1, self.seg_len, self.hop)
        segs2 = segment_signal(ch2, self.seg_len, self.hop)
        n = min(len(segs1), len(segs2))

        proc1 = np.stack([preprocess_segment(s, self.do_mean_removal) for s in segs1[:n]])
        proc2 = np.stack([preprocess_segment(s, self.do_mean_removal) for s in segs2[:n]])
        X = np.stack([proc1, proc2], axis=1)

        F_stat = extract_stat_features(X)
        F_stat = self.scaler.transform(F_stat)

        xb = torch.tensor(F_stat, dtype=torch.float32).to(self.device)
        probs = torch.softmax(self.model(xb), dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        majority = int(np.bincount(preds).argmax())
        return {
            "predictions": preds,
            "class_names": [self.fault_names[p] for p in preds],
            "probabilities": probs,
            "majority_vote": self.fault_names[majority],
        }
