"""
Training loops for both branches of the PG-AMF pipeline.

Branch (a) — :func:`train_basic`:
    Plain Adam, fixed LR, cross-entropy, early stopping on val accuracy.

Branch (b) — :func:`train_smooth`:
    AdamW + CosineAnnealingLR, focal CE + diversity + cosine compactness,
    gradient clipping, best-checkpoint restoration.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.model import PGAMFClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# DataLoader factory
# =============================================================================

def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test :class:`DataLoader` objects.

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test:
        NumPy arrays.  ``X`` can be float features ``(N, D)`` for the stat
        branch or raw segments ``(N, 2, T)`` for PG-AMF.
    batch_size:
        Mini-batch size.
    device:
        If provided, tensors are moved to this device (in-memory transfer).

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    dev = device or torch.device("cpu")

    def to_tensor(a: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(a, dtype=dtype).to(dev)

    def make_ds(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(to_tensor(X, torch.float32), to_tensor(y, torch.long))

    return (
        DataLoader(make_ds(X_train, y_train), batch_size, shuffle=True, drop_last=True),
        DataLoader(make_ds(X_val, y_val), batch_size, shuffle=False),
        DataLoader(make_ds(X_test, y_test), batch_size, shuffle=False),
    )


# =============================================================================
# Shared evaluation helper
# =============================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run inference on *loader* and return accuracy, predictions, and ground truth.

    Returns
    -------
    (accuracy, predictions, true_labels) — both arrays are int32.
    """
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        logits = model(xb)
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(yb.cpu().numpy())
    p, t = np.array(preds, dtype=np.int32), np.array(trues, dtype=np.int32)
    return float((p == t).mean()), p, t


# =============================================================================
# Branch (a) — Basic training
# =============================================================================

def train_basic(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str = "Model",
    epochs: int = 200,
    patience: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """
    Train the statistical MLP baseline (Branch a).

    Uses a fixed Adam learning rate and plain cross-entropy loss.  Early
    stopping is applied on validation accuracy.  No gradient clipping or
    learning-rate scheduling — intentionally simple for a fair comparison.

    Parameters
    ----------
    model:
        :class:`~src.model.MLPBaseline` or compatible ``nn.Module``.
    train_loader, val_loader:
        DataLoaders for training and validation sets.
    name:
        Display name logged during training.
    epochs, patience:
        Maximum training epochs and early-stopping patience.
    lr, weight_decay:
        Adam optimiser hyper-parameters.
    checkpoint_path:
        If given, the best model weights are saved to this path.

    Returns
    -------
    History dict with keys:
        ``tr_loss``, ``val_loss``, ``tr_acc``, ``val_acc``, ``best_val_acc``.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}
    best_acc, best_w, no_imp, t0 = 0.0, None, 0, time.time()

    logger.info("Training (basic): %s | %d epochs | LR=%.0e", name, epochs, lr)

    for ep in range(1, epochs + 1):
        model.train()
        tl = tc = tn = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tc += (model(xb).argmax(-1) == yb).sum().item()
            tn += len(yb)

        model.eval()
        vl = vc = vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                vl += F.cross_entropy(logits, yb).item() * len(yb)
                vc += (logits.argmax(-1) == yb).sum().item()
                vn += len(yb)

        tr_l, val_l = tl / tn, vl / vn
        tr_a, val_a = tc / tn, vc / vn
        history["tr_loss"].append(tr_l)
        history["val_loss"].append(val_l)
        history["tr_acc"].append(tr_a)
        history["val_acc"].append(val_a)

        if val_a > best_acc:
            best_acc, best_w, no_imp = val_a, deepcopy(model.state_dict()), 0
        else:
            no_imp += 1

        if ep % 20 == 0 or ep == 1:
            logger.info(
                "Ep %4d | TrLoss=%.4f TrAcc=%.4f | ValLoss=%.4f ValAcc=%.4f%s",
                ep, tr_l, tr_a, val_l, val_a, "  ★" if no_imp == 0 else "",
            )

        if no_imp >= patience:
            logger.info("Early stopping at epoch %d", ep)
            break

    logger.info(
        "Done in %.1fs | Best Val Acc=%.4f", time.time() - t0, best_acc
    )
    model.load_state_dict(best_w)
    history["best_val_acc"] = best_acc
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info("Checkpoint saved: %s", checkpoint_path)
    return history


# =============================================================================
# Branch (b) — Smooth training (PG-AMF)
# =============================================================================

def train_smooth(
    model: PGAMFClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str = "PG-AMF",
    epochs: int = 200,
    patience: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    grad_clip_norm: float = 2.0,
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """
    Train the PG-AMF classifier (Branch b) with the full smooth schedule.

    Training details:
        - **Optimiser**: AdamW
        - **Schedule**: CosineAnnealingLR (η_min = lr × 0.05)
        - **Loss**: Focal CE + α-diversity + cosine compactness
        - **Gradient clipping**: max_norm = 2.0
        - **Checkpoint**: best validation accuracy

    Parameters
    ----------
    model:
        :class:`~src.model.PGAMFClassifier` instance.
    train_loader, val_loader:
        DataLoaders for the raw segment arrays ``(B, 2, T)``.
    name:
        Display name used in log messages.
    epochs, patience:
        Max training epochs and early-stopping patience.
    lr, weight_decay:
        AdamW hyper-parameters.
    focal_gamma:
        Focusing parameter γ in the focal loss.
    label_smoothing:
        Cross-entropy label smoothing ε.
    grad_clip_norm:
        Maximum gradient norm for clipping.
    checkpoint_path:
        If given, the best model state dict is saved here.

    Returns
    -------
    History dict with keys:
        ``tr_loss``, ``val_loss``, ``tr_acc``, ``val_acc``, ``best_val_acc``.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05
    )
    history: Dict = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}
    best_acc, best_w, no_imp, t0 = 0.0, None, 0, time.time()

    logger.info(
        "Training (smooth): %s | %d epochs | LR=%.0e | γ=%.1f",
        name, epochs, lr, focal_gamma,
    )

    for ep in range(1, epochs + 1):
        model.train()
        tl = tc = tn = 0
        for xb, yb in train_loader:
            opt.zero_grad()

            # Forward with feature return for compactness loss
            out = model(xb, return_features=True)
            logits, z = out["logits"], out["z"]

            # Base focal + diversity loss
            loss = model.focal_loss(logits, yb, label_smoothing, focal_gamma)

            # Cosine compactness: pulls same-class embeddings closer
            z_norm = F.normalize(z, dim=-1)
            sim_mat = z_norm @ z_norm.T                              # (B, B)
            same_cl = (yb.unsqueeze(0) == yb.unsqueeze(1)).float()
            eye_mask = 1.0 - torch.eye(len(yb), device=yb.device)
            same_off = same_cl * eye_mask
            n_pairs = same_off.sum().clamp(min=1)
            L_compact = -(sim_mat * same_off).sum() / n_pairs
            loss = loss + model.lambda_compact * L_compact

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()

            tl += loss.item() * len(yb)
            tc += (logits.argmax(-1) == yb).sum().item()
            tn += len(yb)
        sched.step()

        model.eval()
        vl = vc = vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                vl += F.cross_entropy(logits, yb).item() * len(yb)
                vc += (logits.argmax(-1) == yb).sum().item()
                vn += len(yb)

        tr_l, val_l = tl / tn, vl / vn
        tr_a, val_a = tc / tn, vc / vn
        cur_lr = opt.param_groups[0]["lr"]
        history["tr_loss"].append(tr_l)
        history["val_loss"].append(val_l)
        history["tr_acc"].append(tr_a)
        history["val_acc"].append(val_a)

        if val_a > best_acc:
            best_acc, best_w, no_imp = val_a, deepcopy(model.state_dict()), 0
        else:
            no_imp += 1

        if ep % 20 == 0 or ep == 1:
            logger.info(
                "Ep %4d | TrLoss=%.4f TrAcc=%.4f | ValLoss=%.4f ValAcc=%.4f | LR=%.2e%s",
                ep, tr_l, tr_a, val_l, val_a, cur_lr, "  ★" if no_imp == 0 else "",
            )

        if no_imp >= patience:
            logger.info("Early stopping at epoch %d", ep)
            break

    logger.info(
        "Done in %.1fs | Best Val Acc=%.4f", time.time() - t0, best_acc
    )
    model.load_state_dict(best_w)
    history["best_val_acc"] = best_acc
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info("Checkpoint saved: %s", checkpoint_path)
    return history
