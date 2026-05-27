"""
Evaluation utilities: classification metrics, Fisher Discriminant Ratio,
and 5-fold cross-validation reporting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.trainer import evaluate_model, make_loaders

logger = logging.getLogger(__name__)


# =============================================================================
# Classification metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute accuracy, macro-F1, per-class F1, and a full classification report.

    Parameters
    ----------
    y_true, y_pred:
        Integer label arrays of the same length.
    class_names:
        Optional list of human-readable class names for the report.

    Returns
    -------
    dict with keys:
        ``accuracy``, ``macro_f1``, ``f1_per_class``, ``report``,
        ``confusion_matrix``.
    """
    n_classes = max(y_true.max(), y_pred.max()) + 1
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": f1_score(y_true, y_pred, average=None, labels=labels,
                                  zero_division=0).tolist(),
        "report": classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, normalize="true"),
    }


# =============================================================================
# Fisher Discriminant Ratio
# =============================================================================

def multiclass_fdr(X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute multiclass Fisher Discriminant Ratio: tr(S_W⁻¹ S_B).

    Higher values indicate greater between-class separability relative
    to within-class scatter.

    Parameters
    ----------
    X:
        Feature matrix ``(N, D)``.
    y:
        Integer class labels ``(N,)``.

    Returns
    -------
    float — FDR score.
    """
    classes = np.unique(y)
    mu_all = X.mean(axis=0)
    S_B = np.zeros((X.shape[1],) * 2)
    S_W = np.zeros((X.shape[1],) * 2)
    for c in classes:
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        d = (mu_c - mu_all).reshape(-1, 1)
        S_B += len(Xc) * (d @ d.T)
        S_W += (Xc - mu_c).T @ (Xc - mu_c)
    S_W += np.eye(X.shape[1]) * 1e-6
    try:
        return float(np.trace(np.linalg.inv(S_W) @ S_B))
    except np.linalg.LinAlgError:
        return float(np.trace(np.linalg.pinv(S_W) @ S_B))


def per_feature_fdr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-feature univariate FDR: sb[j] / sw[j].

    Parameters
    ----------
    X:
        Feature matrix ``(N, D)``.
    y:
        Class labels ``(N,)``.

    Returns
    -------
    np.ndarray of shape ``(D,)`` with per-feature FDR values.
    """
    classes = np.unique(y)
    mu_all = X.mean(axis=0)
    sb = np.zeros(X.shape[1])
    sw = np.zeros(X.shape[1])
    for c in classes:
        Xc = X[y == c]
        mu_c = Xc.mean(axis=0)
        sb += len(Xc) * (mu_c - mu_all) ** 2
        sw += ((Xc - mu_c) ** 2).sum(axis=0)
    return sb / (sw + 1e-10)


# =============================================================================
# Feature extraction from a trained model
# =============================================================================

@torch.no_grad()
def extract_features(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract intermediate feature vectors from a trained PGAMFClassifier.

    Parameters
    ----------
    model:
        A :class:`~src.model.PGAMFClassifier` with a ``return_features``
        forward path.
    X:
        Raw segment array ``(N, 2, T)``.
    device:
        Computation device.
    batch_size:
        Inference batch size.

    Returns
    -------
    np.ndarray of shape ``(N, feat_dim)`` — float32.
    """
    model.eval()
    feats: List[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
        out = model(xb, return_features=True)
        feats.append(out["z"].cpu().numpy())
    return np.concatenate(feats, axis=0)


# =============================================================================
# 5-Fold Cross-Validation
# =============================================================================

def cross_validate(
    build_stat_model_fn,
    build_pgamf_model_fn,
    train_stat_fn,
    train_pgamf_fn,
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    F_stat_cv: np.ndarray,
    n_classes: int,
    n_folds: int = 5,
    seed: int = 42,
    device: Optional[torch.device] = None,
    batch_size: int = 64,
    epochs: int = 120,
    patience: int = 20,
) -> Dict:
    """
    Run stratified K-fold cross-validation for both branches.

    Parameters
    ----------
    build_stat_model_fn:
        Callable that returns a fresh :class:`~src.model.MLPBaseline`.
    build_pgamf_model_fn:
        Callable that returns a fresh :class:`~src.model.PGAMFClassifier`.
    train_stat_fn, train_pgamf_fn:
        Training functions — called as ``train_fn(model, tr_loader, val_loader, ...)``.
    X_cv, y_cv:
        Full (train + val) raw segment array and labels for the PG-AMF branch.
    F_stat_cv:
        Full (train + val) statistical feature array for the baseline branch.
    n_classes:
        Number of fault classes.
    n_folds:
        Number of cross-validation folds.
    seed, device, batch_size, epochs, patience:
        Standard hyper-parameters.

    Returns
    -------
    dict with keys ``stat`` and ``pgamf``, each containing:
        ``acc`` (list of fold accuracies),
        ``f1`` (list of fold macro-F1 scores),
        ``f1_per`` (list of per-class F1 arrays).
    """
    dev = device or torch.device("cpu")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results: Dict = {
        "stat":  {"acc": [], "f1": [], "f1_per": []},
        "pgamf": {"acc": [], "f1": [], "f1_per": []},
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), 1):
        logger.info("─── Fold %d / %d ───", fold, n_folds)

        yf_tr, yf_v = y_cv[tr_idx], y_cv[val_idx]

        # ── Branch (a): Statistical ──────────────────────────────────────────
        Fs_tr, Fs_v = F_stat_cv[tr_idx], F_stat_cv[val_idx]
        dl_t, dl_v, _ = make_loaders(Fs_tr, yf_tr, Fs_v, yf_v, Fs_v, yf_v,
                                      batch_size=batch_size, device=dev)
        ms = build_stat_model_fn().to(dev)
        train_stat_fn(ms, dl_t, dl_v,
                      name=f"Stat Fold {fold}",
                      epochs=epochs, patience=patience)
        acc_s, preds_s, true_s = evaluate_model(ms, dl_v)
        f1_s = f1_score(true_s, preds_s, average="macro", zero_division=0)
        f1_per_s = f1_score(true_s, preds_s, average=None,
                             labels=list(range(n_classes)), zero_division=0)
        results["stat"]["acc"].append(acc_s)
        results["stat"]["f1"].append(f1_s)
        results["stat"]["f1_per"].append(f1_per_s)
        logger.info("  Stat : Acc=%.4f  F1=%.4f", acc_s, f1_s)

        # ── Branch (b): PG-AMF ───────────────────────────────────────────────
        Xp_tr, Xp_v = X_cv[tr_idx], X_cv[val_idx]
        dl_t2, dl_v2, _ = make_loaders(Xp_tr, yf_tr, Xp_v, yf_v, Xp_v, yf_v,
                                        batch_size=batch_size, device=dev)
        mp = build_pgamf_model_fn().to(dev)
        train_pgamf_fn(mp, dl_t2, dl_v2,
                       name=f"PG-AMF Fold {fold}",
                       epochs=epochs, patience=patience)
        acc_p, preds_p, true_p = evaluate_model(mp, dl_v2)
        f1_p = f1_score(true_p, preds_p, average="macro", zero_division=0)
        f1_per_p = f1_score(true_p, preds_p, average=None,
                             labels=list(range(n_classes)), zero_division=0)
        results["pgamf"]["acc"].append(acc_p)
        results["pgamf"]["f1"].append(f1_p)
        results["pgamf"]["f1_per"].append(f1_per_p)
        logger.info("  PG-AMF: Acc=%.4f  F1=%.4f", acc_p, f1_p)

    return results


def summarise_cv(results: Dict) -> Dict:
    """
    Compute mean ± std for accuracy, macro-F1, and per-class F1 from CV results.

    Parameters
    ----------
    results:
        Output of :func:`cross_validate`.

    Returns
    -------
    dict with keys ``stat`` and ``pgamf``, each containing:
        ``acc_mean``, ``acc_std``, ``f1_mean``, ``f1_std``,
        ``f1_per_mean``, ``f1_per_std``.
    """
    summary: Dict = {}
    for branch in ("stat", "pgamf"):
        acc = np.array(results[branch]["acc"])
        f1  = np.array(results[branch]["f1"])
        f1p = np.array(results[branch]["f1_per"])
        summary[branch] = {
            "acc_mean": acc.mean(), "acc_std": acc.std(),
            "f1_mean":  f1.mean(),  "f1_std":  f1.std(),
            "f1_per_mean": f1p.mean(axis=0),
            "f1_per_std":  f1p.std(axis=0),
        }
    return summary
