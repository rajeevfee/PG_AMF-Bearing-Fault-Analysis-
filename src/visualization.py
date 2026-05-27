"""
IEEE publication-quality figure generation for the PG-AMF paper pipeline.

All functions accept explicit data arguments and an output directory, so
they can be called independently or from the generate_figures script.

Figures produced (matching the paper):
  Fig 1  — Time-domain waveforms (both channels)
  Fig 2  — Power spectral density (Welch, both channels)
  Fig 3  — Statistical feature distribution boxplots
  Fig 4  — Per-feature Fisher Discriminant Ratio bar chart
  Fig 5  — t-SNE projections pre-training
  Fig 6  — Learning curves (both branches)
  Fig 7  — Confusion matrices
  Fig 8  — Learned α-exponent distribution
  Fig 9  — t-SNE projections post-training
  Fig 10 — Performance bar chart with error bars (CV mean ± std)
  Fig 11 — Per-class F1 grouped bar chart (CV mean ± std)
  Fig 12 — Radar chart (method comparison)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import welch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")  # non-interactive backend — safe for scripts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IEEE style constants
# ---------------------------------------------------------------------------

IEEE_W_SINGLE = 3.5
IEEE_W_DOUBLE = 7.16
IEEE_FONT = 8

IEEE_RC: Dict = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": IEEE_FONT,
    "axes.titlesize": IEEE_FONT,
    "axes.labelsize": IEEE_FONT,
    "xtick.labelsize": IEEE_FONT - 1,
    "ytick.labelsize": IEEE_FONT - 1,
    "legend.fontsize": IEEE_FONT - 1,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
}


def _apply_ieee() -> None:
    matplotlib.rcParams.update(IEEE_RC)


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    logger.info("Saved: %s", path)


# =============================================================================
# Fig 1 — Time-domain waveforms
# =============================================================================

def plot_time_domain(
    vis_segs: List[Dict],
    t_axis: np.ndarray,
    fault_colors: List[str],
    out_dir: Path,
) -> None:
    """
    Plot preprocessed vibration waveforms for both channels, all fault classes.

    Parameters
    ----------
    vis_segs:
        List of dicts with keys ``ch1``, ``ch2``, ``name``.
    t_axis:
        Time axis in ms.
    fault_colors:
        One colour per fault class.
    out_dir:
        Directory where the PNG is saved.
    """
    _apply_ieee()
    n_classes = len(vis_segs)
    ch_colors = ["#1565C0", "#B71C1C"]
    ch_labels = ["Ch1 (Horiz.)", "Ch2 (Vert.)"]

    fig, axes = plt.subplots(n_classes, 2, figsize=(IEEE_W_DOUBLE, 6.5), sharex=True)
    fig.suptitle(
        "Fig. 1. Preprocessed Vibration Signals — Both Channels, All Fault Classes",
        fontsize=IEEE_FONT, fontweight="bold",
    )
    for i, smp in enumerate(vis_segs):
        for ch in range(2):
            ax = axes[i, ch]
            sig = np.asarray(smp[f"ch{ch+1}"], dtype=np.float32)
            ax.plot(t_axis, sig, color=ch_colors[ch], lw=0.5)
            rms = np.sqrt(np.mean(sig**2))
            ax.text(0.97, 0.9, f"RMS={rms:.3f}", transform=ax.transAxes,
                    ha="right", fontsize=6.5)
            if ch == 0:
                ax.set_ylabel(smp["name"], fontsize=7.5,
                              color=fault_colors[i], fontweight="bold")
            if i == 0:
                ax.set_title(ch_labels[ch])
            if i == n_classes - 1:
                ax.set_xlabel("Time [ms]")
            else:
                ax.tick_params(labelbottom=False)
    plt.tight_layout()
    _save(fig, "Fig1_TimeDomain.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 2 — Power Spectral Density
# =============================================================================

def plot_psd(
    vis_segs: List[Dict],
    fs: int,
    char_freqs: Dict[str, float],
    fault_colors: List[str],
    out_dir: Path,
) -> None:
    """PSD (Welch method) for all fault classes and both channels."""
    _apply_ieee()
    n_classes = len(vis_segs)
    ch_colors = ["#1565C0", "#B71C1C"]
    ch_labels = ["Ch1 (Horiz.)", "Ch2 (Vert.)"]
    freq_styles = {
        "fr":   ("k",         "-",  r"$f_r$"),
        "BPFO": ("#F44336",   "--", "BPFO"),
        "BPFI": ("#2196F3",   "-.", "BPFI"),
        "BSF":  ("#4CAF50",   ":",  "BSF"),
        "FTF":  ("#FF9800",   "--", "FTF"),
    }

    fig, axes = plt.subplots(n_classes, 2, figsize=(IEEE_W_DOUBLE, 6.5), sharex=True)
    fig.suptitle(
        "Fig. 2. Power Spectral Density — Welch Method (Both Channels)",
        fontsize=IEEE_FONT, fontweight="bold",
    )
    for i, smp in enumerate(vis_segs):
        for ch in range(2):
            ax = axes[i, ch]
            sig = np.asarray(smp[f"ch{ch+1}"], dtype=np.float64)
            ff, Pxx = welch(sig, fs=fs, nperseg=min(512, len(sig)), window="hann")
            ax.plot(ff / 1e3, 10 * np.log10(Pxx + 1e-20), color=ch_colors[ch], lw=0.7)
            for fname, (fc, ls, _) in freq_styles.items():
                if fname in char_freqs:
                    ax.axvline(char_freqs[fname] / 1e3, color=fc, ls=ls, lw=0.7, alpha=0.6)
            if ch == 0:
                ax.set_ylabel(smp["name"], fontsize=7.5,
                              color=fault_colors[i], fontweight="bold")
            if i == 0:
                ax.set_title(ch_labels[ch])
            if i == n_classes - 1:
                ax.set_xlabel("Frequency [kHz]")
            ax.set_xlim([0, fs / 2 / 1e3])

    legend_handles = [
        Line2D([0], [0], color=fc, ls=ls, lw=1.0, label=lbl)
        for _, (fc, ls, lbl) in freq_styles.items()
    ]
    axes[0, 1].legend(handles=legend_handles, loc="upper right", fontsize=6, ncol=5)
    plt.tight_layout()
    _save(fig, "Fig2_PSD.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 3 — Statistical feature distributions
# =============================================================================

def plot_stat_distributions(
    F_stat_all: np.ndarray,
    y_all: np.ndarray,
    feat_names_1ch: List[str],
    fault_names: List[str],
    fault_colors: List[str],
    n_classes: int,
    out_dir: Path,
) -> None:
    """Boxplot of statistical features per fault class (Ch1 & Ch2)."""
    _apply_ieee()
    fig, axes = plt.subplots(2, 6, figsize=(IEEE_W_DOUBLE, 3.8))
    fig.suptitle(
        "Fig. 3. Statistical Feature Distributions per Fault Class\n"
        "(Top: Ch1 | Bottom: Ch2)",
        fontsize=IEEE_FONT, fontweight="bold",
    )
    for ch_idx in range(2):
        for fi, fname in enumerate(feat_names_1ch):
            ax = axes[ch_idx, fi]
            col = ch_idx * 6 + fi
            data = [F_stat_all[y_all == ci, col] for ci in range(n_classes)]
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={"color": "black", "lw": 1.0},
                            whiskerprops={"lw": 0.6}, capprops={"lw": 0.6},
                            flierprops={"marker": "o", "markersize": 1.5, "alpha": 0.4})
            for patch, clr in zip(bp["boxes"], fault_colors):
                patch.set_facecolor(clr)
                patch.set_alpha(0.65)
            ch_lbl = f"Ch{ch_idx+1}"
            ax.set_title(f"{fname}\n({ch_lbl})", fontsize=6.5, fontweight="bold")
            ax.set_xticklabels([n[:3] for n in fault_names], fontsize=6, rotation=30)
    plt.tight_layout()
    _save(fig, "Fig3_StatDistributions.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 4 — Per-feature FDR
# =============================================================================

def plot_fdr_bar(
    fdr_per: np.ndarray,
    stat_feat_names: List[str],
    fdr_overall: float,
    out_dir: Path,
) -> None:
    """Bar chart of per-feature Fisher Discriminant Ratio values."""
    _apply_ieee()
    fig, ax = plt.subplots(figsize=(IEEE_W_DOUBLE, 2.5))
    n_feat = len(fdr_per)
    colors = ["#2196F3"] * (n_feat // 2) + ["#F44336"] * (n_feat // 2)
    bars = ax.bar(range(n_feat), fdr_per, color=colors, edgecolor="black", lw=0.4, alpha=0.85)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(stat_feat_names, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("FDR")
    ax.set_title(
        f"Fig. 4. Per-Feature FDR — Statistical Baseline (Blue=Ch1, Red=Ch2)  "
        f"[Overall FDR={fdr_overall:.2f}]",
        fontsize=IEEE_FONT,
    )
    for bar, val in zip(bars, fdr_per):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)
    plt.tight_layout()
    _save(fig, "Fig4_FDR.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 5 — Pre-training t-SNE
# =============================================================================

def plot_tsne_pre(
    F_stat_all: np.ndarray,
    F_pgamf_raw: np.ndarray,
    y_all: np.ndarray,
    fault_names: List[str],
    fault_colors: List[str],
    seed: int,
    out_dir: Path,
    n_samples: int = 500,
) -> None:
    """t-SNE projections before training: stat features vs untrained PG-AMF."""
    _apply_ieee()
    idx = np.random.default_rng(seed).choice(len(y_all), min(n_samples, len(y_all)), replace=False)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=seed, n_jobs=-1)
    emb_stat = tsne.fit_transform(F_stat_all[idx])
    emb_pgamf = tsne.fit_transform(F_pgamf_raw[idx])
    y_sub = y_all[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_W_DOUBLE, 3.5))
    fig.suptitle(
        "Fig. 5. t-SNE Projections — Pre-Training\n"
        "(Left: Statistical Baseline; Right: PG-AMF untrained α)",
        fontsize=IEEE_FONT, fontweight="bold",
    )
    for ax, emb, title in [(ax1, emb_stat, "Statistical"), (ax2, emb_pgamf, "PG-AMF (Untrained)")]:
        for ci, (name, clr) in enumerate(zip(fault_names, fault_colors)):
            mask = y_sub == ci
            ax.scatter(emb[mask, 0], emb[mask, 1], c=clr, label=name,
                       s=12, alpha=0.75, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    ax1.legend(loc="upper right", fontsize=6, markerscale=1.5)
    plt.tight_layout()
    _save(fig, "Fig5_tSNE_PreTraining.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 6 — Learning curves
# =============================================================================

def plot_learning_curves(
    hist_stat: Dict,
    hist_pgamf: Dict,
    out_dir: Path,
) -> None:
    """Learning curves (loss & accuracy) for both branches."""
    _apply_ieee()
    fig, axes = plt.subplots(2, 2, figsize=(IEEE_W_DOUBLE, 4.5))
    fig.suptitle(
        "Fig. 6. Learning Curves — Branch (a) Statistical vs Branch (b) PG-AMF",
        fontsize=IEEE_FONT, fontweight="bold",
    )
    pairs = [
        (axes[0, 0], axes[0, 1], hist_stat,  "Branch (a) Statistical MLP", "#2196F3"),
        (axes[1, 0], axes[1, 1], hist_pgamf, "Branch (b) PG-AMF",          "#F44336"),
    ]
    win = 5
    for ax_l, ax_a, hist, lbl, clr in pairs:
        ep = range(1, len(hist["tr_loss"]) + 1)
        ax_l.plot(ep, hist["tr_loss"],  color=clr, lw=1.2, label="Train")
        ax_l.plot(ep, hist["val_loss"], color=clr, lw=1.2, ls="--", label="Val", alpha=0.7)
        if len(hist["val_loss"]) > win:
            ma = np.convolve(hist["val_loss"], np.ones(win) / win, "valid")
            ax_l.plot(range(win, len(hist["val_loss"]) + 1), ma, "k-", lw=0.8, alpha=0.5)
        ax_l.set_title(f"{lbl} — Loss")
        ax_l.set_xlabel("Epoch")
        ax_l.set_ylabel("CE Loss")
        ax_l.legend(fontsize=6)

        tr_pct  = [a * 100 for a in hist["tr_acc"]]
        val_pct = [a * 100 for a in hist["val_acc"]]
        ax_a.plot(ep, tr_pct,  color=clr, lw=1.2, label="Train")
        ax_a.plot(ep, val_pct, color=clr, lw=1.2, ls="--", label="Val", alpha=0.7)
        if len(val_pct) > win:
            ma_a = np.convolve(val_pct, np.ones(win) / win, "valid")
            ax_a.plot(range(win, len(val_pct) + 1), ma_a, "k-", lw=0.8, alpha=0.5)
        best_e = int(np.argmax(hist["val_acc"])) + 1
        ax_a.axvline(best_e, color=clr, ls=":", lw=0.8, alpha=0.5)
        ax_a.set_title(f"{lbl} — Accuracy")
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Accuracy [%]")
        ax_a.set_ylim([0, 105])
        ax_a.legend(fontsize=6)
    plt.tight_layout()
    _save(fig, "Fig6_LearningCurves.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 7 — Confusion matrices
# =============================================================================

def plot_confusion_matrices(
    pred_stat: np.ndarray,
    true_stat: np.ndarray,
    acc_stat: float,
    pred_pgamf: np.ndarray,
    true_pgamf: np.ndarray,
    acc_pgamf: float,
    fault_names: List[str],
    out_dir: Path,
) -> None:
    """Normalised confusion matrices for both branches (test set)."""
    _apply_ieee()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_W_DOUBLE, 3.8))
    fig.suptitle(
        "Fig. 7. Normalised Confusion Matrices — Test Set",
        fontsize=IEEE_FONT, fontweight="bold",
    )
    n = len(fault_names)

    def _draw_cm(ax: plt.Axes, preds, trues, title, cmap="Blues"):
        cm = np.nan_to_num(confusion_matrix(trues, preds, normalize="true"))
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(fault_names, rotation=30, ha="right", fontsize=6)
        ax.set_yticklabels(fault_names, fontsize=6)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("True", fontsize=7)
        ax.set_title(title, fontsize=7, fontweight="bold")
        for i in range(n):
            for j in range(n):
                c = "white" if cm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                        fontsize=5.5, color=c, fontweight="bold")
        return im

    im1 = _draw_cm(ax1, pred_stat, true_stat,
                   f"Branch (a) Statistical MLP\nAcc={acc_stat:.3f}")
    im2 = _draw_cm(ax2, pred_pgamf, true_pgamf,
                   f"Branch (b) PG-AMF\nAcc={acc_pgamf:.3f}", "Reds")
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Recall")
    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Recall")
    plt.tight_layout()
    _save(fig, "Fig7_ConfusionMatrices.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 8 — Learned α-exponent distribution
# =============================================================================

def plot_alpha_distribution(
    alphas_trained: np.ndarray,
    alpha_min: float,
    alpha_max: float,
    out_dir: Path,
) -> None:
    """Scatter plot of learned PG-AMF exponents for both channels."""
    _apply_ieee()
    F = alphas_trained.shape[1]
    fig, ax = plt.subplots(figsize=(IEEE_W_SINGLE * 1.6, 2.2))
    for ch_i, (mk, clr) in enumerate(zip(["o", "s"], ["#2196F3", "#F44336"])):
        a_s = np.sort(alphas_trained[ch_i])
        ax.scatter(range(F), a_s, marker=mk, s=50, color=clr, zorder=3,
                   label=f"Channel {ch_i+1}", edgecolors="black", lw=0.5)
        ax.plot(range(F), a_s, color=clr, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(2.0, color="g", ls=":", lw=1.0, alpha=0.7, label="RMS (α=2)")
    ax.axhline(4.0, color="m", ls=":", lw=1.0, alpha=0.7, label="Kurtosis~(α=4)")
    ax.set_xlabel("Moment Index $i$")
    ax.set_ylabel(r"Learned $\alpha_i$")
    ax.set_title("Fig. 8. Learned PG-AMF Exponents After Training", fontsize=IEEE_FONT)
    ax.set_ylim([alpha_min - 0.2, alpha_max + 0.2])
    ax.set_xticks(range(F))
    ax.legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    _save(fig, "Fig8_AlphaDistribution.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 9 — Post-training t-SNE
# =============================================================================

def plot_tsne_post(
    emb_stat: np.ndarray,
    y_stat: np.ndarray,
    Z_trained: np.ndarray,
    y_all: np.ndarray,
    fault_names: List[str],
    fault_colors: List[str],
    seed: int,
    out_dir: Path,
    n_samples: int = 500,
) -> None:
    """t-SNE projections after training: stat baseline vs trained PG-AMF."""
    _apply_ieee()
    idx = np.random.default_rng(seed).choice(len(y_all), min(n_samples, len(y_all)), replace=False)
    emb_pgamf = TSNE(n_components=2, perplexity=30, random_state=seed, n_jobs=-1).fit_transform(
        Z_trained[idx]
    )
    y_sub = y_all[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_W_DOUBLE, 3.2))
    fig.suptitle(
        "Fig. 9. t-SNE Projections — After Training\n"
        "(Left: Statistical Baseline; Right: PG-AMF trained α)",
        fontsize=IEEE_FONT, fontweight="bold",
    )
    for ax, emb, ys, title in [
        (ax1, emb_stat, y_stat, "Statistical Baseline"),
        (ax2, emb_pgamf, y_sub, "PG-AMF (Trained α)"),
    ]:
        for ci, (name, clr) in enumerate(zip(fault_names, fault_colors)):
            mask = ys == ci
            ax.scatter(emb[mask, 0], emb[mask, 1], c=clr, label=name,
                       s=12, alpha=0.75, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    ax1.legend(loc="upper right", fontsize=6, markerscale=1.5)
    plt.tight_layout()
    _save(fig, "Fig9_tSNE_PostTraining.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 10 — Performance bar chart (CV mean ± std)
# =============================================================================

def plot_performance_bars(
    acc_s_m: float, acc_s_std: float,
    acc_p_m: float, acc_p_std: float,
    f1_s_m: float,  f1_s_std: float,
    f1_p_m: float,  f1_p_std: float,
    out_dir: Path,
) -> None:
    """Bar chart comparing CV accuracy and macro-F1 for both branches."""
    _apply_ieee()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_W_DOUBLE, 3.0))
    fig.suptitle("Fig. 10. Model Performance — Mean ± Std (5-Fold CV)",
                 fontsize=IEEE_FONT, fontweight="bold")
    for ax, (m_s, s_s, m_p, s_p), ylabel in [
        (axes[0], (acc_s_m * 100, acc_s_std * 100, acc_p_m * 100, acc_p_std * 100), "Accuracy [%]"),
        (axes[1], (f1_s_m * 100,  f1_s_std * 100,  f1_p_m * 100,  f1_p_std * 100),  "Macro-F1 [%]"),
    ]:
        bars = ax.bar(
            [0, 1], [m_s, m_p], yerr=[s_s, s_p],
            color=["#2196F3", "#F44336"], edgecolor="black", lw=0.5,
            capsize=5, error_kw={"lw": 1.2, "capthick": 1.2},
            alpha=0.85, width=0.5,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["(a) Statistical\nMLP", "(b) PG-AMF"], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_ylim([max(0, min(m_s, m_p) - 15), 105])
        for bar, m, s in zip(bars, [m_s, m_p], [s_s, s_p]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.5,
                    f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    plt.tight_layout()
    _save(fig, "Fig10_PerformanceMeanStd.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 11 — Per-class F1 grouped bars
# =============================================================================

def plot_per_class_f1(
    f1_per_stat_mean: np.ndarray,
    f1_per_stat_std: np.ndarray,
    f1_per_pgamf_mean: np.ndarray,
    f1_per_pgamf_std: np.ndarray,
    fault_names: List[str],
    out_dir: Path,
) -> None:
    """Grouped bar chart of per-class F1 mean ± std for both branches."""
    _apply_ieee()
    fig, ax = plt.subplots(figsize=(IEEE_W_DOUBLE, 3.2))
    x = np.arange(len(fault_names))
    w = 0.35
    ax.bar(x - w / 2, f1_per_stat_mean * 100, w, yerr=f1_per_stat_std * 100,
           color="#2196F3", alpha=0.85, edgecolor="black", lw=0.5, capsize=3,
           label="(a) Statistical MLP")
    ax.bar(x + w / 2, f1_per_pgamf_mean * 100, w, yerr=f1_per_pgamf_std * 100,
           color="#F44336", alpha=0.85, edgecolor="black", lw=0.5, capsize=3,
           label="(b) PG-AMF")
    ax.set_xticks(x)
    ax.set_xticklabels(fault_names, fontsize=7)
    ax.set_ylabel("F1 Score [%]")
    ax.set_title("Fig. 11. Per-Class F1 — Mean ± Std (5-Fold CV)", fontsize=IEEE_FONT)
    ax.set_ylim([0, 115])
    ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, "Fig11_PerClassF1.png", out_dir)
    plt.close(fig)


# =============================================================================
# Fig 12 — Radar chart
# =============================================================================

def plot_radar_chart(
    acc_s_m: float, f1_s_m: float, fdr_stat: float,
    acc_p_m: float, f1_p_m: float, fdr_pgamf: float,
    n_params_stat: int, n_params_pgamf: int,
    out_dir: Path,
) -> None:
    """Radar chart comparing the two branches on four normalised metrics."""
    _apply_ieee()
    cats = ["Accuracy", "Macro-F1", "FDR\n(norm.)", "Param\nEfficiency"]
    n_cat = len(cats)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    max_fdr = max(fdr_stat, fdr_pgamf) + 1e-10
    max_p = max(n_params_stat, n_params_pgamf) + 1

    sv = [acc_s_m, f1_s_m, fdr_stat / max_fdr, 1 - n_params_stat / max_p] + [acc_s_m]
    pv = [acc_p_m, f1_p_m, fdr_pgamf / max_fdr, 1 - n_params_pgamf / max_p] + [acc_p_m]

    fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.plot(angles, sv, "o-", color="#2196F3", lw=1.5, label="(a) Statistical MLP")
    ax.fill(angles, sv, alpha=0.12, color="#2196F3")
    ax.plot(angles, pv, "s-", color="#F44336", lw=1.5, label="(b) PG-AMF")
    ax.fill(angles, pv, alpha=0.12, color="#F44336")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6.5)
    ax.set_title("Fig. 12. Method Comparison Radar\n(Metrics normalised)",
                 fontsize=IEEE_FONT, fontweight="bold", y=1.10)
    plt.tight_layout()
    _save(fig, "Fig12_RadarChart.png", out_dir)
    plt.close(fig)
