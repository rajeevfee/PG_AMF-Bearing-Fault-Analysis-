#!/usr/bin/env python3
"""
Train both branches of the PG-AMF pipeline.

Usage
-----
# Train with default config (uses synthetic data if real data is absent)
python scripts/train.py

# Point to your XJTU_Gearbox folder
python scripts/train.py --data_root /path/to/XJTU_Gearbox

# Override any config key via CLI
python scripts/train.py --epochs 100 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Allow running from repo root without editable install ────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dataset import BearingDataLoader
from src.evaluate import compute_metrics, multiclass_fdr, per_feature_fdr
from src.model import MLPBaseline, PGAMFClassifier
from src.preprocessing import (
    build_dataset,
    extract_stat_features,
    fit_scaler,
    split_dataset,
    STAT_FEAT_NAMES,
    STAT_FEAT_NAMES_1CH,
)
from src.trainer import evaluate_model, make_loaders, train_basic, train_smooth
from src.utils import compute_char_freqs, get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Statistical MLP (Branch a) and PG-AMF (Branch b)."
    )
    p.add_argument("--config", nargs="*",
                   default=["configs/default.yaml",
                             "configs/data.yaml",
                             "configs/model.yaml",
                             "configs/train.yaml"],
                   help="YAML config files (merged in order).")
    p.add_argument("--data_root",    type=str,  help="Override data root path.")
    p.add_argument("--epochs",       type=int,  help="Override training epochs.")
    p.add_argument("--batch_size",   type=int,  help="Override batch size.")
    p.add_argument("--seed",         type=int,  help="Override random seed.")
    p.add_argument("--device",       type=str,  help="Override device (cpu/cuda/auto).")
    p.add_argument("--no_pgamf",     action="store_true",
                   help="Skip PG-AMF branch (train baseline only).")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_logging()
    args = parse_args()

    # ── Load & patch config ───────────────────────────────────────────────────
    cfg = load_config(*args.config)
    if args.data_root:
        cfg["paths"]["data_root"] = args.data_root
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
        cfg["train"]["stat_mlp"]["epochs"] = args.epochs
        cfg["train"]["pgamf"]["epochs"]    = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.seed:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device

    seed       = cfg.get("seed", 42)
    device     = get_device(cfg.get("device", "auto"))
    epochs     = cfg["train"]["epochs"]
    patience   = cfg["train"]["patience"]
    batch_size = cfg["train"]["batch_size"]

    set_seed(seed)

    out_dir = Path(cfg["paths"]["outputs"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Bearing geometry ─────────────────────────────────────────────────────
    bear = cfg["data"]["bearing"]
    char_freqs = compute_char_freqs(
        n_balls=bear["n_balls"],
        ball_diameter=bear["ball_diameter"],
        pitch_diameter=bear["pitch_diameter"],
        shaft_freq=bear["shaft_freq"],
    )
    logger.info("Characteristic freqs: %s", char_freqs)

    # ── Load data ─────────────────────────────────────────────────────────────
    loader = BearingDataLoader(
        data_root=cfg["paths"]["data_root"],
        fault_types=cfg["fault_types"],
        fault_labels=cfg["fault_labels"],
        channel_files=cfg.get("channel_files", ["Data_Chan1.txt", "Data_Chan2.txt"]),
        skip_header=cfg.get("skip_header", 15),
        fs=cfg["data"]["signal"]["fs"],
        seed=seed,
    )
    data_dict = loader.load_all()
    df_raw    = loader.to_dataframe(data_dict)
    n_classes = len(cfg["fault_types"])

    # ── Segmentation & split ──────────────────────────────────────────────────
    sig = cfg["data"]["signal"]
    X_all, y_all = build_dataset(
        df_raw,
        seg_len=sig["seg_len"],
        hop=sig["hop"],
        min_segs=sig["min_segs"],
        max_segs=sig["max_segs"],
        do_mean_removal=sig["do_mean_removal"],
    )
    splits = cfg["data"]["splits"]
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_all, y_all,
        val_split=splits["val_split"],
        test_split=splits["test_split"],
        seed=seed,
    )

    # ── Statistical features (Branch a) ──────────────────────────────────────
    logger.info("Extracting statistical features …")
    F_stat_train = extract_stat_features(X_train)
    F_stat_val   = extract_stat_features(X_val)
    F_stat_test  = extract_stat_features(X_test)

    scaler = fit_scaler(F_stat_train)
    F_stat_train = scaler.transform(F_stat_train)
    F_stat_val   = scaler.transform(F_stat_val)
    F_stat_test  = scaler.transform(F_stat_test)
    n_stat_feat  = F_stat_train.shape[1]

    # ── DataLoaders ───────────────────────────────────────────────────────────
    dl_stat_tr, dl_stat_v, dl_stat_te = make_loaders(
        F_stat_train, y_train, F_stat_val, y_val, F_stat_test, y_test,
        batch_size=batch_size, device=device,
    )
    dl_pgamf_tr, dl_pgamf_v, dl_pgamf_te = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, device=device,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Branch (a): Statistical MLP — basic training
    # ─────────────────────────────────────────────────────────────────────────
    model_cfg  = cfg["model"]
    train_cfg  = cfg["train"]

    model_stat = MLPBaseline(n_stat_feat,
                             model_cfg["baseline_mlp"]["hidden_size"],
                             n_classes).to(device)
    logger.info("Statistical MLP params: %d", sum(p.numel() for p in model_stat.parameters()))

    hist_stat = train_basic(
        model_stat, dl_stat_tr, dl_stat_v,
        name="Branch (a) Statistical MLP",
        epochs=epochs,
        patience=patience,
        lr=train_cfg["stat_mlp"]["lr"],
        weight_decay=train_cfg["stat_mlp"]["weight_decay"],
        checkpoint_path=ckpt_dir / "stat_mlp_best.pt",
    )
    acc_stat, pred_stat, true_stat = evaluate_model(model_stat, dl_stat_te)
    metrics_stat = compute_metrics(true_stat, pred_stat, cfg["fault_types"])
    logger.info("[Branch a] Test Acc=%.4f  Macro-F1=%.4f",
                acc_stat, metrics_stat["macro_f1"])

    # ─────────────────────────────────────────────────────────────────────────
    # Branch (b): PG-AMF — smooth training
    # ─────────────────────────────────────────────────────────────────────────
    if not args.no_pgamf:
        pg_cfg = model_cfg["pgamf"]
        model_pgamf = PGAMFClassifier(
            n_classes=n_classes,
            F=pg_cfg["F"],
            n_channels=pg_cfg["n_channels"],
            hidden=pg_cfg["hidden_size"],
            lambda_div=pg_cfg["lambda_div"],
            lambda_compact=pg_cfg["lambda_compact"],
            alpha_min=pg_cfg["alpha_min"],
            alpha_max=pg_cfg["alpha_max"],
        ).to(device)
        logger.info("PG-AMF params: %d", sum(p.numel() for p in model_pgamf.parameters()))

        tr_pg = train_cfg["pgamf"]
        hist_pgamf = train_smooth(
            model_pgamf, dl_pgamf_tr, dl_pgamf_v,
            name="Branch (b) PG-AMF",
            epochs=epochs,
            patience=patience,
            lr=tr_pg["lr"],
            weight_decay=tr_pg["weight_decay"],
            focal_gamma=tr_pg["focal_gamma"],
            label_smoothing=tr_pg["label_smoothing"],
            grad_clip_norm=tr_pg["grad_clip_norm"],
            checkpoint_path=ckpt_dir / "pgamf_best.pt",
        )
        acc_pgamf, pred_pgamf, true_pgamf = evaluate_model(model_pgamf, dl_pgamf_te)
        metrics_pgamf = compute_metrics(true_pgamf, pred_pgamf, cfg["fault_types"])
        logger.info("[Branch b] Test Acc=%.4f  Macro-F1=%.4f",
                    acc_pgamf, metrics_pgamf["macro_f1"])
    else:
        logger.info("PG-AMF branch skipped (--no_pgamf).")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "stat":  {"accuracy": float(acc_stat),
                  "macro_f1": float(metrics_stat["macro_f1"]),
                  "report":   metrics_stat["report"]},
    }
    if not args.no_pgamf:
        results["pgamf"] = {"accuracy": float(acc_pgamf),
                             "macro_f1": float(metrics_pgamf["macro_f1"]),
                             "report":   metrics_pgamf["report"]}

    out_json = results_dir / "test_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved: %s", out_json)

    # ── Print summary table ───────────────────────────────────────────────────
    sep = "═" * 60
    print(f"\n{sep}")
    print("  TEST SET RESULTS")
    print(sep)
    print(f"  {'Method':<28} {'Accuracy':>10} {'Macro-F1':>10}")
    print("─" * 60)
    print(f"  {'Branch (a) Statistical MLP':<28} {acc_stat*100:>9.2f}%  {metrics_stat['macro_f1']:>9.4f}")
    if not args.no_pgamf:
        print(f"  {'Branch (b) PG-AMF':<28} {acc_pgamf*100:>9.2f}%  {metrics_pgamf['macro_f1']:>9.4f}")
    print(sep)


if __name__ == "__main__":
    main()
