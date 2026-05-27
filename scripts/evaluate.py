#!/usr/bin/env python3
"""
Evaluate trained models and run 5-fold cross-validation.

Usage
-----
python scripts/evaluate.py                          # full CV + test eval
python scripts/evaluate.py --no_cv                  # test-set only (fast)
python scripts/evaluate.py --data_root /path/to/...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dataset import BearingDataLoader
from src.evaluate import (
    compute_metrics,
    cross_validate,
    multiclass_fdr,
    per_feature_fdr,
    summarise_cv,
)
from src.model import MLPBaseline, PGAMFClassifier
from src.preprocessing import (
    build_dataset,
    extract_stat_features,
    fit_scaler,
    split_dataset,
    STAT_FEAT_NAMES,
)
from src.trainer import evaluate_model, make_loaders, train_basic, train_smooth
from src.utils import compute_char_freqs, get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate models and run cross-validation.")
    p.add_argument("--config", nargs="*",
                   default=["configs/default.yaml",
                             "configs/data.yaml",
                             "configs/model.yaml",
                             "configs/train.yaml"])
    p.add_argument("--data_root", type=str)
    p.add_argument("--ckpt_stat",  type=str, default="outputs/checkpoints/stat_mlp_best.pt")
    p.add_argument("--ckpt_pgamf", type=str, default="outputs/checkpoints/pgamf_best.pt")
    p.add_argument("--no_cv", action="store_true", help="Skip cross-validation.")
    p.add_argument("--device", type=str)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    cfg = load_config(*args.config)
    if args.data_root:
        cfg["paths"]["data_root"] = args.data_root
    if args.device:
        cfg["device"] = args.device

    seed    = cfg.get("seed", 42)
    device  = get_device(cfg.get("device", "auto"))
    n_classes = len(cfg["fault_types"])
    fault_names = list(cfg["fault_labels"].values())

    set_seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    loader = BearingDataLoader(
        data_root=cfg["paths"]["data_root"],
        fault_types=cfg["fault_types"],
        fault_labels=cfg["fault_labels"],
        fs=cfg["data"]["signal"]["fs"],
        seed=seed,
    )
    data_dict = loader.load_all()
    df_raw    = loader.to_dataframe(data_dict)

    sig = cfg["data"]["signal"]
    X_all, y_all = build_dataset(df_raw, sig["seg_len"], sig["hop"],
                                  sig["min_segs"], sig["max_segs"],
                                  sig["do_mean_removal"])
    splits = cfg["data"]["splits"]
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_all, y_all, splits["val_split"], splits["test_split"], seed
    )

    F_stat_train = extract_stat_features(X_train)
    F_stat_val   = extract_stat_features(X_val)
    F_stat_test  = extract_stat_features(X_test)
    scaler = fit_scaler(F_stat_train)
    F_stat_train = scaler.transform(F_stat_train)
    F_stat_val   = scaler.transform(F_stat_val)
    F_stat_test  = scaler.transform(F_stat_test)
    n_stat_feat  = F_stat_train.shape[1]

    batch_size = cfg["train"]["batch_size"]
    dl_stat_tr, dl_stat_v, dl_stat_te = make_loaders(
        F_stat_train, y_train, F_stat_val, y_val, F_stat_test, y_test,
        batch_size=batch_size, device=device,
    )
    dl_pgamf_tr, dl_pgamf_v, dl_pgamf_te = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, device=device,
    )

    mc = cfg["model"]
    pg = mc["pgamf"]

    # ── Load checkpoints ──────────────────────────────────────────────────────
    model_stat = MLPBaseline(n_stat_feat, mc["baseline_mlp"]["hidden_size"], n_classes).to(device)
    model_pgamf = PGAMFClassifier(
        n_classes=n_classes, F=pg["F"], n_channels=pg["n_channels"],
        hidden=pg["hidden_size"], lambda_div=pg["lambda_div"],
        lambda_compact=pg["lambda_compact"],
        alpha_min=pg["alpha_min"], alpha_max=pg["alpha_max"],
    ).to(device)

    import torch
    for model, ckpt, name in [
        (model_stat,  args.ckpt_stat,  "Statistical MLP"),
        (model_pgamf, args.ckpt_pgamf, "PG-AMF"),
    ]:
        ckpt_path = Path(ckpt)
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.info("Loaded checkpoint: %s", ckpt_path)
        else:
            logger.warning("Checkpoint not found (%s) — using randomly initialised weights.", ckpt_path)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    acc_stat, pred_stat, true_stat = evaluate_model(model_stat, dl_stat_te)
    acc_pgamf, pred_pgamf, true_pgamf = evaluate_model(model_pgamf, dl_pgamf_te)

    metrics_stat  = compute_metrics(true_stat, pred_stat, fault_names)
    metrics_pgamf = compute_metrics(true_pgamf, pred_pgamf, fault_names)

    sep = "═" * 65
    print(f"\n{sep}")
    print("  TABLE III. HOLD-OUT TEST SET RESULTS")
    print(sep)
    print(f"  {'Method':<30} {'Accuracy':>10} {'Macro-F1':>10}")
    print("─" * 65)
    print(f"  {'Branch (a) Statistical MLP':<30} {acc_stat*100:>9.2f}% {metrics_stat['macro_f1']:>10.4f}")
    print(f"  {'Branch (b) PG-AMF':<30} {acc_pgamf*100:>9.2f}% {metrics_pgamf['macro_f1']:>10.4f}")
    print(sep)
    print("\nPG-AMF Classification Report:\n")
    print(metrics_pgamf["report"])

    # ── FDR ──────────────────────────────────────────────────────────────────
    F_stat_all = np.concatenate([F_stat_train, F_stat_val, F_stat_test], axis=0)
    fdr_stat  = multiclass_fdr(F_stat_all, y_all)
    fdr_per   = per_feature_fdr(F_stat_all, y_all)
    print(f"Multiclass FDR — Statistical: {fdr_stat:.4f}")
    for name, val in zip(STAT_FEAT_NAMES, fdr_per):
        print(f"  {name:<22} FDR = {val:.4f}")

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    if not args.no_cv:
        logger.info("Running %d-fold cross-validation …", cfg["train"]["n_folds"])
        X_cv  = np.concatenate([X_train, X_val])
        y_cv  = np.concatenate([y_train, y_val])
        Fs_cv = np.concatenate([F_stat_train, F_stat_val])

        tc = cfg["train"]
        pg_t = tc["pgamf"]

        cv_results = cross_validate(
            build_stat_model_fn=lambda: MLPBaseline(n_stat_feat, mc["baseline_mlp"]["hidden_size"], n_classes),
            build_pgamf_model_fn=lambda: PGAMFClassifier(
                n_classes=n_classes, F=pg["F"], n_channels=pg["n_channels"],
                hidden=pg["hidden_size"], lambda_div=pg["lambda_div"],
                lambda_compact=pg["lambda_compact"],
                alpha_min=pg["alpha_min"], alpha_max=pg["alpha_max"],
            ),
            train_stat_fn=partial(train_basic, lr=tc["stat_mlp"]["lr"],
                                   weight_decay=tc["stat_mlp"]["weight_decay"]),
            train_pgamf_fn=partial(train_smooth, lr=pg_t["lr"],
                                    weight_decay=pg_t["weight_decay"],
                                    focal_gamma=pg_t["focal_gamma"],
                                    label_smoothing=pg_t["label_smoothing"],
                                    grad_clip_norm=pg_t["grad_clip_norm"]),
            X_cv=X_cv, y_cv=y_cv, F_stat_cv=Fs_cv,
            n_classes=n_classes,
            n_folds=tc["n_folds"],
            seed=seed,
            device=device,
            batch_size=tc["batch_size"],
            epochs=120, patience=20,
        )
        summary = summarise_cv(cv_results)

        print(f"\n{sep}")
        print(f"  TABLE IV. {cfg['train']['n_folds']}-FOLD CROSS-VALIDATION — MEAN ± STD")
        print(sep)
        for branch, label in [("stat", "Branch (a) Statistical MLP"),
                               ("pgamf", "Branch (b) PG-AMF")]:
            s = summary[branch]
            print(f"  {label:<30} Acc={s['acc_mean']*100:.2f}±{s['acc_std']*100:.2f}%  "
                  f"F1={s['f1_mean']*100:.2f}±{s['f1_std']*100:.2f}%")
        print(sep)

        # Save
        results_dir = Path(cfg["paths"]["results"])
        results_dir.mkdir(parents=True, exist_ok=True)
        cv_out = {b: {k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in s.items()}
                  for b, s in summary.items()}
        with open(results_dir / "cv_results.json", "w") as fh:
            json.dump(cv_out, fh, indent=2)
        logger.info("CV results saved to %s/cv_results.json", results_dir)


if __name__ == "__main__":
    main()
