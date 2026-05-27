#!/usr/bin/env python3
"""
Run inference on new raw vibration data using a trained PG-AMF or Stat MLP model.

Usage
-----
# PG-AMF inference on two-channel text files
python scripts/infer.py --ch1 data/sample/ch1.txt --ch2 data/sample/ch2.txt

# Use the statistical MLP instead
python scripts/infer.py --ch1 data/sample/ch1.txt --ch2 data/sample/ch2.txt --branch stat

# Specify custom checkpoint
python scripts/infer.py --ch1 ch1.txt --ch2 ch2.txt --ckpt outputs/checkpoints/pgamf_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.inference import PGAMFInference, StatMLPInference, load_checkpoint
from src.model import MLPBaseline, PGAMFClassifier
from src.preprocessing import STAT_FEAT_NAMES_1CH, extract_stat_features, fit_scaler
from src.utils import get_device, load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer bearing fault class from raw vibration data.")
    p.add_argument("--ch1",    type=str, required=True,  help="Path to Channel 1 text file.")
    p.add_argument("--ch2",    type=str, required=True,  help="Path to Channel 2 text file.")
    p.add_argument("--branch", type=str, default="pgamf", choices=["pgamf", "stat"],
                   help="Which branch to use for inference.")
    p.add_argument("--ckpt",   type=str, default=None,   help="Checkpoint path (auto-detected if omitted).")
    p.add_argument("--config", nargs="*",
                   default=["configs/default.yaml",
                             "configs/data.yaml",
                             "configs/model.yaml",
                             "configs/train.yaml"])
    p.add_argument("--skip_header", type=int, default=None, help="Override header lines to skip.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def read_txt_channel(path: str | Path, skip_header: int = 15) -> np.ndarray:
    """Read a single-column vibration text file into a float32 array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    values = []
    with open(path, "r", errors="replace") as fh:
        for _ in range(skip_header):
            try:
                next(fh)
            except StopIteration:
                break
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line.split()[0]))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric data found in {path}")
    return np.array(values, dtype=np.float32)


def main() -> None:
    setup_logging()
    args = parse_args()

    cfg = load_config(*args.config)
    if args.device:
        cfg["device"] = args.device

    device      = get_device(cfg.get("device", "auto"))
    fault_names = list(cfg["fault_labels"].values())
    n_classes   = len(cfg["fault_types"])
    sig         = cfg["data"]["signal"]
    skip_header = args.skip_header if args.skip_header is not None else cfg.get("skip_header", 15)

    # ── Load raw signals ──────────────────────────────────────────────────────
    logger.info("Reading channel files …")
    ch1 = read_txt_channel(args.ch1, skip_header)
    ch2 = read_txt_channel(args.ch2, skip_header)
    logger.info("  Ch1: %d samples  |  Ch2: %d samples", len(ch1), len(ch2))

    mc = cfg["model"]
    pg = mc["pgamf"]

    # ── Load model & run inference ────────────────────────────────────────────
    if args.branch == "pgamf":
        ckpt_path = args.ckpt or "outputs/checkpoints/pgamf_best.pt"
        model = PGAMFClassifier(
            n_classes=n_classes, F=pg["F"], n_channels=pg["n_channels"],
            hidden=pg["hidden_size"], lambda_div=pg["lambda_div"],
            lambda_compact=pg["lambda_compact"],
            alpha_min=pg["alpha_min"], alpha_max=pg["alpha_max"],
        )
        load_checkpoint(model, ckpt_path)
        runner = PGAMFInference(
            model=model,
            fault_names=fault_names,
            device=device,
            seg_len=sig["seg_len"],
            hop=sig["hop"],
            do_mean_removal=sig["do_mean_removal"],
        )
    else:
        ckpt_path = args.ckpt or "outputs/checkpoints/stat_mlp_best.pt"
        # For stat branch we need a fitted scaler — load a dummy one from scratch
        # (users should pickle and reload a fitted scaler for production use)
        from src.preprocessing import preprocess_segment, segment_signal
        segs1 = segment_signal(ch1, sig["seg_len"], sig["hop"])
        segs2 = segment_signal(ch2, sig["seg_len"], sig["hop"])
        n = min(len(segs1), len(segs2))
        import numpy as _np
        proc1 = _np.stack([preprocess_segment(s, sig["do_mean_removal"]) for s in segs1[:n]])
        proc2 = _np.stack([preprocess_segment(s, sig["do_mean_removal"]) for s in segs2[:n]])
        X_tmp = _np.stack([proc1, proc2], axis=1)
        F_tmp = extract_stat_features(X_tmp)
        scaler = fit_scaler(F_tmp)  # fit on this file — replace with saved scaler in production

        n_stat_feat = F_tmp.shape[1]
        model = MLPBaseline(n_stat_feat, mc["baseline_mlp"]["hidden_size"], n_classes)
        load_checkpoint(model, ckpt_path)
        runner = StatMLPInference(
            model=model,
            scaler=scaler,
            fault_names=fault_names,
            device=device,
            seg_len=sig["seg_len"],
            hop=sig["hop"],
            do_mean_removal=sig["do_mean_removal"],
        )

    # ── Predict ───────────────────────────────────────────────────────────────
    result = runner.predict(ch1, ch2)

    sep = "═" * 50
    print(f"\n{sep}")
    print(f"  INFERENCE RESULT  [{args.branch.upper()} branch]")
    print(sep)
    print(f"  File Ch1        : {args.ch1}")
    print(f"  File Ch2        : {args.ch2}")
    print(f"  Segments analysed: {len(result['predictions'])}")
    print(f"  Majority vote   : {result['majority_vote']}")
    print(sep)
    print("\n  Segment-level predictions:")
    for i, (pred, name) in enumerate(zip(result["predictions"], result["class_names"])):
        probs = result["probabilities"][i]
        conf = probs[pred] * 100
        print(f"  Seg {i+1:>3} → {name:<15}  ({conf:.1f}% confidence)")
    print(sep)


if __name__ == "__main__":
    main()
