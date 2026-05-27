"""
Unit and integration tests for the PG-AMF pipeline.

Run with:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import BearingDataLoader, make_synthetic_signal
from src.evaluate import multiclass_fdr, per_feature_fdr
from src.model import CrossChannelFusion, MLPBaseline, PGAMFClassifier, PGAMFLayer
from src.preprocessing import (
    build_dataset,
    extract_stat_features,
    fit_scaler,
    preprocess_segment,
    segment_signal,
    split_dataset,
)
from src.trainer import evaluate_model, make_loaders, train_basic, train_smooth
from src.utils import compute_char_freqs, get_device, load_config, set_seed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CLASSES  = 3   # use 3 classes for speed
SEG_LEN    = 512
HOP        = 256
BATCH_SIZE = 8
DEVICE     = torch.device("cpu")


@pytest.fixture(scope="session")
def dummy_dataset():
    """Small synthetic (X, y) dataset for fast tests."""
    set_seed(0)
    rng = np.random.default_rng(0)
    N_per_class = 20
    X = rng.standard_normal((N_per_class * N_CLASSES, 2, SEG_LEN)).astype(np.float32)
    y = np.repeat(np.arange(N_CLASSES), N_per_class).astype(np.int64)
    return X, y


@pytest.fixture(scope="session")
def split_data(dummy_dataset):
    X, y = dummy_dataset
    return split_dataset(X, y, val_split=0.2, test_split=0.15, seed=42)


@pytest.fixture(scope="session")
def stat_features(split_data):
    X_train, y_train, X_val, y_val, X_test, y_test = split_data
    F_tr = extract_stat_features(X_train)
    F_v  = extract_stat_features(X_val)
    F_te = extract_stat_features(X_test)
    scaler = fit_scaler(F_tr)
    return (scaler.transform(F_tr), y_train,
            scaler.transform(F_v),  y_val,
            scaler.transform(F_te), y_test)


# =============================================================================
# Dataset & preprocessing
# =============================================================================

class TestPreprocessing:

    def test_preprocess_mean_removal(self):
        seg = np.ones(100, dtype=np.float32) * 5.0
        out = preprocess_segment(seg, do_mean_removal=True)
        assert np.allclose(out, 0.0), "Mean-removal should zero a constant signal."

    def test_preprocess_no_mean_removal(self):
        seg = np.ones(100, dtype=np.float32) * 3.0
        out = preprocess_segment(seg, do_mean_removal=False)
        assert np.allclose(out, 3.0), "No mean-removal should preserve values."

    def test_segment_signal_shape(self):
        signal = np.random.randn(10000).astype(np.float32)
        segs = segment_signal(signal, seg_len=SEG_LEN, hop=HOP)
        expected_n = (10000 - SEG_LEN) // HOP + 1
        assert segs.shape == (expected_n, SEG_LEN)

    def test_segment_values(self):
        signal = np.arange(1000, dtype=np.float32)
        segs = segment_signal(signal, seg_len=10, hop=5)
        assert np.allclose(segs[0], signal[:10])
        assert np.allclose(segs[1], signal[5:15])

    def test_stat_features_shape(self, dummy_dataset):
        X, _ = dummy_dataset
        F = extract_stat_features(X)
        assert F.shape == (len(X), 12), "Expected 6 features × 2 channels = 12."

    def test_stat_features_dtype(self, dummy_dataset):
        X, _ = dummy_dataset
        F = extract_stat_features(X)
        assert F.dtype == np.float32

    def test_split_sizes(self, dummy_dataset):
        X, y = dummy_dataset
        X_tr, y_tr, X_v, y_v, X_te, y_te = split_dataset(X, y, 0.2, 0.15, seed=0)
        total = len(y_tr) + len(y_v) + len(y_te)
        assert total == len(y)


# =============================================================================
# Synthetic data generation
# =============================================================================

class TestSyntheticData:

    def test_shape(self):
        ch1, ch2 = make_synthetic_signal("1ndBearing_Normal", label=0, n=4096)
        assert ch1.shape == (4096,)
        assert ch2.shape == (4096,)

    def test_dtype(self):
        ch1, ch2 = make_synthetic_signal("1ndBearing_ball", label=1, n=2048)
        assert ch1.dtype == np.float32
        assert ch2.dtype == np.float32

    def test_different_faults_differ(self):
        ch1_n, _ = make_synthetic_signal("1ndBearing_Normal", label=0, n=4096)
        ch1_b, _ = make_synthetic_signal("1ndBearing_ball",   label=1, n=4096)
        assert not np.allclose(ch1_n, ch1_b), "Different fault types must differ."


# =============================================================================
# Models
# =============================================================================

class TestPGAMFLayer:

    def test_output_shape(self):
        layer = PGAMFLayer(F=5, n_channels=2)
        x = torch.randn(4, 2, SEG_LEN)
        feats = layer(x)
        assert len(feats) == 2
        assert feats[0].shape == (4, 15), "Expected (B, F*3)."

    def test_alpha_bounds(self):
        layer = PGAMFLayer(F=8, alpha_min=0.5, alpha_max=5.0)
        alphas = layer.alphas
        assert (alphas >= 0.5).all()
        assert (alphas <= 5.0).all()

    def test_diversity_loss_scalar(self):
        layer = PGAMFLayer(F=5)
        loss = layer.diversity_loss()
        assert loss.ndim == 0

    def test_no_nan(self):
        layer = PGAMFLayer(F=6)
        x = torch.randn(8, 2, SEG_LEN)
        feats = layer(x)
        for f in feats:
            assert not torch.isnan(f).any(), "NaN in PGAMFLayer output."


class TestCrossChannelFusion:

    def test_output_shape(self):
        fpc = 15
        fusion = CrossChannelFusion(fpc)
        f1 = torch.randn(4, fpc)
        f2 = torch.randn(4, fpc)
        out = fusion(f1, f2)
        assert out.shape == (4, fpc * 2)


class TestMLPBaseline:

    def test_forward_shape(self):
        model = MLPBaseline(12, 64, N_CLASSES)
        x = torch.randn(8, 12)
        logits = model(x)
        assert logits.shape == (8, N_CLASSES)


class TestPGAMFClassifier:

    def test_forward_logits(self):
        model = PGAMFClassifier(N_CLASSES, F=4, hidden=32)
        x = torch.randn(6, 2, SEG_LEN)
        logits = model(x)
        assert logits.shape == (6, N_CLASSES)

    def test_return_features(self):
        model = PGAMFClassifier(N_CLASSES, F=4, hidden=32)
        x = torch.randn(6, 2, SEG_LEN)
        out = model(x, return_features=True)
        assert "logits"  in out
        assert "z"       in out
        assert "alphas"  in out
        assert out["logits"].shape == (6, N_CLASSES)

    def test_focal_loss_scalar(self):
        model = PGAMFClassifier(N_CLASSES, F=4, hidden=32)
        x = torch.randn(6, 2, SEG_LEN)
        logits = model(x)
        targets = torch.randint(0, N_CLASSES, (6,))
        loss = model.focal_loss(logits, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_no_nan_in_loss(self):
        model = PGAMFClassifier(N_CLASSES, F=4, hidden=32)
        x = torch.randn(16, 2, SEG_LEN)
        logits = model(x)
        targets = torch.randint(0, N_CLASSES, (16,))
        loss = model.focal_loss(logits, targets)
        assert not torch.isnan(loss)


# =============================================================================
# Training & evaluation (smoke tests — 2 epochs)
# =============================================================================

class TestTraining:

    def test_train_basic_runs(self, stat_features):
        F_tr, y_tr, F_v, y_v, F_te, y_te = stat_features
        dl_tr, dl_v, dl_te = make_loaders(F_tr, y_tr, F_v, y_v, F_te, y_te,
                                           batch_size=BATCH_SIZE, device=DEVICE)
        model = MLPBaseline(12, 32, N_CLASSES).to(DEVICE)
        hist = train_basic(model, dl_tr, dl_v, name="test", epochs=2, patience=5)
        assert "best_val_acc" in hist
        assert 0.0 <= hist["best_val_acc"] <= 1.0

    def test_train_smooth_runs(self, split_data):
        X_tr, y_tr, X_v, y_v, X_te, y_te = split_data
        dl_tr, dl_v, dl_te = make_loaders(X_tr, y_tr, X_v, y_v, X_te, y_te,
                                           batch_size=BATCH_SIZE, device=DEVICE)
        model = PGAMFClassifier(N_CLASSES, F=4, hidden=32).to(DEVICE)
        hist = train_smooth(model, dl_tr, dl_v, name="test", epochs=2, patience=5)
        assert "best_val_acc" in hist

    def test_evaluate_returns_correct_shapes(self, stat_features):
        F_tr, y_tr, F_v, y_v, F_te, y_te = stat_features
        dl_tr, dl_v, dl_te = make_loaders(F_tr, y_tr, F_v, y_v, F_te, y_te,
                                           batch_size=BATCH_SIZE, device=DEVICE)
        model = MLPBaseline(12, 32, N_CLASSES).to(DEVICE)
        acc, preds, trues = evaluate_model(model, dl_te)
        assert 0.0 <= acc <= 1.0
        assert len(preds) == len(trues) == len(y_te)


# =============================================================================
# Evaluation metrics
# =============================================================================

class TestMetrics:

    def test_fdr_positive(self, stat_features):
        F_tr, y_tr, *_ = stat_features
        fdr = multiclass_fdr(F_tr, y_tr)
        assert fdr >= 0.0

    def test_per_feature_fdr_shape(self, stat_features):
        F_tr, y_tr, *_ = stat_features
        fdr_per = per_feature_fdr(F_tr, y_tr)
        assert fdr_per.shape == (12,)

    def test_per_feature_fdr_nonneg(self, stat_features):
        F_tr, y_tr, *_ = stat_features
        fdr_per = per_feature_fdr(F_tr, y_tr)
        assert (fdr_per >= 0).all()


# =============================================================================
# Utilities
# =============================================================================

class TestUtils:

    def test_char_freqs(self):
        freqs = compute_char_freqs(7, 2.1, 14.5, 30.0)
        assert set(freqs.keys()) == {"BPFO", "BPFI", "BSF", "FTF", "fr"}
        assert all(v > 0 for v in freqs.values())

    def test_get_device_cpu(self):
        dev = get_device("cpu")
        assert dev.type == "cpu"

    def test_load_config(self, tmp_path):
        import yaml
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("seed: 99\n")
        cfg = load_config(cfg_file)
        assert cfg["seed"] == 99

    def test_set_seed_reproducible(self):
        set_seed(7)
        a = np.random.randn(5)
        set_seed(7)
        b = np.random.randn(5)
        assert np.allclose(a, b)
