"""
Dataset utilities: raw file loading, synthetic signal generation,
preprocessing, and segmentation into (N, 2, seg_len) arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Raw file I/O
# =============================================================================

class BearingDataLoader:
    """
    Load multi-channel vibration data from XJTU Gearbox dataset folders.

    Each fault class lives in a sub-folder under ``data_root``.  Inside each
    sub-folder the two channel files (``Data_Chan1.txt`` / ``Data_Chan2.txt``)
    contain one sample per line with ``skip_header`` header rows.

    If a folder is missing, a physics-informed synthetic signal is generated
    automatically so that the pipeline can run without real data.

    Parameters
    ----------
    data_root:
        Root path that contains one sub-folder per fault type.
    fault_types:
        List of folder names (fault class identifiers).
    fault_labels:
        Mapping from folder name → human-readable class name.
    channel_files:
        File names for each channel inside each fault folder.
    skip_header:
        Number of header rows to skip when reading text files.
    fs:
        Sampling frequency [Hz] — needed for synthetic generation.
    seed:
        RNG seed for synthetic signals.
    """

    def __init__(
        self,
        data_root: str | Path,
        fault_types: List[str],
        fault_labels: Dict[str, str],
        channel_files: Optional[List[str]] = None,
        skip_header: int = 15,
        fs: int = 20480,
        seed: int = 42,
    ) -> None:
        self.root = Path(data_root)
        self.fault_types = fault_types
        self.fault_labels = fault_labels
        self.channel_files = channel_files or ["Data_Chan1.txt", "Data_Chan2.txt"]
        self.skip_header = skip_header
        self.fs = fs
        self.seed = seed
        self.label_map: Dict[str, int] = {ft: i for i, ft in enumerate(fault_types)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_channel(self, path: Path) -> np.ndarray:
        """Read a single-column text file into a 1-D float32 array."""
        if not path.exists():
            raise FileNotFoundError(f"Channel file not found: {path}")
        try:
            return np.loadtxt(str(path), skiprows=self.skip_header, usecols=(0,),
                              dtype=np.float32).ravel()
        except Exception:
            pass
        # Fallback: line-by-line parsing for exotic number formats
        values: List[float] = []
        with open(path, "r", errors="replace") as fh:
            for _ in range(self.skip_header):
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_fault(self, fault_type: str) -> Dict:
        """
        Load all channel files for a single fault class.

        Returns
        -------
        dict with keys:
            ``data``     — np.ndarray of shape ``(N_samples, N_channels)``
            ``metadata`` — dict with label, fault names, and sample count
        """
        folder = self.root / fault_type
        if not folder.exists():
            raise FileNotFoundError(f"Fault folder missing: {folder}")
        channels = [self._read_channel(folder / cf) for cf in self.channel_files]
        min_len = min(len(c) for c in channels)
        data = np.stack([c[:min_len] for c in channels], axis=1)  # (N, 2)
        return {
            "data": data,
            "metadata": {
                "fault_type": fault_type,
                "fault_name": self.fault_labels[fault_type],
                "label": self.label_map[fault_type],
                "n_samples": data.shape[0],
                "synthetic": False,
            },
        }

    def load_all(self) -> Dict[str, Dict]:
        """
        Load every fault class, falling back to synthetic data on failure.

        Returns
        -------
        dict
            Mapping from fault_type string → entry dict (same structure as
            :meth:`load_fault`).
        """
        logger.info("Loading bearing vibration data from %s", self.root)
        out: Dict[str, Dict] = {}
        for fault in self.fault_types:
            try:
                entry = self.load_fault(fault)
                logger.info(
                    "  ✓ %-25s %d samples",
                    entry["metadata"]["fault_name"],
                    entry["data"].shape[0],
                )
            except Exception as exc:
                label = self.label_map[fault]
                logger.warning(
                    "  ⚠ %s — %s — using synthetic fallback.", fault, exc
                )
                ch1, ch2 = make_synthetic_signal(
                    fault_type=fault,
                    label=label,
                    fs=self.fs,
                    seed=self.seed,
                )
                n = min(len(ch1), len(ch2))
                entry = {
                    "data": np.stack([ch1[:n], ch2[:n]], axis=1),
                    "metadata": {
                        "fault_type": fault,
                        "fault_name": self.fault_labels[fault],
                        "label": label,
                        "n_samples": n,
                        "synthetic": True,
                    },
                }
                logger.info(
                    "  ⚠ %-25s synthetic  n=%d", self.fault_labels[fault], n
                )
            out[fault] = entry
        return out

    def to_dataframe(self, data_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Convert the output of :meth:`load_all` to a flat :class:`pd.DataFrame`.

        Columns: ``fault_type``, ``label``, ``fault_name``, ``n_samples``,
        ``ch1`` (ndarray), ``ch2`` (ndarray).
        """
        records = []
        for fault_type, entry in data_dict.items():
            m = entry["metadata"]
            records.append(
                {
                    "fault_type": fault_type,
                    "label": m["label"],
                    "fault_name": m["fault_name"],
                    "n_samples": m["n_samples"],
                    "ch1": entry["data"][:, 0],
                    "ch2": entry["data"][:, 1],
                    "synthetic": m.get("synthetic", False),
                }
            )
        return pd.DataFrame(records)


# =============================================================================
# Synthetic signal generation
# =============================================================================

def make_synthetic_signal(
    fault_type: str,
    label: int,
    fs: int = 20480,
    n: int = 65536,
    fr: float = 30.0,
    bpfo: float = 91.875,
    bpfi: float = 118.125,
    bsf: float = 59.856,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a physics-informed synthetic bearing vibration signal.

    Impulse trains are modulated by the shaft frequency and convolved with a
    resonance ring-down to simulate bearing fault signatures.

    Parameters
    ----------
    fault_type:
        Folder name containing fault type keywords (``ball``, ``inner``,
        ``outer``, ``mix``).
    label:
        Integer class label used to offset the RNG seed for diversity.
    fs, n, fr, bpfo, bpfi, bsf:
        Signal parameters matching :class:`BearingDataLoader` geometry.
    seed:
        Base RNG seed.

    Returns
    -------
    (ch1, ch2) both float32 arrays of length *n*.
    """
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed + label)

    # Base shaft harmonics + noise
    s = (
        0.30 * np.sin(2 * np.pi * fr * t)
        + 0.15 * np.sin(4 * np.pi * fr * t)
        + 0.05 * rng.standard_normal(n).astype(np.float32)
    )

    def impulse_train(
        f_fault: float,
        f_mod: Optional[float] = None,
        snr_db: float = 12,
    ) -> np.ndarray:
        period = int(fs / f_fault)
        impulse = np.zeros(n)
        impulse[::period] = 1.0
        omega_r, decay = 2 * np.pi * 3500, 1200
        t_ring = np.linspace(0, 1 / f_fault, period, endpoint=False)
        ring = np.exp(-decay * t_ring) * np.sin(omega_r * t_ring)
        fault_s = np.convolve(impulse, ring, mode="same")
        if f_mod is not None:
            fault_s *= 1.0 + 0.35 * np.sin(2 * np.pi * f_mod * t)
        amp = np.sqrt(np.mean(fault_s**2) * 10 ** (snr_db / 10))
        return (fault_s / (np.std(fault_s) + 1e-8) * amp * 0.10).astype(np.float32)

    if "ball" in fault_type:
        s += impulse_train(bsf)
    if "inner" in fault_type:
        s += impulse_train(bpfi, fr)
    if "outer" in fault_type:
        s += impulse_train(bpfo)
    if "mix" in fault_type:
        s += impulse_train(bsf) + impulse_train(bpfi, fr) + impulse_train(bpfo)

    ch2 = (0.88 * s + 0.03 * rng.standard_normal(n).astype(np.float32)).astype(
        np.float32
    )
    return s.astype(np.float32), ch2
