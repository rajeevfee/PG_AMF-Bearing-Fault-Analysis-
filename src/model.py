"""
Neural network architectures.

Branch (a) — Baseline:
    :class:`MLPBaseline`  — simple MLP operating on 12 statistical features.

Branch (b) — PG-AMF:
    :class:`PGAMFLayer`          — physics-guided adaptive moment feature layer (v4).
    :class:`CrossChannelFusion`  — bilinear cross-channel fusion with gated residual.
    :class:`PGAMFClassifier`     — full PG-AMF v4 model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Branch (a) — Statistical MLP Baseline
# =============================================================================

class MLPBaseline(nn.Module):
    """
    Simple feed-forward classifier for hand-crafted statistical features.

    Architecture: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear.

    Parameters
    ----------
    input_dim:
        Number of input features (12 for 2-channel 6-feature stat. baseline).
    hidden_dim:
        Width of the two hidden layers.
    n_classes:
        Number of output classes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, input_dim) → (B, n_classes)
        return self.net(x)


# =============================================================================
# Branch (b) — PG-AMF components
# =============================================================================

class PGAMFLayer(nn.Module):
    """
    Physics-Guided Adaptive Moment Feature Layer — v4 (Enhanced).

    For each channel *c* and each learnable exponent α_i, three moment
    families are computed:

    * **Absolute**   : ``f_abs[i]  = (1/T) Σ |x_t|^αᵢ``
    * **Signed**     : ``f_sgn[i]  = (1/T) Σ sign(x_t)|x_t|^αᵢ``  — asymmetry-aware
    * **AC-coupled** : ``f_ac[i]   = (1/T) Σ |x_t − x̄|^αᵢ``       — DC-suppressed

    The exponents α ∈ (alpha_min, alpha_max) are parameterised via
    sigmoid: ``α = alpha_min + (alpha_max - alpha_min) * σ(θ)``.

    Output per channel: ``F × 3`` features.
    Total output: list of ``n_channels`` tensors, each of shape ``(B, F*3)``.

    Parameters
    ----------
    F:
        Number of learnable α exponents per channel.
    n_channels:
        Number of input channels (default 2).
    alpha_min, alpha_max:
        Bounds on the learnable exponents.
    """

    def __init__(
        self,
        F: int,
        n_channels: int = 2,
        alpha_min: float = 0.5,
        alpha_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.F = F
        self.n_channels = n_channels
        self.alpha_min = alpha_min
        self.alpha_range = alpha_max - alpha_min
        # θ shape: (n_channels, F) — one set of exponents per channel
        self.theta = nn.Parameter(
            torch.linspace(-2, 2, F).unsqueeze(0).repeat(n_channels, 1)
        )

    @property
    def alphas(self) -> torch.Tensor:
        """Learnable α exponents, constrained to (alpha_min, alpha_max)."""
        return self.alpha_min + self.alpha_range * torch.sigmoid(self.theta)

    def _moments(self, x_base: torch.Tensor, alphas_ch: torch.Tensor) -> torch.Tensor:
        """
        Compute generalised moments for a prepared 1-D channel batch.

        Parameters
        ----------
        x_base:
            Shape ``(B, T)`` — already prepared (abs / signed / ac).
        alphas_ch:
            Shape ``(F,)`` — exponents for this channel.

        Returns
        -------
        Tensor of shape ``(B, F)``.
        """
        # (B, T, 1) ** (1, 1, F) → (B, T, F) → mean over T → (B, F)
        return (
            x_base.unsqueeze(-1) ** alphas_ch.unsqueeze(0).unsqueeze(0)
        ).mean(dim=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        x:
            Tensor of shape ``(B, n_channels, T)`` or ``(B, T)`` for single channel.

        Returns
        -------
        list of ``n_channels`` tensors, each ``(B, F*3)``.
        Negatives are never raised to fractional powers (|x|^α is always
        computed first, sign restored separately) — NaN-safe.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        _B, C, _T = x.shape
        alphas = self.alphas  # (n_channels, F)

        feats_per_ch: list[torch.Tensor] = []
        for ch in range(C):
            sig = x[:, ch, :]    # (B, T)
            a_ch = alphas[ch]    # (F,)
            sign = sig.sign()    # (B, T)

            # Absolute moments: (1/T) Σ |x|^α
            x_abs = sig.abs() + 1e-8
            f_abs = self._moments(x_abs, a_ch)                              # (B, F)

            # Signed moments: (1/T) Σ sign(x) · |x|^α  — NaN-safe
            x_pow = x_abs.unsqueeze(-1) ** a_ch.unsqueeze(0).unsqueeze(0)  # (B, T, F)
            f_sgn = (sign.unsqueeze(-1) * x_pow).mean(dim=1)               # (B, F)

            # AC-coupled moments: (1/T) Σ |x − x̄|^α
            x_ac = (sig - sig.mean(dim=-1, keepdim=True)).abs() + 1e-8
            f_ac = self._moments(x_ac, a_ch)                               # (B, F)

            feats_per_ch.append(torch.cat([f_abs, f_sgn, f_ac], dim=-1))   # (B, 3F)

        return feats_per_ch

    def diversity_loss(self) -> torch.Tensor:
        """
        Margin-based repulsion loss to prevent α-exponent collapse.

        Penalises pairs of α values within the same channel that are
        closer than a margin of δ = 0.3, encouraging diverse coverage of
        the moment-order spectrum.
        """
        alphas = self.alphas
        loss = torch.tensor(0.0, device=alphas.device)
        F = self.F
        if F < 2:
            return loss
        for ch in range(self.n_channels):
            a = alphas[ch]                                      # (F,)
            diff = (a.unsqueeze(0) - a.unsqueeze(1)).abs()     # (F, F)
            mask = 1 - torch.eye(F, device=a.device)
            loss = loss + (torch.relu(0.3 - diff) * mask).sum() / (F * (F - 1))
        return loss / self.n_channels


class CrossChannelFusion(nn.Module):
    """
    Bilinear cross-channel fusion with gated residual.

    Replaces simple concatenation of channel features with:

    .. code-block:: text

        interaction = tanh(Bilinear(f1, f2))
        gates       = softmax(Linear([f1, f2]))     shape (B, 2)
        f1_out      = f1 + gates[:,0] * interaction
        f2_out      = f2 + gates[:,1] * interaction
        output      = concat([f1_out, f2_out])       shape (B, 2*feat_per_ch)

    Parameters
    ----------
    feat_per_ch:
        Number of features per channel (F * 3 for PGAMFLayer).
    """

    def __init__(self, feat_per_ch: int) -> None:
        super().__init__()
        self.W = nn.Bilinear(feat_per_ch, feat_per_ch, feat_per_ch, bias=True)
        self.gate = nn.Linear(feat_per_ch * 2, 2)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        interaction = torch.tanh(self.W(f1, f2))
        gates = torch.softmax(self.gate(torch.cat([f1, f2], dim=-1)), dim=-1)
        f1_out = f1 + gates[:, 0:1] * interaction
        f2_out = f2 + gates[:, 1:2] * interaction
        return torch.cat([f1_out, f2_out], dim=-1)  # (B, 2*feat_per_ch)


class PGAMFClassifier(nn.Module):
    """
    Full PG-AMF v4 Classifier.

    Architecture:
        ``x (B, 2, T)``
        → :class:`PGAMFLayer`         (3 moment families × F × 2 channels)
        → :class:`CrossChannelFusion` (bilinear + gated residual)
        → BatchNorm
        → MLP (BN → GELU → Dropout → BN → GELU → Dropout → Linear)

    Loss:
        ``L = L_focal + λ_div · L_div + λ_compact · L_compact``

    Parameters
    ----------
    n_classes:
        Number of fault classes.
    F:
        Adaptive moment exponents per channel.
    n_channels:
        Number of vibration channels (default 2).
    hidden:
        MLP hidden layer width.
    lambda_div:
        Weight for diversity (α-repulsion) regularisation.
    lambda_compact:
        Weight for cosine compactness regularisation (computed in trainer).
    alpha_min, alpha_max:
        Bounds for learnable α exponents.
    """

    def __init__(
        self,
        n_classes: int,
        F: int = 10,
        n_channels: int = 2,
        hidden: int = 128,
        lambda_div: float = 0.01,
        lambda_compact: float = 0.005,
        alpha_min: float = 0.5,
        alpha_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.lambda_div = lambda_div
        self.lambda_compact = lambda_compact

        self.pgamf = PGAMFLayer(F, n_channels, alpha_min, alpha_max)
        feat_per_ch = F * 3
        self.fusion = CrossChannelFusion(feat_per_ch)
        feat_dim = feat_per_ch * n_channels

        self.bn = nn.BatchNorm1d(feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, n_classes),
        )

        # Expose dims for external reference (e.g. trainer, evaluation)
        self.feat_dim = feat_dim
        self.feat_per_ch = feat_per_ch

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | dict:
        """
        Parameters
        ----------
        x:
            Input tensor ``(B, 2, T)``.
        return_features:
            If ``True``, return a dict ``{logits, z, alphas}`` instead of
            just logits.

        Returns
        -------
        logits ``(B, n_classes)`` or dict when *return_features* is ``True``.
        """
        feats_per_ch = self.pgamf(x)                              # [f_ch1, f_ch2] each (B, 3F)
        z_raw = self.fusion(feats_per_ch[0], feats_per_ch[1])    # (B, 6F)
        z = self.bn(z_raw)
        logits = self.classifier(z)
        if return_features:
            return {"logits": logits, "z": z, "alphas": self.pgamf.alphas}
        return logits

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float = 0.05,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """
        Focal cross-entropy loss plus diversity regularisation.

        Cosine compactness is **not** computed here — it requires the
        intermediate feature tensor *z*, which is injected by the trainer.

        Returns
        -------
        Scalar loss tensor.
        """
        ce_raw = F.cross_entropy(
            logits, targets,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_raw)
        L_focal = ((1.0 - pt) ** gamma * ce_raw).mean()
        L_div = self.pgamf.diversity_loss()
        return L_focal + self.lambda_div * L_div
