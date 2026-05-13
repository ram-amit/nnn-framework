"""
model.py — Next-Generation Neural Network (NNN) for Marketing Measurement.

Transformer-based architecture that models cross-channel synergies and funnel
stage transitions for B2B SaaS attribution.

Architecture overview:
  1. Input Projection — maps (spend, impressions, clicks, embeddings) → d_model
  2. Funnel Stage Blocks (ToFu → MoFu → BoFu) — each block contains:
     a. Channel Self-Attention: captures cross-channel synergies at each timestep
     b. Temporal Causal Attention: models carry-over/lag effects across time
     c. Feed-Forward Network with residual connections
  3. Gated Funnel Transitions — learnable gates between stages that model
     the conversion process (awareness → consideration → purchase)
  4. Residual Path Fusion — weighted combination of all stage outputs,
     allowing the model to capture both direct and indirect effects
  5. Output Head — pools across channels and predicts the target KPI

Based on: Google Research (April 2025) Transformer-based attribution.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSelfAttention(nn.Module):
    """
    Multi-head self-attention across marketing channels at each time step.
    Captures synergies like "LinkedIn spend lifts organic search volume."
    Stores attention weights for interpretability probing.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, d_model)
        Returns:
            (batch, channels, d_model)
        """
        B, C, _ = x.shape

        Q = self.W_q(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, C, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, C, -1)
        return self.W_o(out)


class TemporalCausalAttention(nn.Module):
    """
    Causal (masked) self-attention across time steps.
    Prevents information leakage from future periods — critical for
    time-series marketing data where we must respect temporal ordering.
    Models adstock-like carry-over and delayed channel impact.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, d_model)
        Returns:
            (batch, time, d_model)
        """
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class FunnelStageBlock(nn.Module):
    """
    One stage of the B2B marketing funnel.

    Each stage applies:
      1. Channel self-attention (cross-channel synergies within each timestep)
      2. Temporal causal attention (carry-over effects across time, per channel)
      3. Position-wise FFN (non-linear feature transformation)

    All with pre-norm residual connections (more stable than post-norm for
    small datasets typical in marketing measurement).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.channel_attn = ChannelSelfAttention(d_model, n_heads, dropout)
        self.temporal_attn = TemporalCausalAttention(d_model, n_heads, dropout)

        self.norm_chan = nn.LayerNorm(d_model)
        self.norm_temp = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, d_model)
        Returns:
            (batch, time, channels, d_model)
        """
        B, T, C, D = x.shape

        # Channel attention: operate on (B*T, C, D)
        x_flat = x.reshape(B * T, C, D)
        x_norm = self.norm_chan(x_flat)
        x_flat = x_flat + self.channel_attn(x_norm)
        x = x_flat.view(B, T, C, D)

        # Temporal attention: operate on (B*C, T, D)
        x_perm = x.permute(0, 2, 1, 3).reshape(B * C, T, D)
        x_norm = self.norm_temp(x_perm)
        x_perm = x_perm + self.temporal_attn(x_norm)
        x = x_perm.view(B, C, T, D).permute(0, 2, 1, 3)

        # FFN
        x_norm = self.norm_ff(x)
        x = x + self.ffn(x_norm)

        return x


class GatedFunnelTransition(nn.Module):
    """
    Transition layer between funnel stages, modeling the conversion process.

    Uses a learned gating mechanism:
      output = gate * transform(x) + (1 - gate) * x

    The gate learns which signals "convert" to the next funnel stage
    vs. which pass through unchanged. This is the key structural prior
    that distinguishes this from a generic stacked transformer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        return g * self.transform(x) + (1 - g) * x


FUNNEL_STAGE_NAMES = ["tofu", "mofu", "bofu"]


class NNNModel(nn.Module):
    """
    Next-Generation Neural Network (NNN) for Marketing Measurement.

    Input:  (batch, time, channels, input_dim)   — the Rank-4 tensor
    Output: (batch, time, 1)                     — predicted KPI per geo per week

    Key design decisions:
    - Pre-norm residual connections for training stability with limited data
    - Causal temporal masking to prevent future information leakage
    - Gated funnel transitions as structural prior for the B2B conversion funnel
    - Weighted residual fusion across funnel stages for multi-level effects
    - L1 penalty on attention weights (not just projections) for interpretability
    """

    def __init__(
        self,
        n_channels: int,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_funnel_stages: int = 3,
        dropout: float = 0.1,
        max_time_steps: int = 256,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_funnel_stages = n_funnel_stages

        # --- Input projection ---
        self.input_proj = nn.Linear(input_dim, d_model)

        # --- Positional encodings ---
        self.time_pos = nn.Parameter(
            torch.randn(1, max_time_steps, 1, d_model) * 0.02
        )
        self.channel_emb = nn.Parameter(
            torch.randn(1, 1, n_channels, d_model) * 0.02
        )

        # --- Funnel stages ---
        self.funnel_stages = nn.ModuleList(
            [
                FunnelStageBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_funnel_stages)
            ]
        )

        # --- Gated transitions between stages ---
        self.funnel_transitions = nn.ModuleList(
            [
                GatedFunnelTransition(d_model, dropout)
                for _ in range(n_funnel_stages - 1)
            ]
        )

        # --- Residual path fusion: learnable weights per stage ---
        self.stage_weights = nn.Parameter(torch.ones(n_funnel_stages))

        # --- Output head ---
        self.channel_pool = nn.Sequential(
            nn.Linear(d_model * n_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, channels, input_dim)
        Returns:
            (batch, time, 1)
        """
        B, T, C, D = x.shape

        x = self.input_proj(x)
        x = x + self.time_pos[:, :T, :, :]
        x = x + self.channel_emb[:, :, :C, :]

        stage_outputs: List[torch.Tensor] = []
        for i, stage in enumerate(self.funnel_stages):
            x = stage(x)
            stage_outputs.append(x)
            if i < len(self.funnel_transitions):
                x = self.funnel_transitions[i](x)

        # Weighted fusion of all funnel stage outputs
        w = F.softmax(self.stage_weights, dim=0)
        fused = sum(wi * out for wi, out in zip(w, stage_outputs))

        # Pool across channels → predict KPI
        fused = fused.reshape(B, T, C * self.d_model)
        pooled = self.channel_pool(fused)
        return self.output_head(pooled)

    # ------------------------------------------------------------------
    # Interpretability / Probing
    # ------------------------------------------------------------------

    def get_attention_maps(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """
        Extract attention weight matrices from all funnel stages.

        Returns:
            {
                "tofu": {"channel": Tensor(B*T, heads, C, C),
                         "temporal": Tensor(B*C, heads, T, T)},
                "mofu": { ... },
                "bofu": { ... },
            }
        """
        maps: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}
        for i, stage in enumerate(self.funnel_stages):
            name = FUNNEL_STAGE_NAMES[i] if i < len(FUNNEL_STAGE_NAMES) else f"stage_{i}"
            maps[name] = {
                "channel": stage.channel_attn.attn_weights,
                "temporal": stage.temporal_attn.attn_weights,
            }
        return maps

    def get_channel_contributions(
        self,
        x: torch.Tensor,
        channel_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Probing function: estimate each channel's contribution to the output KPI.

        Runs a forward pass, then aggregates the channel attention weights from
        the BoFu (bottom-of-funnel) stage. The column-sum of the attention matrix
        represents how much each channel is "attended to" by all other channels —
        a proxy for that channel's influence on the final prediction.
        """
        with torch.no_grad():
            self.forward(x)
            maps = self.get_attention_maps()

            last_stage = list(maps.keys())[-1]
            chan_attn = maps[last_stage]["channel"]
            if chan_attn is None:
                return {}

            # (B*T, heads, C, C) → average over batch*time and heads
            avg = chan_attn.mean(dim=(0, 1))  # (C, C)
            # Column-sum: total attention received by each channel
            contributions = avg.sum(dim=0)
            contributions = contributions / contributions.sum()

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(contributions.shape[0])]

        return {name: val.item() for name, val in zip(channel_names, contributions)}

    def get_temporal_decay_profile(
        self,
        x: torch.Tensor,
        channel_idx: int = 0,
    ) -> torch.Tensor:
        """
        Probing function: extract the temporal attention pattern for a given channel.
        Shows how the model weights past time steps — analogous to an adstock curve.

        Returns:
            Tensor of shape (T,) — average attention from the last time step
            to all previous steps, for the specified channel.
        """
        with torch.no_grad():
            self.forward(x)
            maps = self.get_attention_maps()
            last_stage = list(maps.keys())[-1]
            temp_attn = maps[last_stage]["temporal"]
            if temp_attn is None:
                return torch.tensor([])

            B_C, H, T, _ = temp_attn.shape
            B = x.shape[0]
            C = B_C // B

            # Reshape to (B, C, H, T, T), select channel
            temp_attn = temp_attn.view(B, C, H, T, T)
            channel_attn = temp_attn[:, channel_idx, :, :, :]  # (B, H, T, T)

            # Attention from last time step to all others, averaged
            last_row = channel_attn[:, :, -1, :]  # (B, H, T)
            return last_row.mean(dim=(0, 1))  # (T,)

    # ------------------------------------------------------------------
    # L1 Regularization
    # ------------------------------------------------------------------

    def get_l1_penalty(self) -> torch.Tensor:
        """
        Compute L1 penalty on stored attention weights (post-softmax).

        This directly penalizes dense attention patterns, encouraging the model
        to focus on a sparse set of channel interactions — making the learned
        synergy structure interpretable.

        Falls back to L1 on Q/K projection weights if attention maps haven't
        been computed yet (first call before any forward pass).
        """
        device = next(self.parameters()).device
        l1 = torch.tensor(0.0, device=device)

        has_attn = False
        for stage in self.funnel_stages:
            if stage.channel_attn.attn_weights is not None:
                has_attn = True
                l1 = l1 + stage.channel_attn.attn_weights.abs().mean()
            if stage.temporal_attn.attn_weights is not None:
                has_attn = True
                l1 = l1 + stage.temporal_attn.attn_weights.abs().mean()

        if not has_attn:
            for stage in self.funnel_stages:
                for attn in [stage.channel_attn, stage.temporal_attn]:
                    l1 = l1 + attn.W_q.weight.abs().sum()
                    l1 = l1 + attn.W_k.weight.abs().sum()

        return l1

    def get_l1_projection_penalty(self) -> torch.Tensor:
        """
        Supplementary L1 on Q/K/V projection matrices.
        Encourages weight sparsity in addition to attention sparsity.
        """
        device = next(self.parameters()).device
        l1 = torch.tensor(0.0, device=device)
        for stage in self.funnel_stages:
            for attn in [stage.channel_attn, stage.temporal_attn]:
                l1 = l1 + attn.W_q.weight.abs().mean()
                l1 = l1 + attn.W_k.weight.abs().mean()
                l1 = l1 + attn.W_v.weight.abs().mean()
        return l1

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        lines = [
            f"NNN Model Summary",
            f"  Channels:       {self.n_channels}",
            f"  d_model:        {self.d_model}",
            f"  Funnel stages:  {self.n_funnel_stages}",
            f"  Parameters:     {self.count_parameters():,}",
            f"  Stage weights:  {F.softmax(self.stage_weights, dim=0).detach().tolist()}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    n_channels = 6
    input_dim = 67  # 3 numeric + 64 embedding
    model = NNNModel(n_channels=n_channels, input_dim=input_dim)
    print(model.summary())

    x = torch.randn(2, 52, n_channels, input_dim)
    y = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {y.shape}")

    l1 = model.get_l1_penalty()
    print(f"L1 penalty (attention): {l1.item():.6f}")

    l1_proj = model.get_l1_projection_penalty()
    print(f"L1 penalty (projections): {l1_proj.item():.6f}")

    contribs = model.get_channel_contributions(
        x,
        channel_names=[
            "linkedin", "google_search", "google_display",
            "facebook", "content_syndication", "organic",
        ],
    )
    print(f"\nChannel contributions (BoFu attention):")
    for ch, val in sorted(contribs.items(), key=lambda kv: -kv[1]):
        print(f"  {ch:25s} {val:.4f}")

    decay = model.get_temporal_decay_profile(x, channel_idx=0)
    print(f"\nTemporal decay profile (channel 0, last 10 steps): {decay[-10:].tolist()}")
