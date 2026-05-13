"""
train.py — Training loop for the NNN marketing measurement model.

Supports both real (Snowflake-sourced) and synthetic data.

Key features:
  - 5-fold Time-Series Split validation (no future leakage)
  - L1 regularization on attention weights + projection matrices
  - MAPE and R-Squared metrics
  - Synergy matrix extraction from cross-channel attention
  - Per-channel sparsity scoring
  - Early stopping on validation loss
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import NNNModel


@dataclass
class TrainConfig:
    target: str = "Closed_Won"
    epochs: int = 300
    lr: float = 5e-4
    weight_decay: float = 1e-4
    l1_attn_lambda: float = 0.01
    l1_proj_lambda: float = 0.001
    dropout: float = 0.15
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_funnel_stages: int = 3
    n_splits: int = 5
    patience: int = 30
    min_delta: float = 1e-4
    device: str = "cpu"
    data_mode: str = "real"  # "real" or "synthetic"
    log_transform_target: bool = True


# =========================================================================
# Metrics
# =========================================================================

def mape(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    mask = y_true.abs() > eps
    if mask.sum() == 0:
        return torch.tensor(0.0)
    return ((y_true[mask] - y_pred[mask]).abs() / y_true[mask].abs()).mean() * 100


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-8)


# =========================================================================
# Time-Series CV
# =========================================================================

def time_series_split(
    n_time_steps: int, n_splits: int = 5, min_train: int = 20
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    fold_size = (n_time_steps - min_train) // n_splits

    for i in range(n_splits):
        train_end = min_train + i * fold_size
        val_end = min(train_end + fold_size, n_time_steps)
        if train_end >= n_time_steps or val_end <= train_end:
            break
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        splits.append((train_idx, val_idx))

    return splits


# =========================================================================
# Training loop
# =========================================================================

def train_one_epoch(
    model: NNNModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()

    pred = model(X_train).squeeze(-1)
    mse_loss = nn.functional.mse_loss(pred, y_train)

    l1_attn = model.get_l1_penalty()
    l1_proj = model.get_l1_projection_penalty()
    total_loss = mse_loss + config.l1_attn_lambda * l1_attn + config.l1_proj_lambda * l1_proj

    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        train_mape = mape(y_train, pred).item()
        train_r2 = r_squared(y_train, pred).item()

    return {
        "loss": total_loss.item(),
        "mse": mse_loss.item(),
        "l1_attn": l1_attn.item(),
        "l1_proj": l1_proj.item(),
        "mape": train_mape,
        "r2": train_r2,
    }


@torch.no_grad()
def evaluate(
    model: NNNModel, X_val: torch.Tensor, y_val: torch.Tensor
) -> Dict[str, float]:
    model.eval()
    pred = model(X_val).squeeze(-1)
    mse_loss = nn.functional.mse_loss(pred, y_val)
    val_mape = mape(y_val, pred).item()
    val_r2 = r_squared(y_val, pred).item()
    return {"loss": mse_loss.item(), "mape": val_mape, "r2": val_r2}


def train_fold(
    model: NNNModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    config: TrainConfig,
    fold: int,
) -> Dict:
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history: List[Dict] = []

    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(model, X_train, y_train, optimizer, config)
        val_metrics = evaluate(model, X_val, y_val)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        if val_metrics["loss"] < best_val_loss - config.min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0 or patience_counter == 0:
            print(
                f"  Fold {fold} | Epoch {epoch:3d} | "
                f"Train MAPE: {train_metrics['mape']:5.1f}% R²: {train_metrics['r2']:.4f} | "
                f"Val MAPE: {val_metrics['mape']:5.1f}% R²: {val_metrics['r2']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "fold": fold,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
        "history": history,
        "final_val": evaluate(model, X_val, y_val),
    }


# =========================================================================
# Probing: synergy matrix, sparsity, channel contributions
# =========================================================================

def extract_synergy_matrix(
    model: NNNModel,
    X: torch.Tensor,
    channel_names: List[str],
) -> Dict:
    """
    Extract the cross-channel synergy matrix from attention weights.

    For each funnel stage, the channel attention matrix A[i,j] represents
    how much channel j attends to channel i — i.e., channel i's influence
    on channel j's representation.

    Returns the averaged synergy matrix across all stages and heads,
    plus the top-K strongest cross-channel relationships.
    """
    model.eval()
    with torch.no_grad():
        model.forward(X)
        attn_maps = model.get_attention_maps()

    C = len(channel_names)
    combined = torch.zeros(C, C)
    n_stages = 0

    for stage_name, maps in attn_maps.items():
        chan_attn = maps["channel"]
        if chan_attn is None:
            continue
        # chan_attn: (B*T, heads, C, C) — average across batch*time and heads
        avg = chan_attn.mean(dim=(0, 1))  # (C, C)
        combined += avg
        n_stages += 1

    if n_stages > 0:
        combined /= n_stages

    # Build named matrix
    synergy = {}
    for i, src in enumerate(channel_names):
        for j, tgt in enumerate(channel_names):
            synergy[(src, tgt)] = combined[i, j].item()

    # Top cross-channel relationships (exclude diagonal)
    off_diag = {
        k: v for k, v in synergy.items() if k[0] != k[1]
    }
    top_synergies = sorted(off_diag.items(), key=lambda x: -x[1])[:10]

    return {
        "matrix": combined.numpy(),
        "named": synergy,
        "top_synergies": top_synergies,
        "channel_names": channel_names,
    }


def compute_channel_sparsity(
    model: NNNModel,
    X: torch.Tensor,
    channel_names: List[str],
    threshold: float = 0.05,
) -> Dict[str, Dict]:
    """
    Compute per-channel sparsity scores from attention weights.

    For each channel, measures what fraction of its attention weights
    (both incoming and outgoing) fall below the threshold.
    High sparsity = the model is effectively ignoring this channel.
    """
    model.eval()
    with torch.no_grad():
        model.forward(X)
        attn_maps = model.get_attention_maps()

    C = len(channel_names)
    total_incoming = torch.zeros(C)
    total_outgoing = torch.zeros(C)
    sparse_incoming = torch.zeros(C)
    sparse_outgoing = torch.zeros(C)
    count = 0

    for stage_name, maps in attn_maps.items():
        chan_attn = maps["channel"]
        if chan_attn is None:
            continue
        # (B*T, heads, C, C) → average over batch*time, keep heads
        avg = chan_attn.mean(dim=0)  # (heads, C, C)
        for h in range(avg.shape[0]):
            A = avg[h]  # (C, C) — A[i,j] = how much j attends to i
            for c in range(C):
                col = A[:, c]  # incoming attention to channel c
                row = A[c, :]  # outgoing attention from channel c
                total_incoming[c] += col.numel()
                total_outgoing[c] += row.numel()
                sparse_incoming[c] += (col < threshold).sum().float()
                sparse_outgoing[c] += (row < threshold).sum().float()
            count += 1

    results = {}
    for i, name in enumerate(channel_names):
        inc = sparse_incoming[i] / (total_incoming[i] + 1e-8)
        out = sparse_outgoing[i] / (total_outgoing[i] + 1e-8)
        results[name] = {
            "incoming_sparsity": inc.item(),
            "outgoing_sparsity": out.item(),
            "overall_sparsity": ((inc + out) / 2).item(),
        }

    return results


def probe_model(
    model: NNNModel,
    X: torch.Tensor,
    channel_names: List[str],
) -> Dict:
    model.eval()

    contributions = model.get_channel_contributions(X, channel_names)

    temporal_profiles = {}
    for i, name in enumerate(channel_names):
        decay = model.get_temporal_decay_profile(X, channel_idx=i)
        temporal_profiles[name] = decay.tolist() if len(decay) > 0 else []

    attn_maps = model.get_attention_maps()
    stage_summary = {}
    for stage_name, maps in attn_maps.items():
        chan = maps["channel"]
        stage_summary[stage_name] = {
            "channel_attn_sparsity": (
                (chan < 0.05).float().mean().item() if chan is not None else None
            ),
        }

    synergy = extract_synergy_matrix(model, X, channel_names)
    sparsity = compute_channel_sparsity(model, X, channel_names)

    return {
        "channel_contributions": contributions,
        "temporal_decay_profiles": temporal_profiles,
        "stage_summary": stage_summary,
        "synergy": synergy,
        "channel_sparsity": sparsity,
    }


# =========================================================================
# Data loading
# =========================================================================

def load_real_data(config: TrainConfig):
    from data_prep import RealDataConfig, prepare_real_data

    real_config = RealDataConfig(
        csv_path=str(Path(__file__).parent / "data" / "real_marketing_data.csv"),
    )
    data = prepare_real_data(real_config)

    X = torch.tensor(data["tensor"], dtype=torch.float32)
    y_raw = torch.tensor(data["targets"][config.target], dtype=torch.float32)

    metadata = data["metadata"]
    n_channels = len(metadata["channels"])
    input_dim = metadata["input_dim"]
    channel_names = metadata["channels"]
    n_weeks = X.shape[1]

    # Log-transform target to stabilize training on heavy-tailed CRM data
    y_mean = None
    y_std = None
    if config.log_transform_target:
        y = torch.log1p(y_raw)
        y_mean = y.mean().item()
        y_std = y.std().item()
        y = (y - y_mean) / (y_std + 1e-8)
    else:
        y_mean = y_raw.mean().item()
        y_std = y_raw.std().item()
        y = (y_raw - y_mean) / (y_std + 1e-8)

    return {
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "y_mean": y_mean,
        "y_std": y_std,
        "n_channels": n_channels,
        "input_dim": input_dim,
        "channel_names": channel_names,
        "n_weeks": n_weeks,
        "log_transform": config.log_transform_target,
    }


def load_synthetic_data(config: TrainConfig):
    from data_prep import FunnelConfig, generate_synthetic_data

    funnel_config = FunnelConfig()
    data = generate_synthetic_data(funnel_config)

    target_key = config.target.lower()
    if target_key not in data["targets"]:
        target_key = "closed_won"

    X = torch.tensor(data["tensor"], dtype=torch.float32)
    y_raw = torch.tensor(data["targets"][target_key], dtype=torch.float32)

    y_mean = y_raw.mean().item()
    y_std = y_raw.std().item()
    y = (y_raw - y_mean) / (y_std + 1e-8)

    return {
        "X": X,
        "y": y,
        "y_raw": y_raw,
        "y_mean": y_mean,
        "y_std": y_std,
        "n_channels": funnel_config.n_channels,
        "input_dim": funnel_config.input_dim,
        "channel_names": funnel_config.channels,
        "n_weeks": funnel_config.n_weeks,
        "log_transform": False,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    import sys
    config = TrainConfig()

    if len(sys.argv) > 1 and sys.argv[1] == "synthetic":
        config.data_mode = "synthetic"
        config.target = "closed_won"
        config.log_transform_target = False

    print("=" * 70)
    print("NNN - Next-Generation Neural Network for Marketing Measurement")
    print("=" * 70)
    print(f"  Data mode:    {config.data_mode}")
    print(f"  Target KPI:   {config.target}")
    print(f"  Log-transform: {config.log_transform_target}")
    print(f"  L1 (attn):    {config.l1_attn_lambda}")
    print(f"  L1 (proj):    {config.l1_proj_lambda}")
    print(f"  CV folds:     {config.n_splits}")
    print()

    # --- Load data ---
    if config.data_mode == "real":
        print("Loading real marketing data from Snowflake ETL output...")
        data = load_real_data(config)
    else:
        print("Generating synthetic B2B SaaS funnel data...")
        data = load_synthetic_data(config)

    X_all = data["X"].to(config.device)
    y_all = data["y"].to(config.device)

    print(f"  Input tensor: {X_all.shape}  (geos, weeks, channels, dims)")
    print(f"  Target shape: {y_all.shape}  (geos, weeks)")
    print(f"  Target stats (raw): mean={data['y_raw'].mean():.1f}, "
          f"std={data['y_raw'].std():.1f}, "
          f"range=[{data['y_raw'].min():.1f}, {data['y_raw'].max():.1f}]")
    print(f"  Target stats (transformed): mean={y_all.mean():.4f}, std={y_all.std():.4f}")
    print(f"  Channels ({data['n_channels']}): {', '.join(data['channel_names'])}")
    print()

    # --- Time-series CV ---
    splits = time_series_split(data["n_weeks"], config.n_splits)
    print(f"Time-Series Cross-Validation: {len(splits)} folds")
    for i, (tr, va) in enumerate(splits):
        print(f"  Fold {i}: train weeks 0-{tr[-1]} ({len(tr)}w), "
              f"val weeks {va[0]}-{va[-1]} ({len(va)}w)")
    print()

    fold_results = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        print(f"--- Fold {fold_i} ---")

        X_train = X_all[:, train_idx, :, :]
        y_train = y_all[:, train_idx]
        X_val = X_all[:, val_idx, :, :]
        y_val = y_all[:, val_idx]

        model = NNNModel(
            n_channels=data["n_channels"],
            input_dim=data["input_dim"],
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_funnel_stages=config.n_funnel_stages,
            dropout=config.dropout,
        ).to(config.device)

        if fold_i == 0:
            print(model.summary())
            print()

        result = train_fold(model, X_train, y_train, X_val, y_val, config, fold_i)
        fold_results.append(result)
        print()

    # --- CV results ---
    print("=" * 70)
    print("Cross-Validation Results")
    print("=" * 70)
    val_mapes = [r["final_val"]["mape"] for r in fold_results]
    val_r2s = [r["final_val"]["r2"] for r in fold_results]
    val_losses = [r["final_val"]["loss"] for r in fold_results]

    for r in fold_results:
        v = r["final_val"]
        print(
            f"  Fold {r['fold']}: Loss={v['loss']:.4f}  "
            f"MAPE={v['mape']:.1f}%  R²={v['r2']:.4f}  "
            f"(trained {r['epochs_trained']} epochs)"
        )
    print(f"\n  Mean MAPE:  {np.mean(val_mapes):.1f}% (+/- {np.std(val_mapes):.1f}%)")
    print(f"  Mean R²:    {np.mean(val_r2s):.4f} (+/- {np.std(val_r2s):.4f})")
    print(f"  Mean Loss:  {np.mean(val_losses):.4f}")
    print()

    # --- Final model: train on all data ---
    print("Training final model on full time range...")
    final_model = NNNModel(
        n_channels=data["n_channels"],
        input_dim=data["input_dim"],
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_funnel_stages=config.n_funnel_stages,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = AdamW(
        final_model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    final_model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        pred = final_model(X_all).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_all)
        l1 = config.l1_attn_lambda * final_model.get_l1_penalty()
        l1_proj = config.l1_proj_lambda * final_model.get_l1_projection_penalty()
        (loss + l1 + l1_proj).backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            with torch.no_grad():
                r2_val = r_squared(y_all, pred).item()
                mape_val = mape(y_all, pred).item()
            print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f}  "
                  f"R²={r2_val:.4f}  MAPE={mape_val:.1f}%")

    # Final metrics
    final_model.eval()
    with torch.no_grad():
        final_pred = final_model(X_all).squeeze(-1)
        final_r2 = r_squared(y_all, final_pred).item()
        final_mape = mape(y_all, final_pred).item()
    print(f"  Final: R²={final_r2:.4f}  MAPE={final_mape:.1f}%")

    # --- Probing ---
    print()
    print("=" * 70)
    print("Model Probing - Interpretability Analysis")
    print("=" * 70)
    probe_results = probe_model(final_model, X_all, data["channel_names"])

    # Channel contributions
    print(f"\nChannel Contributions to '{config.target}' (BoFu attention):")
    sorted_contribs = sorted(
        probe_results["channel_contributions"].items(), key=lambda kv: -kv[1]
    )
    for rank, (ch, val) in enumerate(sorted_contribs, 1):
        bar = "#" * int(val * 60)
        print(f"  {rank:2d}. {ch:20s} {val:.4f}  {bar}")

    # Synergy matrix — top 5 cross-channel relationships
    print("\nTop 10 Cross-Channel Synergies (source -> target, attention weight):")
    syn = probe_results["synergy"]
    for i, ((src, tgt), weight) in enumerate(syn["top_synergies"][:10], 1):
        print(f"  {i:2d}. {src:20s} -> {tgt:20s}  {weight:.4f}")

    # Per-channel sparsity
    print(f"\nChannel Sparsity Scores (threshold=0.05):")
    print(f"  {'Channel':20s} {'Incoming':>10s} {'Outgoing':>10s} {'Overall':>10s}  Verdict")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}  {'-'*20}")
    sparsity = probe_results["channel_sparsity"]
    for ch in sorted(sparsity.keys(), key=lambda k: -sparsity[k]["overall_sparsity"]):
        s = sparsity[ch]
        verdict = "IGNORED by L1" if s["overall_sparsity"] > 0.7 else \
                  "sparse" if s["overall_sparsity"] > 0.4 else "active"
        print(f"  {ch:20s} {s['incoming_sparsity']:10.2%} "
              f"{s['outgoing_sparsity']:10.2%} {s['overall_sparsity']:10.2%}  {verdict}")

    # Stage-level sparsity
    print(f"\nAttention Sparsity by Funnel Stage:")
    for stage, info in probe_results["stage_summary"].items():
        sp = info["channel_attn_sparsity"]
        if sp is not None:
            print(f"  {stage:6s}: {sp:.2%}")

    # Temporal decay for top 3 channels
    print(f"\nTemporal Decay Profiles (last 10 weeks, top 3 channels):")
    top3 = [ch for ch, _ in sorted_contribs[:3]]
    for ch in top3:
        profile = probe_results["temporal_decay_profiles"].get(ch, [])
        if len(profile) >= 10:
            recent = [f"{v:.3f}" for v in profile[-10:]]
            print(f"  {ch:20s} [{', '.join(recent)}]")

    # --- Summary table ---
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Data:          {config.data_mode} ({X_all.shape[0]} geos, "
          f"{X_all.shape[1]} weeks, {X_all.shape[2]} channels)")
    print(f"  Target:        {config.target}")
    print(f"  CV Mean MAPE:  {np.mean(val_mapes):.1f}% (+/- {np.std(val_mapes):.1f}%)")
    print(f"  CV Mean R²:    {np.mean(val_r2s):.4f} (+/- {np.std(val_r2s):.4f})")
    print(f"  Final R²:      {final_r2:.4f}")
    print(f"  Final MAPE:    {final_mape:.1f}%")
    print(f"  Top 3 channels for '{config.target}':")
    for rank, (ch, val) in enumerate(sorted_contribs[:3], 1):
        print(f"    {rank}. {ch} ({val:.4f})")
    print(f"  Top 5 synergies:")
    for i, ((src, tgt), w) in enumerate(syn["top_synergies"][:5], 1):
        print(f"    {i}. {src} -> {tgt} ({w:.4f})")

    # L1 recommendation
    stage_sparsities = [
        v["channel_attn_sparsity"]
        for v in probe_results["stage_summary"].values()
        if v["channel_attn_sparsity"] is not None
    ]
    avg_sparsity = np.mean(stage_sparsities) if stage_sparsities else 0
    ignored_count = sum(
        1 for s in sparsity.values() if s["overall_sparsity"] > 0.7
    )

    print(f"\n  L1 REGULARIZATION ASSESSMENT:")
    print(f"    Avg attention sparsity: {avg_sparsity:.2%}")
    print(f"    Channels ignored (>70% sparse): {ignored_count}/{len(sparsity)}")
    if avg_sparsity < 0.15:
        print(f"    -> RECOMMENDATION: INCREASE L1 penalty (current attn={config.l1_attn_lambda})")
        print(f"       Attention is too dense — model is spreading weight across all channels")
        print(f"       uniformly instead of learning sparse, interpretable patterns.")
        print(f"       Try l1_attn_lambda=0.05 for next run.")
    elif avg_sparsity > 0.60:
        print(f"    -> RECOMMENDATION: DECREASE L1 penalty (current attn={config.l1_attn_lambda})")
        print(f"       Too many channels are being zeroed out. The model may be")
        print(f"       underfitting by ignoring potentially useful signals.")
        print(f"       Try l1_attn_lambda=0.005 for next run.")
    else:
        print(f"    -> RECOMMENDATION: Current L1 penalty is well-calibrated.")
        print(f"       Sparsity is in the 15-60% sweet spot for interpretability")
        print(f"       without sacrificing predictive power.")

    # --- Save ---
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    torch.save(final_model.state_dict(), out_dir / "nnn_model.pt")

    # Save probe results (convert numpy arrays to lists for JSON)
    save_probe = {
        "channel_contributions": probe_results["channel_contributions"],
        "channel_sparsity": probe_results["channel_sparsity"],
        "stage_summary": probe_results["stage_summary"],
        "synergy_top": [
            {"source": s, "target": t, "weight": w}
            for (s, t), w in syn["top_synergies"]
        ],
        "synergy_matrix": syn["matrix"].tolist(),
        "synergy_channel_order": syn["channel_names"],
    }
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(save_probe, f, indent=2, default=str)

    cv_results = {
        "config": {
            "target": config.target,
            "data_mode": config.data_mode,
            "l1_attn_lambda": config.l1_attn_lambda,
            "l1_proj_lambda": config.l1_proj_lambda,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_funnel_stages": config.n_funnel_stages,
            "log_transform": config.log_transform_target,
        },
        "folds": [
            {"fold": r["fold"], "val_metrics": r["final_val"],
             "epochs_trained": r["epochs_trained"]}
            for r in fold_results
        ],
        "mean_mape": float(np.mean(val_mapes)),
        "std_mape": float(np.std(val_mapes)),
        "mean_r2": float(np.mean(val_r2s)),
        "std_r2": float(np.std(val_r2s)),
        "final_r2": final_r2,
        "final_mape": final_mape,
    }
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    # Save synergy matrix as CSV for easy inspection
    import pandas as pd
    syn_df = pd.DataFrame(
        syn["matrix"],
        index=syn["channel_names"],
        columns=syn["channel_names"],
    )
    syn_df.to_csv(out_dir / "synergy_matrix.csv")

    print(f"\nSaved to {out_dir}/:")
    print(f"  nnn_model.pt          — trained model weights")
    print(f"  probe_results.json    — channel contributions, sparsity, synergies")
    print(f"  cv_results.json       — cross-validation metrics")
    print(f"  synergy_matrix.csv    — full 13x13 channel synergy matrix")
    print("\nDone.")


if __name__ == "__main__":
    main()
