"""
optimizer.py — Synergy-Aware Budget Optimizer for the NNN framework.

Uses the trained NNNModel as a differentiable simulator to:
  1. Run Spend Sensitivity Analysis (response curves per channel)
  2. Find the Optimal Budget Allocation via gradient descent

The key insight: because the transformer's attention mechanism processes
all channels jointly, modifying one channel's spend changes the
representation of every other channel. This means synergy effects
(e.g., Podcast -> Display) are captured automatically in the gradients.

Usage:
    python3 optimizer.py
    python3 optimizer.py --budget 1000000 --steps 500
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_prep import RealDataConfig, prepare_real_data
from model import NNNModel


@dataclass
class OptimizerConfig:
    weekly_budget: float = 1_000_000.0
    sensitivity_steps: int = 21
    sensitivity_range: Tuple[float, float] = (0.5, 1.5)
    optim_steps: int = 800
    optim_lr: float = 0.01
    baseline_weeks: int = 8
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    n_funnel_stages: int = 3


# =========================================================================
# Data + model loading
# =========================================================================

def load_model_and_data(config: OptimizerConfig) -> Dict:
    proj_dir = Path(__file__).parent

    real_config = RealDataConfig(
        csv_path=str(proj_dir / "data" / "real_marketing_data.csv"),
    )
    data = prepare_real_data(real_config)
    meta = data["metadata"]

    model = NNNModel(
        n_channels=len(meta["channels"]),
        input_dim=meta["input_dim"],
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        n_funnel_stages=config.n_funnel_stages,
    )
    model.load_state_dict(torch.load(proj_dir / "output" / "nnn_model.pt", weights_only=True))
    model.eval()

    X = torch.tensor(data["tensor"], dtype=torch.float32)
    y_raw = torch.tensor(data["targets"]["Closed_Won"], dtype=torch.float32)

    y_log = torch.log1p(y_raw)
    y_mean = y_log.mean().item()
    y_std = y_log.std().item()

    spend_max = meta["normalization"]["max"][0]

    # Current average weekly spend per channel (in dollars)
    csv = pd.read_csv(
        proj_dir / "data" / "real_marketing_data.csv",
        keep_default_na=False, na_values=[""],
    )
    current_spend = csv.groupby("Channel")["Spend"].mean()
    channel_names = meta["channels"]
    current_spend_vec = np.array([current_spend.get(ch, 0) for ch in channel_names])

    return {
        "model": model,
        "X": X,
        "y_raw": y_raw,
        "y_mean": y_mean,
        "y_std": y_std,
        "spend_max": spend_max,
        "channel_names": channel_names,
        "current_spend": current_spend_vec,
        "n_channels": len(channel_names),
        "input_dim": meta["input_dim"],
    }


def pred_to_closed_won(pred: torch.Tensor, y_mean: float, y_std: float) -> torch.Tensor:
    """Convert model output (standardized log space) back to Closed_Won counts."""
    return torch.expm1(pred * y_std + y_mean)


# =========================================================================
# 1. Spend Sensitivity Analysis
# =========================================================================

def run_sensitivity(bundle: Dict, config: OptimizerConfig) -> Dict:
    """
    For each channel, scale spend from 50% to 150% of current level
    while holding all other channels constant. Records the predicted
    Closed_Won at each level.
    """
    model = bundle["model"]
    X = bundle["X"]
    spend_max = bundle["spend_max"]
    channel_names = bundle["channel_names"]
    y_mean, y_std = bundle["y_mean"], bundle["y_std"]

    B, T, C, D = X.shape
    baseline_start = max(0, T - config.baseline_weeks)
    X_base = X[:, baseline_start:, :, :].clone()

    multipliers = np.linspace(
        config.sensitivity_range[0],
        config.sensitivity_range[1],
        config.sensitivity_steps,
    )

    # Baseline prediction
    with torch.no_grad():
        base_pred = model(X_base).squeeze(-1)
        base_cw = pred_to_closed_won(base_pred, y_mean, y_std)
        baseline_total = base_cw.sum().item()

    results = {}
    for c_idx, ch_name in enumerate(channel_names):
        curve = []
        current_norm = X_base[:, :, c_idx, 0].mean().item()
        current_dollar = current_norm * spend_max

        for mult in multipliers:
            X_mod = X_base.clone()
            X_mod[:, :, c_idx, 0] = current_norm * mult

            with torch.no_grad():
                pred = model(X_mod).squeeze(-1)
                cw = pred_to_closed_won(pred, y_mean, y_std)
                total_cw = cw.sum().item()

            curve.append({
                "multiplier": float(mult),
                "spend_dollar": current_dollar * mult,
                "predicted_closed_won": total_cw,
                "delta_vs_baseline": total_cw - baseline_total,
                "delta_pct": (total_cw - baseline_total) / (baseline_total + 1e-8) * 100,
            })

        # Marginal ROI at current level (finite difference around 1.0x)
        idx_low = config.sensitivity_steps // 2 - 1
        idx_high = config.sensitivity_steps // 2 + 1
        delta_cw = curve[idx_high]["predicted_closed_won"] - curve[idx_low]["predicted_closed_won"]
        delta_spend = curve[idx_high]["spend_dollar"] - curve[idx_low]["spend_dollar"]
        marginal_roi = delta_cw / (delta_spend + 1e-8)

        # Saturation detection: where does the response flatten?
        deltas = [curve[i+1]["predicted_closed_won"] - curve[i]["predicted_closed_won"]
                  for i in range(len(curve)-1)]
        first_half_avg = np.mean(deltas[:len(deltas)//2])
        second_half_avg = np.mean(deltas[len(deltas)//2:])
        saturation_ratio = second_half_avg / (first_half_avg + 1e-8)

        results[ch_name] = {
            "curve": curve,
            "current_spend": current_dollar,
            "marginal_roi": marginal_roi,
            "saturation_ratio": saturation_ratio,
            "saturated": saturation_ratio < 0.5,
        }

    return {"channels": results, "baseline_total_cw": baseline_total}


# =========================================================================
# 2. Synergy-Aware Budget Optimizer
# =========================================================================

def optimize_budget(
    bundle: Dict,
    config: OptimizerConfig,
) -> Dict:
    """
    Find the optimal spend allocation across 13 channels that maximizes
    predicted Closed_Won deals, subject to a fixed total budget.

    The model's attention mechanism captures cross-channel synergies
    automatically: when we change one channel's spend, the attention
    weights redistribute, affecting all other channels' representations.
    This means the gradients naturally encode synergy effects.

    Uses projected gradient descent with softmax parameterization
    to maintain valid budget allocation.
    """
    model = bundle["model"]
    X = bundle["X"]
    spend_max = bundle["spend_max"]
    channel_names = bundle["channel_names"]
    y_mean, y_std = bundle["y_mean"], bundle["y_std"]
    current_spend = bundle["current_spend"]
    n_channels = bundle["n_channels"]

    B, T, C, D = X.shape
    baseline_start = max(0, T - config.baseline_weeks)
    X_base = X[:, baseline_start:, :, :].clone()

    budget_norm = config.weekly_budget / spend_max

    # Initialize allocation logits from current spend proportions
    current_total = current_spend.sum()
    current_props = current_spend / (current_total + 1e-8)
    # Clamp small values to prevent -inf in log
    current_props = np.clip(current_props, 0.01, None)
    current_props = current_props / current_props.sum()

    alloc_logits = nn.Parameter(torch.tensor(
        np.log(current_props), dtype=torch.float32,
    ))

    optimizer = torch.optim.Adam([alloc_logits], lr=config.optim_lr)

    # Freeze model weights
    for p in model.parameters():
        p.requires_grad_(False)

    history = []
    best_cw = -float("inf")
    best_alloc = None

    for step in range(config.optim_steps):
        optimizer.zero_grad()

        # Softmax → proportions → spend per channel (in normalized units)
        proportions = torch.softmax(alloc_logits, dim=0)
        spend_per_channel = proportions * budget_norm

        # Inject optimized spend into tensor
        X_opt = X_base.clone().detach()
        for c in range(n_channels):
            X_opt[:, :, c, 0] = spend_per_channel[c]

        pred = model(X_opt).squeeze(-1)
        total_cw = pred_to_closed_won(pred, y_mean, y_std).sum()

        # Maximize closed-won → minimize negative
        loss = -total_cw
        loss.backward()
        optimizer.step()

        cw_val = total_cw.item()
        if cw_val > best_cw:
            best_cw = cw_val
            best_alloc = proportions.detach().clone()

        if step % 100 == 0 or step == config.optim_steps - 1:
            dollar_alloc = (proportions.detach().numpy() * config.weekly_budget)
            top3 = np.argsort(-dollar_alloc)[:3]
            print(f"  Step {step:4d}: Predicted CW={cw_val:,.0f}  "
                  f"Top: {channel_names[top3[0]]}=${dollar_alloc[top3[0]]:,.0f}, "
                  f"{channel_names[top3[1]]}=${dollar_alloc[top3[1]]:,.0f}, "
                  f"{channel_names[top3[2]]}=${dollar_alloc[top3[2]]:,.0f}")

        history.append({"step": step, "predicted_cw": cw_val})

    # Re-enable model gradients
    for p in model.parameters():
        p.requires_grad_(True)

    # Build final allocation
    optimal_dollars = (best_alloc.numpy() * config.weekly_budget)

    return {
        "optimal_allocation": {
            ch: float(optimal_dollars[i])
            for i, ch in enumerate(channel_names)
        },
        "optimal_proportions": {
            ch: float(best_alloc[i])
            for i, ch in enumerate(channel_names)
        },
        "predicted_closed_won": best_cw,
        "budget": config.weekly_budget,
        "history": history,
    }


# =========================================================================
# Reporting
# =========================================================================

def print_budget_shift_table(
    current_spend: np.ndarray,
    optimal: Dict,
    channel_names: List[str],
    sensitivity: Dict,
) -> pd.DataFrame:
    """Print and return a comparison table: Current vs. NNN-Optimized allocation."""
    rows = []
    current_total = current_spend.sum()
    optimal_total = sum(optimal["optimal_allocation"].values())

    for i, ch in enumerate(channel_names):
        curr = current_spend[i]
        curr_pct = curr / (current_total + 1e-8) * 100
        opt = optimal["optimal_allocation"][ch]
        opt_pct = opt / (optimal_total + 1e-8) * 100
        delta = opt - curr
        delta_pct = delta / (curr + 1e-8) * 100

        sens = sensitivity["channels"].get(ch, {})
        mroi = sens.get("marginal_roi", 0)
        sat = sens.get("saturated", False)

        rows.append({
            "Channel": ch,
            "Current $/wk": curr,
            "Current %": curr_pct,
            "Optimized $/wk": opt,
            "Optimized %": opt_pct,
            "Delta $": delta,
            "Delta %": delta_pct,
            "Marginal ROI": mroi,
            "Saturated": sat,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Delta $", ascending=False)

    print(f"\n{'='*110}")
    print(f"  RECOMMENDED BUDGET SHIFT: Current vs. NNN-Optimized (${optimal_total:,.0f}/week)")
    print(f"{'='*110}")
    print(f"  {'Channel':20s} {'Current':>12s} {'Curr%':>7s}  {'Optimized':>12s} {'Opt%':>7s}  {'Delta':>12s} {'Shift':>8s}  {'mROI':>8s} {'Sat?':>5s}")
    print(f"  {'-'*20} {'-'*12} {'-'*7}  {'-'*12} {'-'*7}  {'-'*12} {'-'*8}  {'-'*8} {'-'*5}")

    for _, r in df.iterrows():
        arrow = "+" if r["Delta $"] > 0 else ""
        sat_str = "YES" if r["Saturated"] else ""
        print(
            f"  {r['Channel']:20s} "
            f"${r['Current $/wk']:>11,.0f} {r['Current %']:>6.1f}%  "
            f"${r['Optimized $/wk']:>11,.0f} {r['Optimized %']:>6.1f}%  "
            f"{arrow}${abs(r['Delta $']):>10,.0f} {arrow}{r['Delta %']:>6.1f}%  "
            f"{r['Marginal ROI']:>8.5f} {sat_str:>5s}"
        )

    print(f"  {'-'*20} {'-'*12} {'-'*7}  {'-'*12} {'-'*7}  {'-'*12} {'-'*8}  {'-'*8} {'-'*5}")
    totals = df[["Current $/wk", "Optimized $/wk"]].sum()
    print(
        f"  {'TOTAL':20s} "
        f"${totals['Current $/wk']:>11,.0f} {'100.0':>6s}%  "
        f"${totals['Optimized $/wk']:>11,.0f} {'100.0':>6s}%"
    )

    return df


def print_response_curves(sensitivity: Dict, channel_names: List[str]):
    """Print ASCII response curves for each channel."""
    print(f"\n{'='*80}")
    print(f"  SPEND RESPONSE CURVES (Closed-Won vs. Spend Multiplier)")
    print(f"{'='*80}")

    baseline = sensitivity["baseline_total_cw"]

    # Sort channels by marginal ROI
    sorted_channels = sorted(
        channel_names,
        key=lambda ch: sensitivity["channels"][ch]["marginal_roi"],
        reverse=True,
    )

    for ch in sorted_channels:
        info = sensitivity["channels"][ch]
        curve = info["curve"]
        vals = [p["predicted_closed_won"] for p in curve]
        min_v, max_v = min(vals), max(vals)
        spread = max_v - min_v

        sat_tag = " [SATURATED]" if info["saturated"] else ""
        print(f"\n  {ch} (mROI: {info['marginal_roi']:.5f}, "
              f"current: ${info['current_spend']:,.0f}/wk){sat_tag}")

        bar_width = 40
        for p in curve[::2]:  # every other point to save space
            mult = p["multiplier"]
            cw = p["predicted_closed_won"]
            if spread > 0:
                filled = int((cw - min_v) / spread * bar_width)
            else:
                filled = bar_width // 2
            bar = "#" * filled + "." * (bar_width - filled)
            marker = " <-- current" if abs(mult - 1.0) < 0.06 else ""
            print(f"    {mult:.2f}x  |{bar}| {cw:>8,.0f} CW{marker}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="NNN Budget Optimizer")
    parser.add_argument("--budget", type=float, default=1_000_000,
                        help="Fixed weekly budget in dollars")
    parser.add_argument("--steps", type=int, default=800,
                        help="Gradient descent optimization steps")
    args = parser.parse_args()

    config = OptimizerConfig(
        weekly_budget=args.budget,
        optim_steps=args.steps,
    )

    print("=" * 70)
    print("NNN Budget Optimizer - Synergy-Aware Allocation")
    print("=" * 70)
    print(f"  Weekly budget: ${config.weekly_budget:,.0f}")
    print(f"  Optimization steps: {config.optim_steps}")
    print(f"  Baseline window: last {config.baseline_weeks} weeks")
    print()

    # --- Load ---
    print("Loading trained model and data...")
    bundle = load_model_and_data(config)
    print(f"  Model loaded: {bundle['n_channels']} channels, "
          f"input_dim={bundle['input_dim']}")
    print(f"  Current total spend (avg/wk): "
          f"${bundle['current_spend'].sum():,.0f}")
    print()

    # --- Sensitivity Analysis ---
    print("Running Spend Sensitivity Analysis...")
    sensitivity = run_sensitivity(bundle, config)
    print_response_curves(sensitivity, bundle["channel_names"])

    # --- Budget Optimization ---
    print(f"\n{'='*70}")
    print("Running Synergy-Aware Budget Optimization...")
    print(f"{'='*70}")
    optimal = optimize_budget(bundle, config)

    # --- Results ---
    table_df = print_budget_shift_table(
        bundle["current_spend"],
        optimal,
        bundle["channel_names"],
        sensitivity,
    )

    # Summary insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")

    baseline_cw = sensitivity["baseline_total_cw"]
    optimal_cw = optimal["predicted_closed_won"]
    uplift = (optimal_cw - baseline_cw) / (baseline_cw + 1e-8) * 100

    print(f"  Baseline Closed-Won (last {config.baseline_weeks}wk): {baseline_cw:,.0f}")
    print(f"  Optimized Closed-Won (predicted):    {optimal_cw:,.0f}")
    print(f"  Potential uplift:                     {uplift:+.1f}%")

    # Top increases / decreases
    table_sorted = table_df.sort_values("Delta $")
    top_cuts = table_sorted.head(3)
    top_adds = table_sorted.tail(3).iloc[::-1]

    print(f"\n  INCREASE spend on:")
    for _, r in top_adds.iterrows():
        if r["Delta $"] > 0:
            print(f"    {r['Channel']:20s}  +${r['Delta $']:>10,.0f}/wk  "
                  f"(mROI: {r['Marginal ROI']:.5f})")

    print(f"\n  DECREASE spend on:")
    for _, r in top_cuts.iterrows():
        if r["Delta $"] < 0:
            print(f"    {r['Channel']:20s}  -${abs(r['Delta $']):>10,.0f}/wk  "
                  f"({'saturated' if r['Saturated'] else 'low mROI'})")

    # Synergy note
    print(f"\n  SYNERGY ALERT:")
    probe_path = Path(__file__).parent / "output" / "probe_results.json"
    if probe_path.exists():
        with open(probe_path) as f:
            probe = json.load(f)
        top_syn = probe.get("synergy_top", [])[:3]
        for s in top_syn:
            src, tgt = s["source"], s["target"]
            src_delta = table_df.loc[table_df["Channel"] == src, "Delta $"]
            if len(src_delta) > 0 and src_delta.values[0] < -1000:
                print(f"    WARNING: Cutting {src} may hurt {tgt} "
                      f"(synergy weight: {s['weight']:.4f})")

    # --- Save ---
    out_dir = Path(__file__).parent / "output"

    with open(out_dir / "optimization_results.json", "w") as f:
        json.dump({
            "config": {"budget": config.weekly_budget, "steps": config.optim_steps},
            "optimal_allocation": optimal["optimal_allocation"],
            "optimal_proportions": optimal["optimal_proportions"],
            "predicted_closed_won": optimal["predicted_closed_won"],
            "baseline_closed_won": baseline_cw,
            "uplift_pct": uplift,
            "sensitivity": {
                ch: {
                    "marginal_roi": float(info["marginal_roi"]),
                    "saturated": bool(info["saturated"]),
                    "saturation_ratio": float(info["saturation_ratio"]),
                    "current_spend": float(info["current_spend"]),
                }
                for ch, info in sensitivity["channels"].items()
            },
        }, f, indent=2)

    table_df.to_csv(out_dir / "budget_shift_table.csv", index=False)

    print(f"\nSaved to {out_dir}/:")
    print(f"  optimization_results.json  — full results + sensitivity")
    print(f"  budget_shift_table.csv     — current vs optimized allocation")
    print("\nDone.")


if __name__ == "__main__":
    main()
