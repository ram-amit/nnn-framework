"""
data_prep.py — Data preparation for the NNN marketing measurement framework.

Two data paths:
  1. Synthetic: generates fake B2B SaaS funnel data for development/testing
  2. Real: ingests CSV exports from a data warehouse, runs text through the
     embedding pipeline (embed.py), and constructs the Rank-4 tensor

Both paths produce the same output shape:
  (Geographies, Time, Channels, Dimensions)
where Dimensions = numeric_features + embedding_dims.
"""

import logging
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FunnelConfig:
    """Configuration for synthetic B2B SaaS funnel data."""

    geos: list = field(default_factory=lambda: ["US", "UK", "DE", "FR", "AU"])
    n_weeks: int = 104
    channels: list = field(
        default_factory=lambda: [
            "linkedin",
            "google_search",
            "google_display",
            "facebook",
            "content_syndication",
            "organic",
        ]
    )
    numeric_features: list = field(
        default_factory=lambda: ["spend", "impressions", "clicks"]
    )
    embedding_dim: int = 64
    seed: int = 42

    # Funnel conversion rates (base rates, vary by geo)
    base_branded_search_rate: float = 0.15
    base_trial_rate: float = 0.04
    base_close_rate: float = 0.12

    # Adstock decay rates per channel
    adstock_decay: dict = field(
        default_factory=lambda: {
            "linkedin": 0.7,
            "google_search": 0.3,
            "google_display": 0.6,
            "facebook": 0.5,
            "content_syndication": 0.8,
            "organic": 0.9,
        }
    )

    # Saturation half-points (Hill function)
    saturation_k: dict = field(
        default_factory=lambda: {
            "linkedin": 50000,
            "google_search": 30000,
            "google_display": 40000,
            "facebook": 35000,
            "content_syndication": 20000,
            "organic": 10000,
        }
    )

    # Cross-channel synergy matrix (row channel boosts col channel)
    # e.g., linkedin spend boosts google_search effectiveness
    synergy_matrix: Optional[np.ndarray] = None

    @property
    def n_geos(self) -> int:
        return len(self.geos)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def input_dim(self) -> int:
        return len(self.numeric_features) + self.embedding_dim

    def __post_init__(self):
        if self.synergy_matrix is None:
            n = self.n_channels
            self.synergy_matrix = np.eye(n)
            ch = {name: i for i, name in enumerate(self.channels)}
            # LinkedIn boosts branded search
            self.synergy_matrix[ch["linkedin"], ch["google_search"]] = 0.25
            # Content syndication boosts LinkedIn and branded search
            self.synergy_matrix[ch["content_syndication"], ch["linkedin"]] = 0.15
            self.synergy_matrix[ch["content_syndication"], ch["google_search"]] = 0.10
            # Display boosts branded search (awareness → search)
            self.synergy_matrix[ch["google_display"], ch["google_search"]] = 0.20
            # Facebook boosts display and branded search
            self.synergy_matrix[ch["facebook"], ch["google_search"]] = 0.12
            self.synergy_matrix[ch["facebook"], ch["google_display"]] = 0.08


def _hill_saturation(x: np.ndarray, k: float, n: float = 2.0) -> np.ndarray:
    return x**n / (k**n + x**n)


def _apply_adstock(series: np.ndarray, decay: float) -> np.ndarray:
    result = np.zeros_like(series)
    result[0] = series[0]
    for t in range(1, len(series)):
        result[t] = series[t] + decay * result[t - 1]
    return result


def generate_synthetic_embeddings(
    n_geos: int, n_weeks: int, n_channels: int, embed_dim: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Simulate campaign metadata embeddings.
    In production these would come from Llama-3 / Vertex AI encoding of
    creative text, targeting criteria, and search intent signals.

    Each channel gets a base embedding that drifts slowly over time
    (simulating creative refreshes) and varies slightly by geo.
    """
    base = rng.standard_normal((n_channels, embed_dim))
    base = base / np.linalg.norm(base, axis=1, keepdims=True)

    embeddings = np.zeros((n_geos, n_weeks, n_channels, embed_dim))
    for g in range(n_geos):
        geo_shift = rng.standard_normal(embed_dim) * 0.1
        for c in range(n_channels):
            drift = np.cumsum(rng.standard_normal((n_weeks, embed_dim)) * 0.01, axis=0)
            embeddings[g, :, c, :] = base[c] + geo_shift + drift

    return embeddings


def generate_synthetic_data(config: FunnelConfig) -> dict:
    """
    Generate a full synthetic dataset mimicking a B2B SaaS funnel.

    Returns dict with:
        tensor: np.ndarray of shape (n_geos, n_weeks, n_channels, input_dim)
        targets: dict of np.ndarray, each (n_geos, n_weeks)
            - branded_search_volume
            - trial_signups
            - closed_won
        config: the FunnelConfig used
        raw_spend: np.ndarray (n_geos, n_weeks, n_channels) — pre-transform spend
    """
    rng = np.random.default_rng(config.seed)
    G, T, C = config.n_geos, config.n_weeks, config.n_channels

    # --- Generate raw spend patterns ---
    spend = np.zeros((G, T, C))
    for g in range(G):
        geo_scale = 0.5 + rng.random() * 1.5  # geo budget multiplier
        for c, ch_name in enumerate(config.channels):
            if ch_name == "organic":
                spend[g, :, c] = 0.0  # organic has no spend
                continue
            base_weekly = rng.uniform(5000, 80000) * geo_scale
            # Seasonality (quarterly patterns common in B2B)
            weeks = np.arange(T)
            seasonal = 1 + 0.2 * np.sin(2 * np.pi * weeks / 13)
            # Trend (budget growth)
            trend = 1 + 0.003 * weeks
            noise = rng.normal(1, 0.1, T).clip(0.5, 1.5)
            spend[g, :, c] = base_weekly * seasonal * trend * noise

    # --- Derived metrics: impressions, clicks ---
    cpm = rng.uniform(5, 50, (G, C))  # cost per 1000 impressions
    ctr = rng.uniform(0.005, 0.05, (G, C))

    impressions = np.zeros((G, T, C))
    clicks = np.zeros((G, T, C))
    for g in range(G):
        for c, ch_name in enumerate(config.channels):
            if ch_name == "organic":
                impressions[g, :, c] = rng.uniform(50000, 200000) * (
                    1 + 0.005 * np.arange(T)
                )
                clicks[g, :, c] = impressions[g, :, c] * rng.uniform(0.02, 0.08)
                continue
            impressions[g, :, c] = spend[g, :, c] / cpm[g, c] * 1000
            clicks[g, :, c] = impressions[g, :, c] * ctr[g, c]

    # --- Apply adstock to spend ---
    adstocked_spend = np.zeros_like(spend)
    for g in range(G):
        for c, ch_name in enumerate(config.channels):
            decay = config.adstock_decay[ch_name]
            adstocked_spend[g, :, c] = _apply_adstock(spend[g, :, c], decay)

    # --- Apply saturation ---
    saturated = np.zeros_like(adstocked_spend)
    for c, ch_name in enumerate(config.channels):
        k = config.saturation_k[ch_name]
        saturated[:, :, c] = _hill_saturation(adstocked_spend[:, :, c], k)

    # --- Apply cross-channel synergies ---
    synergized = np.zeros_like(saturated)
    for g in range(G):
        for t in range(T):
            channel_vec = saturated[g, t, :]
            synergized[g, t, :] = channel_vec @ config.synergy_matrix

    # --- Generate funnel targets ---
    geo_rates = rng.uniform(0.7, 1.3, (G, 1))

    # Branded search volume (driven by all channels, especially LinkedIn + display)
    channel_weights_search = np.array([0.30, 0.05, 0.25, 0.15, 0.10, 0.15])
    branded_search = np.zeros((G, T))
    for g in range(G):
        base = (synergized[g] * channel_weights_search).sum(axis=1)
        branded_search[g] = (
            base
            * config.base_branded_search_rate
            * geo_rates[g]
            * 10000
        )
        branded_search[g] += rng.normal(0, branded_search[g].std() * 0.1, T)
        branded_search[g] = np.maximum(branded_search[g], 0)

    # Trial signups (driven by branded search + direct channel effects)
    trial_signups = np.zeros((G, T))
    for g in range(G):
        direct_effect = (synergized[g] * np.array([0.10, 0.35, 0.05, 0.10, 0.20, 0.20])).sum(axis=1)
        search_effect = _apply_adstock(branded_search[g] / branded_search[g].max(), 0.3)
        trial_signups[g] = (
            (0.4 * direct_effect + 0.6 * search_effect)
            * config.base_trial_rate
            * geo_rates[g]
            * 500
        )
        trial_signups[g] += rng.normal(0, trial_signups[g].std() * 0.15, T)
        trial_signups[g] = np.maximum(trial_signups[g], 0)

    # Closed won (driven by trials with lag + some direct channel influence)
    closed_won = np.zeros((G, T))
    for g in range(G):
        lagged_trials = np.zeros(T)
        lagged_trials[4:] = trial_signups[g, :-4]  # 4-week sales cycle
        lagged_trials = _apply_adstock(lagged_trials, 0.4)
        direct = (synergized[g] * np.array([0.05, 0.20, 0.02, 0.03, 0.05, 0.10])).sum(axis=1)
        closed_won[g] = (
            (0.7 * lagged_trials / (lagged_trials.max() + 1e-8) + 0.3 * direct)
            * config.base_close_rate
            * geo_rates[g]
            * 100
        )
        closed_won[g] += rng.normal(0, closed_won[g].std() * 0.2, T)
        closed_won[g] = np.maximum(closed_won[g], 0)

    # --- Generate embeddings ---
    embeddings = generate_synthetic_embeddings(G, T, C, config.embedding_dim, rng)

    # --- Assemble Rank-4 Tensor ---
    # Normalize numeric features
    spend_norm = spend / (spend.max() + 1e-8)
    imp_norm = impressions / (impressions.max() + 1e-8)
    clicks_norm = clicks / (clicks.max() + 1e-8)

    numeric = np.stack([spend_norm, imp_norm, clicks_norm], axis=-1)  # (G, T, C, 3)
    tensor = np.concatenate([numeric, embeddings], axis=-1)  # (G, T, C, 3 + embed_dim)

    return {
        "tensor": tensor,
        "targets": {
            "branded_search_volume": branded_search,
            "trial_signups": trial_signups,
            "closed_won": closed_won,
        },
        "config": config,
        "raw_spend": spend,
    }


def csv_to_tensor(
    df: pd.DataFrame,
    geo_col: str = "geo",
    time_col: str = "week",
    channel_col: str = "channel",
    numeric_cols: Optional[list] = None,
    embedding_cols: Optional[list] = None,
) -> torch.Tensor:
    """
    Transform a long-format CSV/DataFrame into the Rank-4 Tensor.
    Used by the synthetic data path where embedding columns are already in the DF.

    Returns:
        torch.Tensor of shape (n_geos, n_time_steps, n_channels, n_dimensions)
    """
    if numeric_cols is None:
        numeric_cols = ["spend", "impressions", "clicks"]
    if embedding_cols is None:
        embedding_cols = [c for c in df.columns if c.startswith("emb_")]

    feature_cols = numeric_cols + embedding_cols

    geos = sorted(df[geo_col].unique())
    times = sorted(df[time_col].unique())
    channels = sorted(df[channel_col].unique())

    geo_idx = {g: i for i, g in enumerate(geos)}
    time_idx = {t: i for i, t in enumerate(times)}
    chan_idx = {c: i for i, c in enumerate(channels)}

    tensor = np.zeros((len(geos), len(times), len(channels), len(feature_cols)))

    for _, row in df.iterrows():
        g = geo_idx[row[geo_col]]
        t = time_idx[row[time_col]]
        c = chan_idx[row[channel_col]]
        tensor[g, t, c, :] = row[feature_cols].values.astype(float)

    return torch.tensor(tensor, dtype=torch.float32)


# =========================================================================
# Real-data ingestion path
# =========================================================================

@dataclass
class RealDataConfig:
    """Configuration for loading real marketing CSV data."""

    csv_path: str = "data/sample_marketing_data.csv"
    date_col: str = "Date"
    geo_col: str = "Geography"
    channel_col: str = "Channel"
    spend_col: str = "Spend"
    text_col: str = "Campaign_Metadata"
    target_cols: List[str] = field(
        default_factory=lambda: ["Enterprise_Trials", "Closed_Won"]
    )
    numeric_features: List[str] = field(
        default_factory=lambda: ["Spend", "Enterprise_Trials"]
    )
    embedding_dim: int = 256
    use_api_embeddings: bool = False

    @property
    def input_dim(self) -> int:
        return len(self.numeric_features) + self.embedding_dim


def load_real_csv(config: RealDataConfig) -> pd.DataFrame:
    """
    Load and validate a real marketing CSV.

    Expected schema:
        Date, Geography, Channel, Spend, Enterprise_Trials, Closed_Won, Campaign_Metadata

    Returns a cleaned DataFrame with Date parsed and sorted.
    """
    path = Path(config.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Marketing data CSV not found: {path}")

    df = pd.read_csv(path, keep_default_na=False, na_values=[""])

    required = [
        config.date_col,
        config.geo_col,
        config.channel_col,
        config.spend_col,
        config.text_col,
    ] + config.target_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[config.date_col] = pd.to_datetime(df[config.date_col])
    df = df.sort_values([config.geo_col, config.date_col, config.channel_col])
    df = df.reset_index(drop=True)

    logger.info(
        "Loaded %d rows: %d geos, %d weeks, %d channels",
        len(df),
        df[config.geo_col].nunique(),
        df[config.date_col].nunique(),
        df[config.channel_col].nunique(),
    )
    return df


def build_real_tensor(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    config: RealDataConfig,
) -> Dict:
    """
    Construct the Rank-4 Tensor from a real marketing DataFrame + embeddings.

    Steps:
      1. Normalize numeric features (per-feature min-max scaling)
      2. Concatenate [normalized_numerics, embeddings] along the last axis
      3. Pivot into shape (Geographies, Time, Channels, Dimensions)
      4. Extract targets aggregated per (Geo, Time)

    Args:
        df: marketing DataFrame with Date, Geography, Channel, etc.
        embeddings: np.ndarray of shape (len(df), embedding_dim)
        config: RealDataConfig

    Returns:
        dict with keys:
            tensor: np.ndarray (G, T, C, D)
            targets: dict of np.ndarray, each (G, T)
            metadata: dict with geos, dates, channels, input_dim
    """
    assert len(embeddings) == len(df), (
        f"Embedding rows ({len(embeddings)}) must match DataFrame rows ({len(df)})"
    )

    geos = sorted(df[config.geo_col].unique())
    dates = sorted(df[config.date_col].unique())
    channels = sorted(df[config.channel_col].unique())

    geo_idx = {g: i for i, g in enumerate(geos)}
    date_idx = {d: i for i, d in enumerate(dates)}
    chan_idx = {c: i for i, c in enumerate(channels)}

    G, T, C = len(geos), len(dates), len(channels)
    n_numeric = len(config.numeric_features)
    embed_dim = embeddings.shape[1]
    D = n_numeric + embed_dim

    # Normalize numeric features (min-max per feature)
    numeric_values = df[config.numeric_features].values.astype(np.float64)
    feat_min = numeric_values.min(axis=0)
    feat_max = numeric_values.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range == 0] = 1.0
    numeric_norm = (numeric_values - feat_min) / feat_range

    # Build the tensor
    tensor = np.zeros((G, T, C, D), dtype=np.float32)
    for row_i, (_, row) in enumerate(df.iterrows()):
        g = geo_idx[row[config.geo_col]]
        t = date_idx[row[config.date_col]]
        c = chan_idx[row[config.channel_col]]
        tensor[g, t, c, :n_numeric] = numeric_norm[row_i]
        tensor[g, t, c, n_numeric:] = embeddings[row_i]

    # Build targets: aggregate per (geo, week)
    targets = {}
    for target_col in config.target_cols:
        target_arr = np.zeros((G, T), dtype=np.float32)
        grouped = df.groupby([config.geo_col, config.date_col])[target_col].sum()
        for (geo, date), val in grouped.items():
            g = geo_idx[geo]
            t = date_idx[date]
            target_arr[g, t] = val
        targets[target_col] = target_arr

    return {
        "tensor": tensor,
        "targets": targets,
        "metadata": {
            "geos": geos,
            "dates": [str(d) for d in dates],
            "channels": channels,
            "input_dim": D,
            "n_numeric": n_numeric,
            "embed_dim": embed_dim,
            "normalization": {
                "features": config.numeric_features,
                "min": feat_min.tolist(),
                "max": feat_max.tolist(),
            },
        },
    }


def prepare_real_data(config: Optional[RealDataConfig] = None) -> Dict:
    """
    End-to-end pipeline: CSV -> embeddings -> Rank-4 Tensor.

    Set config.use_api_embeddings=True to call OpenAI, otherwise uses
    deterministic hash-based pseudo-embeddings (safe for local dev).
    """
    from embed import EmbeddingConfig, EmbeddingPipeline

    if config is None:
        config = RealDataConfig()

    df = load_real_csv(config)

    embed_config = EmbeddingConfig(dimensions=config.embedding_dim)
    pipeline = EmbeddingPipeline(embed_config)

    if config.use_api_embeddings:
        logger.info("Embedding Campaign_Metadata via OpenAI API...")
        embeddings = pipeline.embed_dataframe(df, text_col=config.text_col)
    else:
        logger.info("Using offline (hash-based) embeddings for local dev")
        embeddings = pipeline.embed_dataframe_offline(df, text_col=config.text_col)

    result = build_real_tensor(df, embeddings, config)
    return result


def tensor_to_dataloader(
    tensor: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 4,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """Wrap tensor + targets into a DataLoader. Geo dimension becomes batch."""
    X = torch.tensor(tensor, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    if y.dim() == 2:
        y = y.unsqueeze(-1)  # (G, T) → (G, T, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("synthetic", "both"):
        print("=" * 60)
        print("PATH 1: Synthetic Data")
        print("=" * 60)
        config = FunnelConfig()
        data = generate_synthetic_data(config)

        print(f"Tensor shape: {data['tensor'].shape}")
        print(f"  Geos:       {config.n_geos} ({', '.join(config.geos)})")
        print(f"  Weeks:      {config.n_weeks}")
        print(f"  Channels:   {config.n_channels} ({', '.join(config.channels)})")
        print(f"  Dimensions: {config.input_dim} "
              f"({len(config.numeric_features)} numeric + {config.embedding_dim} embedding)")
        for name, arr in data["targets"].items():
            print(f"  Target '{name}': mean={arr.mean():.2f}, std={arr.std():.2f}")
        print()

    if mode in ("real", "both"):
        print("=" * 60)
        print("PATH 2: Real Data (sample CSV + embeddings)")
        print("=" * 60)
        real_config = RealDataConfig(
            csv_path=str(Path(__file__).parent / "data" / "sample_marketing_data.csv"),
        )
        real_data = prepare_real_data(real_config)

        t = real_data["tensor"]
        m = real_data["metadata"]
        print(f"Tensor shape: {t.shape}")
        print(f"  Geos:       {len(m['geos'])} ({', '.join(m['geos'])})")
        print(f"  Weeks:      {len(m['dates'])}")
        print(f"  Channels:   {len(m['channels'])} ({', '.join(m['channels'])})")
        print(f"  Dimensions: {m['input_dim']} "
              f"({m['n_numeric']} numeric + {m['embed_dim']} embedding)")
        print()
        for tgt_name, tgt_arr in real_data["targets"].items():
            print(f"  Target '{tgt_name}': shape={tgt_arr.shape}, "
                  f"mean={tgt_arr.mean():.2f}, sum={tgt_arr.sum():.0f}")

        # Verify tensor is compatible with model.py
        print(f"\n  Tensor dtype: {t.dtype}")
        print(f"  Any NaN: {np.isnan(t).any()}")
        print(f"  Value range: [{t.min():.4f}, {t.max():.4f}]")
        print(f"  Ready for NNNModel(n_channels={len(m['channels'])}, "
              f"input_dim={m['input_dim']})")
        print()
