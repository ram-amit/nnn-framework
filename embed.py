"""
embed.py — Text embedding pipeline for the NNN marketing measurement framework.

Converts Campaign_Metadata text (ad copy, campaign names, audience targeting)
into high-dimensional vectors using OpenAI's text-embedding-3-small API.

Features:
  - Batched API calls (configurable batch size)
  - Exponential backoff with jitter for rate-limit handling
  - Disk-based caching to avoid re-embedding identical text
  - Deterministic hashing for cache keys
  - Graceful fallback to zero vectors when API is unavailable

Usage:
  from embed import EmbeddingPipeline, EmbeddingConfig
  pipeline = EmbeddingPipeline(EmbeddingConfig())
  embeddings = pipeline.embed_dataframe(df, text_col="Campaign_Metadata")
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 256
    batch_size: int = 64
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    cache_dir: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.cache_dir is None:
            self.cache_dir = str(
                Path(__file__).parent / "data" / ".embedding_cache"
            )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingPipeline:

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._client = None
        self._cache: dict = {}
        self._cache_loaded = False

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError(
                    "openai package required: pip install openai"
                )
        return self._client

    def _load_cache(self) -> None:
        if self._cache_loaded:
            return
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        index_file = cache_path / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                self._cache = json.load(f)
            logger.info("Loaded %d cached embeddings", len(self._cache))
        self._cache_loaded = True

    def _save_cache(self) -> None:
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        index_file = cache_path / "index.json"
        with open(index_file, "w") as f:
            json.dump(self._cache, f)

    def _get_cached(self, text: str) -> Optional[List[float]]:
        self._load_cache()
        key = _text_hash(text)
        entry = self._cache.get(key)
        if entry is None:
            return None
        npy_path = Path(self.config.cache_dir) / f"{key}.npy"
        if npy_path.exists():
            return np.load(npy_path).tolist()
        return None

    def _set_cached(self, text: str, embedding: List[float]) -> None:
        key = _text_hash(text)
        npy_path = Path(self.config.cache_dir) / f"{key}.npy"
        np.save(npy_path, np.array(embedding, dtype=np.float32))
        self._cache[key] = {
            "model": self.config.model,
            "dims": self.config.dimensions,
            "text_preview": text[:80],
        }

    def _embed_batch_api(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embedding API with exponential backoff retry."""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts,
                    dimensions=self.config.dimensions,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                last_error = e
                error_name = type(e).__name__

                if "AuthenticationError" in error_name:
                    raise

                delay = min(
                    self.config.base_delay * (2 ** attempt)
                    + np.random.uniform(0, 1),
                    self.config.max_delay,
                )
                logger.warning(
                    "Embedding API attempt %d/%d failed (%s: %s), "
                    "retrying in %.1fs",
                    attempt + 1,
                    self.config.max_retries,
                    error_name,
                    str(e)[:120],
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Embedding API failed after {self.config.max_retries} attempts: "
            f"{last_error}"
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts, using cache where available.

        Returns:
            np.ndarray of shape (len(texts), config.dimensions)
        """
        self._load_cache()
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        n_cached = len(texts) - len(uncached_indices)
        if n_cached > 0:
            logger.info(
                "Cache hit: %d/%d texts, embedding %d new",
                n_cached, len(texts), len(uncached_indices),
            )

        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]

            for batch_start in range(0, len(uncached_texts), self.config.batch_size):
                batch_end = min(
                    batch_start + self.config.batch_size, len(uncached_texts)
                )
                batch = uncached_texts[batch_start:batch_end]
                batch_indices = uncached_indices[batch_start:batch_end]

                logger.info(
                    "Embedding batch %d-%d of %d",
                    batch_start, batch_end, len(uncached_texts),
                )
                embeddings = self._embed_batch_api(batch)

                for idx, text, emb in zip(batch_indices, batch, embeddings):
                    results[idx] = emb
                    self._set_cached(text, emb)

            self._save_cache()

        return np.array(results, dtype=np.float32)

    def embed_dataframe(
        self,
        df,
        text_col: str = "Campaign_Metadata",
    ) -> np.ndarray:
        """
        Embed the text column of a DataFrame.

        Deduplicates texts before calling the API to minimize cost,
        then maps results back to the original row order.

        Returns:
            np.ndarray of shape (len(df), config.dimensions)
        """
        texts = df[text_col].fillna("").astype(str).tolist()

        unique_texts = list(dict.fromkeys(texts))
        logger.info(
            "%d rows, %d unique texts to embed", len(texts), len(unique_texts)
        )
        unique_embeddings = self.embed_texts(unique_texts)

        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        result = np.zeros(
            (len(texts), self.config.dimensions), dtype=np.float32
        )
        for i, text in enumerate(texts):
            result[i] = unique_embeddings[text_to_idx[text]]

        return result

    def embed_dataframe_offline(
        self,
        df,
        text_col: str = "Campaign_Metadata",
    ) -> np.ndarray:
        """
        Fallback: generate deterministic pseudo-embeddings from text hashes.
        Use when the OpenAI API is unavailable (local dev, CI, tests).

        Produces vectors that are consistent for identical text but carry
        no real semantic meaning. Shape matches the API output exactly.
        """
        texts = df[text_col].fillna("").astype(str).tolist()
        result = np.zeros(
            (len(texts), self.config.dimensions), dtype=np.float32
        )

        for i, text in enumerate(texts):
            seed = int(_text_hash(text), 16) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.config.dimensions).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            result[i] = vec

        return result


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    csv_path = Path(__file__).parent / "data" / "sample_marketing_data.csv"
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique texts: {df['Campaign_Metadata'].nunique()}")
    print()

    config = EmbeddingConfig(dimensions=256)
    pipeline = EmbeddingPipeline(config)

    has_key = bool(config.api_key)
    print(f"OPENAI_API_KEY set: {has_key}")

    if has_key:
        print("Running live API embedding...")
        embeddings = pipeline.embed_dataframe(df)
    else:
        print("No API key found. Running offline (hash-based) embedding...")
        embeddings = pipeline.embed_dataframe_offline(df)

    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    print(f"Sample norms: {np.linalg.norm(embeddings[:3], axis=1)}")
    print(f"Sample cosine sim (row 0 vs 1): "
          f"{np.dot(embeddings[0], embeddings[1]):.4f}")
    print(f"Sample cosine sim (row 0 vs 5): "
          f"{np.dot(embeddings[0], embeddings[5]):.4f}")

    out_path = Path(__file__).parent / "data" / "embeddings.npy"
    np.save(out_path, embeddings)
    print(f"\nSaved embeddings to {out_path}")
