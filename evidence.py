"""Frozen text encoding and construction of intent bridge B and matching S.

Use only training Mashups as reference history. Training queries must be in
the same row order as that history and set train_targets=True to exclude
self-retrieval. Embeddings passed to build_evidence() must be L2-normalized.
The API columns of train_y and rows of api_embeddings must have identical order.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def encode_texts(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> np.ndarray:
    """Encode descriptions with frozen, L2-normalized sentence embeddings.

    Requires sentence-transformers. Model weights are downloaded by that
    library on first use when they are not already cached.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name, device=device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder.encode(
        list(texts),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def build_evidence(
    target_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_y: np.ndarray,
    api_embeddings: np.ndarray,
    neighbors: int = 10,
    temperature: float = 0.1,
    train_targets: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return B, S, and retrieved history indices; never read query labels."""
    if neighbors < 1 or temperature <= 0:
        raise ValueError("neighbors and temperature must be positive")
    similarities = target_embeddings @ train_embeddings.T
    if train_targets:
        if len(target_embeddings) != len(train_embeddings):
            raise ValueError("Training targets must align with the train embeddings.")
        np.fill_diagonal(similarities, -np.inf)

    k = min(neighbors, similarities.shape[1] - int(train_targets))
    if k < 1:
        raise ValueError("At least one eligible historical neighbor is required")
    neighbor_ids = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    neighbor_sims = np.take_along_axis(similarities, neighbor_ids, axis=1)
    order = np.argsort(-neighbor_sims, axis=1)
    neighbor_ids = np.take_along_axis(neighbor_ids, order, axis=1)
    neighbor_sims = np.take_along_axis(neighbor_sims, order, axis=1)

    logits = neighbor_sims / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)

    neighbor_y = train_y[neighbor_ids]
    bridge = np.einsum("nk,nkm->nm", weights, neighbor_y).astype(np.float32)
    functional = (target_embeddings @ api_embeddings.T).astype(np.float32)
    return bridge, functional, neighbor_ids.astype(np.int64)
