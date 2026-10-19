"""Paper training objective and inference score recovery.

Compute loss on r_0 = y - B with weighted reconstruction and pairwise ranking.
Provide one observed positive and 20 unobserved negatives per query, sampled
without replacement from the candidate API set. This module contains model
operations only; dataset loading, experiment orchestration and metrics are
intentionally outside this release.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def loss_weights(y: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
    return 0.10 + 0.90 * ((y > 0) | (bridge > 1e-7)).float() + 1.50 * y


def training_loss(
    model,
    y: torch.Tensor,
    bridge: torch.Tensor,
    functional: torch.Tensor,
    time: torch.Tensor,
    noise: torch.Tensor,
    positive_ids: torch.Tensor,
    negative_ids: torch.Tensor,
):
    """Return total loss, weighted reconstruction, and pairwise ranking loss.

    y, bridge, functional, noise: [batch, number of APIs].
    time: [batch] integer indices in [0, T - 1].
    positive_ids: [batch, 1]; negative_ids: [batch, number of negatives].
    """
    clean = y - bridge
    state = model.q_sample(clean, time, noise)
    prediction = model.denoiser(
        state, time, bridge, functional, bridge.new_zeros((len(bridge), 4))
    )
    weights = loss_weights(y, bridge)
    reconstruction = ((prediction - clean).square() * weights).sum() / weights.sum()
    scores = bridge + prediction
    ranking = F.softplus(
        scores.gather(1, negative_ids) - scores.gather(1, positive_ids)
    ).mean()
    return reconstruction + 0.2 * ranking, reconstruction, ranking


@torch.no_grad()
def predict_scores(
    model,
    bridge: torch.Tensor,
    functional: torch.Tensor,
    sampling_steps: int = 20,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover clean residual by DDIM, then return z = B + predicted r_0.

    J = sampling_steps is the number of selected time points, not the maximum
    training time T. Pass fixed initial_noise for reproducible inference.
    """
    if not 2 <= sampling_steps <= model.steps:
        raise ValueError("sampling_steps must satisfy 2 <= J <= T")
    model.eval()
    residual = model.ddim_sample(
        bridge,
        functional,
        bridge.new_zeros((len(bridge), 4)),
        sampling_steps,
        initial_noise,
        model_type="diffusion",
    )
    return bridge + residual
