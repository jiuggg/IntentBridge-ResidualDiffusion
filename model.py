"""Dual-condition residual diffusion for cold-start API recommendation.

The denoiser predicts r_0 = y - B. DDIM returns a clean residual; add B once
after sampling to obtain recommendation scores. Training time indices are
zero-based: index T - 1 in this implementation corresponds to paper step T.

Dependencies: numpy and torch. build_model() uses the paper's default model.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((points / steps) + offset) / (1 + offset) * math.pi * 0.5
    ) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=torch.float32) * -scale
        )
        values = time.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([values.sin(), values.cos()], dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.net = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(self.norm(values))


class BridgeSemanticCrossAttentionDenoiser(nn.Module):
    """Recover a clean residual from B, functional matching S, and noisy state.

    Conditions supply queries; the time-aware noisy state supplies keys and
    values. Separate attention branches are fused with the time embedding.
    The internal name 'semantic' denotes functional matching S. The unused
    confidence argument preserves the original checkpoint-facing interface;
    this model has only the two evidence conditions B and S.
    """

    def __init__(
        self,
        num_apis: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
        attention_tokens: int = 16,
        attention_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % attention_tokens != 0:
            raise ValueError("hidden_dim must be divisible by attention_tokens")
        token_dim = hidden_dim // attention_tokens
        if token_dim % attention_heads != 0:
            raise ValueError(
                "hidden_dim / attention_tokens must be divisible by attention_heads"
            )

        self.num_apis = num_apis
        self.hidden_dim = hidden_dim
        self.attention_tokens = attention_tokens
        self.token_dim = token_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noisy_encoder = nn.Linear(num_apis, hidden_dim)
        self.bridge_encoder = nn.Linear(num_apis, hidden_dim)
        self.semantic_encoder = nn.Linear(num_apis, hidden_dim)
        self.noisy_norm = nn.LayerNorm(token_dim)
        self.bridge_norm = nn.LayerNorm(token_dim)
        self.semantic_norm = nn.LayerNorm(token_dim)
        self.bridge_attention = nn.MultiheadAttention(
            token_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.semantic_attention = nn.MultiheadAttention(
            token_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(depth)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_apis),
            nn.Tanh(),
        )

    def _tokens(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(-1, self.attention_tokens, self.token_dim)

    def forward(
        self,
        noisy_residual: torch.Tensor,
        time: torch.Tensor,
        bridge: torch.Tensor,
        semantic: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        del confidence
        noisy_state = self.noisy_encoder(noisy_residual)
        time_state = self.time_embed(time)

        noisy_tokens = self.noisy_norm(
            self._tokens(noisy_state + time_state)
        )
        bridge_query = self.bridge_norm(
            self._tokens(self.bridge_encoder(bridge) + time_state)
        )
        semantic_query = self.semantic_norm(
            self._tokens(self.semantic_encoder(semantic) + time_state)
        )
        bridge_attended, _ = self.bridge_attention(
            bridge_query, noisy_tokens, noisy_tokens, need_weights=False
        )
        semantic_attended, _ = self.semantic_attention(
            semantic_query, noisy_tokens, noisy_tokens, need_weights=False
        )
        condition_state = self.condition_fusion(
            torch.cat(
                [
                    bridge_attended.flatten(1),
                    semantic_attended.flatten(1),
                    time_state,
                ],
                dim=1,
            )
        )
        state = noisy_state + condition_state
        for block in self.blocks:
            state = block(state)
        return self.output(state)


class ResidualDiffusion(nn.Module):
    def __init__(self, denoiser: nn.Module, steps: int = 100):
        super().__init__()
        self.denoiser = denoiser
        self.steps = steps
        betas = cosine_beta_schedule(steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar)
        )

    @staticmethod
    def _extract(values: torch.Tensor, time: torch.Tensor, shape) -> torch.Tensor:
        return values.gather(0, time).reshape(time.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self, clean_residual: torch.Tensor, time: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alpha_bar, time, clean_residual.shape)
            * clean_residual
            + self._extract(self.sqrt_one_minus_alpha_bar, time, clean_residual.shape)
            * noise
        )

    def training_loss(
        self,
        clean_residual: torch.Tensor,
        bridge: torch.Tensor,
        semantic: torch.Tensor,
        confidence: torch.Tensor,
        target: torch.Tensor,
        ranking_weight: float,
        negative_samples: int,
        model_type: str = "diffusion",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch = clean_residual.shape[0]
        if model_type == "diffusion":
            time = torch.randint(0, self.steps, (batch,), device=clean_residual.device)
            noise = torch.randn_like(clean_residual)
            noisy = self.q_sample(clean_residual, time, noise)
        elif model_type == "mlp":
            time = torch.zeros(batch, device=clean_residual.device, dtype=torch.long)
            noisy = 0.5 * (bridge + semantic)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        prediction = self.denoiser(noisy, time, bridge, semantic, confidence)

        important = ((target > 0) | (bridge > 1e-7)).float()
        weights = 0.10 + 0.90 * important + 1.50 * target
        diffusion_loss = ((prediction - clean_residual).square() * weights).sum()
        diffusion_loss = diffusion_loss / weights.sum().clamp_min(1.0)

        scores = bridge + prediction
        positive_probabilities = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
        negative_probabilities = (1.0 - target)
        negative_probabilities = negative_probabilities / negative_probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        positive_ids = torch.multinomial(positive_probabilities, 1)
        negative_ids = torch.multinomial(
            negative_probabilities, negative_samples, replacement=False
        )
        positive_scores = scores.gather(1, positive_ids)
        negative_scores = scores.gather(1, negative_ids)
        ranking_loss = F.softplus(negative_scores - positive_scores).mean()
        total = diffusion_loss + ranking_weight * ranking_loss
        return total, {
            "diffusion": float(diffusion_loss.detach().cpu()),
            "ranking": float(ranking_loss.detach().cpu()),
        }

    @torch.no_grad()
    def ddim_sample(
        self,
        bridge: torch.Tensor,
        semantic: torch.Tensor,
        confidence: torch.Tensor,
        sampling_steps: int,
        initial_noise: Optional[torch.Tensor] = None,
        model_type: str = "diffusion",
    ) -> torch.Tensor:
        if model_type == "mlp":
            time = torch.zeros(
                bridge.shape[0], device=bridge.device, dtype=torch.long
            )
            proxy = 0.5 * (bridge + semantic)
            return self.denoiser(proxy, time, bridge, semantic, confidence)
        if model_type != "diffusion":
            raise ValueError(f"Unknown model_type: {model_type}")
        if initial_noise is None:
            current = torch.randn_like(bridge)
        else:
            current = initial_noise.clone()

        times = np.linspace(self.steps - 1, 0, sampling_steps, dtype=np.int64)
        times = np.unique(times)[::-1].copy()
        for index, time_value in enumerate(times):
            time = torch.full(
                (bridge.shape[0],), int(time_value), device=bridge.device, dtype=torch.long
            )
            clean = self.denoiser(current, time, bridge, semantic, confidence)
            clean = clean.clamp(-1.0, 1.0)
            if index == len(times) - 1:
                current = clean
                continue

            previous_time = int(times[index + 1])
            alpha_now = self.alpha_bar[time_value]
            alpha_previous = self.alpha_bar[previous_time]
            predicted_noise = (
                current - torch.sqrt(alpha_now) * clean
            ) / torch.sqrt(1.0 - alpha_now).clamp_min(1e-8)
            current = (
                torch.sqrt(alpha_previous) * clean
                + torch.sqrt(1.0 - alpha_previous) * predicted_noise
            )
        return current


def build_model(num_apis, hidden_dim=256, depth=3, steps=50):
    """Build the paper model: 16 tokens, 4 heads, 3 blocks, and T = 50."""
    return ResidualDiffusion(BridgeSemanticCrossAttentionDenoiser(
        num_apis, hidden_dim=hidden_dim, depth=depth, dropout=0.1,
        attention_tokens=16, attention_heads=4,
    ), steps=steps)
