from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 64
    token_count: int = 45
    dropout: float = .1
    manifold_weight: float = .1
    hyper_beta: float = .5
    ffn_gamma: float = .3

    def __post_init__(self):
        if self.d_model < 2 or self.d_model % 2 or self.nhead < 1 or self.d_model % self.nhead:
            raise ValueError("d_model must be even and divisible by nhead.")
        if min(self.num_layers, self.dim_feedforward, self.token_count) < 1:
            raise ValueError("Layer, hidden and token counts must be positive.")
        if not 0 <= self.dropout < 1 or self.manifold_weight < 0:
            raise ValueError("Invalid dropout or manifold weight.")


class ManifoldAttention(nn.Module):
    """Multi-head attention with a learned positive-definite distance bias.

    G = (AA^T + eps I) / trace(AA^T + eps I) * head_dim.
    The manuscript's duplicated/undefined metric equation is resolved explicitly
    as M(q,k) = -(q-k)^T G (q-k) / head_dim. This geometry does not by itself
    enforce physical constraints; feasibility is handled by the decoder.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.weight = config.manifold_weight
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.metric_factor = nn.Parameter(torch.eye(self.head_dim).repeat(self.heads, 1, 1))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, tokens, dimension = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        eye = torch.eye(self.head_dim, dtype=x.dtype, device=x.device)
        metric = self.metric_factor @ self.metric_factor.transpose(-1, -2) + 1e-4 * eye
        metric = metric * self.head_dim / metric.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        qg, kg = q @ metric, k @ metric
        distance = ((qg * q).sum(-1).unsqueeze(-1) + (kg * k).sum(-1).unsqueeze(-2)
                    - 2 * (qg @ k.transpose(-1, -2))).clamp_min(0) / self.head_dim
        attention = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
                                  - self.weight * distance, dim=-1)
        # A differentiable regularizer whose scale is independent of the batch size.
        manifold_loss = (attention * distance).sum(-1).mean()
        output = (self.dropout(attention) @ v).transpose(1, 2).reshape(batch, tokens, dimension)
        return self.projection(output), manifold_loss


class HyperConnectionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = ManifoldAttention(config)
        self.hyper = nn.Linear(config.d_model, config.d_model)
        self.norm1, self.norm2 = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.dim_feedforward), nn.ReLU(),
                                 nn.Dropout(config.dropout), nn.Linear(config.dim_feedforward, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        self.beta, self.gamma = config.hyper_beta, config.ffn_gamma

    def forward(self, previous, current):
        attended, regularizer = self.attention(current)
        hidden = self.norm1(current + self.dropout(attended)
                            + self.beta * self.hyper(previous * current))
        output = self.norm2(hidden + self.dropout(self.ffn(hidden)) + self.gamma * hidden)
        return output, regularizer


class MHCMTN(nn.Module):
    """Instance-specific solution updater: 15 decisions plus 36 context values.

    Each output contains [12,3] categorical logits and three incentive logits.
    45 internal tokens follow the earlier compact prototype convention. They
    are latent tokens, not 45 chronological observations or 45 output decisions.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.solution_projection = nn.Linear(15, config.token_count)
        self.embedding = nn.Linear(1, config.d_model)
        self.context_projection = nn.Linear(36, config.token_count * config.d_model)
        position = torch.arange(config.token_count, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, config.d_model, 2) * (-math.log(10000) / config.d_model))
        encoding = torch.zeros(config.token_count, config.d_model)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        self.blocks = nn.ModuleList(HyperConnectionBlock(config) for _ in range(config.num_layers))
        self.fusion = nn.Linear(config.d_model * (config.num_layers + 1), config.d_model)
        self.output = nn.Linear(config.token_count * config.d_model, 45)
        nn.init.normal_(self.output.weight, std=.005)
        nn.init.zeros_(self.output.bias)

    def forward(self, solution, context):
        if solution.ndim != 2 or solution.shape[-1] != 15 or context.shape != (len(solution), 36):
            raise ValueError("Expected solution [batch,15] and context [batch,36].")
        normalized = torch.cat((solution[:, :12] - 2, solution[:, 12:] * 4 - 1), -1)
        hidden = self.embedding(self.solution_projection(normalized).unsqueeze(-1))
        hidden = hidden + self.context_projection(context).reshape_as(hidden) + self.position_encoding
        states, previous, regularizers = [hidden], hidden, []
        for block in self.blocks:
            updated, regularizer = block(previous, hidden)
            previous, hidden = hidden, updated
            states.append(hidden)
            regularizers.append(regularizer)
        fused = F.relu(self.fusion(torch.cat(states, dim=-1)))
        raw = self.output(fused.flatten(1)).reshape(-1, 3, 15)
        reference_labels = F.one_hot(solution[:, :12].long() - 1, 3).to(raw.dtype)
        logits = raw[:, :, :12].transpose(1, 2) + 2 * reference_labels
        return logits, raw[:, :, 12:].mean(1), torch.stack(regularizers).mean()

    def configuration(self):
        return asdict(self.config)


class MLPO(nn.Module):
    def __init__(self, hidden_size=15):
        super().__init__()
        self.hidden = nn.Linear(15, hidden_size)
        self.output = nn.Linear(hidden_size, 15)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, solution, context):
        normalized = torch.cat((solution[:, :12] - 2, solution[:, 12:] * 4 - 1), -1)
        raw = self.output(torch.tanh(self.hidden(normalized)))
        soft = (solution[:, :12] + torch.tanh(raw[:, :12])).clamp(1, 3)
        rounded = soft + (soft.round() - soft).detach()
        # Lagrange memberships are exactly one-hot in the forward pass while
        # retaining the legacy STE derivative through the ordinal variable.
        memberships = torch.stack((.5 * (rounded - 2) * (rounded - 3),
                                   -(rounded - 1) * (rounded - 3),
                                   .5 * (rounded - 1) * (rounded - 2)), dim=-1)
        return memberships, raw[:, 12:], raw.sum() * 0
