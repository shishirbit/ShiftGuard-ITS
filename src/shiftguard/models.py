"""Reproducible baselines and the provisional ShiftGuard backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Persistence(nn.Module):
    def __init__(self, horizon: int = 12) -> None:
        super().__init__()
        self.horizon = horizon

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        return x[:, -1:, :, :1].expand(-1, self.horizon, -1, -1)


class LinearBaseline(nn.Module):
    """Per-node shared linear projection from history to all horizons."""

    def __init__(self, history: int = 12, horizon: int = 12) -> None:
        super().__init__()
        self.projection = nn.Linear(history, horizon)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        values = x[..., 0].transpose(1, 2)
        return self.projection(values).transpose(1, 2).unsqueeze(-1)


class MLPBaseline(nn.Module):
    """Node-shared nonlinear history encoder with direct multi-horizon output."""

    def __init__(self, history: int = 12, horizon: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(history, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        values = x[..., 0].transpose(1, 2)
        return self.network(values).transpose(1, 2).unsqueeze(-1)


class STIDBaseline(nn.Module):
    """Compact identity/time-feature baseline inspired by STID."""

    def __init__(self, nodes: int, history: int = 12, horizon: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(nodes, hidden) * 0.02)
        self.time_embedding = nn.Embedding(288, hidden)
        self.day_embedding = nn.Embedding(7, hidden)
        self.history = nn.Linear(history, hidden)
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, horizon, 1),
        )

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        values = x[..., 0].transpose(1, 2)
        temporal = self.history(values).transpose(1, 2).unsqueeze(-1)
        identity = self.node_embedding.T.unsqueeze(0).unsqueeze(-1).expand(len(x), -1, -1, -1)
        if x.shape[-1] >= 4:
            time_index = (x[:, -1, 0, 2] * 288).long().clamp(0, 287)
            day_index = (x[:, -1, 0, 3] * 6).round().long().clamp(0, 6)
            calendar = self.time_embedding(time_index) + self.day_embedding(day_index)
            identity = identity + calendar[:, :, None, None]
        return self.mlp(torch.cat([temporal, identity], dim=1)).permute(0, 1, 2, 3)


class AxisTransformerBlock(nn.Module):
    """Pre-norm transformer block applied along one spatiotemporal axis."""

    def __init__(self, width: int, heads: int, feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feed_forward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + self.dropout(attended)
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class STAEformerBaseline(nn.Module):
    """Faithful adaptive-embedding temporal/spatial transformer baseline."""

    def __init__(
        self,
        nodes: int,
        history: int = 12,
        horizon: int = 12,
        input_embedding: int = 24,
        calendar_embedding: int = 24,
        adaptive_embedding: int = 80,
        feed_forward: int = 256,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.nodes, self.history, self.horizon = nodes, history, horizon
        width = input_embedding + 2 * calendar_embedding + adaptive_embedding
        self.input_projection = nn.Linear(1, input_embedding)
        self.time_embedding = nn.Embedding(288, calendar_embedding)
        self.day_embedding = nn.Embedding(7, calendar_embedding)
        self.adaptive_embedding = nn.Parameter(torch.empty(history, nodes, adaptive_embedding))
        nn.init.xavier_uniform_(self.adaptive_embedding)
        self.temporal_blocks = nn.ModuleList(
            [AxisTransformerBlock(width, heads, feed_forward, dropout) for _ in range(layers)]
        )
        self.spatial_blocks = nn.ModuleList(
            [AxisTransformerBlock(width, heads, feed_forward, dropout) for _ in range(layers)]
        )
        self.output_projection = nn.Linear(history * width, horizon)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        batch, history, nodes, _ = x.shape
        time_index = (x[..., 2] * 288).long().clamp(0, 287)
        day_index = (x[..., 3] * 6).round().long().clamp(0, 6)
        adaptive = self.adaptive_embedding[None].expand(batch, -1, -1, -1)
        features = torch.cat(
            [
                self.input_projection(x[..., :1]),
                self.time_embedding(time_index),
                self.day_embedding(day_index),
                adaptive,
            ],
            dim=-1,
        )
        width = features.shape[-1]
        for block in self.temporal_blocks:
            temporal = features.transpose(1, 2).reshape(batch * nodes, history, width)
            features = block(temporal).reshape(batch, nodes, history, width).transpose(1, 2)
        for block in self.spatial_blocks:
            spatial = features.reshape(batch * history, nodes, width)
            features = block(spatial).reshape(batch, history, nodes, width)
        flattened = features.transpose(1, 2).reshape(batch, nodes, history * width)
        return self.output_projection(flattened).transpose(1, 2).unsqueeze(-1)


class GRUBaseline(nn.Module):
    """Shared per-node GRU followed by a direct multi-horizon head."""

    def __init__(self, horizon: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        batch, history, nodes, _ = x.shape
        series = x[..., :1].permute(0, 2, 1, 3).reshape(batch * nodes, history, 1)
        _, state = self.gru(series)
        output = self.head(state[-1]).reshape(batch, nodes, -1)
        return output.transpose(1, 2).unsqueeze(-1)


class STGCNBaseline(nn.Module):
    """Compact spectral spatial-temporal convolution baseline."""

    def __init__(self, adjacency: torch.Tensor, horizon: int = 12, hidden: int = 64) -> None:
        super().__init__()
        nodes = adjacency.shape[0]
        normalized = adjacency + torch.eye(nodes, device=adjacency.device)
        normalized = normalized / normalized.sum(-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("adjacency", normalized.float())
        self.temporal_in = nn.Conv2d(1, hidden, kernel_size=(3, 1), padding=(1, 0))
        self.spatial = nn.Linear(hidden, hidden)
        self.temporal_out = nn.Conv2d(hidden, hidden, kernel_size=(3, 1), padding=(1, 0))
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        features = F.gelu(self.temporal_in(x[..., :1].permute(0, 3, 1, 2)))
        features = features.permute(0, 2, 3, 1)
        features = torch.einsum("nm,btmd->btnd", self.adjacency, features)
        features = F.gelu(self.spatial(features)).permute(0, 3, 1, 2)
        features = F.gelu(self.temporal_out(features)).mean(dim=2).transpose(1, 2)
        return self.head(features).transpose(1, 2).unsqueeze(-1)


class GraphWaveNetBaseline(nn.Module):
    """Small Graph WaveNet-style adaptive graph and gated dilated TCN."""

    def __init__(
        self, nodes: int, adjacency: torch.Tensor, horizon: int = 12, hidden: int = 32
    ) -> None:
        super().__init__()
        static = adjacency + torch.eye(nodes, device=adjacency.device)
        self.register_buffer("static", static / static.sum(-1, keepdim=True).clamp_min(1e-6))
        self.source = nn.Parameter(torch.randn(nodes, 10) * 0.02)
        self.target = nn.Parameter(torch.randn(nodes, 10) * 0.02)
        self.start = nn.Conv2d(1, hidden, 1)
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.residual = nn.ModuleList()
        for dilation in (1, 2, 4):
            self.filters.append(
                nn.Conv2d(hidden, hidden, (2, 1), dilation=(dilation, 1), padding=(dilation, 0))
            )
            self.gates.append(
                nn.Conv2d(hidden, hidden, (2, 1), dilation=(dilation, 1), padding=(dilation, 0))
            )
            self.residual.append(nn.Conv2d(hidden, hidden, 1))
        self.end = nn.Sequential(nn.Conv2d(hidden, hidden * 2, 1), nn.ReLU(), nn.Conv2d(hidden * 2, horizon, 1))

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        features = self.start(x[..., :1].permute(0, 3, 1, 2))
        adaptive = torch.softmax(F.relu(self.source @ self.target.T), dim=-1)
        adjacency = 0.5 * self.static + 0.5 * adaptive
        for convolution, gate, residual in zip(self.filters, self.gates, self.residual, strict=True):
            previous = features
            features = torch.tanh(convolution(features)) * torch.sigmoid(gate(features))
            features = features[:, :, : previous.shape[2], :]
            graph = torch.einsum("nm,bctm->bctn", adjacency, features)
            features = residual(graph) + previous
        return self.end(features[:, :, -1:, :]).squeeze(2).unsqueeze(-1)


class DiffusionGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden: int, supports: int = 2, order: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.order = order
        mixed = (input_dim + hidden) * (1 + supports * order)
        self.gates = nn.Linear(mixed, hidden * 2)
        self.candidate = nn.Linear(mixed, hidden)

    def _diffuse(self, values: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        outputs = [values]
        for support in supports:
            propagated = values
            for _ in range(self.order):
                propagated = torch.einsum("nm,bmf->bnf", support, propagated)
                outputs.append(propagated)
        return torch.cat(outputs, dim=-1)

    def forward(
        self, values: torch.Tensor, state: torch.Tensor, supports: list[torch.Tensor]
    ) -> torch.Tensor:
        reset, update = torch.sigmoid(
            self.gates(self._diffuse(torch.cat([values, state], dim=-1), supports))
        ).chunk(2, dim=-1)
        candidate_input = self._diffuse(torch.cat([values, reset * state], dim=-1), supports)
        candidate = torch.tanh(self.candidate(candidate_input))
        return update * state + (1 - update) * candidate


class DCRNNBaseline(nn.Module):
    """Diffusion-convolution recurrent baseline with a direct horizon decoder."""

    def __init__(self, adjacency: torch.Tensor, horizon: int = 12, hidden: int = 64) -> None:
        super().__init__()
        forward = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1e-6)
        backward = adjacency.T / adjacency.T.sum(-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("forward_support", forward.float())
        self.register_buffer("backward_support", backward.float())
        self.cell = DiffusionGRUCell(1, hidden)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        state = x.new_zeros((len(x), x.shape[2], self.cell.hidden))
        supports = [self.forward_support, self.backward_support]
        for step in range(x.shape[1]):
            state = self.cell(x[:, step, :, :1], state, supports)
        return self.head(state).transpose(1, 2).unsqueeze(-1)


class AGCRNBaseline(nn.Module):
    """Node-adaptive recurrent baseline using an embedding-derived graph."""

    def __init__(self, nodes: int, horizon: int = 12, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.embedding = nn.Parameter(torch.randn(nodes, 16) * 0.02)
        self.gates = nn.Linear(1 + hidden, hidden * 2)
        self.candidate = nn.Linear(1 + hidden, hidden)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, observed: torch.Tensor | None = None) -> torch.Tensor:
        state = x.new_zeros((len(x), x.shape[2], self.hidden))
        adjacency = torch.softmax(F.relu(self.embedding @ self.embedding.T), dim=-1)
        for step in range(x.shape[1]):
            values = x[:, step, :, :1]
            mixed = torch.cat([values, state], dim=-1)
            mixed = torch.einsum("nm,bmf->bnf", adjacency, mixed)
            reset, update = torch.sigmoid(self.gates(mixed)).chunk(2, dim=-1)
            candidate_input = torch.cat([values, reset * state], dim=-1)
            candidate_input = torch.einsum("nm,bmf->bnf", adjacency, candidate_input)
            candidate = torch.tanh(self.candidate(candidate_input))
            state = update * state + (1 - update) * candidate
        return self.head(state).transpose(1, 2).unsqueeze(-1)


class ReliabilityGraphBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden * 2 + 1, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor, reliability: torch.Tensor
    ) -> torch.Tensor:
        # features: B,N,D; adjacency: B,N,N; reliability: B,N,1
        messages = torch.bmm(adjacency, self.message(features))
        gate = torch.sigmoid(self.gate(torch.cat([features, messages, reliability], dim=-1)))
        updated = gate * messages + (1 - gate) * features
        return self.norm(features + self.dropout(F.gelu(updated)))


class ShiftGuard(nn.Module):
    """Reliability-conditioned graph-temporal forecaster.

    The model is deliberately small. Observation masks affect both temporal
    encoding and sender reliability in graph propagation. A regime router mixes
    compact output experts. It emits a median and positive Laplace scale; the
    latter is subsequently calibrated and used for selective forecasting.
    """

    def __init__(
        self,
        nodes: int,
        adjacency: torch.Tensor,
        history: int = 12,
        horizon: int = 12,
        hidden: int = 64,
        graph_layers: int = 3,
        regimes: int = 4,
        dropout: float = 0.1,
        use_reliability: bool = True,
        use_static_adjacency: bool = True,
        use_adaptive_adjacency: bool = True,
    ) -> None:
        super().__init__()
        if adjacency.shape != (nodes, nodes):
            raise ValueError("adjacency has wrong shape")
        self.nodes = nodes
        self.horizon = horizon
        self.use_reliability = use_reliability
        self.use_static_adjacency = use_static_adjacency
        self.use_adaptive_adjacency = use_adaptive_adjacency
        if not (use_static_adjacency or use_adaptive_adjacency):
            raise ValueError("at least one adjacency source is required")
        self.register_buffer("static_adjacency", adjacency.float())
        self.node_embedding = nn.Parameter(torch.randn(nodes, hidden) * 0.02)
        self.node_source = nn.Parameter(torch.randn(nodes, 16) * 0.02)
        self.node_target = nn.Parameter(torch.randn(nodes, 16) * 0.02)
        self.time_embedding = nn.Embedding(288, hidden)
        self.day_embedding = nn.Embedding(7, hidden)
        self.temporal = nn.Sequential(
            nn.Linear(history * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList(
            [ReliabilityGraphBlock(hidden, dropout) for _ in range(graph_layers)]
        )
        self.regime_router = nn.Sequential(
            nn.Linear(4, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, regimes)
        )
        self.median_heads = nn.ModuleList([nn.Linear(hidden, horizon) for _ in range(regimes)])
        self.scale_heads = nn.ModuleList([nn.Linear(hidden, horizon) for _ in range(regimes)])
        self.history_skip = nn.Linear(history, horizon, bias=False)
        nn.init.zeros_(self.history_skip.weight)

    def _adjacency(self, reliability: torch.Tensor) -> torch.Tensor:
        adaptive = torch.softmax(F.relu(self.node_source @ self.node_target.T), dim=-1)
        static = self.static_adjacency + torch.eye(self.nodes, device=adaptive.device)
        static = static / static.sum(-1, keepdim=True).clamp_min(1e-6)
        if self.use_static_adjacency and self.use_adaptive_adjacency:
            base = 0.5 * static + 0.5 * adaptive
        elif self.use_static_adjacency:
            base = static
        else:
            base = adaptive
        sender_reliability = reliability if self.use_reliability else torch.ones_like(reliability)
        weighted = base.unsqueeze(0) * sender_reliability.transpose(1, 2)
        return weighted / weighted.sum(-1, keepdim=True).clamp_min(1e-6)

    def forward(
        self, x: torch.Tensor, observed: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intrinsic = x[..., 1:2] if x.shape[-1] >= 4 else torch.ones_like(x[..., :1])
        if observed is None:
            observed = torch.ones_like(x[..., :1])
        observed = observed * intrinsic
        values = x[..., :1] * observed
        reliability = observed.mean(dim=1).squeeze(-1).unsqueeze(-1)
        graph_reliability = reliability if self.use_reliability else torch.ones_like(reliability)
        encoded = torch.cat([values[..., 0], observed[..., 0]], dim=1)
        encoded = encoded.permute(0, 2, 1)
        features = self.temporal(encoded)
        features = features + self.node_embedding[None]
        if x.shape[-1] >= 4:
            time_index = (x[:, -1, 0, 2] * 288).long().clamp(0, 287)
            day_index = (x[:, -1, 0, 3] * 6).round().long().clamp(0, 6)
            features = features + (
                self.time_embedding(time_index) + self.day_embedding(day_index)
            )[:, None, :]
        adjacency = self._adjacency(graph_reliability)
        for block in self.blocks:
            features = block(features, adjacency, graph_reliability)

        last = values[:, -1, :, 0]
        diff = values[:, 1:, :, 0] - values[:, :-1, :, 0]
        regime_stats = torch.stack(
            [
                values[..., 0].mean((1, 2)),
                values[..., 0].std((1, 2)),
                diff.abs().mean((1, 2)),
                reliability.mean((1, 2)),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.regime_router(regime_stats), dim=-1)
        medians = torch.stack([head(features) for head in self.median_heads], dim=-1)
        scales = torch.stack([F.softplus(head(features)) + 1e-4 for head in self.scale_heads], dim=-1)
        weights = weights[:, None, None, :]
        median = (medians * weights).sum(-1).permute(0, 2, 1).unsqueeze(-1)
        scale = (scales * weights).sum(-1).permute(0, 2, 1).unsqueeze(-1)
        # Residual parameterization makes persistence an easy initial solution.
        history_residual = self.history_skip(values[..., 0].transpose(1, 2))
        history_residual = history_residual.transpose(1, 2).unsqueeze(-1)
        median = median + last[:, None, :, None] + history_residual
        return median, scale, weights.squeeze(1).squeeze(1)


def laplace_loss(
    median: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = torch.abs(target - median) / scale + torch.log(2 * scale)
    if observed is None:
        return loss.mean()
    return (loss * observed).sum() / observed.sum().clamp_min(1)
