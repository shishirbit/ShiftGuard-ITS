import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from shiftguard.models import (
    AGCRNBaseline,
    DCRNNBaseline,
    GraphWaveNetBaseline,
    GRUBaseline,
    LinearBaseline,
    MLPBaseline,
    ShiftGuard,
    STAEformerBaseline,
    STGCNBaseline,
)


def test_model_shapes_and_gradients() -> None:
    batch, history, horizon, nodes = 2, 12, 12, 8
    x = torch.randn(batch, history, nodes, 1)
    observed = torch.ones_like(x)
    adjacency = torch.eye(nodes)
    linear = LinearBaseline(history, horizon)
    assert linear(x).shape == (batch, horizon, nodes, 1)
    assert MLPBaseline(history, horizon, hidden=8)(x).shape == (batch, horizon, nodes, 1)
    assert GRUBaseline(horizon, hidden=8)(x).shape == (batch, horizon, nodes, 1)
    assert STGCNBaseline(adjacency, horizon, hidden=8)(x).shape == (batch, horizon, nodes, 1)
    assert GraphWaveNetBaseline(nodes, adjacency, horizon, hidden=8)(x).shape == (
        batch,
        horizon,
        nodes,
        1,
    )
    assert DCRNNBaseline(adjacency, horizon, hidden=8)(x).shape == (batch, horizon, nodes, 1)
    assert AGCRNBaseline(nodes, horizon, hidden=8)(x).shape == (batch, horizon, nodes, 1)
    assert STAEformerBaseline(
        nodes,
        history,
        horizon,
        input_embedding=4,
        calendar_embedding=2,
        adaptive_embedding=4,
        feed_forward=16,
        heads=2,
        layers=1,
    )(torch.randn(batch, history, nodes, 4)).shape == (batch, horizon, nodes, 1)
    model = ShiftGuard(nodes, adjacency, history, horizon, hidden=16, graph_layers=2)
    median, scale, weights = model(x, observed)
    assert median.shape == scale.shape == (batch, horizon, nodes, 1)
    assert weights.shape == (batch, 4)
    assert torch.all(scale > 0)
    median.mean().backward()
    assert model.node_source.grad is not None


def test_shiftguard_ablation_configurations() -> None:
    x = torch.randn(2, 12, 8, 1)
    observed = torch.ones_like(x)
    adjacency = torch.eye(8)
    variants = (
        ShiftGuard(8, adjacency, hidden=16, use_reliability=False),
        ShiftGuard(8, adjacency, hidden=16, use_adaptive_adjacency=False),
        ShiftGuard(8, adjacency, hidden=16, use_static_adjacency=False),
        ShiftGuard(8, adjacency, hidden=16, regimes=1),
    )
    for model in variants:
        median, scale, weights = model(x, observed)
        assert median.shape == scale.shape == (2, 12, 8, 1)
        assert weights.shape[0] == 2
