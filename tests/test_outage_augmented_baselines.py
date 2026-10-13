import importlib.util
from pathlib import Path

import torch


def load_train():
    spec = importlib.util.spec_from_file_location("training_script", Path("scripts/train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    return train


def test_outage_augmented_baselines_share_base_architecture():
    train = load_train()
    adjacency = torch.eye(6)
    for name in train.OUTAGE_AUGMENTED_BASELINES:
        augmented = f"{name}{train.OUTAGE_AUGMENTED_SUFFIX}"
        assert augmented in train.MODEL_CHOICES
        assert train.base_model_name(augmented) == name
        base = train.build_model(name, 6, adjacency)
        variant = train.build_model(augmented, 6, adjacency)
        assert type(base) is type(variant)
        assert sum(p.numel() for p in base.parameters()) == sum(
            p.numel() for p in variant.parameters()
        )


def test_laplace_variants_emit_positive_scale():
    train = load_train()
    nodes = 6
    x = torch.randn(2, 12, nodes, 4)
    x[..., 2] = torch.rand(2, 12, nodes)
    x[..., 3] = torch.rand(2, 12, nodes)
    for name in train.OUTAGE_AUGMENTED_BASELINES:
        variant = f"{name}{train.OUTAGE_AUGMENTED_SUFFIX}{train.LAPLACE_SUFFIX}"
        assert variant in train.MODEL_CHOICES
        assert train.base_model_name(variant) == name
        median, scale = train.unpack(train.build_model(variant, nodes, torch.eye(nodes))(x))
        assert median.shape == scale.shape == (2, 12, nodes, 1)
        assert torch.all(scale > 0)
        loss = train.laplace_loss(median, scale, torch.randn_like(median))
        loss.backward()


def test_mask_inputs_matches_stress_evaluation():
    train = load_train()
    x = torch.randn(3, 12, 5, 4)
    observed = (torch.rand(3, 12, 5, 1) > 0.5).float()
    masked = train.mask_inputs(x, observed)
    expected = x.clone()
    expected[..., :1] = expected[..., :1] * observed
    torch.testing.assert_close(masked, expected)
    torch.testing.assert_close(masked[..., 1:], x[..., 1:])


def test_shiftguard_names_map_to_flags():
    train = load_train()
    from shiftguard.registry import REFERENCE_ABLATIONS, REFERENCE_MODEL

    adjacency = torch.eye(6)
    reference = train.build_model(REFERENCE_MODEL, 6, adjacency)
    assert reference.use_reliability is False
    assert train.build_model("shiftguard", 6, adjacency).use_reliability is True
    for name in REFERENCE_ABLATIONS:
        assert name in train.MODEL_CHOICES
        train.build_model(name, 6, adjacency)
    # Legacy names must rebuild exactly the architectures their checkpoints hold.
    for name in ("shiftguard_norouter", "shiftguard_staticgraph", "shiftguard_adaptivegraph"):
        legacy = train.build_model(name, 6, adjacency)
        assert legacy.use_reliability is True
