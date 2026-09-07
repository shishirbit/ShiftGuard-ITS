import copy
import importlib.util
from pathlib import Path

import torch


def test_microbatch_preserves_masked_gradient():
    spec = importlib.util.spec_from_file_location("training_script", Path("scripts/train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    torch.manual_seed(4)
    model = train.MLPBaseline()
    split_model = copy.deepcopy(model)
    x = torch.randn(5, 12, 3, 1)
    y = torch.randn(5, 12, 3, 2)
    y[..., 1] = (torch.rand(5, 12, 3) > 0.4).float()
    y[0, ..., 1] = 0
    observed = torch.ones_like(x)
    train.backward_batch(model, x, y, observed)
    train.backward_batch(split_model, x, y, observed, microbatch_size=2)
    for full, split in zip(model.parameters(), split_model.parameters(), strict=True):
        torch.testing.assert_close(full.grad, split.grad, rtol=1e-5, atol=1e-7)
