import random

import numpy as np
import pytest
import torch

from shiftguard.checkpointing import restore_training_state, save_training_state


def test_training_recovery_reproduces_next_update(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(0.2))
    optimizer = torch.optim.AdamW(model.parameters())

    def step():
        optimizer.zero_grad()
        model(torch.randn(4, 3)).square().mean().backward()
        optimizer.step()

    step()
    path = tmp_path / "last.pt"
    save_training_state(path, model, optimizer, config={"seed": 7}, epoch=1)
    step()
    expected = [parameter.detach().clone() for parameter in model.parameters()]
    random_expected = (random.random(), np.random.rand())
    restored = restore_training_state(path, model, optimizer, {"seed": 7})
    assert restored["epoch"] == 1
    step()
    for actual, reference in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert (random.random(), np.random.rand()) == random_expected
    assert not (tmp_path / "last.pt.tmp").exists()
    with pytest.raises(ValueError, match="configuration"):
        restore_training_state(path, model, optimizer, {"seed": 8})
