"""Atomic, epoch-boundary training recovery including optimizer and RNG state."""

import random
from pathlib import Path

import numpy as np
import torch


def save_training_state(path, model, optimizer, **metadata):
    numpy_state = np.random.get_state()
    state = {
        **metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(path, model, optimizer, config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != config:
        raise ValueError("Resume configuration differs from saved training configuration")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    name, keys, position, gaussian, cached = state["numpy_rng"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    return state
