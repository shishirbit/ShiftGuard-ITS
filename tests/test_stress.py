import numpy as np

from shiftguard.stress import apply_outage, burst_mask, hub_mask, region_mask


def test_structured_masks_have_expected_shape_and_effect() -> None:
    shape = (5, 12, 8, 1)
    adjacency = np.eye(8, k=1) + np.eye(8, k=-1)
    rng = np.random.default_rng(4)
    for mask in [
        burst_mask(shape, 0.2, rng),
        region_mask(shape, 0.25, adjacency, rng),
        hub_mask(shape, 0.25, adjacency),
    ]:
        assert mask.shape == shape
        assert mask.dtype == bool
        assert np.count_nonzero(~mask) > 0
        assert np.all(apply_outage(np.ones(shape), mask)[~mask] == 0)

