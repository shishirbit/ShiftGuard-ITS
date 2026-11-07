import numpy as np
import pytest

from shiftguard.calibration import prequential_calibration
from shiftguard.outages import (
    HISTORY,
    HORIZON,
    episode_durations,
    historical_profile,
    hub_episode_timeline,
    region_episode_timeline,
    sensor_renewal_timeline,
    trace_replay_timeline,
    windows_from_timeline,
)


def _adjacency(nodes: int = 20) -> np.ndarray:
    return np.random.default_rng(0).random((nodes, nodes))


def test_episode_durations_measure_missing_runs() -> None:
    observed = np.array([[1, 0], [0, 0], [0, 1], [1, 1], [0, 1]], dtype=bool)
    assert sorted(episode_durations(observed).tolist()) == [1, 2, 2]


def test_windows_share_one_physical_timeline() -> None:
    timeline = np.random.default_rng(1).random((60, 5)) > 0.5
    inputs, targets = windows_from_timeline(timeline)
    assert inputs.shape == (60 - HISTORY - HORIZON + 1, HISTORY, 5)
    assert targets.shape == (60 - HISTORY - HORIZON + 1, HORIZON, 5)
    assert (inputs[1:, :-1] == inputs[:-1, 1:]).all()
    assert (targets[0, 0] == timeline[HISTORY]).all()


def test_group_outages_reach_requested_severity() -> None:
    durations = np.array([12, 24, 48])
    for build, severity in ((region_episode_timeline, 0.3), (hub_episode_timeline, 0.15)):
        observed = build(500, severity, durations, _adjacency(), np.random.default_rng(2))
        assert severity <= 1 - observed.mean() < severity + 0.1


def test_hub_outage_rejects_severity_the_pool_cannot_reach() -> None:
    durations = np.array([12, 24, 48])
    with pytest.raises(ValueError):
        hub_episode_timeline(500, 0.3, durations, _adjacency(), np.random.default_rng(2))


def test_sensor_renewal_severity_is_close_to_request() -> None:
    durations = np.array([2, 10, 48])
    observed = sensor_renewal_timeline(5000, 30, 0.2, durations, np.random.default_rng(3))
    assert abs((1 - observed.mean()) - 0.2) < 0.05


def test_trace_replay_superposition_raises_missingness() -> None:
    rng = np.random.default_rng(4)
    trace = rng.random((400, 10)) > 0.1
    once = trace_replay_timeline(100, 10, trace, 1, np.random.default_rng(5))
    twice = trace_replay_timeline(100, 10, trace, 2, np.random.default_rng(5))
    assert once.mean() > twice.mean()
    assert trace_replay_timeline(100, 25, trace, 1, rng).shape == (100, 25)


def test_historical_profile_uses_only_available_readings() -> None:
    speed = np.array([[10.0], [99.0], [30.0]])
    available = np.array([[True], [False], [True]])
    profile = historical_profile(speed, available, np.array([0, 0, 1]), np.array([0, 0, 0]))
    assert profile[0, 0, 0] == 10.0
    assert profile[0, 1, 0] == 30.0
    assert profile[3, 100, 0] == 20.0


def test_feedback_hides_failed_sensor_outcomes_from_calibrator() -> None:
    rng = np.random.default_rng(6)
    cal_target = rng.normal(size=(400, 2, 3))
    target = rng.normal(size=(600, 2, 3))
    point = np.zeros_like(target)
    scale = np.ones_like(target)
    # Sensor 0 becomes huge in the test block, but only while it is failed.
    target[:, :, 0] *= 100
    feedback = target.copy()
    feedback[:, :, 0] = np.nan
    arrays = (cal_target, np.zeros_like(cal_target), np.ones_like(cal_target), target, point, scale)
    oracle = prequential_calibration(*arrays, block_size=100, window=300)
    feasible = prequential_calibration(*arrays, block_size=100, window=300, feedback_target=feedback)
    assert oracle[1][-1, 0, 0] > 10
    assert np.isfinite(feasible[1]).all()
    assert feasible[1][-1, 0, 0] < 10


def test_certificate_stride_changes_only_the_rule() -> None:
    rng = np.random.default_rng(7)
    target = rng.normal(size=(600, 2, 3))
    arrays = (target[:200], np.zeros((200, 2, 3)), np.ones((200, 2, 3)),
              target[200:], np.zeros((400, 2, 3)), np.ones((400, 2, 3)))
    dense = prequential_calibration(*arrays, block_size=100, window=300)
    thinned = prequential_calibration(*arrays, block_size=100, window=300, certificate_stride=12)
    assert np.array_equal(dense[1], thinned[1])
