import numpy as np

from shiftguard.calibration import (
    RollingGroupConformal,
    calibrate_abstention,
    conformal_group_quantiles,
    conformal_quantile,
    interval_metrics,
    prequential_calibration,
    risk_coverage_curve,
)


def test_calibration_and_selective_metrics() -> None:
    rng = np.random.default_rng(2)
    history = rng.normal(size=(100, 3, 4))
    pred = np.zeros((3, 4))
    lower, upper, width = RollingGroupConformal().intervals(pred, history)
    assert np.all(lower <= pred)
    assert np.all(upper >= pred)
    assert np.all(width >= 0)
    metrics = interval_metrics(np.zeros_like(pred), lower, upper, alpha=0.1)
    assert metrics["coverage"] == 1.0
    coverage, risk, aurc = risk_coverage_curve(np.arange(12), width)
    assert len(coverage) == len(risk) == 12
    assert aurc >= 0
    errors = np.concatenate([np.full(900, 1.0), np.full(100, 10.0)])
    uncertainty = np.arange(1000)
    rule = calibrate_abstention(errors, uncertainty, error_tolerance=5, risk_limit=0.1)
    assert 0 < rule.calibration_coverage < 1
    assert rule.upper_failure_bound <= 0.1


def test_conformal_order_statistics() -> None:
    scores = np.arange(10, dtype=float)
    assert conformal_quantile(scores, alpha=0.2) == 8.0
    grouped = np.stack([scores, scores + 1], axis=1)
    assert np.array_equal(conformal_group_quantiles(grouped, alpha=0.2), [8.0, 9.0])


def test_abstention_certification_does_not_reuse_candidate_outcomes() -> None:
    uncertainty = np.linspace(0, 1, 1000)
    errors = np.where(uncertainty < 0.7, 1.0, 10.0)
    changed = errors.copy()
    changed[::2] = 1000.0
    original_rule = calibrate_abstention(errors, uncertainty)
    changed_rule = calibrate_abstention(changed, uncertainty)
    assert original_rule == changed_rule


def test_prequential_calibration_shapes_and_delay() -> None:
    rng = np.random.default_rng(7)
    shape = (40, 3, 2)
    cal_target = rng.normal(size=shape)
    cal_point = cal_target + rng.normal(scale=0.2, size=shape)
    scale = np.full(shape, 0.5)
    test_target = rng.normal(size=shape)
    test_point = test_target + rng.normal(scale=0.3, size=shape)
    lower, upper, accepted, rules = prequential_calibration(
        cal_target,
        cal_point,
        scale,
        test_target,
        test_point,
        scale,
        block_size=10,
        window=30,
    )
    assert lower.shape == upper.shape == accepted.shape == shape
    assert len(rules) == 4
    assert np.all(lower <= upper)
    assert np.all(accepted == accepted[:, :1, :1])

    global_lower, global_upper, accept_all, _ = prequential_calibration(
        cal_target,
        cal_point,
        scale,
        test_target,
        test_point,
        scale,
        block_size=10,
        window=30,
        groupwise=False,
        update=False,
        risk_control=False,
    )
    assert np.all(global_lower <= global_upper)
    assert np.all(accept_all)
