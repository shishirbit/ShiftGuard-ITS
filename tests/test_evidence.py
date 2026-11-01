import numpy as np

from shiftguard.evidence import (
    BLIND,
    FRESH,
    RECENT,
    STALE,
    blend,
    bucket_quantiles,
    evidence_buckets,
    evidence_intervals,
    fit_blend_weights,
    shadow_availability,
)


def test_buckets_follow_time_since_last_report() -> None:
    up = np.ones((1, 12, 4), dtype=bool)
    up[0, -3:, 1] = False  # last report three steps before the origin
    up[0, -8:, 2] = False  # last report eight steps before the origin
    up[0, :, 3] = False
    assert evidence_buckets(up).tolist() == [[FRESH, RECENT, STALE, BLIND]]


def test_shadow_only_removes_availability() -> None:
    up = np.random.default_rng(0).random((50, 12, 30)) > 0.2
    shadow, hidden = shadow_availability(up, np.random.default_rng(1))
    assert not (shadow & ~up).any()
    assert 0.2 < hidden.mean() < 0.4
    assert not (shadow != up)[:, :, :][np.repeat(~hidden[:, None, :], 12, axis=1)].any()


def test_blend_prefers_whichever_source_is_better_per_bucket() -> None:
    rng = np.random.default_rng(2)
    target = rng.normal(size=(400, 2, 10))
    bucket = np.where(rng.random((400, 10)) < 0.5, FRESH, BLIND).astype(np.int8)
    good, bad = target + 0.01, target + rng.normal(5.0, 1.0, size=target.shape)
    cells = bucket[:, None, :]
    point = np.where(cells == FRESH, good, bad)
    profile = np.where(cells == FRESH, bad, good)
    weights = fit_blend_weights(target, point, profile, bucket)
    assert (weights[FRESH] == 1).all()
    assert (weights[BLIND] == 0).all()
    assert np.abs(blend(point, profile, bucket, weights) - target).mean() < 0.02


def test_unseen_bucket_defaults_to_the_model() -> None:
    target = np.zeros((100, 1, 2))
    weights = fit_blend_weights(target, target, target + 1, np.zeros((100, 2), dtype=np.int8))
    assert (weights[BLIND] == 1).all()


def test_thin_bucket_falls_back_to_the_pool() -> None:
    score = np.abs(np.random.default_rng(3).normal(size=(500, 1, 4)))
    bucket = np.full((500, 4), FRESH, dtype=np.int8)
    bucket[:5, 0] = BLIND
    table = bucket_quantiles(score, bucket, 0.1)
    assert table[BLIND, 0] == table[STALE, 0]


def test_evidence_intervals_widen_for_silent_sensors_without_their_outcomes() -> None:
    rng = np.random.default_rng(4)
    origins, nodes = 700, 6
    fresh_error = rng.normal(0, 1, (origins, 1, nodes))
    shadow_error = rng.normal(0, 5, (origins, 1, nodes))
    bucket = np.full((origins, nodes), FRESH, dtype=np.int8)
    bucket[:, 0] = BLIND
    score = np.abs(fresh_error)
    score[:, :, 0] = np.nan  # a silent sensor never reports its outcome
    shadow_bucket = np.full((origins, nodes), BLIND, dtype=np.int8)
    state = {
        "score": score,
        "bucket": bucket,
        "shadow_score": np.abs(shadow_error),
        "shadow_bucket": shadow_bucket,
    }
    point, scale = np.zeros((origins, 1, nodes)), np.ones((origins, 1, nodes))
    lower, upper = evidence_intervals(state, state, point, scale, block_size=100, window=300)
    assert (upper - lower)[-1, 0, 0] > 3 * (upper - lower)[-1, 0, 1]
    assert np.isfinite(upper).all()
