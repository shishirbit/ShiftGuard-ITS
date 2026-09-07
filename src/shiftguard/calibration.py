"""Prequential group-adaptive conformal calibration and selective metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def finite_mean(values: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    """Mean over finite entries, returning NaN for an entirely missing group."""

    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = finite.sum(axis=axis)
    return np.divide(
        np.where(finite, array, 0.0).sum(axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal order statistic."""

    clean = np.asarray(scores, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("inf")
    rank = min(len(clean), int(np.ceil((len(clean) + 1) * (1 - alpha))))
    return float(np.partition(clean, rank - 1)[rank - 1])


def conformal_group_quantiles(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized conformal quantiles over the first (calibration) axis."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("group scores require a calibration axis and group axes")
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    ranks = np.minimum(counts, np.ceil((counts + 1) * (1 - alpha)).astype(int))
    indices = np.maximum(ranks - 1, 0)
    partitioned = np.partition(
        np.where(finite, values, np.inf), np.unique(indices), axis=0
    )
    quantiles = np.take_along_axis(partitioned, indices[None], axis=0)[0]
    quantiles[counts == 0] = np.inf
    return quantiles


@dataclass
class RollingGroupConformal:
    alpha: float = 0.1
    window: int = 2016
    min_group: int = 48
    shrinkage: float = 48.0

    def intervals(
        self, prediction: np.ndarray, residual_history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrate each horizon/node using past residuals and global shrinkage.

        ``prediction`` has shape ``(H,N)`` and history ``(W,H,N)``. A convex
        shrinkage toward the global quantile prevents tiny groups from producing
        unstable intervals. The returned width is also the abstention score.
        """

        pred = np.asarray(prediction, dtype=np.float64)
        history = np.abs(np.asarray(residual_history, dtype=np.float64)[-self.window :])
        if history.ndim != 3 or pred.shape != history.shape[1:]:
            raise ValueError("expected prediction (H,N), history (W,H,N)")
        global_q = conformal_quantile(history.ravel(), self.alpha)
        quantiles = np.empty_like(pred)
        for h in range(pred.shape[0]):
            for n in range(pred.shape[1]):
                group = history[:, h, n]
                group = group[np.isfinite(group)]
                if len(group) < self.min_group:
                    group_q = global_q
                else:
                    group_q = conformal_quantile(group, self.alpha)
                weight = len(group) / (len(group) + self.shrinkage)
                quantiles[h, n] = weight * group_q + (1 - weight) * global_q
        return pred - quantiles, pred + quantiles, 2 * quantiles


def interval_metrics(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict[str, float]:
    target = np.asarray(target)
    finite = np.isfinite(target) & np.isfinite(lower) & np.isfinite(upper)
    target, lower, upper = target[finite], lower[finite], upper[finite]
    covered = (target >= lower) & (target <= upper)
    width = upper - lower
    penalty = (2 / alpha) * (
        np.maximum(lower - target, 0) + np.maximum(target - upper, 0)
    )
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "winkler": float(np.mean(width + penalty)),
    }


def risk_coverage_curve(
    errors: np.ndarray, uncertainty: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return coverage, retained MAE, and normalized area under risk-coverage."""

    errors = np.abs(np.asarray(errors, dtype=np.float64).ravel())
    uncertainty = np.asarray(uncertainty, dtype=np.float64).ravel()
    finite = np.isfinite(errors) & np.isfinite(uncertainty)
    errors, uncertainty = errors[finite], uncertainty[finite]
    order = np.argsort(uncertainty)
    cumulative = np.cumsum(errors[order])
    coverage = np.arange(1, len(order) + 1) / len(order)
    risk = cumulative / np.arange(1, len(order) + 1)
    aurc = float(np.trapezoid(risk, coverage))
    return coverage, risk, aurc


@dataclass(frozen=True)
class AbstentionRule:
    uncertainty_threshold: float
    calibration_coverage: float
    empirical_failure_rate: float
    upper_failure_bound: float

    def accept(self, uncertainty: np.ndarray) -> np.ndarray:
        return np.asarray(uncertainty) <= self.uncertainty_threshold


def prequential_calibration(
    calibration_target: np.ndarray,
    calibration_point: np.ndarray,
    calibration_scale: np.ndarray,
    test_target: np.ndarray,
    test_point: np.ndarray,
    test_scale: np.ndarray,
    *,
    alpha: float = 0.1,
    block_size: int = 288,
    window: int = 2016,
    error_tolerance: float = 5.0,
    risk_limit: float = 0.1,
    delta: float = 0.05,
    groupwise: bool = True,
    update: bool = True,
    risk_control: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[AbstentionRule]]:
    """Leakage-safe block-prequential intervals and acceptance decisions.

    At each block boundary, calibration uses validation residuals and only test
    forecast origins whose complete horizon has already elapsed. Thus no target
    from the evaluated block or its unresolved predecessor forecasts is used.
    """

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            calibration_target,
            calibration_point,
            calibration_scale,
            test_target,
            test_point,
            test_scale,
        )
    ]
    cal_target, cal_point, cal_scale, target, point, scale = arrays
    if cal_target.shape != cal_point.shape or cal_target.shape != cal_scale.shape:
        raise ValueError("calibration arrays must have equal shapes")
    if target.shape != point.shape or target.shape != scale.shape:
        raise ValueError("test arrays must have equal shapes")
    if cal_target.shape[1:] != target.shape[1:]:
        raise ValueError("calibration and test groups must match")
    if block_size < 1 or window < 1:
        raise ValueError("block_size and window must be positive")

    lower = np.empty_like(point)
    upper = np.empty_like(point)
    accepted = np.zeros_like(point, dtype=bool)
    rules: list[AbstentionRule] = []
    horizon = target.shape[1]
    for start in range(0, len(target), block_size):
        stop = min(start + block_size, len(target))
        known_stop = max(0, start - horizon) if update else 0
        history_target = np.concatenate([cal_target, target[:known_stop]])[-window:]
        history_point = np.concatenate([cal_point, point[:known_stop]])[-window:]
        history_scale = np.concatenate([cal_scale, scale[:known_stop]])[-window:]
        scores = np.abs(history_target - history_point) / np.maximum(history_scale, 1e-4)
        if groupwise:
            quantiles = conformal_group_quantiles(scores, alpha)
        else:
            quantiles = np.full(target.shape[1:], conformal_quantile(scores, alpha))
        radius = scale[start:stop] * quantiles[None]
        lower[start:stop], upper[start:stop] = point[start:stop] - radius, point[start:stop] + radius
        history_width = 2 * history_scale * quantiles[None]
        group_axes = tuple(range(1, history_target.ndim))
        origin_error = finite_mean(np.abs(history_target - history_point), group_axes)
        origin_uncertainty = finite_mean(history_width, group_axes)
        if risk_control:
            rule = calibrate_abstention(
                origin_error,
                origin_uncertainty,
                error_tolerance=error_tolerance,
                risk_limit=risk_limit,
                delta=delta,
            )
        else:
            historical_error = origin_error[np.isfinite(origin_error)]
            empirical = float(np.mean(historical_error > error_tolerance))
            rule = AbstentionRule(float(np.finfo(np.float64).max), 1.0, empirical, 1.0)
        block_axes = tuple(range(1, radius.ndim))
        block_uncertainty = finite_mean(2 * radius, block_axes)
        origin_acceptance = rule.accept(block_uncertainty)
        accepted[start:stop] = origin_acceptance.reshape(
            (-1,) + (1,) * (accepted.ndim - 1)
        )
        rules.append(rule)
    return lower, upper, accepted, rules


def calibrate_abstention(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    error_tolerance: float = 5.0,
    risk_limit: float = 0.1,
    delta: float = 0.05,
    candidates: int = 50,
) -> AbstentionRule:
    """Select maximum coverage with a split-sample simultaneous risk bound.

    Even forecast origins construct candidate thresholds and odd origins certify
    them. This prevents the same outcomes from defining and validating a rule.
    Bonferroni-corrected exact Clopper--Pearson bounds permit selection among
    candidates under the stated i.i.d. Bernoulli calibration assumption.
    """

    raw_error = np.abs(np.asarray(errors, dtype=np.float64))
    raw_score = np.asarray(uncertainty, dtype=np.float64)
    if raw_error.shape != raw_score.shape:
        raise ValueError("errors and uncertainty must have equal shapes")
    if raw_error.shape[0] < 2:
        raise ValueError("split risk calibration requires at least two origins")
    candidate_score = raw_score[::2].ravel()
    candidate_score = candidate_score[np.isfinite(candidate_score)]
    error = raw_error[1::2].ravel()
    score = raw_score[1::2].ravel()
    finite = np.isfinite(error) & np.isfinite(score)
    error, score = error[finite], score[finite]
    if not len(error) or not len(candidate_score):
        raise ValueError("no finite calibration examples")
    levels = np.linspace(0.02, 1.0, candidates)
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    cumulative_failures = np.cumsum(error[order] > error_tolerance)
    candidate_score.sort()
    quantile_indices = np.ceil(levels * (len(candidate_score) - 1)).astype(int)
    thresholds = np.unique(candidate_score[quantile_indices])
    per_test_delta = delta / len(thresholds)
    best: AbstentionRule | None = None
    for threshold in thresholds:
        n = int(np.searchsorted(sorted_score, threshold, side="right"))
        if n == 0:
            continue
        failures = int(cumulative_failures[n - 1])
        upper = 1.0 if failures == n else float(beta.ppf(1 - per_test_delta, failures + 1, n - failures))
        rule = AbstentionRule(
            uncertainty_threshold=float(threshold),
            calibration_coverage=n / len(error),
            empirical_failure_rate=failures / n,
            upper_failure_bound=upper,
        )
        if upper <= risk_limit and (best is None or rule.calibration_coverage > best.calibration_coverage):
            best = rule
    if best is None:
        return AbstentionRule(float("-inf"), 0.0, 0.0, 0.0)
    return best
