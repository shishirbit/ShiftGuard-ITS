import numpy as np

from shiftguard.metrics import horizon_metrics, point_metrics


def test_point_metrics() -> None:
    target = np.array([1.0, 2.0, 4.0])
    prediction = np.array([2.0, 2.0, 2.0])
    result = point_metrics(target, prediction)
    assert np.isclose(result["mae"], 1.0)
    assert result["rmse"] >= result["mae"]
    assert "h3" in horizon_metrics(np.ones((2, 12, 3)), np.ones((2, 12, 3)))

