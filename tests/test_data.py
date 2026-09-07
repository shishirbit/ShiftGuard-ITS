import numpy as np
import pandas as pd

from shiftguard.data import prepare


def test_prepare_is_chronological_and_train_normalized() -> None:
    index = pd.date_range("2020-01-01", periods=100, freq="5min")
    frame = pd.DataFrame({"a": np.arange(100), "b": np.arange(100) * 2}, index=index)
    splits, scaler = prepare(frame, history=4, horizon=2)
    assert splits["train"].timestamps.max() < splits["val"].timestamps.min()
    assert splits["val"].timestamps.max() < splits["test"].timestamps.min()
    assert np.allclose(scaler.mean, [[35.0, 70.0]])
    assert splits["train"].x.dtype == np.float32
    assert splits["train"].x.shape[-1] == 4
    assert splits["train"].y.shape[-1] == 2
