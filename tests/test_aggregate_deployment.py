import importlib.util
from pathlib import Path

import pandas as pd


def load_aggregate():
    spec = importlib.util.spec_from_file_location("aggregate_deployment", Path("scripts/aggregate_deployment.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gating_claims_exclude_the_exploratory_run():
    aggregate = load_aggregate()
    aggregate.RESAMPLES = 20
    record = {"daily_mae": [3.0, 3.1], "evidence": {"daily_mae": [2.9, 3.0]}}
    results = {scenario: record for scenario in aggregate.OUTAGES}
    runs = {
        (dataset, model, seed): results
        for dataset in ("METR-LA", "PEMS-BAY")
        for model in ("shiftguard_noreliability", "staeformer")
        for seed in (1, 12)
    }
    frame = pd.DataFrame(
        [{"dataset": key[0], "model": key[1], "seed": key[2]} for key in runs]
    )
    table = aggregate.paired_intervals(runs, frame)
    seeds = table.set_index(["dataset", "scenario", "contrast"]).seeds
    assert seeds["METR-LA", "trace_x1", "gated-model"] == 1
    assert seeds["METR-LA", "mean_outage", "ref-staeformer|gated"] == 1
    # Point-model contrasts and other datasets keep every seed.
    assert seeds["METR-LA", "mean_outage", "ref-staeformer|model"] == 2
    assert seeds["PEMS-BAY", "trace_x1", "gated-model"] == 2
