import numpy as np
import pandas as pd

from src.compare_runs import build_table, to_markdown


def make_runs():
    base = {
        "status": "FINISHED",
        "tags.mlflow.parentRunId": np.nan,
        "params.limit_folds": "0",
        "metrics.std_top1": 0.05,
        "metrics.mean_top5": 0.45,
        "metrics.std_macro_f1": 0.04,
        "metrics.params": 273508.0,
        "metrics.mean_model_mb": 1.1,
        "metrics.mean_latency_ms": 10.0,
        "params.epochs": "60",
        "params.lr": "0.002",
    }
    rows = [
        {**base, "tags.mlflow.runName": "gru_baseline", "tags.model": "gru",
         "start_time": 1, "metrics.mean_top1": 0.18, "metrics.mean_macro_f1": 0.17,
         "params.hand_local": "False"},
        {**base, "tags.mlflow.runName": "transformer_handlocal", "tags.model": "transformer",
         "start_time": 2, "metrics.mean_top1": 0.31, "metrics.mean_macro_f1": 0.30,
         "params.hand_local": "True", "params.delta": "True"},
        # to be dropped: quick test, unfinished run, per-fold child run
        {**base, "tags.mlflow.runName": "gru_baseline", "tags.model": "gru",
         "start_time": 3, "metrics.mean_top1": 0.014, "metrics.mean_macro_f1": 0.004,
         "params.limit_folds": "1"},
        {**base, "tags.mlflow.runName": "gru_killed", "status": "KILLED",
         "start_time": 4, "metrics.mean_top1": 0.5, "metrics.mean_macro_f1": 0.5},
        {**base, "tags.mlflow.runName": "fold0", "tags.mlflow.parentRunId": "abc",
         "start_time": 5, "metrics.mean_top1": 0.9, "metrics.mean_macro_f1": 0.9},
    ]
    return pd.DataFrame(rows)


def test_build_table_keeps_only_complete_runs_sorted_by_top1():
    table = build_table(make_runs())
    assert table["run"].tolist() == ["transformer_handlocal", "gru_baseline"]
    assert table.loc[0, "features"] == "hand-local, delta"
    assert table.loc[1, "features"] == "raw landmarks"
    assert table.loc[1, "top1_mean"] == 0.18  # the quick test with the same name was dropped


def test_markdown_has_one_row_per_run():
    markdown = to_markdown(build_table(make_runs()))
    lines = markdown.strip().splitlines()
    assert len(lines) == 2 + 2  # header, separator, two runs
    assert "31.0 ± 5.0" in markdown
