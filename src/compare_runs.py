"""Collect the finished MLflow runs into one comparison table.

Usage (PowerShell, from the project root, venv active):

    python -m src.compare_runs

Writes reports/model_comparison.csv (raw numbers) and reports/model_comparison.md
(table for the report/slides, best top-1 first) and prints the table.

Only complete cross-validation runs are listed: unfinished runs, quick tests made with
--limit-folds, and the per-fold child runs are skipped. If a run name was used more than
once, only the latest run is kept.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src.utils.mlflow_utils import configure_mlflow, load_params

FEATURE_FLAGS = [
    ("hand_local", "hand-local"),
    ("canonical_hand", "canonical-hand"),
    ("delta", "delta"),
    ("bidirectional", "bidirectional"),
]


def _column(df: pd.DataFrame, name: str, default=None) -> pd.Series:
    if name in df:
        return df[name]
    return pd.Series(default, index=df.index, dtype=object)


def build_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Turn the result of mlflow.search_runs into one row per complete model run."""
    df = runs.copy()
    if "tags.mlflow.parentRunId" in df:
        df = df[df["tags.mlflow.parentRunId"].isna()]  # drop the per-fold child runs
    df = df[df["status"] == "FINISHED"]
    if "params.limit_folds" in df:
        df = df[df["params.limit_folds"].fillna("0").astype(str) == "0"]  # drop quick tests
    df = df[df["metrics.mean_top1"].notna()]
    df = df.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")

    features = []
    for _, row in df.iterrows():
        used = [label for key, label in FEATURE_FLAGS
                if str(row.get(f"params.{key}", "False")) == "True"]
        features.append(", ".join(used) if used else "raw landmarks")

    table = pd.DataFrame({
        "run": df["tags.mlflow.runName"],
        "model": _column(df, "tags.model").fillna("gru"),
        "features": features,
        "epochs": _column(df, "params.epochs"),
        "lr": _column(df, "params.lr"),
        "top1_mean": df["metrics.mean_top1"],
        "top1_std": df["metrics.std_top1"],
        "top5_mean": df["metrics.mean_top5"],
        "macro_f1_mean": df["metrics.mean_macro_f1"],
        "macro_f1_std": df["metrics.std_macro_f1"],
        "params": _column(df, "metrics.params"),
        "model_mb": df["metrics.mean_model_mb"],
        "latency_ms": df["metrics.mean_latency_ms"],
    })
    return table.sort_values("top1_mean", ascending=False).reset_index(drop=True)


def to_markdown(table: pd.DataFrame) -> str:
    header = ["Run", "Model", "Features", "Top-1 (%)", "Top-5 (%)", "Macro-F1",
              "Params", "Size (MB)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, r in table.iterrows():
        params = "" if pd.isna(r["params"]) else f"{int(r['params']):,}"
        lines.append(
            f"| {r['run']} | {r['model']} | {r['features']} "
            f"| {100 * r['top1_mean']:.1f} ± {100 * r['top1_std']:.1f} "
            f"| {100 * r['top5_mean']:.1f} "
            f"| {r['macro_f1_mean']:.3f} ± {r['macro_f1_std']:.3f} "
            f"| {params} | {r['model_mb']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    params = load_params()
    configure_mlflow(params)
    runs = mlflow.search_runs(experiment_names=[params.get("experiment_name", "signbridge")],
                              output_format="pandas")
    if runs.empty:
        print("No MLflow runs found.")
        return 1
    table = build_table(runs)
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    markdown = to_markdown(table)
    (out_dir / "model_comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Latency is not in the table: measure it once in a clean run before quoting it.")
    print(f"saved: {out_dir / 'model_comparison.csv'} and {out_dir / 'model_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
