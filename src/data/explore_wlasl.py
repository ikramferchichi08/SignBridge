"""Create simple WLASL metadata plots and a summary report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def explore(metadata_path: Path = Path("data/raw/wlasl/metadata.csv")) -> Path:
    metadata = pd.read_csv(metadata_path)
    report_dir = Path("reports/wlasl")
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = metadata["label"].value_counts().sort_values(ascending=False)
    counts.plot(kind="bar", figsize=(16, 6), color="#227ee6")
    plt.ylabel("Videos")
    plt.xlabel("Sign")
    plt.tight_layout()
    plt.savefig(report_dir / "class_distribution.png", dpi=150)
    plt.close()
    metadata["duration_sec"].plot(kind="hist", bins=20, color="#1eb8a6")
    plt.xlabel("Duration (seconds)")
    plt.tight_layout()
    plt.savefig(report_dir / "duration_hist.png", dpi=150)
    plt.close()
    signer_note = "signer information is missing"
    signer_count = 0
    signer_lines = ""
    if "signer_id" in metadata.columns and metadata["signer_id"].notna().any():
        signer_count = metadata["signer_id"].nunique()
        signer_note = f"{signer_count} distinct signers; signer-aware splitting is feasible"
        per_sign = metadata.groupby("label")["signer_id"].nunique().sort_values()
        signer_lines = "\n".join(f"- `{label}`: {count} signers" for label, count in per_sign.items())
    unreadable = metadata[(metadata["width"] <= 0) | (metadata["height"] <= 0)]
    summary = f"""# WLASL subset exploration

- Videos: {len(metadata)}
- Signs: {metadata["label"].nunique()}
- Signers: {signer_note}
- Duration (seconds): mean {metadata["duration_sec"].mean():.2f}, min {metadata["duration_sec"].min():.2f}, max {metadata["duration_sec"].max():.2f}
- FPS: mean {metadata["fps"].mean():.2f}, min {metadata["fps"].min():.2f}, max {metadata["fps"].max():.2f}
- Corrupted/unreadable videos: {len(unreadable)}

## Signers per sign

{signer_lines or "Signer IDs were unavailable, so signer-aware splitting is not possible from this metadata."}
"""
    (report_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return report_dir


if __name__ == "__main__":
    explore()
