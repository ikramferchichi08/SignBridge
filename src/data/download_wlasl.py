"""Download and prepare a bounded WLASL subset from Hugging Face."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Voxel51/WLASL"
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/"
    "start_kit/WLASL_v0.3.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--name", default="wlasl_test")
    parser.add_argument("--top-k-signs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/wlasl"))
    parser.add_argument("--discover-only", action="store_true")
    return parser.parse_args()


def discover_with_fiftyone(max_samples: int, name: str) -> bool:
    """Try the requested FiftyOne loader and print its actual schema."""
    try:
        import fiftyone.utils.huggingface as fouh

        dataset = fouh.load_from_hub(
            REPO_ID, max_samples=max_samples, persistent=True, name=name
        )
        print("DATASET:")
        print(dataset)
        print("\nFIELD SCHEMA:")
        print(dataset.get_field_schema())
        print("\nFIRST SAMPLE:")
        print(dataset.first())
        return True
    except Exception as error:
        print(f"FiftyOne discovery failed: {error}")
        return False


def load_annotations() -> list[dict]:
    with urllib.request.urlopen(ANNOTATIONS_URL) as response:
        groups = json.load(response)
    annotations = []
    for group in groups:
        for instance in group["instances"]:
            annotations.append({"label": group["gloss"], **instance})
    return annotations


def hf_video_index() -> dict[str, tuple[str, int | None]]:
    api = HfApi()
    index = {}
    for item in api.list_repo_tree(REPO_ID, path_in_repo="data", repo_type="dataset", recursive=True):
        if item.path.endswith(".mp4"):
            index[Path(item.path).stem] = (item.path, getattr(item, "size", None))
    return index


def select_annotations(annotations: list[dict], index: dict, max_samples: int, top_k: int) -> list[dict]:
    available = [row for row in annotations if row["video_id"] in index]
    counts = Counter(row["label"] for row in available)
    kept_labels = [label for label, _ in counts.most_common(top_k)]
    per_label = max(1, math.ceil(max_samples / len(kept_labels)))
    selected = []
    for label in kept_labels:
        rows = [row for row in available if row["label"] == label]
        selected.extend(rows[:per_label])
    return selected[:max_samples]


def video_metadata(path: Path) -> tuple[float, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0, 0.0, 0, 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return (frames / fps if fps else 0.0), fps, width, height


def export_subset(selected: list[dict], index: dict, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, annotation in enumerate(selected, start=1):
        repo_path, _ = index[annotation["video_id"]]
        source = Path(hf_hub_download(REPO_ID, repo_path, repo_type="dataset"))
        destination = output_dir / annotation["label"] / f"{annotation['video_id']}.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        duration, fps, width, height = video_metadata(destination)
        rows.append(
            {
                "video_path": destination.resolve().relative_to(Path.cwd().resolve()).as_posix(),
                "label": annotation["label"],
                "signer_id": annotation.get("signer_id", ""),
                "original_split": annotation.get("split", ""),
                "duration_sec": duration,
                "fps": fps,
                "width": width,
                "height": height,
            }
        )
        if number % 25 == 0:
            print(f"Downloaded {number}/{len(selected)} videos")
    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    return metadata


def main() -> int:
    args = parse_args()
    if args.discover_only:
        if discover_with_fiftyone(args.max_samples, args.name):
            return 0
        print("Fallback fields: label, signer_id, split, fps, video_id, source, bbox")
        return 0

    annotations = load_annotations()
    index = hf_video_index()
    selected = select_annotations(annotations, index, args.max_samples, args.top_k_signs)
    counts = Counter(row["label"] for row in selected)
    print(f"Indexed Hugging Face videos: {len(index)}")
    print(f"Selected videos: {len(selected)}; signs: {len(counts)}")
    print("Top selected labels:", counts.most_common(20))
    print(
        "Kept-sign count range:",
        min(counts.values()) if counts else 0,
        "to",
        max(counts.values()) if counts else 0,
    )
    metadata = export_subset(selected, index, args.output_dir)
    print(f"Wrote {len(metadata)} rows to {args.output_dir / 'metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
