"""Download a WLASL subset (only the glosses you need) and rebuild metadata.csv.

Usage (PowerShell, from the SignBridge project root, venv active):

    python src/download_wlasl_subset.py ^
        --wlasl-json "C:/path/to/WLASL-master/WLASL-master/start_kit/WLASL_v0.3.json"

What it does
1. Reads the glosses from data/raw/wlasl/metadata.csv (your current 100 signs).
   If that file does not exist, it takes the top --top glosses of the JSON.
2. For every video of those glosses that is not already in data/raw/wlasl/<gloss>/,
   downloads the source (YouTube via yt-dlp, other sites via plain HTTP),
   cuts it to the sign (frame_start / frame_end) and saves <video_id>.mp4.
3. Rebuilds data/raw/wlasl/metadata.csv for ALL videos on disk.
4. Writes reports/wlasl_download_log.csv (status of every attempted video).

Existing videos are never overwritten, so it is safe to run it again.
Many source links are dead: a partial result is normal.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

UA = {"User-Agent": "Mozilla/5.0 (compatible; SignBridge-student-project)"}


# ----------------------------------------------------------------- selection
def load_glosses(wlasl: list[dict], meta_csv: Path, top: int) -> list[str]:
    if meta_csv.exists():
        with meta_csv.open(newline="", encoding="utf-8") as f:
            labels = sorted({row["label"] for row in csv.DictReader(f)})
        print(f"{len(labels)} glosses taken from {meta_csv}")
        return labels
    labels = [entry["gloss"] for entry in wlasl[:top]]
    print(f"{meta_csv} not found: using the top {top} glosses of the JSON")
    return labels


# ------------------------------------------------------------------ download
def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def download_raw(url: str, stem: Path) -> Path:
    """Download url to a file starting with `stem`; return the file path."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    if is_youtube(url):
        import yt_dlp

        opts = {
            "outtmpl": str(stem) + ".%(ext)s",
            "format": "mp4/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 20,
            "retries": 2,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        found = sorted(stem.parent.glob(stem.name + ".*"))
        if not found:
            raise RuntimeError("yt-dlp produced no file")
        return found[0]

    dst = stem.with_suffix(".mp4")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp, dst.open("wb") as out:
        shutil.copyfileobj(resp, out)
    if dst.stat().st_size < 1024:
        raise RuntimeError("downloaded file is too small")
    return dst


def cut_clip(src: Path, dst: Path, frame_start: int, frame_end: int) -> None:
    """Keep frames [frame_start, frame_end] (1-based, -1 = until the end)."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open the video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10**9
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = max(frame_start - 1, 0)
    end = total if frame_end is None or frame_end <= 0 else min(frame_end, total)

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    written, idx = 0, start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        idx += 1
    cap.release()
    writer.release()
    if written < 5:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"only {written} frames written")


def process_instance(job: dict, cache: Path, keep_raw: bool) -> dict:
    vid, gloss, url = job["video_id"], job["gloss"], job["url"]
    dst = job["dst"]
    raw = None
    try:
        raw = download_raw(url, cache / vid)
        cut_clip(raw, dst, job["frame_start"], job["frame_end"])
        status, detail = "ok", ""
    except Exception as exc:  # noqa: BLE001 - we want to log every failure
        status, detail = "failed", str(exc).splitlines()[0][:150]
    finally:
        if raw is not None and not keep_raw:
            raw.unlink(missing_ok=True)
    host = url.split("/")[2] if "//" in url else url
    return {"video_id": vid, "gloss": gloss, "host": host,
            "status": status, "detail": detail}


# ------------------------------------------------------------------ metadata
def rebuild_metadata(out_dir: Path, info: dict[str, dict]) -> int:
    rows = []
    for mp4 in sorted(out_dir.glob("*/*.mp4")):
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        meta = info.get(mp4.stem, {})
        rows.append({
            "video_path": mp4.as_posix(),
            "label": mp4.parent.name,
            "signer_id": meta.get("signer_id", ""),
            "original_split": meta.get("split", ""),
            "duration_sec": (frames / fps) if fps else 0.0,
            "fps": fps,
            "width": w,
            "height": h,
        })
    cols = ["video_path", "label", "signer_id", "original_split",
            "duration_sec", "fps", "width", "height"]
    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--wlasl-json", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/raw/wlasl"))
    ap.add_argument("--cache", type=Path, default=Path("data/_wlasl_cache"))
    ap.add_argument("--log", type=Path,
                    default=Path("reports/wlasl_download_log.csv"))
    ap.add_argument("--top", type=int, default=100,
                    help="used only when metadata.csv does not exist")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="try only N new videos (quick test)")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--metadata-only", action="store_true",
                    help="skip downloads, just rebuild metadata.csv")
    args = ap.parse_args()

    wlasl = json.loads(args.wlasl_json.read_text(encoding="utf-8"))
    glosses = set(load_glosses(wlasl, args.out / "metadata.csv", args.top))

    info, jobs = {}, []
    for entry in wlasl:
        gloss = entry["gloss"]
        if gloss not in glosses:
            continue
        for inst in entry["instances"]:
            vid = str(inst["video_id"])
            info[vid] = inst
            dst = args.out / gloss / f"{vid}.mp4"
            if dst.exists():
                continue
            jobs.append({
                "video_id": vid, "gloss": gloss, "url": inst["url"],
                "frame_start": inst.get("frame_start", 1),
                "frame_end": inst.get("frame_end", -1), "dst": dst,
            })

    present = sum(1 for _ in args.out.glob("*/*.mp4"))
    print(f"already on disk: {present} | new videos to try: {len(jobs)}")
    if args.limit:
        jobs = jobs[: args.limit]
        print(f"--limit: trying only {len(jobs)}")

    results = []
    if jobs and not args.metadata_only:
        args.cache.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(process_instance, j, args.cache, args.keep_raw)
                    for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                results.append(res)
                if i % 25 == 0 or i == len(futs):
                    ok = sum(r["status"] == "ok" for r in results)
                    print(f"  {i}/{len(futs)} tried, {ok} ok", flush=True)
        if not args.keep_raw:
            shutil.rmtree(args.cache, ignore_errors=True)

        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(
                f, fieldnames=["video_id", "gloss", "host", "status", "detail"])
            wr.writeheader()
            wr.writerows(results)
        print(f"log written to {args.log}")

    n = rebuild_metadata(args.out, info)
    per_gloss = {}
    for p in args.out.glob("*/*.mp4"):
        per_gloss[p.parent.name] = per_gloss.get(p.parent.name, 0) + 1
    if per_gloss:
        counts = sorted(per_gloss.values())
        print(f"\nmetadata.csv rebuilt: {n} videos, {len(per_gloss)} glosses, "
              f"{counts[0]}-{counts[-1]} videos per gloss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
