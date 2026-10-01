#!/usr/bin/env python3
"""Convert MoWorld source/latent CSV metadata to MindSpeed-MM JSONL."""

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--mode", choices=("raw", "feature"), required=True)
    parser.add_argument("--prompt-column", default="prompt1")
    parser.add_argument("--num-frames", type=int, default=125)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--limit", type=int, default=0, help="Write only N records (0 = all)")
    parser.add_argument("--check-files", action="store_true")
    return parser.parse_args()


def first_value(row, *keys, default=None):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def raw_record(row, prompt_column, num_frames, height, width):
    root = os.path.expanduser(first_value(row, "obs_path", "folder_path", default=""))
    video = first_value(row, "video", "video_path")
    if not video:
        video = os.path.join(root, "video.mp4")
    start = int(float(first_value(row, "start_frame", default=0)))
    end = int(float(first_value(row, "end_frame", default=start + num_frames)))
    prompt = first_value(row, prompt_column, "prompt", "caption", "cap", default="")
    return {
        "file": video,
        "path": video,
        "captions": prompt,
        "cap": prompt,
        "cut": [start],
        "start_frame": start,
        "end_frame": end,
        "num_frames": num_frames,
        "sample_num_frames": num_frames,
        "sample_size": f"{num_frames}x{height}x{width}",
    }


def feature_record(row, prompt_column, num_frames, height, width):
    feature = first_value(row, "latent_path", "feature_path", "file")
    if not feature:
        raise ValueError("feature mode requires latent_path, feature_path, or file")
    start = int(float(first_value(row, "start_frame", default=0)))
    end = int(float(first_value(row, "end_frame", default=start + num_frames)))
    prompt = first_value(row, prompt_column, "prompt", "caption", "cap", default="")
    return {
        "file": os.path.expanduser(feature),
        "captions": prompt,
        "cap": prompt,
        "start_frame": start,
        "end_frame": end,
        "sample_num_frames": num_frames,
        "sample_size": f"{num_frames}x{height}x{width}",
    }


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_record = raw_record if args.mode == "raw" else feature_record
    written = 0
    missing = []
    with open(args.input, newline="", encoding="utf-8-sig") as src, output.open("w", encoding="utf-8") as dst:
        for line_no, row in enumerate(csv.DictReader(src), start=2):
            if args.limit and written >= args.limit:
                break
            try:
                record = make_record(
                    row, args.prompt_column, args.num_frames, args.height, args.width
                )
            except Exception as exc:
                raise ValueError(f"invalid CSV row {line_no}: {exc}") from exc
            if args.check_files and not os.path.isfile(record["file"]):
                missing.append(record["file"])
                continue
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} input files are missing; first entries:\n{preview}")
    print(f"wrote {written} records to {output}")


if __name__ == "__main__":
    main()

