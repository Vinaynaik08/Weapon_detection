#!/usr/bin/env python3
"""
Convert YOLO segmentation polygon annotations (class x1 y1 x2 y2 ...)
to YOLO detection bbox format (class x_center y_center width height).
"""

import os
import glob
import argparse
from tqdm import tqdm

def polygon_to_bbox(points):
    """
    Convert polygon points to YOLO bbox format (normalized).
    points = [x1, y1, x2, y2, ...] all between 0-1
    """
    xs = points[0::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Clamp values between 0 and 1
    x_min = max(0.0, min(1.0, x_min))
    x_max = max(0.0, min(1.0, x_max))
    y_min = max(0.0, min(1.0, y_min))
    y_max = max(0.0, min(1.0, y_max))

    # Convert to YOLO bbox format
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def convert_split_labels(input_dir, output_dir):
    """
    Convert all polygon .txt files in one split (train/valid/test).
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    for txt_file in tqdm(txt_files, desc=f"Converting {input_dir}"):
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        new_lines = []
        for line in lines:
            parts = line.split()
            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            if len(coords) < 4 or len(coords) % 2 != 0:
                continue  # skip malformed polygons

            x_center, y_center, w, h = polygon_to_bbox(coords)
            if w <= 0 or h <= 0:
                continue  # skip degenerate boxes

            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        out_path = os.path.join(output_dir, os.path.basename(txt_file))
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)


def main(dataset_root):
    splits = ["train", "valid", "test"]
    for split in splits:
        input_dir = os.path.join(dataset_root, split, "labels")
        output_dir = os.path.join(dataset_root, split, "labels_bbox")

        if not os.path.exists(input_dir):
            print(f"⚠️ Skipping {split}, no labels folder found.")
            continue

        convert_split_labels(input_dir, output_dir)
        print(f"✅ Finished {split}. Converted labels saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO polygons → YOLO bboxes")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset folder (contains train/valid/test)")
    args = parser.parse_args()
    main(args.dataset_root)
