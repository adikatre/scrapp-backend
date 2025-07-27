#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_taco_local.py

Convert a locally downloaded TACO dataset (COCO-style) into YOLO format.
No downloading occurs here. You must set --src-dir to your local dataset,
for example: --src-dir taco-trash-dataset

The script:
  1) Reads annotations.json (COCO).
  2) Optionally reads meta_df.csv to get image width/height if needed.
  3) Converts segmentations/bboxes to YOLO [cls cx cy w h] labels.
  4) Splits into train/val/test.
  5) Writes datasets/taco_yolo/taco.yaml for Ultralytics.

USAGE:
  python scripts/01_prepare_taco_local.py \
      --src-dir taco-trash-dataset \
      --out-dir datasets/taco_yolo \
      --train 0.8 --val 0.1 --test 0.1 --seed 1337
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True,
                        help="Path to local TACO dataset root (contains annotations.json, meta_df.csv, data/...).")
    parser.add_argument("--out-dir", default="datasets/taco_yolo",
                        help="Output directory to write YOLO dataset.")
    parser.add_argument("--train", type=float, default=0.8,
                        help="Fraction of images for training.")
    parser.add_argument("--val", type=float, default=0.1,
                        help="Fraction of images for validation.")
    parser.add_argument("--test", type=float, default=0.1,
                        help="Fraction of images for test.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for splits.")
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    images_train = out_dir / "images" / "train"
    images_val = out_dir / "images" / "val"
    images_test = out_dir / "images" / "test"

    labels_train = out_dir / "labels" / "train"
    labels_val = out_dir / "labels" / "val"
    labels_test = out_dir / "labels" / "test"

    for directory in [images_train, images_val, images_test,
                      labels_train, labels_val, labels_test]:
        directory.mkdir(parents=True, exist_ok=True)

    dirs = {
        "images_train": images_train,
        "images_val": images_val,
        "images_test": images_test,
        "labels_train": labels_train,
        "labels_val": labels_val,
        "labels_test": labels_test
    }
    return dirs


def load_coco(src_dir: Path) -> Dict:
    ann_path = src_dir / "annotations.json"
    if not ann_path.exists():
        message = f"annotations.json not found at {ann_path}"
        raise FileNotFoundError(message)

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return coco


def build_category_mappings(coco: Dict) -> Tuple[List[int], List[str], Dict[int, int]]:
    cat_ids_sorted = sorted([cat["id"] for cat in coco["categories"]])
    class_names = []
    for cat_id in cat_ids_sorted:
        # find category name by id
        name = None
        for c in coco["categories"]:
            if c["id"] == cat_id:
                name = c["name"]
                break
        if name is None:
            name = f"class_{cat_id}"
        class_names.append(name)

    catid_to_yoloid = {}
    yolo_index = 0
    for cat_id in cat_ids_sorted:
        catid_to_yoloid[cat_id] = yolo_index
        yolo_index += 1

    return cat_ids_sorted, class_names, catid_to_yoloid


def load_meta_df_sizes(src_dir: Path) -> Dict[str, Tuple[int, int]]:
    """
    Read meta_df.csv if present. Build a dict: {img_file -> (width, height)}.
    The screenshot shows columns: img_file, img_width, img_height, etc.
    """
    sizes = {}
    meta_path = src_dir / "meta_df.csv"
    if not meta_path.exists():
        return sizes

    df = pd.read_csv(meta_path)
    if "img_file" not in df.columns or "img_width" not in df.columns or "img_height" not in df.columns:
        return sizes

    for _, row in df.iterrows():
        key = str(row["img_file"])
        width = int(row["img_width"])
        height = int(row["img_height"])
        sizes[key] = (width, height)
    return sizes


def segmentation_to_bbox(segmentation: List[List[float]], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute YOLO normalized bbox from segmentation polygons.
    """
    xs = []
    ys = []

    for seg in segmentation:
        arr = np.array(seg, dtype=np.float32)
        arr = arr.reshape(-1, 2)
        for point in arr:
            xs.append(point[0])
            ys.append(point[1])

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min = float(np.min(xs))
    y_min = float(np.min(ys))
    x_max = float(np.max(xs))
    y_max = float(np.max(ys))

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = (x_max - x_min)
    h = (y_max - y_min)

    if w <= 0 or h <= 0:
        return None

    cx = cx / float(img_w)
    cy = cy / float(img_h)
    w = w / float(img_w)
    h = h / float(img_h)

    if cx < 0.0:
        cx = 0.0
    if cx > 1.0:
        cx = 1.0
    if cy < 0.0:
        cy = 0.0
    if cy > 1.0:
        cy = 1.0
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    if h < 0.0:
        h = 0.0
    if h > 1.0:
        h = 1.0

    return (cx, cy, w, h)


def coco_bbox_to_yolo(bbox_xywh: List[float], img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert COCO [x, y, w, h] to YOLO normalized (cx, cy, w, h).
    """
    if len(bbox_xywh) < 4:
        return None

    x = float(bbox_xywh[0])
    y = float(bbox_xywh[1])
    w = float(bbox_xywh[2])
    h = float(bbox_xywh[3])

    if w <= 0 or h <= 0:
        return None

    cx = x + w / 2.0
    cy = y + h / 2.0

    cx = cx / float(img_w)
    cy = cy / float(img_h)
    w = w / float(img_w)
    h = h / float(img_h)

    if cx < 0.0:
        cx = 0.0
    if cx > 1.0:
        cx = 1.0
    if cy < 0.0:
        cy = 0.0
    if cy > 1.0:
        cy = 1.0
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    if h < 0.0:
        h = 0.0
    if h > 1.0:
        h = 1.0

    return (cx, cy, w, h)


def resolve_image_path(src_dir: Path, file_name: str) -> Optional[Path]:
    """
    Try to resolve an image path from a TACO file_name.
    TACO files usually live under src_dir/data/batch_*/<file>.
    """
    candidate = src_dir / file_name
    if candidate.exists():
        return candidate

    base_name = Path(file_name).name
    data_dir = src_dir / "data"
    if not data_dir.exists():
        return None

    matches = []
    for p in data_dir.rglob(base_name):
        matches.append(p)

    if len(matches) == 0:
        return None

    # Choose the first match found
    result = matches[0]
    return result


def write_dataset_yaml(out_dir: Path, class_names: List[str]) -> None:
    yaml_path = out_dir / "taco.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# datasets/taco_yolo/taco.yaml\n")
        f.write("# Auto-generated by 01_prepare_taco_local.py\n")
        f.write(f"path: {str(out_dir.resolve())}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"  - {name}\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    dirs = ensure_dirs(out_dir)
    coco = load_coco(src_dir)

    cat_ids_sorted, class_names, catid_to_yoloid = build_category_mappings(coco)

    images_meta = {}
    for img in coco["images"]:
        images_meta[img["id"]] = img

    annotations_by_image = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Optional: pull width/height from meta_df.csv when missing
    meta_sizes = load_meta_df_sizes(src_dir)

    img_ids = []
    for key in images_meta.keys():
        img_ids.append(key)

    random.shuffle(img_ids)

    total = len(img_ids)
    n_train = int(total * args.train)
    n_val = int(total * args.val)
    # Remaining goes to test
    n_test = total - n_train - n_val

    train_ids = set()
    val_ids = set()
    test_ids = set()

    index = 0
    while index < n_train:
        train_ids.add(img_ids[index])
        index += 1
    end_val = n_train + n_val
    while index < end_val:
        val_ids.add(img_ids[index])
        index += 1
    while index < total:
        test_ids.add(img_ids[index])
        index += 1

    print("==> Converting COCO → YOLO and copying images...")
    for img_id in tqdm(img_ids, total=len(img_ids)):
        img_meta = images_meta[img_id]
        file_name = img_meta["file_name"]

        src_img_path = resolve_image_path(src_dir, file_name)
        if src_img_path is None or not src_img_path.exists():
            # Skip if image cannot be resolved
            continue

        if img_id in train_ids:
            dst_img_dir = dirs["images_train"]
            dst_lbl_dir = dirs["labels_train"]
        elif img_id in val_ids:
            dst_img_dir = dirs["images_val"]
            dst_lbl_dir = dirs["labels_val"]
        else:
            dst_img_dir = dirs["images_test"]
            dst_lbl_dir = dirs["labels_test"]

        dst_img_path = dst_img_dir / src_img_path.name
        shutil.copy2(src_img_path, dst_img_path)

        # Determine image size
        img_w = img_meta.get("width", None)
        img_h = img_meta.get("height", None)

        if img_w is None or img_h is None:
            # Try meta_df.csv
            size_key = str(file_name)
            if size_key in meta_sizes:
                size_tuple = meta_sizes[size_key]
                img_w = size_tuple[0]
                img_h = size_tuple[1]
            else:
                # Fallback to reading the image
                image_mat = cv2.imread(str(dst_img_path))
                if image_mat is not None:
                    img_h = int(image_mat.shape[0])
                    img_w = int(image_mat.shape[1])

        if img_w is None or img_h is None:
            # Cannot compute a label without dimensions
            # Create an empty label file and continue
            dst_lbl_path = dst_lbl_dir / (dst_img_path.stem + ".txt")
            with open(dst_lbl_path, "w") as f:
                pass
            continue

        # Build label lines
        lines = []
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                cat_id = ann.get("category_id", None)
                if cat_id is None:
                    continue
                if cat_id not in catid_to_yoloid:
                    continue

                yolo_id = catid_to_yoloid[cat_id]
                yolo_box = None

                seg = ann.get("segmentation", None)
                if isinstance(seg, list) and len(seg) > 0:
                    yolo_box = segmentation_to_bbox(seg, img_w, img_h)

                if yolo_box is None:
                    bbox = ann.get("bbox", None)
                    if bbox is not None and len(bbox) >= 4:
                        yolo_box = coco_bbox_to_yolo(bbox, img_w, img_h)

                if yolo_box is None:
                    continue

                cx, cy, w, h = yolo_box
                line = f"{yolo_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                lines.append(line)

        # Write label file (empty file is fine if no annotations)
        dst_lbl_path = dst_lbl_dir / (dst_img_path.stem + ".txt")
        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
                f.write("\n")

    write_dataset_yaml(out_dir, class_names)

    print("\n✅ Done.")
    print(f"   YOLO dataset at: {str(out_dir.resolve())}")
    print(f"   Data file: {str((out_dir / 'taco.yaml').resolve())}")
    print(f"   Classes: {len(class_names)}")


if __name__ == "__main__":
    main()
