#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_yolo.py

Fine-tune an Ultralytics YOLOv11 model on the YOLO-formatted TACO dataset.

USAGE:
  python scripts/02_train_yolo.py \
      --data datasets/taco_yolo/taco.yaml \
      --model yolo11m.pt --epochs 50 --imgsz 640 --batch 16
"""

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/taco_yolo/taco.yaml",
                        help="Path to the data YAML created by 01_prepare_taco_local.py")
    parser.add_argument("--model", default="yolo11m.pt",
                        help="Pretrained YOLO model to fine-tune (e.g., yolo11s.pt, yolo11m.pt).")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training and validation.")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--project", default="runs/detect",
                        help="Where to save runs.")
    parser.add_argument("--name", default="taco_yolo11m",
                        help="Run name.")
    parser.add_argument("--device", default="auto",
                        help="Device selection: 'cpu', 'cuda', 'mps', or 'auto'.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs).")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=False,
        patience=args.patience,
        pretrained=True,
        verbose=True
    )

    # Optional: evaluate with validation metrics (mAP)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name + "_val"
    )

    print("\n✅ Training complete.")
    print(f"Best weights expected at: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
