#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_predict_and_route.py

Predict with a trained YOLO model on:
  - images / directories / videos (one-off), or
  - a webcam stream (continuous; --source 0)

Adds:
  - Robust device selection (auto -> CPU fallback if no CUDA)
  - Per-frame bin guidance overlay
  - Optional live window display (--show)
  - Optional MP4 recording (--save-vid)

USAGE (webcam, CPU-safe):
  python scripts/03_predict_and_route.py \
      --weights runs/detect/taco_yolo11m/weights/best.pt \
      --source 0 \
      --bin-map bin_map.yaml \
      --conf 0.25 --imgsz 640 \
      --device cpu \
      --show \
      --save-vid runs/predict/webcam_out.mp4

USAGE (single image file):
  python scripts/03_predict_and_route.py \
      --weights runs/detect/taco_yolo11m/weights/best.pt \
      --source ../test_imgs/battery.png \
      --bin-map bin_map.yaml \
      --conf 0.25 --imgsz 640 \
      --device cpu
"""

import argparse
from collections import Counter
import os
import sys
import time
import yaml
import cv2
from ultralytics import YOLO

# Import torch only to check CUDA availability.
try:
    import torch
except Exception:
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained YOLO weights (.pt)."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, directory, video path, or webcam index (e.g. '0')."
    )
    parser.add_argument(
        "--bin-map",
        default="bin_map.yaml",
        help="YAML mapping from class name -> disposal route."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size used at prediction."
    )
    parser.add_argument(
        "--project",
        default="runs/predict",
        help="Output project directory (used for non-webcam inputs)."
    )
    parser.add_argument(
        "--name",
        default="taco_demo",
        help="Run name subfolder (used for non-webcam inputs)."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'cpu', '0' for first CUDA GPU, '0,1' for multi-GPU, or 'auto' to pick automatically."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames in a window (useful for webcam)."
    )
    parser.add_argument(
        "--save-vid",
        default="",
        help="Optional MP4 path to save the annotated stream (webcam or video input)."
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """
    Decide which device to pass to Ultralytics.
    - 'cpu' -> 'cpu'
    - 'auto' -> '0' if CUDA available, else 'cpu'
    - anything else (e.g., '0' or '0,1') is passed through
    """
    if requested is None:
        return "cpu"

    text = str(requested).strip().lower()
    if text == "cpu":
        return "cpu"

    if text == "auto":
        if torch is None:
            return "cpu"
        if not torch.cuda.is_available():
            return "cpu"
        return "0"

    return requested


def load_bin_map(bin_map_path: str):
    """
    Load the YAML mapping of class names -> disposal routes.
    Normalize keys to lowercase for case-insensitive lookup.
    """
    if not os.path.exists(bin_map_path):
        print(f"ERROR: bin_map file not found: {bin_map_path}")
        sys.exit(1)

    with open(bin_map_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_map = data.get("mapping", {})
    default_route = data.get("default", "Landfill / Check local rules")

    mapping = {}
    for k, v in raw_map.items():
        key = str(k).strip().lower()
        mapping[key] = v

    return mapping, default_route


def is_webcam_source(source_str: str) -> bool:
    """
    Heuristic: treat as webcam if the source is a small non-negative integer string
    and not an existing file or directory.
    """
    if os.path.exists(source_str):
        return False

    if not source_str.isdigit():
        return False

    index = int(source_str)
    if index < 0:
        return False

    return True


def overlay_bin_summary(frame, route_counter: Counter, x: int = 10, y: int = 30):
    """
    Draw a per-frame bin summary in the top-left corner of the frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (0, 255, 0)
    thickness = 2
    line_height = 22

    if len(route_counter) == 0:
        text = "No items detected"
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return

    header = "Bin guidance (this frame):"
    cv2.putText(frame, header, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    y_cursor = y + line_height
    for route, count in route_counter.most_common():
        text = f"{route}: {count}"
        cv2.putText(frame, text, (x, y_cursor), font, scale, color, thickness, cv2.LINE_AA)
        y_cursor += line_height


def main():
    args = parse_args()

    # Resolve device robustly.
    chosen_device = resolve_device(args.device)
    if args.device == "auto":
        if chosen_device == "cpu":
            print("No CUDA device detected by PyTorch. Falling back to CPU for prediction.")
        else:
            print("CUDA detected. Using GPU device index:", chosen_device)
    else:
        print("Using device as requested:", chosen_device)

    if torch is not None:
        print("torch.cuda.is_available():", torch.cuda.is_available())
        print("torch.cuda.device_count():", torch.cuda.device_count())
    else:
        print("torch is not available; prediction will proceed on CPU if possible.")

    # Ensure weights exist.
    if not os.path.exists(args.weights):
        print(f"ERROR: Weights file not found: {args.weights}")
        sys.exit(1)

    # Load YOLO model once.
    model = YOLO(args.weights)

    # Load bin map.
    class_to_route, default_route = load_bin_map(args.bin_map)

    # Webcam/streaming mode:
    if is_webcam_source(args.source):
        print("==> Starting webcam stream...")
        # We let Ultralytics open the webcam internally by passing source=0 with stream=True.
        # We will consume frames as a generator and render them ourselves to inject bin guidance.
        route_counter = Counter()
        fps_smooth = None
        last_time = time.time()

        # Optional video writer setup (only if a path was provided).
        writer = None
        if len(args.save_vid.strip()) > 0:
            # We do not know the frame size until we get the first result.
            # We'll create the writer lazily after the first annotated frame is produced.
            writer = "PENDING"  # placeholder to indicate delayed init

        for result in model.predict(
            source=int(args.source),          # 0 -> default camera index
            conf=args.conf,
            imgsz=args.imgsz,
            device=chosen_device,
            stream=True,                      # crucial: yields results continuously
            verbose=False
        ):
            # Compute simple FPS (smoothed)
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0.0:
                fps_inst = 1.0 / dt
                if fps_smooth is None:
                    fps_smooth = fps_inst
                else:
                    # simple exponential smoothing
                    fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst

            # Aggregate bin routes for this frame
            route_counter.clear()
            id_to_name = result.names

            if result.boxes is not None:
                for b in result.boxes:
                    cls_id = int(b.cls[0].cpu().numpy())
                    if cls_id in id_to_name:
                        cname = id_to_name[cls_id]
                    else:
                        cname = f"class_{cls_id}"

                    key = cname.lower()
                    if key in class_to_route:
                        route = class_to_route[key]
                    else:
                        route = default_route

                    route_counter[route] += 1

            # Render YOLO's annotated frame
            frame_annotated = result.plot()

            # Overlay FPS
            if fps_smooth is not None:
                fps_text = f"FPS: {fps_smooth:.1f}"
                cv2.putText(frame_annotated, fps_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # Overlay per-frame bin summary
            overlay_bin_summary(frame_annotated, route_counter, x=10, y=45)

            # Lazy-init video writer if requested
            if writer == "PENDING":
                h, w = frame_annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_vid, fourcc, 20.0, (w, h))

            # Write frame if recording
            if writer not in [None, "PENDING"]:
                writer.write(frame_annotated)

            # Show window if requested
            if args.show:
                cv2.imshow("YOLO TrashCam", frame_annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        # Cleanup
        if writer not in [None, "PENDING"]:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

        print("Webcam stream stopped.")
        return

    # Non-webcam path (images/dirs/videos): keep previous batch behavior.
    print("==> Running on file/directory/video source (non-webcam).")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=chosen_device,
        save=True,
        save_txt=True,
        verbose=False
    )

    aggregate_bins = Counter()
    per_input_summaries = []

    for r in results:
        id_to_name = r.names
        object_entries = []

        if r.boxes is not None:
            for b in r.boxes:
                cls_id = int(b.cls[0].cpu().numpy())
                conf_val = float(b.conf[0].cpu().numpy())

                if cls_id in id_to_name:
                    class_name = id_to_name[cls_id]
                else:
                    class_name = f"class_{cls_id}"

                key = class_name.lower()
                if key in class_to_route:
                    route = class_to_route[key]
                else:
                    route = default_route

                object_entries.append((class_name, conf_val, route))
                aggregate_bins[route] += 1

        summary = {
            "path": r.path,
            "objects": object_entries
        }
        per_input_summaries.append(summary)

    print("\n=== Detection Summary ===")
    for item in per_input_summaries:
        print(f"\nFile: {item['path']}")
        if len(item["objects"]) == 0:
            print("  No objects detected at current confidence threshold.")
        else:
            for class_name, conf_val, route in item["objects"]:
                print(f"  - {class_name:20s} {conf_val:5.2f}  ->  {route}")

    print("\n=== Bin Guidance (aggregated) ===")
    if len(aggregate_bins) == 0:
        print("  No items detected.")
    else:
        for route, count in aggregate_bins.most_common():
            print(f"  {route}: {count} item(s)")

    print(f"\nAnnotated outputs saved under: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
