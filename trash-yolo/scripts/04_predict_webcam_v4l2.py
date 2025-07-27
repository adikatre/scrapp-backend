#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_predict_webcam_v4l2.py

Continuous YOLO inference from a Linux V4L2 webcam in WSL2.
This script explicitly opens the camera with OpenCV's CAP_V4L2 backend and
forces a reasonable FOURCC / resolution / FPS to avoid 'select() timeout' issues.

USAGE (CPU-safe):
  python scripts/04_predict_webcam_v4l2.py \
      --weights runs/detect/taco_yolo11m/weights/best.pt \
      --device-id /dev/video0 \
      --bin-map bin_map.yaml \
      --conf 0.25 --imgsz 640 \
      --width 1280 --height 720 --fps 30 \
      --device cpu \
      --show \
      --save-vid runs/predict/webcam_out.mp4
"""

import argparse
import os
import sys
import time
from collections import Counter
from typing import Tuple

import cv2
import yaml
import numpy as np
from ultralytics import YOLO

# Import torch just to check CUDA availability.
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
        "--device-id",
        default="/dev/video0",
        help="Camera device node or index, e.g., '/dev/video0' or '/dev/video1'."
    )
    parser.add_argument(
        "--bin-map",
        default="bin_map.yaml",
        help="YAML mapping of class name -> disposal route."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size fed to YOLO."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested capture FPS."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'cpu', '0' for first CUDA GPU, or 'auto' to pick automatically."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live window."
    )
    parser.add_argument(
        "--save-vid",
        default="",
        help="Optional MP4 path to save the annotated stream."
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    """
    Decide which device to pass to Ultralytics.
    """
    if requested is None:
        return "cpu"

    text = str(requested).strip().lower()
    if text == "cpu":
        return "cpu"

    if text == "auto":
        if torch is None or not torch.cuda.is_available():
            return "cpu"
        return "0"

    return requested


def load_bin_map(path: str):
    """
    Load YAML mapping class_name -> route (normalized to lowercase keys).
    """
    if not os.path.exists(path):
        print(f"ERROR: bin_map file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    raw_map = data.get("mapping", {})
    default = data.get("default", "Landfill / Check local rules")

    mapping = {}
    for k, v in raw_map.items():
        key = str(k).strip().lower()
        mapping[key] = v

    return mapping, default


def try_set_fourcc(cap: cv2.VideoCapture, fourcc_str: str) -> bool:
    """
    Try to set a FOURCC (e.g., 'MJPG' or 'YUYV') on the capture device.
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    ok = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    return bool(ok)


def open_v4l2_camera(device_id: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """
    Open a V4L2 camera and configure pixel format / resolution / fps.
    Try MJPG first (common for USB cams), then fall back to YUYV.
    """
    # Use CAP_V4L2 to force the backend on Linux
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device_id}")

    # Set buffer size small to reduce latency, if supported
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # Set resolution and FPS first (some drivers require this before FOURCC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    # Try MJPG, then YUYV
    ok_mjpg = try_set_fourcc(cap, "MJPG")
    if not ok_mjpg:
        ok_yuyv = try_set_fourcc(cap, "YUYV")
        if not ok_yuyv:
            print("WARNING: Could not set FOURCC to MJPG or YUYV; continuing with driver default.")

    # Verify we can read at least one frame
    warm_ok, frame = cap.read()
    if not warm_ok or frame is None:
        cap.release()
        raise RuntimeError("Camera opened but failed to deliver a frame (format mismatch or device busy).")

    return cap


def overlay_bin_summary(frame: np.ndarray, route_counts: Counter) -> None:
    """
    Draw a simple per-frame bin summary and FPS in the corner.
    """
    x = 10
    y = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    thickness = 2
    scale = 0.7
    line_h = 24

    if len(route_counts) == 0:
        text = "No items detected"
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return

    header = "Bin guidance (this frame):"
    cv2.putText(frame, header, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    y_cursor = y + line_h
    for route, count in route_counts.most_common():
        text = f"{route}: {count}"
        cv2.putText(frame, text, (x, y_cursor), font, scale, color, thickness, cv2.LINE_AA)
        y_cursor += line_h


def main():
    args = parse_args()

    # Resolve compute device for YOLO
    chosen_device = resolve_device(args.device)
    if args.device == "auto" and chosen_device == "cpu":
        print("No CUDA device detected by PyTorch. Using CPU.")
    else:
        print("Using device:", chosen_device)

    # Load YOLO once
    if not os.path.exists(args.weights):
        print(f"ERROR: weights not found: {args.weights}")
        sys.exit(1)
    model = YOLO(args.weights)

    # Load class -> route mapping
    class_to_route, default_route = load_bin_map(args.bin_map)

    # Open webcam with explicit V4L2 settings
    print(f"Opening camera {args.device-id if hasattr(args,'device-id') else args.device_id} ...")
    try:
        cap = open_v4l2_camera(args.device_id, args.width, args.height, args.fps)
    except Exception as e:
        print("ERROR opening camera:", e)
        print("Hints:")
        print("  - Try the other node (e.g., --device-id /dev/video1)")
        print("  - Close any Windows apps using the camera, then re-attach with usbipd")
        print("  - Check supported formats: v4l2-ctl --device=/dev/video0 --list-formats-ext")
        sys.exit(1)

    # Optional writer
    writer = None
    if len(args.save_vid.strip()) > 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Query actual size from capture
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_vid, fourcc, float(args.fps), (out_w, out_h))

    # Timing for FPS
    last = time.time()
    fps_smooth = None

    # Main loop
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: Failed to read frame (camera disconnected or busy).")
                break

            # Compute simple FPS
            now = time.time()
            dt = now - last
            last = now
            if dt > 0.0:
                fps = 1.0 / dt
                if fps_smooth is None:
                    fps_smooth = fps
                else:
                    # Simple smoothing
                    fps_smooth = 0.9 * fps_smooth + 0.1 * fps

            # Run YOLO on this frame (numpy array). We pass a list [frame]
            # so Ultralytics returns a list of results.
            results = model.predict(
                source=[frame],
                conf=args.conf,
                imgsz=args.imgsz,
                device=chosen_device,
                verbose=False
            )

            route_counts = Counter()
            annotated = frame

            # Process detections for this single frame
            if len(results) > 0:
                r = results[0]
                id_to_name = r.names

                # Boxes and overlay from Ultralytics
                annotated = r.plot()

                if r.boxes is not None:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].cpu().numpy())
                        if cls_id in id_to_name:
                            cname = id_to_name[cls_id]
                        else:
                            cname = f"class_{cls_id}"

                        key = cname.lower()
                        route = class_to_route.get(key, default_route)
                        route_counts[route] += 1

            # Overlay FPS
            if fps_smooth is not None:
                text = f"FPS: {fps_smooth:.1f}"
                cv2.putText(annotated, text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            # Overlay per-frame bin guidance
            overlay_bin_summary(annotated, route_counts)

            # Show window
            if args.show:
                cv2.imshow("YOLO TrashCam (V4L2)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Save
            if writer is not None:
                writer.write(annotated)

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
