## 01) Repository Layout

trash-yolo/
├─ README.md
├─ requirements.txt
├─ bin_map.yaml
├─ scripts/
│ ├─ 01_prepare_taco_local.py # Convert local TACO COCO→YOLO (no downloads)
│ ├─ 02_train_yolo.py # Train YOLO (safe CPU fallback)
│ ├─ 03_predict_and_route.py # Predict on file/dir/video; map to bin guidance
│ └─ 04_predict_webcam_v4l2.py # Continuous webcam (explicit V4L2), overlays guidance
└─ datasets/
└─ taco_yolo/ # created by script 01 (images/, labels/, taco.yaml)

taco-trash-dataset/
├─ data/
│ ├─ batch_1/
│ ├─ batch_2/
│ └─ ... (image folders)
├─ annotations.json
├─ meta_df.csv

python scripts/01_prepare_taco_local.py \
  --src-dir taco-trash-dataset \
  --out-dir datasets/taco_yolo \
  --train 0.8 --val 0.1 --test 0.1 --seed 1337

## 02) Train YOLO

`yolo11s.pt` for speed
`yolo11m.pt` for accuracy

**For CPU:**

python scripts/02_train_yolo.py \
  --data datasets/taco_yolo/taco.yaml \
  --model yolo11s.pt \
  --epochs 40 \
  --imgsz 640 \
  --batch 8 \
  --device cpu

**For GPU**

python scripts/02_train_yolo.py \
  --data datasets/taco_yolo/taco.yaml \
  --model yolo11m.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0

This will create a resulting file at `runs/detect/<run-name>/weights/best.pt`

## 03) Predict

python scripts/03_predict_and_route.py \
  --weights runs/detect/taco_yolo11m/weights/best.pt \
  --source path/to/image_or_dir_or_video.mp4 \
  --bin-map bin_map.yaml \
  --conf 0.25 --imgsz 640 \
  --device cpu \
  --show \
  --save-vid runs/predict/output.mp4

### To Attach a webcam to WSL:

**In PowerShell:**

`usbipd list` # note down the BUSID of your webcam
`usbipd bind --busid <BUSUID>` 
`usbipd attach --wsl --busid <BUSUID> --auto-attach`

**In WSL** (Ubuntu 24 in my case)**:**
`sudo apt install -y v4l-utils`
`v4l2-ctl --list-devices`
Restart WSL
`ls -l /dev/video*` # You should see /dev/video0 or something similar
`sudo usermod -aG video <username>`

## 04) Run the webcam script

python scripts/04_predict_webcam_v4l2.py   --weights runs/detect/taco_yolo11m/weights/best.pt   --device-id /dev/video0   --bin-map bin_map.yaml   --conf 0.25 --imgsz 640   --width 1280 --height 720 --fps 30   --device cpu   --show   --save-vid runs/predict/webcam_out.mp4

