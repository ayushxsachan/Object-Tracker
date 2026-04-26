# Object-Tracker
AI system for real-time object tracking and alerts.

## Features

- Detects everyday objects with YOLOv8 by default, with an optional MediaPipe object-detector backend.
- Tracks multiple objects and keeps position history.
- Saves an original baseline layout automatically after a short calibration period.
- Displays bounding boxes, labels, object IDs, confidence scores, and status.
- Shows `Object Missing: [Object Name]` when a baseline object disappears.
- Shows `Object Misplaced: [Object Name]` when a baseline object moves too far.
- Plays an alert sound when an object becomes missing or misplaced.
- Saves timestamped CSV logs in `logs/events.csv`.
- Uses detection skipping, smaller model size, camera buffer control, and FPS smoothing for better real-time performance.

## Project Structure

```text
obj_track/
├── main.py              # Webcam loop, UI, alerts, and keyboard controls
├── object_detector.py   # YOLOv8 and MediaPipe detector backends
├── tracker.py           # Object identity, history, missing/misplaced logic
├── utils.py             # Drawing, logging, math, FPS, and alert helpers
├── config.py            # Easy-to-edit thresholds and settings
├── requirements.txt     # Python dependencies
└── README.md            # Setup and usage instructions
```

## 1. Install Python

Install Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/).

On Windows, enable **Add python.exe to PATH** during installation. Then confirm:

```powershell
python --version
pip --version
```

If `python` is not found on Windows, try:

```powershell
py --version
```

## 2. Create a Virtual Environment

Open this folder in PowerShell:

```powershell
cd D:\obj_track
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation scripts, run this once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## 3. Install Required Libraries

```powershell
pip install -r requirements.txt
```

Libraries used:

- `opencv-python` for webcam video, drawing, and UI.
- `mediapipe` for the optional MediaPipe object detector backend.
- `numpy`, installed because OpenCV and model libraries use it heavily.
- `ultralytics` for YOLOv8 object detection.

YOLOv8 will download `yolov8n.pt` the first time it runs if the model is not already cached.

## 4. Webcam Permissions

On Windows:

- Open **Settings > Privacy & security > Camera**.
- Enable camera access.
- Enable camera access for desktop apps.
- Close apps like Teams, Zoom, OBS, or browsers if they are already using the webcam.

On macOS:

- Open **System Settings > Privacy & Security > Camera**.
- Allow your terminal, VS Code, or Python app to use the camera.

## 5. Run the Project

```powershell
python main.py
```

Keyboard controls:

- Press `S` to save or re-save the current object positions as the baseline.
- Press `R` to reset tracking and calibrate again.
- Press `Q` or `Esc` to quit.

## 6. Run in VS Code

1. Open VS Code.
2. Choose **File > Open Folder** and select `D:\obj_track`.
3. Install the Microsoft Python extension if prompted.
4. Press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select `.venv\Scripts\python.exe`.
5. Open `main.py`.
6. Click **Run Python File** or run `python main.py` in the VS Code terminal.

## 7. How the Baseline Works

Place your objects on the desk before starting the app. During the first few seconds, the system calibrates and learns the original positions. After calibration, it locks the baseline automatically.

You can press `S` anytime to save the current layout again. This is useful after rearranging the desk.

## 8. Configuration

Edit `config.py` for common changes:

- `CAMERA_INDEX`: change to `1` or `2` if the wrong camera opens.
- `WATCHED_CLASSES`: choose which labels should be tracked.
- `YOLO_CONFIDENCE_THRESHOLD`: lower it if detections are missed; raise it if false detections appear.
- `MISSING_SECONDS_THRESHOLD`: how long an object must be unseen before a missing alert.
- `MISPLACED_DISTANCE_RATIO`: how far an object must move before a misplaced alert.
- `DETECTION_INTERVAL`: increase for smoother FPS; decrease for faster detection updates.

## 9. MediaPipe Backend

YOLOv8 is the easiest default for general objects. To use MediaPipe instead:

1. Download a MediaPipe object detection `.tflite` model, such as EfficientDet Lite.
2. Create a `models` folder.
3. Place the model at `models/efficientdet_lite0.tflite`.
4. Set this in `config.py`:

```python
DETECTOR_BACKEND = "mediapipe"
```

Then run:

```powershell
python main.py
```

## 10. Notes About Wallets and Keys

The default YOLOv8 COCO model can detect objects like bottles, cups, phones, laptops, books, keyboards, mice, backpacks, and handbags. Wallets and keys usually require a custom-trained YOLO model because they are not reliable default COCO classes.

To use a custom model, put it in the `models` folder and change:

```python
YOLO_MODEL_NAME = "models/my_wallet_keys_model.pt"
```

Also add your custom labels to `WATCHED_CLASSES` if needed.

## 11. Troubleshooting

If the camera does not open, check permissions, close other webcam apps, and try `CAMERA_INDEX = 1` in `config.py`.

If FPS is low, increase `DETECTION_INTERVAL`, lower `FRAME_WIDTH` and `FRAME_HEIGHT`, keep `YOLO_MODEL_NAME = "yolov8n.pt"`, and close other heavy apps.

If objects flicker between normal and misplaced, increase `MISPLACED_DISTANCE_RATIO` in `config.py`.

If no objects are detected, lower `YOLO_CONFIDENCE_THRESHOLD`, improve lighting, move objects apart, or use a custom model for small items.
