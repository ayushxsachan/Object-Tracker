"""Object detector backends for YOLOv8 and MediaPipe.

The rest of the app talks to the ObjectDetector class only. This keeps the
tracker independent from whichever model you choose.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2

import config
from utils import Detection, clamp_box, normalize_label, should_watch_label


class ObjectDetector:
    """Detect everyday objects in an OpenCV frame."""

    def __init__(self, backend: Optional[str] = None) -> None:
        self.requested_backend = backend or config.DETECTOR_BACKEND
        self.backend_name = ""
        self._model = None

        if self.requested_backend not in {"auto", "yolo", "mediapipe"}:
            raise ValueError(
                "DETECTOR_BACKEND must be one of: 'auto', 'yolo', or 'mediapipe'."
            )

        self._initialize_backend()

    def detect(self, frame) -> List[Detection]:
        """Return detected objects for the current BGR frame."""

        if self.backend_name == "yolo":
            return self._detect_with_yolo(frame)
        if self.backend_name == "mediapipe":
            return self._detect_with_mediapipe(frame)

        raise RuntimeError("Object detector backend was not initialized.")

    def _initialize_backend(self) -> None:
        errors = []

        if self.requested_backend in {"auto", "yolo"}:
            try:
                self._initialize_yolo()
                return
            except Exception as exc:
                errors.append(f"YOLO unavailable: {exc}")
                if self.requested_backend == "yolo":
                    raise RuntimeError(errors[-1]) from exc

        if self.requested_backend in {"auto", "mediapipe"}:
            try:
                self._initialize_mediapipe()
                return
            except Exception as exc:
                errors.append(f"MediaPipe unavailable: {exc}")
                if self.requested_backend == "mediapipe":
                    raise RuntimeError(errors[-1]) from exc

        help_text = (
            "No detector backend could be started.\n"
            "Install dependencies with: pip install -r requirements.txt\n"
            "YOLO downloads yolov8n.pt on first use. For MediaPipe, place a .tflite "
            f"model at {config.MEDIAPIPE_MODEL_PATH}."
        )
        raise RuntimeError(help_text + "\n" + "\n".join(errors))

    def _initialize_yolo(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(config.YOLO_MODEL_NAME)
        self.backend_name = "yolo"

    def _initialize_mediapipe(self) -> None:
        model_path = Path(config.MEDIAPIPE_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. "
                "Download an EfficientDet Lite .tflite model or use YOLO."
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=config.MEDIAPIPE_CONFIDENCE_THRESHOLD,
        )

        self._mp = mp
        self._model = vision.ObjectDetector.create_from_options(options)
        self.backend_name = "mediapipe"

    def _detect_with_yolo(self, frame) -> List[Detection]:
        height, width = frame.shape[:2]
        results = self._model.predict(
            frame,
            imgsz=config.YOLO_IMAGE_SIZE,
            conf=config.YOLO_CONFIDENCE_THRESHOLD,
            device=config.YOLO_DEVICE,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names

        for box in result.boxes:
            class_id = int(box.cls[0])
            raw_label = str(names[class_id])
            label = normalize_label(raw_label)

            if not should_watch_label(label):
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=clamp_box((x1, y1, x2, y2), width, height),
                )
            )

        return detections

    def _detect_with_mediapipe(self, frame) -> List[Detection]:
        height, width = frame.shape[:2]

        # OpenCV frames are BGR. MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._model.detect(mp_image)

        detections: List[Detection] = []
        for item in result.detections:
            if not item.categories:
                continue

            category = item.categories[0]
            label = normalize_label(category.category_name)

            if not should_watch_label(label):
                continue

            bbox = item.bounding_box
            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = bbox.origin_x + bbox.width
            y2 = bbox.origin_y + bbox.height

            detections.append(
                Detection(
                    label=label,
                    confidence=float(category.score),
                    box=clamp_box((x1, y1, x2, y2), width, height),
                )
            )

        return detections
