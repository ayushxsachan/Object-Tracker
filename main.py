"""Real-time Object Tracking and Missing/Misplaced Object Detection.

Run:
    python main.py

Controls:
    S = save/re-save the current desk layout as the baseline
    R = reset tracking and calibrate again
    Q or Esc = quit
"""

from __future__ import annotations

import time

import cv2

import config
from object_detector import ObjectDetector
from tracker import ObjectTracker
from utils import AlertManager, EventLogger, FPSCounter, camera_backend, draw_alert_banner, draw_panel


WINDOW_NAME = "Smart Desk Object Monitor"


def open_camera():
    """Open the webcam and request the configured resolution."""

    capture = cv2.VideoCapture(config.CAMERA_INDEX, camera_backend())
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check camera permissions and make sure no "
            "other app is using the camera."
        )

    return capture


def main() -> int:
    print("Starting Smart Desk Object Monitor...")
    print("Press S to save baseline, R to reset, Q/Esc to quit.")

    try:
        detector = ObjectDetector()
        print(f"Detector backend: {detector.backend_name}")
    except Exception as exc:
        print(f"Detector startup failed:\n{exc}")
        return 1

    try:
        capture = open_camera()
    except Exception as exc:
        print(f"Camera startup failed:\n{exc}")
        return 1

    tracker = ObjectTracker()
    fps_counter = FPSCounter()
    alert_manager = AlertManager()
    event_logger = EventLogger(config.LOG_FILE) if config.ENABLE_EVENT_LOGGING else None

    frame_index = 0
    latest_detections = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            now = time.time()
            fps = fps_counter.update()

            # Detecting every frame can be heavy. We skip some frames for smoother
            # FPS, while the tracker keeps displaying the most recent boxes.
            should_detect = frame_index % max(1, config.DETECTION_INTERVAL) == 0
            if should_detect:
                latest_detections = detector.detect(frame)
                detections_for_tracker = latest_detections
            else:
                detections_for_tracker = None

            tracks, events = tracker.update(detections_for_tracker, frame.shape, now)

            for event in events:
                if event.event_type in {"missing", "misplaced"}:
                    alert_manager.notify(event)
                    if event_logger is not None:
                        event_logger.write(event)
                    print(event.message)

            tracker.draw_tracks(frame)

            panel_lines = build_panel_lines(
                detector_name=detector.backend_name,
                fps=fps,
                track_count=len(tracks),
                detection_count=len(latest_detections or []),
                tracker=tracker,
                now=now,
            )
            draw_panel(frame, panel_lines)

            active_messages = unique_messages(
                alert_manager.active_messages() + tracker.active_problem_messages()
            )
            draw_alert_banner(frame, active_messages[:4])

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                saved_count = tracker.lock_baseline(now)
                print(f"Baseline saved for {saved_count} object(s).")
            if key == ord("r"):
                tracker.reset()
                latest_detections = None
                print("Tracker reset. Recalibrating baseline...")

            frame_index += 1

    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def build_panel_lines(
    detector_name: str,
    fps: float,
    track_count: int,
    detection_count: int,
    tracker: ObjectTracker,
    now: float,
):
    """Create concise UI text for the top-left status panel."""

    if tracker.baseline_locked:
        baseline_line = "Baseline: locked | press S to resave"
    else:
        seconds_left = tracker.calibration_seconds_left(now)
        baseline_line = f"Calibrating baseline: {seconds_left:.1f}s | press S now"

    return [
        f"FPS: {fps:.1f} | Detector: {detector_name} | Detections: {detection_count}",
        f"Tracked objects: {track_count}",
        baseline_line,
        "Controls: S save baseline | R reset | Q/Esc quit",
    ]


def unique_messages(messages):
    """Keep alert banners readable when a recent alert is also still active."""

    unique = []
    seen = set()
    for message in messages:
        if message in seen:
            continue
        seen.add(message)
        unique.append(message)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
