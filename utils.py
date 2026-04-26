"""Utility helpers shared by the object detector, tracker, and main loop."""

from __future__ import annotations

import csv
import math
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2

import config


Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class Detection:
    """Single object returned by a detector backend."""

    label: str
    confidence: float
    box: Box

    @property
    def center(self) -> Point:
        return box_center(self.box)


@dataclass
class TrackerEvent:
    """Event emitted when an object becomes missing, misplaced, or recovered."""

    event_type: str
    object_id: int
    label: str
    message: str
    confidence: float
    distance_pixels: float
    timestamp: str


class FPSCounter:
    """Small moving-average FPS counter for smoother display."""

    def __init__(self, smoothing: float = 0.90) -> None:
        self.smoothing = smoothing
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        elapsed = max(now - self._last_time, 1e-6)
        instant_fps = 1.0 / elapsed
        self._last_time = now

        if self._fps == 0.0:
            self._fps = instant_fps
        else:
            self._fps = (self.smoothing * self._fps) + (
                (1.0 - self.smoothing) * instant_fps
            )
        return self._fps


class EventLogger:
    """CSV event logger with a stable, spreadsheet-friendly format."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_file.exists():
            with self.log_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "timestamp",
                        "event_type",
                        "object_id",
                        "label",
                        "confidence",
                        "distance_pixels",
                        "message",
                    ]
                )

    def write(self, event: TrackerEvent) -> None:
        with self.log_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    event.timestamp,
                    event.event_type,
                    event.object_id,
                    event.label,
                    f"{event.confidence:.3f}",
                    f"{event.distance_pixels:.1f}",
                    event.message,
                ]
            )


class AlertManager:
    """Handles throttled alert sounds so the UI does not become noisy."""

    def __init__(self) -> None:
        self._last_alert_by_key = {}
        self.recent_alerts: List[Tuple[float, str]] = []

    def notify(self, event: TrackerEvent) -> None:
        key = (event.object_id, event.event_type)
        now = time.time()
        last_alert_at = self._last_alert_by_key.get(key, 0.0)

        if now - last_alert_at < config.ALERT_COOLDOWN_SECONDS:
            return

        self._last_alert_by_key[key] = now
        self.recent_alerts.append((now, event.message))
        self.recent_alerts = self.recent_alerts[-5:]

        if config.ENABLE_SOUND_ALERT:
            thread = threading.Thread(target=play_alert_sound, daemon=True)
            thread.start()

    def active_messages(self) -> List[str]:
        now = time.time()
        self.recent_alerts = [
            (created_at, message)
            for created_at, message in self.recent_alerts
            if now - created_at <= config.ALERT_BANNER_SECONDS
        ]
        return [message for _, message in self.recent_alerts]


def normalize_label(label: str) -> str:
    """Normalize detector class names into friendly display names."""

    clean_label = label.strip().lower()
    return config.DISPLAY_NAME_OVERRIDES.get(clean_label, clean_label)


def should_watch_label(label: str) -> bool:
    """Return True if this label should be tracked by the app."""

    return not config.WATCHED_CLASSES or label in config.WATCHED_CLASSES


def clamp_box(box: Sequence[float], frame_width: int, frame_height: int) -> Box:
    """Clip a bounding box so drawing never goes outside the frame."""

    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width - 1))
    y2 = max(0, min(int(y2), frame_height - 1))
    return x1, y1, x2, y2


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def box_iou(box_a: Box, box_b: Box) -> float:
    """Compute intersection-over-union for two boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def distance(point_a: Point, point_b: Point) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def frame_diagonal(frame_shape: Sequence[int]) -> float:
    height, width = frame_shape[:2]
    return math.hypot(width, height)


def current_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def draw_label(
    frame,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    """Draw text with a filled background for readability."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

    top_left = (x, max(0, y - text_height - baseline - 6))
    bottom_right = (x + text_width + 8, y)
    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, y - baseline - 3),
        font,
        scale,
        config.COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame, lines: Iterable[str]) -> None:
    """Draw the top-left information panel."""

    lines = list(lines)
    if not lines:
        return

    panel_width = 500
    panel_height = 24 + (len(lines) * 24)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), config.COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    y = 38
    for line in lines:
        cv2.putText(
            frame,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            config.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        y += 24


def draw_alert_banner(frame, messages: Sequence[str]) -> None:
    """Draw a red banner for recent missing/misplaced alerts."""

    if not messages:
        return

    height, width = frame.shape[:2]
    banner_height = 44 + (len(messages) - 1) * 26

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - banner_height), (width, height), config.COLOR_MISSING, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    y = height - banner_height + 30
    for message in messages:
        cv2.putText(
            frame,
            message,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            config.COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )
        y += 26


def play_alert_sound() -> None:
    """Play a short alert sound without adding heavy dependencies."""

    try:
        if platform.system() == "Windows":
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            # Terminal bell fallback for macOS/Linux.
            print("\a", end="", flush=True)
    except Exception:
        # Audio should never crash the surveillance loop.
        pass


def camera_backend() -> int:
    """Use DirectShow on Windows for faster camera startup when available."""

    if os.name == "nt":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY
