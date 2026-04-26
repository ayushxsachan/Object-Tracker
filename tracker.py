"""Tracking and missing/misplaced object logic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2

import config
from utils import (
    Detection,
    TrackerEvent,
    box_center,
    box_iou,
    current_timestamp,
    distance,
    draw_label,
    frame_diagonal,
)


@dataclass
class TrackedObject:
    """Represents one object identity across frames."""

    object_id: int
    label: str
    box: Tuple[int, int, int, int]
    confidence: float
    last_seen: float
    created_at: float
    original_center: Optional[Tuple[int, int]] = None
    original_box: Optional[Tuple[int, int, int, int]] = None
    baseline_member: bool = False
    status: str = "new"
    previous_status: str = "new"
    history: Deque[Tuple[float, Tuple[int, int], float]] = field(
        default_factory=lambda: deque(maxlen=config.HISTORY_LENGTH)
    )

    @property
    def center(self) -> Tuple[int, int]:
        return box_center(self.box)

    def remember_detection(self, detection: Detection, seen_at: float) -> None:
        self.box = detection.box
        self.confidence = detection.confidence
        self.last_seen = seen_at
        self.history.append((seen_at, detection.center, detection.confidence))


class ObjectTracker:
    """Maintains object history and compares each object to its saved baseline."""

    def __init__(self) -> None:
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_object_id = 1
        self.started_at: Optional[float] = None
        self.baseline_locked = False
        self.baseline_locked_at: Optional[float] = None

    def update(
        self,
        detections: Optional[Sequence[Detection]],
        frame_shape: Sequence[int],
        now: float,
    ) -> Tuple[List[TrackedObject], List[TrackerEvent]]:
        """Update tracker state.

        detections=None means the detector was intentionally skipped this frame
        for FPS. In that case, we keep existing boxes but still refresh statuses.
        """

        if self.started_at is None:
            self.started_at = now

        if detections is not None:
            self._match_detections_to_tracks(detections, frame_shape, now)

        if not self.baseline_locked and self._should_auto_lock_baseline(now):
            self.lock_baseline(now)

        events = self._refresh_statuses(frame_shape, now)
        self._forget_old_non_baseline_tracks(now)
        return list(self.tracks.values()), events

    def lock_baseline(self, now: float) -> int:
        """Save the current position of every visible object as the baseline."""

        saved_count = 0
        for track in self.tracks.values():
            if now - track.last_seen > config.MISSING_SECONDS_THRESHOLD:
                continue

            track.original_center = track.center
            track.original_box = track.box
            track.baseline_member = True
            track.status = "ok"
            track.previous_status = "ok"
            saved_count += 1

        self.baseline_locked = saved_count > 0
        self.baseline_locked_at = now if self.baseline_locked else None
        return saved_count

    def reset(self) -> None:
        """Clear all object history and start calibration again."""

        self.tracks.clear()
        self.next_object_id = 1
        self.started_at = None
        self.baseline_locked = False
        self.baseline_locked_at = None

    def calibration_seconds_left(self, now: float) -> float:
        if self.baseline_locked:
            return 0.0
        if self.started_at is None:
            return config.AUTO_BASELINE_SECONDS
        elapsed = now - self.started_at
        return max(0.0, config.AUTO_BASELINE_SECONDS - elapsed)

    def draw_tracks(self, frame) -> None:
        """Draw visible tracks and original baseline markers."""

        for track in self.tracks.values():
            if track.status == "missing":
                continue

            color = self._status_color(track.status)
            x1, y1, x2, y2 = track.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            display_name = f"{track.label} #{track.object_id}"
            label = f"{display_name} {track.confidence:.2f}"
            if track.status == "misplaced":
                label += " MISPLACED"
            elif track.status == "new":
                label += " NEW"

            draw_label(frame, label, x1, max(20, y1), color)

            if track.original_center is not None:
                cv2.circle(frame, track.original_center, 5, config.COLOR_OK, -1)
                cv2.line(frame, track.original_center, track.center, color, 1)

    def active_problem_messages(self) -> List[str]:
        """Return messages for objects currently missing or misplaced."""

        messages = []
        for track in self.tracks.values():
            if track.status == "missing":
                messages.append(f"Object Missing: {track.label} #{track.object_id}")
            elif track.status == "misplaced":
                messages.append(f"Object Misplaced: {track.label} #{track.object_id}")
        return messages

    def _should_auto_lock_baseline(self, now: float) -> bool:
        if self.started_at is None or not self.tracks:
            return False
        return now - self.started_at >= config.AUTO_BASELINE_SECONDS

    def _match_detections_to_tracks(
        self,
        detections: Sequence[Detection],
        frame_shape: Sequence[int],
        now: float,
    ) -> None:
        matched_track_ids: Set[int] = set()
        match_distance = frame_diagonal(frame_shape) * config.MATCH_DISTANCE_RATIO

        # Highest-confidence detections get first chance at a track.
        sorted_detections = sorted(detections, key=lambda item: item.confidence, reverse=True)

        for detection in sorted_detections:
            best_track = None
            best_score = float("inf")

            for track in self.tracks.values():
                if track.object_id in matched_track_ids:
                    continue
                if track.label != detection.label:
                    continue

                center_distance = distance(track.center, detection.center)
                iou = box_iou(track.box, detection.box)

                # Either close centers or overlapping boxes can continue a track.
                if center_distance > match_distance and iou < 0.10:
                    continue

                # Lower score is better. IoU helps prefer overlapping boxes when
                # two same-class objects are near each other.
                score = center_distance - (iou * 100.0)
                if score < best_score:
                    best_score = score
                    best_track = track

            if best_track is None:
                best_track = self._create_track(detection, now)

            best_track.remember_detection(detection, now)
            matched_track_ids.add(best_track.object_id)

    def _create_track(self, detection: Detection, now: float) -> TrackedObject:
        track = TrackedObject(
            object_id=self.next_object_id,
            label=detection.label,
            box=detection.box,
            confidence=detection.confidence,
            last_seen=now,
            created_at=now,
        )
        track.history.append((now, detection.center, detection.confidence))
        self.tracks[track.object_id] = track
        self.next_object_id += 1
        return track

    def _refresh_statuses(
        self,
        frame_shape: Sequence[int],
        now: float,
    ) -> List[TrackerEvent]:
        events: List[TrackerEvent] = []
        misplaced_distance = frame_diagonal(frame_shape) * config.MISPLACED_DISTANCE_RATIO

        for track in self.tracks.values():
            old_status = track.status
            track.previous_status = old_status

            if not self.baseline_locked or not track.baseline_member:
                track.status = "new"
            else:
                seconds_since_seen = now - track.last_seen
                if seconds_since_seen >= config.MISSING_SECONDS_THRESHOLD:
                    track.status = "missing"
                else:
                    moved_pixels = self._distance_from_baseline(track)
                    track.status = "misplaced" if moved_pixels >= misplaced_distance else "ok"

            if track.status != old_status:
                event = self._event_for_status_change(track)
                if event is not None:
                    events.append(event)

        return events

    def _event_for_status_change(self, track: TrackedObject) -> Optional[TrackerEvent]:
        moved_pixels = self._distance_from_baseline(track)

        if track.status == "missing":
            message = f"Object Missing: {track.label} #{track.object_id}"
            event_type = "missing"
        elif track.status == "misplaced":
            message = f"Object Misplaced: {track.label} #{track.object_id}"
            event_type = "misplaced"
        elif track.status == "ok" and track.previous_status in {"missing", "misplaced"}:
            message = f"Object Recovered: {track.label} #{track.object_id}"
            event_type = "recovered"
        else:
            return None

        return TrackerEvent(
            event_type=event_type,
            object_id=track.object_id,
            label=track.label,
            message=message,
            confidence=track.confidence,
            distance_pixels=moved_pixels,
            timestamp=current_timestamp(),
        )

    def _distance_from_baseline(self, track: TrackedObject) -> float:
        if track.original_center is None:
            return 0.0
        return distance(track.original_center, track.center)

    def _forget_old_non_baseline_tracks(self, now: float) -> None:
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if not track.baseline_member
            and now - track.last_seen > config.FORGET_UNBASELINED_AFTER_SECONDS
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def _status_color(self, status: str) -> Tuple[int, int, int]:
        if status == "ok":
            return config.COLOR_OK
        if status == "misplaced":
            return config.COLOR_MISPLACED
        if status == "missing":
            return config.COLOR_MISSING
        return config.COLOR_NEW
