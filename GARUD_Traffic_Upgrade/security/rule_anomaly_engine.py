from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional, Tuple
import math
import time
import uuid

import cv2
import numpy as np

Point = Tuple[int, int]


@dataclass
class SecurityEvent:
    id: str
    camera_id: str
    event_type: str
    severity: str
    timestamp: float
    confidence: float
    message: str
    track_id: int = -1
    duration_seconds: float = 0.0
    snapshot: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def point_inside_polygon(point: Point, polygon: List[Point]) -> bool:
    polygon_array = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon_array, point, False) >= 0


class RuleAnomalyEngine:
    """Rule-based CCTV anomaly engine using YOLO/ByteTrack detections.

    Supported events:
    - CROWD_THRESHOLD_EXCEEDED
    - SUDDEN_CROWD_SURGE
    - LOITERING
    - RESTRICTED_ZONE_ENTRY
    - STOPPED_VEHICLE
    """

    PERSON_CLASS_NAMES = {"person"}
    VEHICLE_CLASS_NAMES = {
        "car",
        "motorcycle",
        "motorbike",
        "bus",
        "truck",
        "bicycle",
        "auto_rickshaw",
        "autorickshaw",
    }

    def __init__(self, camera_id: str, config: dict):
        self.camera_id = camera_id
        self.config = config
        history_size = int(config.get("maximum_history_points", 1800))

        self.person_history: Dict[int, Deque[Tuple[float, Point]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.vehicle_history: Dict[int, Deque[Tuple[float, Point]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.count_history: Deque[Tuple[float, int]] = deque(maxlen=history_size)
        self.last_seen: Dict[int, float] = {}
        self.last_alert: Dict[Tuple[str, int], float] = {}
        self.restricted_entry_memory: Dict[int, bool] = {}

    @staticmethod
    def get_ground_point(bounding_box: List[float]) -> Point:
        x1, _, x2, y2 = map(float, bounding_box)
        return int((x1 + x2) / 2.0), int(y2)

    @staticmethod
    def normalized_polygon_to_pixels(
        polygon: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> List[Point]:
        return [
            (int(float(x) * frame_width), int(float(y) * frame_height))
            for x, y in polygon
        ]

    def emit_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        confidence: float,
        track_id: int = -1,
        duration_seconds: float = 0.0,
    ) -> List[SecurityEvent]:
        now = time.time()
        cooldown = float(self.config.get("alert_cooldown_seconds", 15))
        key = (event_type, int(track_id))

        if now - self.last_alert.get(key, 0.0) < cooldown:
            return []

        self.last_alert[key] = now
        return [
            SecurityEvent(
                id=str(uuid.uuid4()),
                camera_id=self.camera_id,
                event_type=event_type,
                severity=severity,
                timestamp=now,
                confidence=max(0.0, min(float(confidence), 0.99)),
                message=message,
                track_id=int(track_id),
                duration_seconds=float(duration_seconds),
            )
        ]

    def process(self, detections: List[dict], frame_shape) -> List[SecurityEvent]:
        frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
        now = time.time()
        events: List[SecurityEvent] = []
        persons = []
        vehicles = []

        for detection in detections:
            if detection.get("track_id") is None:
                continue

            class_name = str(detection.get("class_name", "")).lower()
            track_id = int(detection["track_id"])
            ground_point = self.get_ground_point(detection["bbox"])
            self.last_seen[track_id] = now

            if class_name in self.PERSON_CLASS_NAMES:
                persons.append((track_id, ground_point, detection))
                self.person_history[track_id].append((now, ground_point))
            elif class_name in self.VEHICLE_CLASS_NAMES:
                vehicles.append((track_id, ground_point, detection))
                self.vehicle_history[track_id].append((now, ground_point))

        events.extend(self.detect_crowd_threshold(persons, frame_width, frame_height))
        events.extend(self.detect_crowd_surge(len(persons), now))

        for track_id, ground_point, _ in persons:
            events.extend(
                self.detect_loitering(
                    track_id, ground_point, frame_width, frame_height, now
                )
            )
            events.extend(
                self.detect_restricted_entry(
                    track_id, ground_point, frame_width, frame_height
                )
            )

        for track_id, ground_point, detection in vehicles:
            events.extend(
                self.detect_stopped_vehicle(
                    track_id,
                    ground_point,
                    frame_width,
                    frame_height,
                    now,
                    str(detection.get("class_name", "vehicle")),
                )
            )

        self.cleanup_stale_tracks(now)
        return events

    def detect_crowd_threshold(
        self, persons, frame_width: int, frame_height: int
    ) -> List[SecurityEvent]:
        cfg = self.config.get("crowd_zone")
        if not cfg:
            return []

        threshold = int(cfg.get("threshold", 20))
        polygon_cfg = cfg.get("polygon")

        if polygon_cfg:
            polygon = self.normalized_polygon_to_pixels(
                polygon_cfg, frame_width, frame_height
            )
            count = sum(
                1
                for _, ground_point, _ in persons
                if point_inside_polygon(ground_point, polygon)
            )
        else:
            count = len(persons)

        if count < threshold:
            return []

        confidence = min(0.99, 0.60 + (count / max(threshold, 1)) * 0.20)
        return self.emit_event(
            "CROWD_THRESHOLD_EXCEEDED",
            "amber",
            f"{count} tracked people detected in the monitored crowd zone; threshold is {threshold}.",
            confidence,
        )

    def detect_crowd_surge(
        self, person_count: int, current_time: float
    ) -> List[SecurityEvent]:
        cfg = self.config.get("crowd_surge")
        if not cfg:
            return []

        window = float(cfg.get("window_seconds", 8))
        minimum_rise = int(cfg.get("minimum_rise", 8))
        minimum_current = int(cfg.get("minimum_current_count", 10))
        self.count_history.append((current_time, person_count))

        while self.count_history and current_time - self.count_history[0][0] > window * 2:
            self.count_history.popleft()

        older_counts = [
            count
            for timestamp, count in self.count_history
            if current_time - timestamp >= window * 0.70
        ]
        if not older_counts:
            return []

        baseline = min(older_counts)
        increase = person_count - baseline
        if increase < minimum_rise or person_count < minimum_current:
            return []

        confidence = min(0.99, 0.68 + (increase / max(minimum_rise, 1)) * 0.12)
        return self.emit_event(
            "SUDDEN_CROWD_SURGE",
            "red",
            f"Person count rose from approximately {baseline} to {person_count} within {window:.0f} seconds.",
            confidence,
        )

    def detect_loitering(
        self,
        track_id: int,
        ground_point: Point,
        frame_width: int,
        frame_height: int,
        current_time: float,
    ) -> List[SecurityEvent]:
        cfg = self.config.get("loitering_zone")
        if not cfg or not cfg.get("polygon"):
            return []

        polygon = self.normalized_polygon_to_pixels(
            cfg["polygon"], frame_width, frame_height
        )
        if not point_inside_polygon(ground_point, polygon):
            return []

        required_seconds = float(cfg.get("seconds", 30))
        recent = [
            (timestamp, point)
            for timestamp, point in self.person_history[track_id]
            if current_time - timestamp <= required_seconds + 3.0
        ]
        if len(recent) < 5:
            return []

        duration = current_time - recent[0][0]
        if duration < required_seconds:
            return []

        start = recent[0][1]
        maximum_displacement = max(math.dist(start, point) for _, point in recent)
        allowed_displacement = frame_width * float(
            cfg.get("max_displacement_ratio", 0.08)
        )
        if maximum_displacement > allowed_displacement:
            return []

        return self.emit_event(
            "LOITERING",
            "yellow",
            f"Person track {track_id} remained near the monitored location for approximately {duration:.1f} seconds.",
            0.88,
            track_id,
            duration,
        )

    def detect_restricted_entry(
        self,
        track_id: int,
        ground_point: Point,
        frame_width: int,
        frame_height: int,
    ) -> List[SecurityEvent]:
        cfg = self.config.get("restricted_zone")
        if not cfg or not cfg.get("polygon"):
            return []

        polygon = self.normalized_polygon_to_pixels(
            cfg["polygon"], frame_width, frame_height
        )
        is_inside = point_inside_polygon(ground_point, polygon)
        was_inside = self.restricted_entry_memory.get(track_id, False)
        self.restricted_entry_memory[track_id] = is_inside

        if not is_inside or was_inside:
            return []

        name = cfg.get("name", "the configured restricted area")
        return self.emit_event(
            "RESTRICTED_ZONE_ENTRY",
            "red",
            f"Person track {track_id} entered {name}.",
            0.94,
            track_id,
        )

    def detect_stopped_vehicle(
        self,
        track_id: int,
        ground_point: Point,
        frame_width: int,
        frame_height: int,
        current_time: float,
        class_name: str,
    ) -> List[SecurityEvent]:
        cfg = self.config.get("active_lane_zone")
        if not cfg or not cfg.get("polygon"):
            return []

        polygon = self.normalized_polygon_to_pixels(
            cfg["polygon"], frame_width, frame_height
        )
        if not point_inside_polygon(ground_point, polygon):
            return []

        required_seconds = float(cfg.get("stopped_seconds", 10))
        recent = [
            (timestamp, point)
            for timestamp, point in self.vehicle_history[track_id]
            if current_time - timestamp <= required_seconds + 3.0
        ]
        if len(recent) < 5:
            return []

        duration = current_time - recent[0][0]
        if duration < required_seconds:
            return []

        start = recent[0][1]
        maximum_displacement = max(math.dist(start, point) for _, point in recent)
        allowed_displacement = frame_width * float(
            cfg.get("max_displacement_ratio", 0.025)
        )
        if maximum_displacement > allowed_displacement:
            return []

        readable_name = class_name.replace("_", " ").title()
        return self.emit_event(
            "STOPPED_VEHICLE",
            "amber",
            f"{readable_name} track {track_id} remained stationary in the active lane for approximately {duration:.1f} seconds.",
            0.90,
            track_id,
            duration,
        )

    def cleanup_stale_tracks(self, current_time: float) -> None:
        timeout = float(self.config.get("track_cleanup_seconds", 60))
        stale = [
            track_id
            for track_id, last_seen in self.last_seen.items()
            if current_time - last_seen > timeout
        ]
        for track_id in stale:
            self.last_seen.pop(track_id, None)
            self.person_history.pop(track_id, None)
            self.vehicle_history.pop(track_id, None)
            self.restricted_entry_memory.pop(track_id, None)

    def draw_zones(self, frame):
        frame_height, frame_width = frame.shape[:2]
        specifications = [
            ("crowd_zone", (255, 180, 0), "CROWD ZONE"),
            ("loitering_zone", (0, 255, 255), "LOITERING ZONE"),
            ("restricted_zone", (0, 0, 255), "RESTRICTED ZONE"),
            ("active_lane_zone", (0, 165, 255), "ACTIVE LANE"),
        ]

        for config_key, color, label in specifications:
            cfg = self.config.get(config_key)
            if not cfg or not cfg.get("polygon"):
                continue

            polygon = self.normalized_polygon_to_pixels(
                cfg["polygon"], frame_width, frame_height
            )
            polygon_array = np.asarray(polygon, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_array], color)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
            cv2.polylines(frame, [polygon_array], True, color, 2)
            x, y = polygon_array[0]
            cv2.putText(
                frame,
                label,
                (int(x), max(24, int(y) - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
            )

        return frame
