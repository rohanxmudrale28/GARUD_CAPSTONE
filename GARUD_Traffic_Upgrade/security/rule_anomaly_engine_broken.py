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


def point_inside_polygon(
    point: Point,
    polygon: List[Point],
) -> bool:
    """
    Return True when a point is inside or on the edge
    of the configured polygon.
    """

    polygon_array = np.asarray(
        polygon,
        dtype=np.int32,
    )

    result = cv2.pointPolygonTest(
        polygon_array,
        point,
        False,
    )

    return result >= 0


class RuleAnomalyEngine:
    """
    Explainable fixed-CCTV anomaly engine.

    Supported events:

    1. CROWD_THRESHOLD_EXCEEDED
    2. SUDDEN_CROWD_SURGE
    3. LOITERING
    4. RESTRICTED_ZONE_ENTRY
    5. STOPPED_VEHICLE

    Every detection must use this structure:

    {
        "bbox": [x1, y1, x2, y2],
        "track_id": 12,
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.92
    }
    """

    PERSON_CLASS_NAMES = {
        "person",
    }

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

    def __init__(
        self,
        camera_id: str,
        config: dict,
    ):
        self.camera_id = camera_id
        self.config = config

        history_size = int(
            config.get(
                "maximum_history_points",
                1800,
            )
        )

        self.person_history: Dict[
            int,
            Deque[Tuple[float, Point]]
        ] = defaultdict(
            lambda: deque(
                maxlen=history_size
            )
        )

        self.vehicle_history: Dict[
            int,
            Deque[Tuple[float, Point]]
        ] = defaultdict(
            lambda: deque(
                maxlen=history_size
            )
        )

        self.count_history: Deque[
            Tuple[float, int]
        ] = deque(
            maxlen=1800
        )

        self.last_seen: Dict[int, float] = {}

        self.last_alert: Dict[
            Tuple[str, int],
            float
        ] = {}

        self.restricted_entry_memory: Dict[
            int,
            bool
        ] = {}

    @staticmethod
    def get_ground_point(
        bounding_box: List[float],
    ) -> Point:
        """
        Use the bottom-center of a bounding box as the
        approximate ground-contact position.
        """

        x1, y1, x2, y2 = map(
            float,
            bounding_box,
        )

        center_x = int(
            (x1 + x2) / 2
        )

        ground_y = int(y2)

        return center_x, ground_y

    @staticmethod
    def normalized_polygon_to_pixels(
        polygon: List[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> List"""
        Convert normalized coordinates between 0 and 1
        into pixel coordinates.
        """

        pixel_polygon = []

        for x_value, y_value in polygon:
            pixel_x = int(
                x_value * frame_width
            )

            pixel_y = int(
                y_value * frame_height
            )

            pixel_polygon.append(
                (
                    pixel_x,
                    pixel_y,
                )
            )

        return pixel_polygon

    def emit_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        confidence: float,
        track_id: int = -1,
        duration_seconds: float = 0.0,
    ) -> List"""
        Create an event while preventing repeated alerts
        within the configured cooldown period.
        """

        current_time = time.time()

        cooldown_seconds = float(
            self.config.get(
                "alert_cooldown_seconds",
                15,
            )
        )

        alert_key = (
            event_type,
            int(track_id),
        )

        previous_alert_time = (
            self.last_alert.get(
                alert_key,
                0.0,
            )
        )

        if (
            current_time
            - previous_alert_time
            < cooldown_seconds
        ):
            return []

        self.last_alert[
            alert_key
        ] = current_time

        event = SecurityEvent(
            id=str(uuid.uuid4()),
            camera_id=self.camera_id,
            event_type=event_type,
            severity=severity,
            timestamp=current_time,
            confidence=float(
                max(
                    0.0,
                    min(
                        confidence,
                        0.99,
                    ),
                )
            ),
            message=message,
            track_id=int(track_id),
            duration_seconds=float(
                duration_seconds
            ),
        )

        return [event]

    def process(
        self,
        detections: List[dict],
        frame_shape,
    ) -> List"""
        Process all tracked detections from the current
        frame and return any generated anomaly events.
        """

        frame_height = int(
            frame_shape[0]
        )

        frame_width = int(
            frame_shape[1]
        )

        current_time = time.time()

        generated_events: List[
            SecurityEvent
        ] = []

        current_persons = []
        current_vehicles = []

        for detection in detections:
            class_name = str(
                detection.get(
                    "class_name",
                    "",
                )
            ).lower()

            track_id = int(
                detection["track_id"]
            )

            ground_point = (
                self.get_ground_point(
                    detection["bbox"]
                )
            )

            self.last_seen[
                track_id
            ] = current_time

            if (
                class_name
                in self.PERSON_CLASS_NAMES
            ):
                person_data = (
                    track_id,
                    ground_point,
                    detection,
                )

                current_persons.append(
                    person_data
                )

                self.person_history[
                    track_id
                ].append(
                    (
                        current_time,
                        ground_point,
                    )
                )

            elif (
                class_name
                in self.VEHICLE_CLASS_NAMES
            ):
                vehicle_data = (
                    track_id,
                    ground_point,
                    detection,
                )

                current_vehicles.append(
                    vehicle_data
                )

                self.vehicle_history[
                    track_id
                ].append(
                    (
                        current_time,
                        ground_point,
                    )
                )

        generated_events.extend(
            self.detect_crowd_threshold(
                persons=current_persons,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

        generated_events.extend(
            self.detect_crowd_surge(
                person_count=len(
                    current_persons
                ),
                current_time=current_time,
            )
        )

        for (
            track_id,
            ground_point,
            detection,
        ) in current_persons:
            generated_events.extend(
                self.detect_loitering(
                    track_id=track_id,
                    ground_point=ground_point,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    current_time=current_time,
                )
            )

            generated_events.extend(
                self.detect_restricted_entry(
                    track_id=track_id,
                    ground_point=ground_point,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

        for (
            track_id,
            ground_point,
            detection,
        ) in current_vehicles:
            generated_events.extend(
                self.detect_stopped_vehicle(
                    track_id=track_id,
                    ground_point=ground_point,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    current_time=current_time,
                    class_name=detection.get(
                        "class_name",
                        "vehicle",
                    ),
                )
            )

        self.cleanup_stale_tracks(
            current_time
        )

        return generated_events

    def detect_crowd_threshold(
        self,
        persons,
        frame_width: int,
        frame_height: int,
    ) -> Listcrowd_config = self.config.get(
            "crowd_zone"
        )

        if not crowd_config:
            return []

        threshold = int(
            crowd_config.get(
                "threshold",
                20,
            )
        )

        polygon_config = (
            crowd_config.get(
                "polygon"
            )
        )

        if polygon_config:
            crowd_polygon = (
                self.normalized_polygon_to_pixels(
                    polygon=polygon_config,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

            people_inside = []

            for person_data in persons:
                (
                    track_id,
                    ground_point,
                    detection,
                ) = person_data

                if point_inside_polygon(
                    ground_point,
                    crowd_polygon,
                ):
                    people_inside.append(
                        person_data
                    )

            crowd_count = len(
                people_inside
            )

        else:
            crowd_count = len(persons)

        if crowd_count < threshold:
            return []

        confidence = min(
            0.99,
            0.60
            + (
                crowd_count
                / max(
                    threshold,
                    1,
                )
            )
            * 0.20,
        )

        return self.emit_event(
            event_type=(
                "CROWD_THRESHOLD_EXCEEDED"
            ),
            severity="amber",
            message=(
                f"{crowd_count} tracked people "
                f"detected inside the monitored "
                f"crowd zone. Configured threshold "
                f"is {threshold}."
            ),
            confidence=confidence,
        )

    def detect_crowd_surge(
        self,
        person_count: int,
        current_time: float,
    ) -> Listsurge_config = self.config.get(
            "crowd_surge"
        )

        if not surge_config:
            return []

        window_seconds = float(
            surge_config.get(
                "window_seconds",
                8,
            )
        )

        minimum_rise = int(
            surge_config.get(
                "minimum_rise",
                8,
            )
        )

        minimum_current_count = int(
            surge_config.get(
                "minimum_current_count",
                10,
            )
        )

        self.count_history.append(
            (
                current_time,
                person_count,
            )
        )

        while (
            self.count_history
            and current_time
            - self.count_history[0][0]
            > window_seconds * 2
        ):
            self.count_history.popleft()

        older_counts = []

        for timestamp, count in (
            self.count_history
        ):
            age = (
                current_time
                - timestamp
            )

            if age >= (
                window_seconds * 0.70
            ):
                older_counts.append(
                    count
                )

        if not older_counts:
            return []

        baseline_count = min(
            older_counts
        )

        increase = (
            person_count
            - baseline_count
        )

        if (
            increase < minimum_rise
            or person_count
            < minimum_current_count
        ):
            return []

        confidence = min(
            0.99,
            0.68
            + (
                increase
                / max(
                    minimum_rise,
                    1,
                )
            )
            * 0.12,
        )

        return self.emit_event(
            event_type=(
                "SUDDEN_CROWD_SURGE"
            ),
            severity="red",
            message=(
                "Person count rose from "
                f"approximately {baseline_count} "
                f"to {person_count} within "
                f"{window_seconds:.0f} seconds."
            ),
            confidence=confidence,
        )

    def detect_loitering(
        self,
        track_id: int,
        ground_point: Point,
        frame_width: int,
        frame_height: int,
        current_time: float,
    ) -> Listloiter_config = self.config.get(
            "loitering_zone"
        )

        if not loiter_config:
            return []

        polygon_config = (
            loiter_config.get(
                "polygon"
            )
        )

        if not polygon_config:
            return []

        loiter_polygon = (
            self.normalized_polygon_to_pixels(
                polygon=polygon_config,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

        if not point_inside_polygon(
            ground_point,
            loiter_polygon,
        ):
            return []

        required_seconds = float(
            loiter_config.get(
                "seconds",
                30,
            )
        )

        recent_history = []

        for (
            timestamp,
            tracked_point,
        ) in self.person_history[
            track_id
        ]:
            if (
                current_time
                - timestamp
                <= required_seconds + 3
            ):
                recent_history.append(
                    (
                        timestamp,
                        tracked_point,
                    )
                )

        if len(recent_history) < 5:
            return []

        tracked_duration = (
            current_time
            - recent_history[0][0]
        )

        if (
            tracked_duration
            < required_seconds
        ):
            return []

        starting_point = (
            recent_history[0][1]
        )

        maximum_displacement = 0.0

        for (
            timestamp,
            tracked_point,
        ) in recent_history:
            displacement = math.dist(
                starting_point,
                tracked_point,
            )

            maximum_displacement = max(
                maximum_displacement,
                displacement,
            )

        allowed_ratio = float(
            loiter_config.get(
                "max_displacement_ratio",
                0.08,
            )
        )

        allowed_displacement = (
            frame_width
            * allowed_ratio
        )

        if (
            maximum_displacement
            > allowed_displacement
        ):
            return []

        return self.emit_event(
            event_type="LOITERING",
            severity="yellow",
            message=(
                f"Person track {track_id} "
                f"remained near the monitored "
                f"location for approximately "
                f"{tracked_duration:.1f} seconds."
            ),
            confidence=0.88,
            track_id=track_id,
            duration_seconds=tracked_duration,
        )

    def detect_restricted_entry(
        self,
        track_id: int,
        ground_point: Point,
        frame_width: int,
        frame_height: int,
    ) -> Listrestricted_config = (
            self.config.get(
                "restricted_zone"
            )
        )

        if not restricted_config:
            return []

        polygon_config = (
            restricted_config.get(
                "polygon"
            )
        )

        if not polygon_config:
            return []

        restricted_polygon = (
            self.normalized_polygon_to_pixels(
                polygon=polygon_config,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

        is_inside = point_inside_polygon(
            ground_point,
            restricted_polygon,
        )

        was_inside = (
            self.restricted_entry_memory.get(
                track_id,
                False,
            )
        )

        self.restricted_entry_memory[
            track_id
        ] = is_inside

        if not is_inside:
            return []

        if was_inside:
            return []

        zone_name = (
            restricted_config.get(
                "name",
                "restricted area",
            )
        )

        return self.emit_event(
            event_type=(
                "RESTRICTED_ZONE_ENTRY"
            ),
            severity="red",
            message=(
                f"Person track {track_id} "
                f"entered {zone_name}."
            ),
            confidence=0.94,
            track_id=track_id,
        )

    def detect_stopped_vehicle(
        self,
        track_id: int,
        ground_point: Point,
        frame_width: int,
        frame_height: int,
        current_time: float,
        class_name: str,
    ) -> Listlane_config = self.config.get(
            "active_lane_zone"
        )

        if not lane_config:
            return []

        polygon_config = (
            lane_config.get(
                "polygon"
            )
        )

        if not polygon_config:
            return []

        active_lane_polygon = (
            self.normalized_polygon_to_pixels(
                polygon=polygon_config,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

        if not point_inside_polygon(
            ground_point,
            active_lane_polygon,
        ):
            return []

        required_seconds = float(
            lane_config.get(
                "stopped_seconds",
                10,
            )
        )

        recent_history = []

        for (
            timestamp,
            tracked_point,
        ) in self.vehicle_history[
            track_id
        ]:
            if (
                current_time
                - timestamp
                <= required_seconds + 3
            ):
                recent_history.append(
                    (
                        timestamp,
                        tracked_point,
                    )
                )

        if len(recent_history) < 5:
            return []

        tracked_duration = (
            current_time
            - recent_history[0][0]
        )

        if (
            tracked_duration
            < required_seconds
        ):
            return []

        starting_point = (
            recent_history[0][1]
        )

        maximum_displacement = 0.0

        for (
            timestamp,
            tracked_point,
        ) in recent_history:
            displacement = math.dist(
                starting_point,
                tracked_point,
            )

            maximum_displacement = max(
                maximum_displacement,
                displacement,
            )

        movement_ratio = float(
            lane_config.get(
                "max_displacement_ratio",
                0.025,
            )
        )

        maximum_allowed_movement = (
            frame_width
            * movement_ratio
        )

        if (
            maximum_displacement
            > maximum_allowed_movement
        ):
            return []

        readable_class_name = (
            str(class_name)
            .replace("_", " ")
            .title()
        )

        return self.emit_event(
            event_type=(
                "STOPPED_VEHICLE"
            ),
            severity="amber",
            message=(
                f"{readable_class_name} "
                f"track {track_id} remained "
                f"stationary inside the active "
                f"lane for approximately "
                f"{tracked_duration:.1f} seconds."
            ),
            confidence=0.90,
            track_id=track_id,
            duration_seconds=tracked_duration,
        )

    def cleanup_stale_tracks(
        self,
        current_time: float,
    ) -> None:
        stale_timeout = float(
            self.config.get(
                "track_cleanup_seconds",
                60,
            )
        )

        stale_track_ids = []

        for (
            track_id,
            last_seen_time,
        ) in self.last_seen.items():
            if (
                current_time
                - last_seen_time
                > stale_timeout
            ):
                stale_track_ids.append(
                    track_id
                )

        for track_id in stale_track_ids:
            self.last_seen.pop(
                track_id,
                None,
            )

            self.person_history.pop(
                track_id,
                None,
            )

            self.vehicle_history.pop(
                track_id,
                None,
            )

            self.restricted_entry_memory.pop(
                track_id,
                None,
            )

    def draw_zones(
        self,
        frame,
    ):
        """
        Draw all configured anomaly zones on the
        output frame.
        """

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        zone_specifications = [
            (
                "crowd_zone",
                (255, 180, 0),
                "CROWD ZONE",
            ),
            (
                "loitering_zone",
                (0, 255, 255),
                "LOITERING ZONE",
            ),
            (
                "restricted_zone",
                (0, 0, 255),
                "RESTRICTED ZONE",
            ),
            (
                "active_lane_zone",
                (0, 165, 255),
                "ACTIVE LANE",
            ),
        ]

        for (
            config_key,
            color,
            label,
        ) in zone_specifications:
            zone_config = self.config.get(
                config_key
            )

            if not zone_config:
                continue

            polygon_config = (
                zone_config.get(
                    "polygon"
                )
            )

            if not polygon_config:
                continue

            pixel_polygon = (
                self.normalized_polygon_to_pixels(
                    polygon=polygon_config,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            )

            polygon_array = np.asarray(
                pixel_polygon,
                dtype=np.int32,
            )

            overlay = frame.copy()

            cv2.fillPoly(
                overlay,
                [polygon_array],
                color,
            )

            cv2.addWeighted(
                overlay,
                0.08,
                frame,
                0.92,
                0,
                frame,
            )

            cv2.polylines(
                frame,
                [polygon_array],
                True,
                color,
                2,
            )

            label_x = int(
                polygon_array[0][0]
            )

            label_y = max(
                24,
                int(
                    polygon_array[0][1]
                )
                - 7,
            )

            cv2.putText(
                frame,
                label,
                (
                    label_x,
                    label_y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
            )

        return frame