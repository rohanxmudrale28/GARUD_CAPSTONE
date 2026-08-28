from __future__ import annotations

import json
import os
import threading
import time

from collections import deque
from pathlib import Path

import cv2

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from ultralytics import YOLO

from security import RuleAnomalyEngine
from traffic import TrafficViolationEngine
from traffic.signal_controller import SignalController


ROOT = Path(
    __file__
).resolve().parent


CAMERA_CONFIG_PATH = (
    ROOT
    / "config_data"
    / "cameras.json"
)


SECURITY_CONFIG_PATH = (
    ROOT
    / "config_data"
    / "security_zones.json"
)


if not CAMERA_CONFIG_PATH.exists():
    raise FileNotFoundError(
        "Camera configuration was not found: "
        f"{CAMERA_CONFIG_PATH}"
    )


if not SECURITY_CONFIG_PATH.exists():
    raise FileNotFoundError(
        "Security anomaly configuration was "
        f"not found: {SECURITY_CONFIG_PATH}"
    )


CAMERAS = json.loads(
    CAMERA_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)


SECURITY_CONFIG = json.loads(
    SECURITY_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)


SNAPSHOT_DIRECTORY = (
    ROOT
    / "snapshots"
)


SNAPSHOT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


EVENTS = deque(
    maxlen=1000
)


EVENT_LOCK = threading.Lock()


PROCESSORS = {}


def resolve_camera_source(
    source,
):
    """
    Resolve relative video paths against the GARUD
    project directory.
    """

    if isinstance(
        source,
        int,
    ):
        return source

    if not isinstance(
        source,
        str,
    ):
        return source

    lowered_source = (
        source.lower()
    )

    if (
        lowered_source.startswith(
            "rtsp://"
        )
        or lowered_source.startswith(
            "http://"
        )
        or lowered_source.startswith(
            "https://"
        )
    ):
        return source

    possible_path = Path(source)

    if not possible_path.is_absolute():
        possible_path = (
            ROOT
            / possible_path
        )

    return str(
        possible_path.resolve()
    )


class CameraProcessor:
    def __init__(
        self,
        camera_id: str,
        camera_config: dict,
    ):
        self.camera_id = camera_id
        self.camera_config = camera_config

        self.latest_frame = None
        self.online = False
        self.fps = 0.0
        self.last_error = ""

        self.counts = {
            "WRONG_WAY": 0,
            "RED_LIGHT_JUMP": 0,
            "RESTRICTED_LANE": 0,
            "NO_ENTRY": 0,
            "ILLEGAL_STOPPING": 0,
            "CROWD_THRESHOLD_EXCEEDED": 0,
            "SUDDEN_CROWD_SURGE": 0,
            "LOITERING": 0,
            "RESTRICTED_ZONE_ENTRY": 0,
            "STOPPED_VEHICLE": 0,
        }

        self.signal_controller = (
            SignalController()
        )

        self.traffic_engine = (
            TrafficViolationEngine(
                camera_id,
                camera_config,
            )
        )

        self.security_engine = (
            RuleAnomalyEngine(
                camera_id=camera_id,
                config=SECURITY_CONFIG,
            )
        )

        model_path = (
            ROOT
            / "yolov8m.pt"
        )

        if not model_path.exists():
            model_path = (
                ROOT
                / "yolov8n.pt"
            )

        if not model_path.exists():
            raise FileNotFoundError(
                "Neither yolov8m.pt nor yolov8n.pt "
                "was found in the project root."
            )

        print(
            f"[{camera_id}] Loading model: "
            f"{model_path.name}"
        )

        self.model = YOLO(
            str(model_path)
        )

        self.running = True

        self.worker_thread = threading.Thread(
            target=self.run,
            daemon=True,
        )

        self.worker_thread.start()

    def create_capture(
        self,
    ):
        configured_source = (
            self.camera_config.get(
                "source",
                0,
            )
        )

        resolved_source = (
            resolve_camera_source(
                configured_source
            )
        )

        print(
            f"[{self.camera_id}] Opening source: "
            f"{resolved_source}"
        )

        capture = cv2.VideoCapture(
            resolved_source
        )

        return (
            capture,
            resolved_source,
        )

    def save_event(
        self,
        event,
        frame,
    ):
        """
        Save an event snapshot and place the event in
        the shared GARUD incident stream.
        """

        snapshot_name = (
            f"{self.camera_id}_"
            f"{int(event.timestamp)}_"
            f"{event.event_type}_"
            f"{event.id[:8]}.jpg"
        )

        snapshot_path = (
            SNAPSHOT_DIRECTORY
            / snapshot_name
        )

        cv2.imwrite(
            str(snapshot_path),
            frame,
        )

        event.snapshot = (
            f"/snapshots/{snapshot_name}"
        )

        event_data = (
            event.to_dict()
        )

        with EVENT_LOCK:
            EVENTS.appendleft(
                event_data
            )

            self.counts[
                event.event_type
            ] = (
                self.counts.get(
                    event.event_type,
                    0,
                )
                + 1
            )

        print(
            f"[{self.camera_id}] "
            f"{event.event_type}: "
            f"{event.message}"
        )

    def run(
        self,
    ):
        capture = None
        resolved_source = None
        previous_frame_time = (
            time.time()
        )

        while self.running:
            try:
                if (
                    capture is None
                    or not capture.isOpened()
                ):
                    if capture is not None:
                        capture.release()

                    (
                        capture,
                        resolved_source,
                    ) = self.create_capture()

                    if not capture.isOpened():
                        self.online = False
                        self.last_error = (
                            "Unable to open camera or video"
                        )

                        time.sleep(2)
                        continue

                success, frame = (
                    capture.read()
                )

                if not success:
                    self.online = False

                    configured_source = (
                        self.camera_config.get(
                            "source",
                            0,
                        )
                    )

                    if isinstance(
                        configured_source,
                        str,
                    ):
                        capture.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            0,
                        )

                        time.sleep(0.15)
                        continue

                    capture.release()
                    capture = None
                    time.sleep(1)
                    continue

                self.online = True
                self.last_error = ""

                signal_state = (
                    self.signal_controller.update()
                )

                result = self.model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[
                        0,
                        1,
                        2,
                        3,
                        5,
                        7,
                    ],
                    conf=0.30,
                    iou=0.50,
                    verbose=False,
                )[0]

                detections = []

                if (
                    result.boxes is not None
                    and result.boxes.id
                    is not None
                ):
                    boxes = (
                        result.boxes.xyxy
                        .cpu()
                        .tolist()
                    )

                    track_ids = (
                        result.boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

                    class_ids = (
                        result.boxes.cls
                        .int()
                        .cpu()
                        .tolist()
                    )

                    confidences = (
                        result.boxes.conf
                        .cpu()
                        .tolist()
                    )

                    for (
                        bounding_box,
                        track_id,
                        class_id,
                        confidence,
                    ) in zip(
                        boxes,
                        track_ids,
                        class_ids,
                        confidences,
                    ):
                        class_name = (
                            result.names[
                                class_id
                            ]
                        )

                        detection = {
                            "bbox": bounding_box,
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                        }

                        detections.append(
                            detection
                        )

                        (
                            x1,
                            y1,
                            x2,
                            y2,
                        ) = map(
                            int,
                            bounding_box,
                        )

                        if (
                            class_name
                            == "person"
                        ):
                            box_color = (
                                255,
                                255,
                                0,
                            )
                        else:
                            box_color = (
                                0,
                                220,
                                255,
                            )

                        cv2.rectangle(
                            frame,
                            (
                                x1,
                                y1,
                            ),
                            (
                                x2,
                                y2,
                            ),
                            box_color,
                            2,
                        )

                        label = (
                            f"{class_name} "
                            f"#{track_id} "
                            f"{confidence:.2f}"
                        )

                        cv2.putText(
                            frame,
                            label,
                            (
                                x1,
                                max(
                                    24,
                                    y1 - 8,
                                ),
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            box_color,
                            2,
                        )

                traffic_events = (
                    self.traffic_engine.process(
                        detections=detections,
                        frame_shape=frame.shape,
                        signal_state=signal_state,
                    )
                )

                for event in traffic_events:
                    self.save_event(
                        event,
                        frame,
                    )

                security_events = (
                    self.security_engine.process(
                        detections=detections,
                        frame_shape=frame.shape,
                    )
                )

                for event in security_events:
                    self.save_event(
                        event,
                        frame,
                    )

                self.traffic_engine.draw_rules(
                    frame,
                    signal_state,
                )

                self.security_engine.draw_zones(
                    frame
                )

                current_time = time.time()

                elapsed_time = max(
                    current_time
                    - previous_frame_time,
                    0.001,
                )

                current_fps = (
                    1.0
                    / elapsed_time
                )

                self.fps = (
                    0.90
                    * self.fps
                    + 0.10
                    * current_fps
                )

                previous_frame_time = (
                    current_time
                )

                signal_color = (
                    (0, 0, 255)
                    if signal_state == "red"
                    else (
                        (0, 165, 255)
                        if signal_state
                        == "amber"
                        else (0, 255, 0)
                    )
                )

                cv2.putText(
                    frame,
                    (
                        "SIGNAL: "
                        f"{signal_state.upper()}"
                    ),
                    (
                        20,
                        34,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    signal_color,
                    2,
                )

                person_count = sum(
                    1
                    for detection
                    in detections
                    if detection[
                        "class_name"
                    ]
                    == "person"
                )

                vehicle_count = sum(
                    1
                    for detection
                    in detections
                    if detection[
                        "class_name"
                    ]
                    != "person"
                )

                cv2.putText(
                    frame,
                    (
                        f"PEOPLE: "
                        f"{person_count}"
                    ),
                    (
                        20,
                        66,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"VEHICLES: "
                        f"{vehicle_count}"
                    ),
                    (
                        20,
                        94,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 220, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    (
                        f"FPS: "
                        f"{self.fps:.1f}"
                    ),
                    (
                        20,
                        122,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                )

                encode_success, jpeg = (
                    cv2.imencode(
                        ".jpg",
                        frame,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            82,
                        ],
                    )
                )

                if encode_success:
                    self.latest_frame = (
                        jpeg.tobytes()
                    )

            except Exception as error:
                self.online = False
                self.last_error = str(error)

                print(
                    f"[{self.camera_id}] "
                    f"Processing error: {error}"
                )

                time.sleep(1)

        if capture is not None:
            capture.release()

    def stream(
        self,
    ):
        while True:
            if self.latest_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + self.latest_frame
                    + b"\r\n"
                )

            time.sleep(0.04)


for (
    camera_id,
    camera_config,
) in CAMERAS.items():
    PROCESSORS[
        camera_id
    ] = CameraProcessor(
        camera_id,
        camera_config,
    )


@app.get("/")
def dashboard():
    return render_template(
        "dashboard.html"
    )


@app.get("/api/cameras")
def get_cameras():
    response_data = []

    current_time = time.time()

    with EVENT_LOCK:
        events_snapshot = list(
            EVENTS
        )

    for (
        camera_id,
        camera_config,
    ) in CAMERAS.items():
        processor = (
            PROCESSORS[
                camera_id
            ]
        )

        recent_event = next(
            (
                event
                for event
                in events_snapshot
                if event[
                    "camera_id"
                ]
                == camera_id
            ),
            None,
        )

        if (
            recent_event
            and current_time
            - recent_event[
                "timestamp"
            ]
            < 30
        ):
            severity = (
                recent_event[
                    "severity"
                ]
            )
        else:
            severity = "clear"

        response_data.append(
            {
                "id": camera_id,
                "name": camera_config.get(
                    "name",
                    camera_id,
                ),
                "city": camera_config.get(
                    "city",
                    "",
                ),
                "lat": camera_config.get(
                    "lat",
                    20.5937,
                ),
                "lng": camera_config.get(
                    "lng",
                    78.9629,
                ),
                "online": processor.online,
                "fps": round(
                    processor.fps,
                    1,
                ),
                "signal": (
                    processor
                    .signal_controller
                    .state
                ),
                "severity": severity,
                "counts": processor.counts,
                "error": processor.last_error,
            }
        )

    return jsonify(
        response_data
    )


@app.get("/api/events")
def get_events():
    requested_limit = int(
        request.args.get(
            "limit",
            100,
        )
    )

    requested_limit = max(
        1,
        min(
            requested_limit,
            500,
        ),
    )

    with EVENT_LOCK:
        event_data = list(
            EVENTS
        )[
            :requested_limit
        ]

    return jsonify(
        event_data
    )


@app.post(
    "/api/signal/<camera_id>"
)
def change_signal(
    camera_id,
):
    if camera_id not in PROCESSORS:
        return jsonify(
            {
                "error": (
                    "Camera was not found"
                )
            }
        ), 404

    request_data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    requested_state = str(
        request_data.get(
            "state",
            "",
        )
    ).lower()

    if requested_state not in {
        "red",
        "amber",
        "green",
    }:
        return jsonify(
            {
                "error": (
                    "State must be red, "
                    "amber, or green"
                )
            }
        ), 400

    processor = (
        PROCESSORS[
            camera_id
        ]
    )

    processor.signal_controller.set(
        requested_state
    )

    return jsonify(
        {
            "camera_id": camera_id,
            "state": (
                processor
                .signal_controller
                .state
            ),
        }
    )


@app.get(
    "/api/stream/<camera_id>"
)
def camera_stream(
    camera_id,
):
    if camera_id not in PROCESSORS:
        return jsonify(
            {
                "error": (
                    "Camera was not found"
                )
            }
        ), 404

    return Response(
        PROCESSORS[
            camera_id
        ].stream(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.get(
    "/snapshots/<path:filename>"
)
def get_snapshot(
    filename,
):
    return send_from_directory(
        SNAPSHOT_DIRECTORY,
        filename,
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "cameras": len(
                PROCESSORS
            ),
            "events": len(
                EVENTS
            ),
        }
    )


if __name__ == "__main__":
    port = int(
        os.getenv(
            "PORT",
            "5001",
        )
    )

    print(
        "[GARUD] Starting command "
        f"server on port {port}"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        threaded=True,
        debug=False,
    )