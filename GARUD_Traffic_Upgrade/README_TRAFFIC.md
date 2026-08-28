# GARUD Traffic Intelligence Upgrade

This folder is a clean, runnable traffic-violation extension for GARUD. It adds ByteTrack-based vehicle tracking, wrong-way detection, red-light jumping, restricted-lane use, no-entry crossing, illegal stopping, snapshots, a REST/MJPEG API, map nodes, and a HUD dashboard.

## 1. Install

```bash
cd /Users/rohanmudrale/GARUD
source garud_v2-3/.venv/bin/activate  # use your actual venv path
cp -R /path/to/GARUD_Traffic_Upgrade/* .
pip install -r requirements-traffic.txt
```

Place `yolov8n.pt` in the GARUD root. Your repository already contains this model.

## 2. Configure the camera and road geometry

Edit `config_data/cameras.json`.

- `source`: `0` for a webcam, a video path, or an RTSP URL.
- `lat/lng`: exact physical camera coordinates.
- All polygon/line coordinates are normalized. `[0,0]` is top-left and `[1,1]` is bottom-right.
- `wrong_way_lane.polygon`: lane region.
- `allowed_direction`: image direction vector. `[0,-1]` means vehicles should move upward in the image.
- `stop_line`: red-light stop line.
- `restricted_lane`: bus lane or other controlled lane.
- `no_stopping_zone`: area where a stationary tracked vehicle becomes a violation.

Start with a recorded road video. Pause on one frame, estimate points, run, and tune. Camera angle must stay fixed after calibration.

## 3. Run

```bash
python api_server.py
```

Open `http://localhost:5001`. Do not open the HTML with `file://`. Flask serves the page so Leaflet, the API, and the video stream use one origin.

## 4. Test red-light jumping

1. Click the camera map node.
2. Press RED in Live Telemetry.
3. Let a tracked vehicle cross the configured stop line.
4. GARUD creates an event, saves a snapshot, makes the node red for 30 seconds, and raises the audible alarm.

Browser audio starts only after a user interaction because modern browsers block autoplay. Click anywhere once after opening the dashboard.

## 5. Important accuracy notes

- This implementation detects rule violations from scene geometry. It does not infer legal guilt.
- Use a stable elevated camera, accurate polygons, and real traffic-light state from a controller/PLC for deployment.
- For production, replace the demo signal timer with a signed signal-controller API or edge GPIO input.
- Calibrate each camera independently and validate with labelled day/night/rain clips.
- License plates require a separate ANPR pipeline and lawful retention controls.

## 6. API

- `GET /api/cameras`
- `GET /api/events?limit=50`
- `GET /api/stream/C001`
- `POST /api/signal/C001` with `{"state":"red"}`
- `GET /snapshots/<name>`

## 7. Merge with your existing GARUD pipeline

The provided server is independently runnable to make installation predictable. Once validated, call `TrafficViolationEngine.process()` immediately after your existing `ObjectEngine` returns tracked vehicle boxes. The required detection schema is:

```python
{
  "bbox": [x1, y1, x2, y2],
  "track_id": 42,
  "class_id": 2,
  "class_name": "car",
  "confidence": 0.91
}
```

Keep the existing crowd, anomaly, gesture, database, and alert engines running beside this module. Stream all event dictionaries to the same dashboard event endpoint.
