"""
WeaponEngine - motion-gated, open-vocabulary weapon detection for GARUD.

Design notes (read this before changing thresholds):

Two off-the-shelf pretrained gun/knife YOLO models were tried first and
both failed hard: they locked onto static desk clutter (a computer mouse,
a pen-holder box) as "guns" with high, persistent confidence on nearly
every frame, because they were trained on close-up product-style images
and don't generalize to real elevated CCTV footage.

This engine instead uses YOLOE, an open-vocabulary detector that is
prompted with plain-text class names (e.g. "pistol", "knife") instead of
being narrowly trained on a small dataset. That generalizes better, but on
its own it *still* occasionally misfires on static objects that happen to
have a gun-like silhouette (e.g. a mouse, scissors, cables).

The fix used here is the same one GARUD's own AnomalyEngine already uses:
frame-differencing motion gating. A raw weapon-class detection only counts
if that region of the frame is also moving between frames. Static clutter
never moves, so it gets filtered out; an object actually being held and
moved by a person passes through. A rolling-window temporal confirmation
(the same pattern used by loitering / stopped-vehicle detection elsewhere
in this project) then requires several motion-gated hits in a short
window before raising a confirmed alert, instead of trusting any single
frame.

This is a proof-of-concept, not a production-hardened detector:
  - Confidence scores from the open-vocabulary model are low (often
    0.05-0.15) even on true positives. Don't rely on confidence alone.
  - Motion gating assumes the camera itself is static (fixed CCTV mount).
    It will not work unmodified on a moving/panning camera.
  - A region of interest (ROI) can be supplied to focus compute and
    improve small-object detectability, but the engine also works on the
    full frame at some cost to sensitivity (the object needs to occupy
    enough pixels to be recognized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLOE


Box = List[int]  # [x1, y1, x2, y2] in full-frame pixel coordinates


@dataclass
class WeaponDetection:
    class_name: str
    confidence: float
    motion_score: float
    box: Box

    @property
    def motion_gated(self) -> bool:
        return self._gate_threshold is not None and self.motion_score >= self._gate_threshold

    _gate_threshold: Optional[float] = None


class WeaponEngine:
    """
    Motion-gated, open-vocabulary weapon detector.

    Usage:
        engine = WeaponEngine(roi=(380, 60, 750, 500))
        for frame in video_frames:
            result = engine.detect(frame)
            if result["confirmed"]:
                raise_alert(...)
    """

    # Plain-language prompts - no dataset-specific training required.
    # Extend this list if you want to catch other object types.
    DEFAULT_CLASSES = [
        "gun", "pistol", "handgun", "revolver",
        "knife", "screwdriver", "crowbar", "metal rod",
    ]

    def __init__(
        self,
        model_name: str = "yoloe-11s-seg.pt",
        classes: Optional[List[str]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        confidence: float = 0.04,
        motion_threshold: float = 15.0,
        confirm_window: int = 10,
        confirm_count: int = 4,
        imgsz: int = 640,
    ):
        """
        Args:
            model_name: YOLOE checkpoint. 'yoloe-11s-seg.pt' (small) is
                recommended over larger variants - in testing, the larger
                'yoloe-11l-seg.pt' produced *more* false positives and
                missed a true detection that the small model caught.
            classes: text prompts for the open-vocabulary detector.
            roi: (x1, y1, x2, y2) region of interest in full-frame pixel
                coordinates. Small weapon-like objects can be missed if
                they occupy too small a fraction of the full frame; a
                tighter ROI around the area of interest (e.g. a counter,
                a doorway) improves detection. Pass None to run on the
                full frame.
            confidence: minimum per-detection confidence from YOLOE.
                Kept intentionally low (0.03-0.08) because true positives
                on small/blurry weapons in real CCTV footage often score
                low; motion gating + temporal confirmation is what
                filters out the resulting noise, not this threshold.
            motion_threshold: mean absolute pixel difference (0-255)
                required inside a detection's box, versus the previous
                frame, for that detection to be treated as "real motion"
                rather than static clutter. Tune per-camera: dim/noisy
                footage may need a lower value, brightly lit footage with
                camera-induced flicker may need a higher one.
            confirm_window: size of the rolling frame window used for
                temporal confirmation.
            confirm_count: number of motion-gated hits required within
                the rolling window before `confirmed` is True.
            imgsz: inference resolution passed to YOLOE.
        """
        self.classes = classes or list(self.DEFAULT_CLASSES)
        self.model = YOLOE(model_name)
        self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))

        self.roi = roi
        self.confidence = confidence
        self.motion_threshold = motion_threshold
        self.imgsz = imgsz

        self.confirm_window = confirm_window
        self.confirm_count = confirm_count

        self._prev_gray = None
        self._history: List[bool] = []

    def _crop(self, frame):
        if self.roi is None:
            return frame, (0, 0)
        x1, y1, x2, y2 = self.roi
        return frame[y1:y2, x1:x2], (x1, y1)

    def detect(self, frame) -> dict:
        """
        Run detection on a single BGR frame (as read by cv2.VideoCapture).

        Returns a dict:
            "detections":       every raw detection, motion score included,
                                 in full-frame coordinates.
            "gated_detections":  the subset that passed the motion gate.
            "confirmed":         True once enough recent frames have a
                                  motion-gated detection to raise an alert.
        """
        crop, (offset_x, offset_y) = self._crop(frame)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        results = self.model.predict(
            crop, conf=self.confidence, imgsz=self.imgsz, verbose=False
        )[0]

        detections: List[WeaponDetection] = []
        for cls, conf, box in zip(results.boxes.cls, results.boxes.conf, results.boxes.xyxy):
            bx1, by1, bx2, by2 = [max(0, round(v)) for v in box.tolist()]

            motion_score = 0.0
            if self._prev_gray is not None:
                region = cv2.absdiff(self._prev_gray, gray)[by1:by2, bx1:bx2]
                motion_score = float(region.mean()) if region.size > 0 else 0.0

            detections.append(
                WeaponDetection(
                    class_name=results.names[int(cls)],
                    confidence=float(conf),
                    motion_score=motion_score,
                    box=[bx1 + offset_x, by1 + offset_y, bx2 + offset_x, by2 + offset_y],
                    _gate_threshold=self.motion_threshold,
                )
            )

        self._prev_gray = gray

        gated = [d for d in detections if d.motion_score >= self.motion_threshold]
        has_gated_detection = len(gated) > 0

        self._history.append(has_gated_detection)
        if len(self._history) > self.confirm_window:
            self._history.pop(0)

        confirmed = sum(self._history) >= self.confirm_count

        return {
            "detections": detections,
            "gated_detections": gated,
            "confirmed": confirmed,
        }

    def reset(self):
        """Call when switching to a new video/camera stream to clear temporal state."""
        self._prev_gray = None
        self._history = []
