"""
GARUD weapon-detection PoC runner.

Run this on any video and it will:
  - detect weapon-like objects using an open-vocabulary model (no
    dataset-specific training required)
  - motion-gate detections to suppress false positives on static
    background clutter
  - require sustained detections (temporal confirmation) before raising
    a "confirmed" alert
  - write an annotated copy of the video and print a live per-frame
    status line to the terminal while it runs

Example - the exact settings tuned for the test1 clip:

    python run_weapon_demo.py \\
        --video indian_cctv_weapon_test1_mp4.mp4 \\
        --output test1_weapon_detected.mp4 \\
        --roi 380,60,750,500 \\
        --start-frame 680 --end-frame 900

Run on a full video with no ROI (slower, and small/distant weapons may
not be detected - see engines/weapon_engine.py for why a tight ROI
helps):

    python run_weapon_demo.py --video some_other_clip.mp4
"""

import argparse
import time

import cv2

from engines.weapon_engine import WeaponEngine


def parse_roi(value):
    if value is None:
        return None
    parts = [int(x) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x1,y1,x2,y2 (four integers)")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="weapon_detection_output.mp4", help="Path to save the annotated output video")
    parser.add_argument("--roi", type=parse_roi, default=None,
                         help="Region of interest as x1,y1,x2,y2 in pixel coordinates. "
                              "Omit to run on the full frame.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Default: process to the end of the video")
    parser.add_argument("--confidence", type=float, default=0.04)
    parser.add_argument("--motion-threshold", type=float, default=15.0)
    parser.add_argument("--confirm-window", type=int, default=10)
    parser.add_argument("--confirm-count", type=int, default=4)
    parser.add_argument("--classes", nargs="+", default=None,
                         help="Override the default text prompts, e.g. --classes pistol knife")
    parser.add_argument("--display", action="store_true",
                         help="Show a live preview window while processing (needs a GUI-capable machine)")
    args = parser.parse_args()

    engine = WeaponEngine(
        classes=args.classes,
        roi=args.roi,
        confidence=args.confidence,
        motion_threshold=args.motion_threshold,
        confirm_window=args.confirm_window,
        confirm_count=args.confirm_count,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    end_frame = args.end_frame if args.end_frame is not None else total_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    idx = args.start_frame
    t0 = time.time()
    confirmed_frames = 0

    print(f"[GARUD WeaponEngine] video: {args.video}")
    print(f"[GARUD WeaponEngine] frames {args.start_frame} -> {end_frame} of {total_frames} total")
    print(f"[GARUD WeaponEngine] roi: {args.roi or 'full frame'}")
    print()

    while idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        result = engine.detect(frame)
        confirmed = result["confirmed"]
        if confirmed:
            confirmed_frames += 1

        if args.roi:
            rx1, ry1, rx2, ry2 = args.roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 200, 0), 1)

        for det in result["detections"]:
            x1, y1, x2, y2 = det.box
            if det.motion_gated:
                color = (0, 0, 255) if confirmed else (0, 140, 255)
                thickness = 3
                label = f"{det.class_name} {det.confidence:.2f} (motion {det.motion_score:.0f})"
            else:
                color = (120, 120, 120)
                thickness = 1
                label = None
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if label:
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        banner = "WEAPON ALERT CONFIRMED - GARUD WeaponEngine (PoC)" if confirmed else "MONITORING..."
        banner_bg = (0, 0, 255) if confirmed else (40, 40, 40)
        banner_fg = (255, 255, 255) if confirmed else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (w, 50), banner_bg, -1)
        cv2.putText(frame, banner, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, banner_fg, 2)

        print(f"Frame: {idx}/{total_frames}  |  confirmed frames so far: {confirmed_frames}", end="\r")

        writer.write(frame)

        if args.display:
            cv2.imshow("GARUD Weapon Detection PoC", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print()
    print()
    print(f"[GARUD WeaponEngine] done in {elapsed:.1f}s")
    print(f"[GARUD WeaponEngine] confirmed frames: {confirmed_frames} / {idx - args.start_frame}")
    print(f"[GARUD WeaponEngine] output saved to: {args.output}")


if __name__ == "__main__":
    main()
