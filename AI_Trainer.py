import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3

# Load YOLO model
model = YOLO("yolov8n-pose.pt")

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say("Hey! Rohan created me. I'm your AI Trainer. Let's count your bicep curls and check your form.")
engine.runAndWait()

# Angle function
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Streamlit UI
st.set_page_config(page_title="AI Gym Trainer by Rohan", layout="centered")
st.title("💪 AI Gym Trainer")
st.markdown("""
Welcome to your personal AI-powered biceps curl counter.  
This app uses your webcam and tracks both **left and right arm reps** using advanced pose detection.  
Press **Start** to begin your workout!
""")

# Video stream processor
class RepCounter(VideoTransformerBase):
    def __init__(self):
        self.counter_left = 0
        self.counter_right = 0
        self.stage_left = None
        self.stage_right = None

    def speak_feedback(self, side, correct=True):
        if correct:
            engine.say(f"Good rep on your {side} arm. Light weight baby!")
        else:
            engine.say(f"Watch your {side} arm form.")
        engine.runAndWait()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        kpts = results[0].keypoints

        if kpts is not None and len(kpts.xy) > 0:
            kpts = kpts.xy[0].cpu().numpy()
            try:
                # Left
                shoulder_l, elbow_l, wrist_l = kpts[5], kpts[7], kpts[9]
                angle_left = calculate_angle(shoulder_l, elbow_l, wrist_l)
                if angle_left > 160:
                    self.stage_left = "down"
                if angle_left < 40 and self.stage_left == "down":
                    self.stage_left = "up"
                    self.counter_left += 1
                    self.speak_feedback("left", True)
                cv2.putText(img, f'{int(angle_left)}°', tuple(elbow_l.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                # Right
                shoulder_r, elbow_r, wrist_r = kpts[6], kpts[8], kpts[10]
                angle_right = calculate_angle(shoulder_r, elbow_r, wrist_r)
                if angle_right > 160:
                    self.stage_right = "down"
                if angle_right < 40 and self.stage_right == "down":
                    self.stage_right = "up"
                    self.counter_right += 1
                    self.speak_feedback("right", True)
                cv2.putText(img, f'{int(angle_right)}°', tuple(elbow_r.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Draw boxes
                cv2.rectangle(img, (0, 0), (320, 120), (50, 50, 50), -1)
                cv2.putText(img, "LEFT REPS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(img, str(self.counter_left), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255), 2)
                cv2.putText(img, "STAGE", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.putText(img, self.stage_left if self.stage_left else "-", (90, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                cv2.putText(img, "RIGHT REPS", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(img, str(self.counter_right), (160, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 2)
                cv2.putText(img, "STAGE", (160, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.putText(img, self.stage_right if self.stage_right else "-", (240, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            except Exception as e:
                print("Keypoint Error:", e)

        return img

# Start stream
webrtc_streamer(
    key="biceps-counter",
    video_processor_factory=RepCounter,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
