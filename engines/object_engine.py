from ultralytics import YOLO
import torch

class ObjectEngine:
    def __init__(self, model_path="yolov8m.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path)
