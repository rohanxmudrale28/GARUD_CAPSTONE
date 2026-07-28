from ultralytics import YOLO
import torch

class ObjectEngine:
    def __init__(self, model_path="yolov8m.pt"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path)

def detect(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            device=self.device,
            conf=0.6,
            iou=0.5,
            verbose=False
        )
        return self.extract_detections(results)
        
def extract_detections(self, results):
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class_name": self.model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })
        return detections
