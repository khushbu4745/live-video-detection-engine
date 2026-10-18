import torch
from ultralytics import YOLO
from src.engines.base import BaseVisionEngine

class YOLOPoseEngine(BaseVisionEngine):
    def __init__(self, model_path="models/yolo11n-pose.pt"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO11 Pose from {model_path} on {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.confidence_threshold = 0.45

    def process_frame(self, frame):
        results_generator = self.model(frame, conf=self.confidence_threshold, stream=True, verbose=False)
        return self._format_results(next(results_generator))

    def set_custom_classes(self, class_list: list):
        pass 

    def _format_results(self, result):
        data = []
        if hasattr(result, 'keypoints') and result.keypoints is not None and result.boxes:
            for box, kp in zip(result.boxes, result.keypoints):
                joints = kp.xy[0].tolist() if len(kp.xy) > 0 else []
                data.append({
                    "task_type": "pose",
                    "keypoints": joints,
                    "conf": box.conf[0].item(),
                    "label": "person"
                })
        return data