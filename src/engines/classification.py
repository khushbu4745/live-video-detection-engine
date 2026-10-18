import torch
from ultralytics import YOLO
from src.engines.base import BaseVisionEngine

class YOLOClassificationEngine(BaseVisionEngine):
    def __init__(self, model_path="models/yolo11n-cls.pt"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO11 Classification from {model_path} on {self.device}...")
        self.model = YOLO(model_path).to(self.device)

    def process_frame(self, frame):
        results_generator = self.model(frame, stream=True, verbose=False)
        return self._format_results(next(results_generator))

    def set_custom_classes(self, class_list: list):
        pass 

    def _format_results(self, result):
        data = []
        if hasattr(result, 'probs') and result.probs is not None:
            top_class_id = result.probs.top1
            confidence = result.probs.top1conf.item()
            data.append({
                "task_type": "classification",
                "conf": confidence,
                "label": result.names[top_class_id]
            })
        return data