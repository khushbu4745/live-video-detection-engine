import torch
from ultralytics import YOLO
from src.engines.base import BaseVisionEngine

class YOLOSegmentationEngine(BaseVisionEngine):
    def __init__(self, model_path="models/yolo11n-seg.pt"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO11 Segmentation from {model_path} on {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.confidence_threshold = 0.45

    def process_frame(self, frame):
        results_generator = self.model(frame, conf=self.confidence_threshold, stream=True, verbose=False)
        return self._format_results(next(results_generator))

    def set_custom_classes(self, class_list: list):
        pass 

    def _format_results(self, result):
        data = []
        if result.boxes and hasattr(result, 'masks') and result.masks is not None:
            for box, mask in zip(result.boxes, result.masks):
                polygon = mask.xy[0].tolist() if len(mask.xy) > 0 else []
                data.append({
                    "task_type": "segmentation",
                    "polygon": polygon,
                    "conf": box.conf[0].item(),
                    "label": result.names[int(box.cls[0].item())]
                })
        return data