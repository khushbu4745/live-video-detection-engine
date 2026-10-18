import torch
import cv2
from ultralytics import YOLO, RTDETR
from src.engines.base import BaseVisionEngine

class StandardYOLOEngine(BaseVisionEngine):
    def __init__(self, model_path="models/yolo11n.pt"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Standard YOLO from {model_path} on {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.confidence_threshold = 0.45

    def process_frame(self, frame):
        results_generator = self.model(frame, conf=self.confidence_threshold, stream=True, verbose=False)
        return self._format_results(next(results_generator))

    def set_custom_classes(self, class_list: list):
        pass 

    def _format_results(self, result):
        boxes_data = []
        if result.boxes:
            for box in result.boxes:
                boxes_data.append({
                    "task_type": "detection",
                    "x1": box.xyxy[0][0].item(), "y1": box.xyxy[0][1].item(),
                    "x2": box.xyxy[0][2].item(), "y2": box.xyxy[0][3].item(),
                    "conf": box.conf[0].item(),
                    "label": result.names[int(box.cls[0].item())]
                })
        return boxes_data

class RTDETREngine(StandardYOLOEngine):
    def __init__(self, model_path="models/rtdetr-l.pt"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading RT-DETR from {model_path} on {self.device}...")
        self.model = RTDETR(model_path).to(self.device)
        self.confidence_threshold = 0.45

class YOLOWorldEngine(StandardYOLOEngine):
    def __init__(self, model_path="models/yolov8s-worldv2.pt"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO-World from {model_path} on {self.device}...")
        self.model = YOLO(model_path).to(self.device)
        self.model.set_classes(["person"]) 
        self.confidence_threshold = 0.15

    def set_custom_classes(self, class_list: list):
        if class_list:
            self.model.set_classes(class_list)
            print(f"YOLO-World classes updated to: {class_list}")

class OWLv2Engine(BaseVisionEngine):
    def __init__(self, model_path="/home/khushbu/industrial_vision_system/shared_models/owlv2"):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Loading OWLv2 from local folder {model_path} on {self.device}...")
        
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.processor = Owlv2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path, local_files_only=True).to(self.device)
        
        self.confidence_threshold = 0.10
        self.original_classes = ["person"]
        self.texts = ["a photo of a person"]

    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processor(text=[self.texts], images=rgb_frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([rgb_frame.shape[:2]]).to(self.device)
            results = self.processor.image_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]
            
            boxes_data = []
            for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
                boxes_data.append({
                    "task_type": "detection",
                    "x1": box[0].item(), "y1": box[1].item(),
                    "x2": box[2].item(), "y2": box[3].item(),
                    "conf": score.item(),
                    "label": self.original_classes[label_id.item()]
                })
            return boxes_data
            
        except Exception as e:
            print(f"[OWLv2 FATAL ERROR]: {e}")
            return []

    def set_custom_classes(self, class_list: list):
        if class_list:
            self.original_classes = class_list
            self.texts = [f"a photo of a {c}" for c in class_list]
            print(f"OWLv2 prompt engineered to: {self.texts}")