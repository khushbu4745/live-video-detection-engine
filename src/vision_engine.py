# src/vision_engine.py

from src.engines.detection import StandardYOLOEngine, RTDETREngine, YOLOWorldEngine, OWLv2Engine
from src.engines.segmentation import YOLOSegmentationEngine
from src.engines.pose import YOLOPoseEngine
from src.engines.obb import YOLOOBBEngine
from src.engines.classification import YOLOClassificationEngine

class VisionManager:
    """Acts as the main traffic controller, routing requests to isolated domain engines."""
    
    def __init__(self):
        # Two-tier registry: [Task Type][Model Name] -> Engine Class
        self.registry = {
            "detection": {
                "standard": StandardYOLOEngine,
                "rtdetr": RTDETREngine,
                "world": YOLOWorldEngine,
                "owlv2": OWLv2Engine
            },
            "segmentation": {
                "standard": YOLOSegmentationEngine
            },
            "pose": {
                "standard": YOLOPoseEngine
            },
            "obb": {
                "standard": YOLOOBBEngine
            },
            "classification": {
                "standard": YOLOClassificationEngine
            }
        }
        
        # In-memory cache for lazy-loaded models
        self.loaded_engines = {}
        
        # Default startup state
        self.active_task = "detection"
        self.active_model = "standard"
        
        # Pre-load the default model into GPU memory
        self._load_engine(self.active_task, self.active_model)

    def _get_cache_key(self, task: str, model: str) -> str:
        return f"{task}_{model}"

    def _load_engine(self, task: str, model: str):
        """Instantiates the requested engine and caches it in VRAM."""
        cache_key = self._get_cache_key(task, model)
        if cache_key not in self.loaded_engines:
            print(f"First-time initialization for {task.upper()} ({model.upper()})...")
            engine_class = self.registry[task][model]
            self.loaded_engines[cache_key] = engine_class()

    def get_active_engine(self):
        """Returns the currently active engine to the WebSocket loop."""
        cache_key = self._get_cache_key(self.active_task, self.active_model)
        return self.loaded_engines[cache_key]

    def switch_pipeline(self, task: str, model: str):
        """Hot-swaps the active pipeline based on the user's Tier 1 and Tier 2 selection."""
        if task in self.registry and model in self.registry[task]:
            self._load_engine(task, model)
            self.active_task = task
            self.active_model = model
            print(f"--> Instantly hot-swapped stream to {task.upper()} task using {model.upper()} model.")
        else:
            print(f"Failed to switch: Task '{task}' with model '{model}' is not registered.")