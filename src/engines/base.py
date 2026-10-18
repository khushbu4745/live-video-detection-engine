from abc import ABC, abstractmethod

class BaseVisionEngine(ABC):
    """Abstract blueprint guaranteeing standard API outputs across all architectures."""
    
    @abstractmethod
    def process_frame(self, frame) -> list:
        pass
    
    @abstractmethod
    def set_custom_classes(self, class_list: list):
        pass