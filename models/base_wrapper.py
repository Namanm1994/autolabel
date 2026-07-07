from abc import ABC, abstractmethod

class BaseLabeler(ABC):
    @abstractmethod
    def load_model(self):
        """Initialize and load weights to GPU."""
        pass

    @abstractmethod
    def predict(self, image, text_prompt, **kwargs):
        """
        Accepts a standard raw NumPy BGR/RGB image.
        Must handle its own specific preprocessing internally.
        Returns: (boxes, scores) or (masks, boxes)
        """
        pass
