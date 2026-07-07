import torch
from transformers import OwlViT2ForObjectDetection, OwlViT2Processor
from .base_wrapper import BaseLabeler

class OwlViT2Wrapper(BaseLabeler):
    def __init__(self, device="cuda"):
        self.device = device
        self.load_model()

    def load_model(self):
        self.processor = OwlViT2Processor.from_pretrained("google/owlvit2-base-patch16-ensemble")
        self.model = OwlViT2ForObjectDetection.from_pretrained("google/owlvit2-base-patch16-ensemble").to(self.device)

    def predict(self, rgb_image, text_prompt, threshold=0.25):
        # --- Model Specific Preprocessing Lives Privately Here ---
        inputs = self.processor(text=[[text_prompt]], images=rgb_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.Tensor([rgb_image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
        
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        return boxes, scores
