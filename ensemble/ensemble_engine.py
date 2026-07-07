from models.owlvit_wrapper import OwlViT2Wrapper
from models.dino_wrapper import GroundingDinoWrapper

class PseudolabelEnsemble:
    def __init__(self):
        self.owl = OwlViT2Wrapper()
        self.dino = GroundingDinoWrapper(config_path="...", weights_path="...")

    def process_frame(self, rgb_image, prompt):
        # Extract clean predictions using identical API endpoints
        owl_boxes, owl_scores = self.owl.predict(rgb_image, prompt, threshold=0.20)
        dino_boxes, dino_scores = self.dino.predict(rgb_image, prompt, box_threshold=0.30)
        
        # Run your Non-Maximum Suppression (NMS) logic cleanly here
        final_boxes = self.merge_predictions(owl_boxes, dino_boxes)
        return final_boxes
