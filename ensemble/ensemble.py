import torch
import numpy as np
from transformers import OwlViT2ForObjectDetection, OwlViT2Processor
# SAM 3 is universally accessed via native tools or ultralytics integration by 2026
from ultralytics import SAM3 

class ModelEnsemble:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading Labeling Ensemble Encoders...")
        
        # 1. Initialize OWL-ViT-2
        self.owl_processor = OwlViT2Processor.from_pretrained("google/owlvit2-base-patch16-ensemble")
        self.owl_model = OwlViT2ForObjectDetection.from_pretrained("google/owlvit2-base-patch16-ensemble").to(device)
        
        # 2. Initialize SAM 3 (Handles both exhaustive text tracking and box prompts)
        self.sam3_model = SAM3("sam3_l.pt") # Options: sam3_b.pt, sam3_l.pt

    def get_owlvit2_boxes(self, rgb_image, text_prompt, threshold=0.25):
        """Generates candidate bounding boxes via open-vocabulary detection."""
        inputs = self.owl_processor(text=[[text_prompt]], images=rgb_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.owl_model(**inputs)
        
        target_sizes = torch.Tensor([rgb_image.shape[:2]]).to(self.device)
        results = self.owl_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
        
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        return boxes, scores

    def run_grounding_dino(self, rgb_image, text_prompt, threshold=0.25):
        """Placeholder for Grounding DINO client interface."""
        # Typically imported via groundingdino.util.inference or a dedicated API client
        # Returns boxes, scores
        return np.empty((0, 4)), np.empty((0,))

    def apply_nms(self, boxes, scores, iou_thresh=0.45):
        """Merges overlapping predictions across detectors using standard NMS."""
        if len(boxes) == 0:
            return []
        import torchvision
        torch_boxes = torch.tensor(boxes, dtype=torch.float32)
        torch_scores = torch.tensor(scores, dtype=torch.float32)
        keep_indices = torchvision.ops.nms(torch_boxes, torch_scores, iou_thresh)
        return boxes[keep_indices.numpy()]

    def generate_pseudolabels(self, rgb_image, text_prompt, metadata):
        """Combines predictions from all models into complete bounding boxes and instance masks."""
        # Step A: Get Box Candidates
        owl_boxes, owl_scores = self.get_owlvit2_boxes(rgb_image, text_prompt)
        dino_boxes, dino_scores = self.run_grounding_dino(rgb_image, text_prompt)
        
        # Combine and consolidate bounding boxes
        all_boxes = np.vstack([owl_boxes, dino_boxes]) if len(dino_boxes) > 0 else owl_boxes
        all_scores = np.concatenate([owl_scores, dino_scores]) if len(dino_scores) > 0 else owl_scores
        
        final_canvas_boxes = self.apply_nms(all_boxes, all_scores)
        
        # Step B: Segment Everything via SAM 3 Promptable Concept Segmentation
        # SAM 3 natively searches the frame using the open-vocabulary concept
        sam3_results = self.sam3_model(rgb_image, bboxes=final_canvas_boxes if len(final_canvas_boxes) > 0 else None, labels=text_prompt)
        
        final_masks = []
        final_boxes = []
        
        if sam3_results and hasattr(sam3_results[0], 'masks') and sam3_results[0].masks is not None:
            # Extract generated pixel segmentation masks
            masks_data = sam3_results[0].masks.data.cpu().numpy()
            boxes_data = sam3_results[0].boxes.xyxy.cpu().numpy()
            
            # Map back to real asset aspect ratios
            for mask, box in zip(masks_data, boxes_data):
                # Resize the binary mask back to raw file shape
                h_orig, w_orig = metadata["original_shape"]
                h_res, w_res = metadata["resized_shape"]
                
                # Crop away letterbox background before resizing
                cropped_mask = mask[:h_res, :w_res]
                restored_mask = cv2.resize(cropped_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                
                final_masks.append(restored_mask)
                
                # Rescale boxes back
                scale = metadata["scale"]
                final_boxes.append([int(coord / scale) for coord in box])
                
        return final_boxes, final_masks
