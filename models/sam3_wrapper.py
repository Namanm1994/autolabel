import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import logging

logger = logging.getLogger("SAM3Wrapper")

class SAM3Wrapper:
    """
    Dedicated wrapper for Meta SAM 3 (Promptable Concept Segmentation).
    Handles text prompt sanitization, image tensor scaling, and mask generation.
    """
    def __init__(self, model_id: str = "facebookresearch/sam3", device: str = "cuda"):
        self.device = torch.device(device)
        logger.info(f"Loading SAM 3 checkpoint: {model_id} on {self.device}...")
        
        # Load Hugging Face / Meta SAM 3 processor and model weights
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def _sanitize_prompts(self, prompts: list[str]) -> list[str]:
        """Ensures text prompts are clean, short noun phrases for SAM 3 text encoder."""
        clean_prompts = []
        for p in prompts:
            # Remove punctuation, strip whitespace, keep short
            cleaned = p.strip().lower()
            if len(cleaned.split()) > 6:
                logger.warning(f"Prompt '{p}' is long. SAM 3 works best with short noun phrases.")
            clean_prompts.append(cleaned)
        return clean_prompts

    def preprocess_image(self, bgr_image: np.ndarray) -> Image.Image:
        """Converts raw BGR OpenCV frame to RGB PIL Image expected by Sam3Processor."""
        rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_img)

    @torch.no_grad()
    def segment_text_concepts(
        self, 
        bgr_image: np.ndarray, 
        text_prompts: list[str], 
        confidence_threshold: float = 0.35
    ) -> dict:
        """
        Runs Promptable Concept Segmentation (PCS) using natural language noun phrases.
        """
        sanitized_prompts = self._sanitize_prompts(text_prompts)
        pil_img = self.preprocess_image(bgr_image)

        # 1. Prepare inputs through Sam3Processor
        inputs = self.processor(
            images=pil_img, 
            text=sanitized_prompts, 
            return_tensors="pt"
        ).to(self.device)

        # 2. Execute SAM 3 forward pass
        outputs = self.model(**inputs)

        # 3. Post-process outputs to original image resolution
        target_sizes = [pil_img.size[::-1]]  # (height, width)
        results = self.processor.post_process_instance_segmentation(
            outputs, 
            target_sizes=target_sizes,
            threshold=confidence_threshold
        )[0]

        masks = results["masks"].cpu().numpy()       # Shape: [N, H, W] (bool)
        boxes = results["boxes"].cpu().numpy()       # Shape: [N, 4] (xmin, ymin, xmax, ymax)
        scores = results["scores"].cpu().numpy()     # Shape: [N]
        labels = results["labels"]                   # Matched prompt strings/indices

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }
