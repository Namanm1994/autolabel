import logging
import json
from typing import Tuple, List, Optional
from pydantic import BaseModel, Field, validator
import torch
import torchvision.transforms.functional as F

# Setup structured JSON logging for ELK/Datadog ingestion
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        }
        return json.dumps(log_record)

logger = logging.getLogger("PipelinePreprocessor")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ==============================================================================
# 1. STRICT CONFIGURATION SCHEMA (Pydantic V2)
# ==============================================================================
class PreprocessConfig(BaseModel):
    target_size: int = Field(default=1024, ge=256, le=4096)
    padding_color: List[float] = Field(default=[114.0, 114.0, 114.0])
    device: str = Field(default="cuda")
    
    @validator("padding_color")
    def validate_rgb(cls, v):
        if len(v) != 3 or not all(0 <= c <= 255 for c in v):
            raise ValueError("padding_color must be a list of 3 floats between 0 and 255")
        return v


# ==============================================================================
# 2. CUDA-ACCELERATED PRODUCTION PREPROCESSOR
# ==============================================================================
class ProductionPreprocessor:
    def __init__(self, config_dict: dict):
        # Validate configuration instantly on instantiation
        self.cfg = PreprocessConfig(**config_dict["preprocessing"])
        self.device = torch.device(self.cfg.device)
        
        # Pre-initialize uniform padding tensor directly on target device
        self.pad_color_tensor = torch.tensor(
            self.cfg.padding_color, dtype=torch.float32, device=self.device
        ).view(3, 1, 1)

    def process_tensor_gpu(self, raw_cpu_tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Processes an already-loaded image tensor directly on the GPU.
        Input tensor shape expected: [C, H, W], uint8
        """
        try:
            # 1. Immediately ship raw bytes to device; perform all math in VRAM
            cuda_tensor = raw_cpu_tensor.to(self.device, non_blocking=True).float()
            
            _, orig_h, orig_w = cuda_tensor.shape
            
            # 2. Calculate proportional scaling factors
            scale = self.cfg.target_size / max(orig_h, orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            
            # 3. GPU Resize via Bilinear/Cubic interpolation
            resized_tensor = F.resize(
                cuda_tensor, [new_h, new_w], 
                interpolation=F.InterpolationMode.BILINEAR, 
                antialias=True
            )
            
            # 4. Canvas allocation & allocation-free padding placement
            # Initializes canvas with ImageNet background gray directly in memory
            canvas = self.pad_color_tensor.expand(3, self.cfg.target_size, self.cfg.target_size).clone()
            canvas[:, :new_h, :new_w] = resized_tensor
            
            # Normalize to [0.0, 1.0] expected by Vision Transformers
            canvas /= 255.0
            
            metadata = {
                "original_shape": (orig_h, orig_w),
                "resized_shape": (new_h, new_w),
                "scale": scale
            }
            
            return canvas, metadata

        except Exception as e:
            logger.error(f"GPU Preprocessing Pipeline Failure: {str(e)}", exc_info=True)
            raise

    def handle_corrupted_asset(self, asset_path: str, destination_dlq: str):
        """Isolates and logs faulty/corrupted assets to a Dead-Letter Directory."""
        import shutil
        import os
        logger.warning(f"Quarantining corrupted asset to Dead-Letter Queue: {asset_path}")
        os.makedirs(destination_dlq, exist_ok=True)
        try:
            shutil.move(asset_path, os.path.join(destination_dlq, os.path.basename(asset_path)))
        except IOError as e:
            logger.critical(f"Failed to isolate corrupted asset: {str(e)}")
