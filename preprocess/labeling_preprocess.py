import cv2
import numpy as np

class LabelingPreprocessor:
    def __init__(self, target_size=1024):
        self.target_size = target_size

    def prepare_image(self, img_path):
        """Loads an image and returns a letterboxed 1024x1024 version along with scale metadata."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create blank canvas and paste image into center/top-left
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        
        # Metadata keeps track of how to warp coordinates back to original resolution
        metadata = {
            "original_shape": (h, w),
            "resized_shape": (new_h, new_w),
            "scale": scale
        }
        return canvas, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), metadata

    def scale_boxes_back(self, boxes, metadata):
        """Converts bounding boxes from 1024x1024 canvas space back to original resolution."""
        scale = metadata["scale"]
        scaled_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            scaled_boxes.append([
                int(xmin / scale),
                int(ymin / scale),
                int(xmax / scale),
                int(ymax / scale)
            ])
        return np.array(scaled_boxes)
