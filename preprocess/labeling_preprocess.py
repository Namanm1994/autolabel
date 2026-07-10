import cv2
import numpy as np

class LabelingPreprocessor:
    def __init__(self, config):
        self.cfg = config['preprocessing']
        self.target_size = self.cfg['target_size']
        self.pad_color = self.cfg['padding_color']

    def _correct_exif_orientation(self, img_path):
        """Reads an image and applies EXIF orientation adjustments if present."""
        img = cv2.imread(img_path)
        if img is None or not self.cfg['correct_exif']:
            return img
            
        # Check orientation via a secondary check or cv2 flag if using modern openCV versions
        # Standard robust workaround to read orientation metadata safely:
        try:
            # Re-read image with EXIF flag evaluation
            img_corrected = cv2.imread(img_path, cv2.IMREAD_IMAGE_ORIENTATION_TIFF)
            if img_corrected is not None:
                return img_corrected
        except Exception:
            pass
        return img

    def _get_interpolation(self, orig_w, orig_h):
        """Chooses the ideal mathematical transform based on scaling direction."""
        if max(orig_w, orig_h) > self.target_size:
            # Downsampling requires INTER_AREA to prevent aliasing
            return cv2.INTER_AREA if self.cfg['downscale_interpolation'] == "area" else cv2.INTER_LINEAR
        else:
            # Upsampling benefits from cubic smoothness
            return cv2.INTER_CUBIC if self.cfg['upscale_interpolation'] == "cubic" else cv2.INTER_LINEAR

    def prepare_image(self, img_path):
        """Loads, rotates, balances, and pads an image into neutral canvas space."""
        img = self._correct_exif_orientation(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Select appropriate interpolation mode
        interp = self._get_interpolation(w, h)
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        # Build canvas filled with ImageNet neutral gray background
        canvas = np.full((self.target_size, self.target_size, 3), self.pad_color, dtype=np.uint8)
        
        # Center or pin the image to top-left (top-left simplifies box math)
        canvas[:new_h, :new_w] = resized
        
        metadata = {
            "original_shape": (h, w),
            "resized_shape": (new_h, new_w),
            "scale": scale
        }
        return canvas, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), metadata

    def scale_boxes_back(self, boxes, metadata):
        """Maps canvas boxes cleanly back to original resolution assets."""
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
