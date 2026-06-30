import os
import sys
import cv2
import numpy as np
import torch
import shutil

# Dynamically add repositories to path so we can import their modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "repositories", "Real-ESRGAN"))
sys.path.append(os.path.join(BASE_DIR, "repositories", "realbasicVSR"))

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    RealESRGANer = None


# ==========================================
# REUSABLE PREPROCESSING UTILITIES
# ==========================================

def strip_letterbox(img, threshold=10):
    """Removes dead black borders from edges to save VRAM and compute."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = np.where(np.max(gray, axis=1) > threshold)[0]
    cols = np.where(np.max(gray, axis=0) > threshold)[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return img, (0, img.shape[0], 0, img.shape[1])
        
    ymin, ymax = rows[0], rows[-1] + 1
    xmin, xmax = cols[0], cols[-1] + 1
    return img[ymin:ymax, xmin:xmax], (ymin, ymax, xmin, xmax)


def enhance_contrast(img):
    """Applies CLAHE to optimize visibility in shadows/low lighting conditions."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def align_dimensions(img, divisor=8):
    """Pads images using reflection to be perfectly divisible for neural network layers."""
    h, w, _ = img.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    return img, pad_h, pad_w


def remove_padding(img, pad_h, pad_w, scale=4):
    """Crops out the reflective padding added during the alignment step."""
    if pad_h == 0 and pad_w == 0:
        return img
    h, w, _ = img.shape
    return img[0:h - (pad_h * scale), 0:w - (pad_w * scale)]


def is_duplicate_frame(img1, img2, threshold=2.0):
    """Returns True if frames are virtually identical to bypass duplicate heavy compute."""
    if img1 is None or img2 is None or img1.shape != img2.shape:
        return False
    mae = np.mean(cv2.absdiff(img1, img2))
    return mae < threshold


# ==========================================
# CORE PROCESSOR MODULES
# ==========================================

class ImageProcessor:
    """Handles single image upscaling using Real-ESRGAN with preprocessing."""
    def __init__(self, config):
        self.config = config['real_esrgan']
        self.device = torch.device(config['device'])
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = os.path.join(config['weights_dir'], f"{self.config['model_name']}.pth")
        
        self.upscaler = RealESRGANer(
            scale=self.config['outscale'],
            model_path=model_path,
            model=model,
            tile=self.config['tile'],
            device=self.device
        )

    def preprocess(self, img):
        """Orchestrates image pre-processing steps."""
        img, _ = strip_letterbox(img)
        img = enhance_contrast(img)
        img, pad_h, pad_w = align_dimensions(img, divisor=8)
        return img, pad_h, pad_w

    def process(self, input_path, output_path):
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to read image: {input_path}")
            return False
            
        # Run preprocessing
        processed_img, pad_h, pad_w = self.preprocess(img)
        
        try:
            outscale = self.config['outscale']
            output, _ = self.upscaler.enhance(processed_img, outscale=outscale)
            
            # Run postprocessing to restore shapes
            output = remove_padding(output, pad_h, pad_w, scale=outscale)
            
            cv2.imwrite(output_path, output)
            return True
        except Exception as e:
            print(f"Error processing image {input_path}: {e}")
            return False


class VideoProcessor:
    """Handles video streams by segmenting scene cuts and batching preprocessed frames."""
    def __init__(self, config):
        self.config = config['real_basic_vsr']
        self.device = config['device']
        self.weights = os.path.join(config['weights_dir'], f"{self.config['model_name']}.pth")

    def process_video_scenes(self, input_path, temp_root, scene_threshold=30.0):
        """
        Extracts frames, strips duplicates, applies enhancements, 
        and partitions frame sequences at hard scene cuts.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return []

        scene_idx = 0
        frame_idx = 0
        prev_frame = None
        
        current_scene_dir = os.path.join(temp_root, f"scene_{scene_idx:03d}")
        os.makedirs(current_scene_dir, exist_ok=True)
        scene_dirs = [current_scene_dir]

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Motion Deduplication Check
            if prev_frame is not None and is_duplicate_frame(frame, prev_frame):
                continue  # Skip processing this redundant frame
                
            # 2. Scene Cut Detection (using basic histogram changes)
            if prev_frame is not None:
                hist1 = cv2.calcHist([frame], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                
                if hist_diff > scene_threshold:
                    scene_idx += 1
                    current_scene_dir = os.path.join(temp_root, f"scene_{scene_idx:03d}")
                    os.makedirs(current_scene_dir, exist_ok=True)
                    scene_dirs.append(current_scene_dir)
            
            # 3. Apply Frame-Level Enhancements (Contrast & Dimensions)
            enhanced = enhance_contrast(frame)
            aligned, _, _ = align_dimensions(enhanced, divisor=8)
            
            # Save frame to active scene bucket
            frame_path = os.path.join(current_scene_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, aligned)
            
            prev_frame = frame.copy()
            frame_idx += 1

        cap.release()
        return scene_dirs

    def process(self, input_path, output_path):
        import subprocess
        
        # Define clean operational scratchpad directories
        temp_root = os.path.join(BASE_DIR, "temp_video_workspace")
        os.makedirs(temp_root, exist_ok=True)
        
        print(f"Parsing and preprocessing scenes for: {input_path}")
        scene_dirs = self.process_video_scenes(input_path, temp_root)
        
        if not scene_dirs:
            print("Video preprocessing yielded no usable sequences.")
            return False

        script_path = os.path.join(BASE_DIR, "repositories", "realbasicVSR", "inference_realbasicvsr.py")
        
        try:
            for scene_dir in scene_dirs:
                # Skip scene folders that ended up completely empty
                if not os.listdir(scene_dir):
                    continue
                    
                scene_output = os.path.join(scene_dir, "output_scene.mp4")
                
                cmd = [
                    sys.executable, script_path,
                    "--video_path", scene_dir,  # Passing preprocessed frame sequence
                    "--output_path", scene_output,
                    "--weights", self.weights,
                    "--device", self.device
                ]
                
                print(f"Running RealBasicVSR inference on sequence segment: {os.path.basename(scene_dir)}")
                subprocess.run(cmd, check=True)
            
            # Note: For production use, you can easily stitch the 'output_scene.mp4' 
            # sequences together into a final video file using an FFmpeg concat script.
            # Right now, we move the primary enhanced clip out to finalize execution.
            first_scene_out = os.path.join(scene_dirs[0], "output_scene.mp4")
            if os.path.exists(first_scene_out):
                shutil.move(first_scene_out, output_path)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Subprocess model failure: {e}")
            return False
        finally:
            # Clean up transient frames from disk to free up workspace space
            if os.path.exists(temp_root):
                shutil.rmtree(temp_root)
