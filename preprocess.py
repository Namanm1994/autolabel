import os
import sys
import torch

# Dynamically add repositories to path so we can import their modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "repositories", "Real-ESRGAN"))
sys.path.append(os.path.join(BASE_DIR, "repositories", "realbasicVSR"))

# Attempt imports from your forked repos (Adjust if your forks have custom entry points)
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    RealESRGANer = None

class ImageProcessor:
    """Handles single image upscaling using Real-ESRGAN."""
    def __init__(self, config):
        self.config = config['real_esrgan']
        self.device = torch.device(config['device'])
        
        # Initialize Real-ESRGAN network architecture
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = os.path.join(config['weights_dir'], f"{self.config['model_name']}.pth")
        
        self.upscaler = RealESRGANer(
            scale=self.config['outscale'],
            model_path=model_path,
            model=model,
            tile=self.config['tile'],
            device=self.device
        )

    def process(self, input_path, output_path):
        import cv2
        img = cv2.imread(input_path)
        try:
            output, _ = self.upscaler.enhance(img, outscale=self.config['outscale'])
            cv2.imwrite(output_path, output)
            print(self.upscaler)
            return True
        except Exception as e:
            print(f"Error processing image {input_path}: {e}")
            return False


class VideoProcessor:
    """Handles video stream upscaling using RealBasicVSR."""
    def __init__(self, config):
        self.config = config['real_basic_vsr']
        self.device = config['device']
        self.weights = os.path.join(config['weights_dir'], f"{self.config['model_name']}.pth")
        # NOTE: RealBasicVSR typically relies on MMEditing/MMMagic tools or a direct runner script.
        # We invoke this using a fallback CLI wrapper or direct inference call depending on repository layout.

    def process(self, input_path, output_path):
        # Because RealBasicVSR has deep dependencies on MMcv/MMEditing, executing its native 
        # inference script via subprocess is often the most stable way to prevent dependency conflicts.
        import subprocess
        
        script_path = os.path.join(BASE_DIR, "repositories", "realbasicVSR", "inference_realbasicvsr.py")
        
        cmd = [
            sys.executable, script_path,
            "--video_path", input_path,
            "--output_path", output_path,
            "--weights", self.weights,
            "--device", self.device
        ]
        
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error processing video {input_path}: {e}")
            return False
