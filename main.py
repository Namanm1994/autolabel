import os
import yaml
from processors import ImageProcessor, VideoProcessor

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Ensure environment folders exist
    os.makedirs(config['input_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)

    print("Initializing Enhancement Models...")
    image_pipeline = ImageProcessor(config)
    video_pipeline = VideoProcessor(config)

    # Scan for files
    files = [f for f in os.listdir(config['input_dir']) if os.path.isfile(os.path.join(config['input_dir'], f))]
    
    if not files:
        print(f"No files found in '{config['input_dir']}'. Add some images/videos and rerun.")
        return

    print(f"Found {len(files)} files to process.")

    for file in files:
        input_path = os.path.join(config['input_dir'], file)
        output_path = os.path.join(config['output_dir'], f"enhanced_{file}")
        _, ext = os.path.splitext(file.lower())

        if ext in IMAGE_EXTENSIONS:
            print(f"Processing Image: {file} -> Real-ESRGAN")
            image_pipeline.process(input_path, output_path)
            
        elif ext in VIDEO_EXTENSIONS:
            print(f"Processing Video: {file} -> RealBasicVSR")
            video_pipeline.process(input_path, output_path)
            
        else:
            print(f"Skipping unsupported file type: {file}")

if __name__ == "__main__":
    main()
