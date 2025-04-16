"""
Inference script for VisionDetect.

This script provides a command-line interface for running inference with trained
object detection models on images or video streams.
"""

import os
import argparse
import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.predictor import Predictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with object detection model")
    
    # Input arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image, directory of images, or video file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Confidence threshold for filtering predictions")
    
    # Other arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for inference (cuda, cpu)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    parser.add_argument("--video", action="store_true",
                        help="Process input as video")
    parser.add_argument("--save-video", action="store_true",
                        help="Save processed video")
    parser.add_argument("--show", action="store_true",
                        help="Show results in window")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def process_image(predictor, image_path, output_dir, show=False):
    """
    Process a single image.
    
    Args:
        predictor: Model predictor
        image_path: Path to image
        output_dir: Directory to save results
        show: Whether to show results in window
    """
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Make prediction
    result = predictor.predict(image, return_visualization=True)
    
    # Save visualization
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    visualization = cv2.cvtColor(result["visualization"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, visualization)
    
    # Show result if requested
    if show:
        cv2.imshow("Detection", visualization)
        cv2.waitKey(0)
    
    return result


def process_directory(predictor, directory, output_dir, show=False):
    """
    Process all images in a directory.
    
    Args:
        predictor: Model predictor
        directory: Directory containing images
        output_dir: Directory to save results
        show: Whether to show results in window
    """
    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(directory).glob(f"*{ext}")))
    
    # Process each image
    results = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        result = process_image(predictor, image_path, output_dir, show=show)
        results.append(result)
    
    return results


def process_video(predictor, video_path, output_dir, save_video=False, show=False):
    """
    Process a video file.
    
    Args:
        predictor: Model predictor
        video_path: Path to video file
        output_dir: Directory to save results
        save_video: Whether to save processed video
        show: Whether to show results in window
    """
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer if requested
    if save_video:
        output_path = os.path.join(output_dir, os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_count = 0
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            result = predictor.predict(frame_rgb, return_visualization=True)
            
            # Convert visualization back to BGR
            visualization = cv2.cvtColor(result["visualization"], cv2.COLOR_RGB2BGR)
            
            # Save frame if requested
            if save_video:
                writer.write(visualization)
            
            # Show frame if requested
            if show:
                cv2.imshow("Detection", visualization)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    if save_video:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        # Override config with command line arguments
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                config_key = key.replace("_", "-")
                if config_key in config:
                    config[config_key] = value
    else:
        # Use command line arguments as config
        config = vars(args)
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Create predictor
    predictor = Predictor(
        model_path=config["model_path"],
        device=config["device"],
        confidence_threshold=config["confidence_threshold"]
    )
    
    # Process input
    input_path = config["input"]
    
    if config.get("video", False) or input_path.endswith((".mp4", ".avi", ".mov")):
        # Process video
        process_video(
            predictor=predictor,
            video_path=input_path,
            output_dir=config["output_dir"],
            save_video=config.get("save_video", False),
            show=config.get("show", False)
        )
    elif os.path.isdir(input_path):
        # Process directory of images
        process_directory(
            predictor=predictor,
            directory=input_path,
            output_dir=config["output_dir"],
            show=config.get("show", False)
        )
    else:
        # Process single image
        process_image(
            predictor=predictor,
            image_path=input_path,
            output_dir=config["output_dir"],
            show=config.get("show", False)
        )


if __name__ == "__main__":
    main()
