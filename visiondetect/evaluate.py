"""
Evaluation script for VisionDetect.

This script provides a command-line interface for evaluating trained object detection models
on test datasets and generating performance metrics and visualizations.
"""

import os
import argparse
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data.preprocessing import DataProcessor
from src.models.predictor import Predictor
from src.utils.metrics import calculate_map, calculate_map_range
from src.utils.visualization import visualize_batch, plot_training_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    
    # Dataset arguments
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing the dataset")
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 640],
                        help="Image size (height, width)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Confidence threshold for filtering predictions")
    
    # Evaluation arguments
    parser.add_argument("--output-dir", type=str, default="evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of predictions")
    parser.add_argument("--max-visualizations", type=int, default=10,
                        help="Maximum number of visualizations to generate")
    
    # Other arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for evaluation (cuda, cpu)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(predictor, dataloader, output_dir, visualize=False, max_visualizations=10):
    """
    Evaluate model on a dataset.
    
    Args:
        predictor: Model predictor
        dataloader: Test dataloader
        output_dir: Directory to save evaluation results
        visualize: Whether to generate visualizations
        max_visualizations: Maximum number of visualizations to generate
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations directory if needed
    if visualize:
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize metrics
    all_predictions = []
    all_targets = []
    
    # Process each batch
    vis_count = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Get images and targets
        images = batch["images"].cpu().numpy()
        targets = batch["targets"]
        
        # Make predictions
        batch_predictions = []
        for i in range(len(images)):
            # Convert image from CHW to HWC format
            image = np.transpose(images[i], (1, 2, 0))
            
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image * std + mean) * 255.0
            image = image.astype(np.uint8)
            
            # Make prediction
            prediction = predictor.predict(image)
            batch_predictions.append(prediction)
        
        # Store predictions and targets for mAP calculation
        all_predictions.extend(batch_predictions)
        all_targets.extend(targets)
        
        # Generate visualizations if requested
        if visualize and vis_count < max_visualizations:
            # Determine number of images to visualize from this batch
            num_vis = min(len(images), max_visualizations - vis_count)
            
            # Generate visualizations
            for i in range(num_vis):
                # Convert image from CHW to HWC format
                image = np.transpose(images[i], (1, 2, 0))
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image * std + mean) * 255.0
                image = image.astype(np.uint8)
                
                # Get prediction
                prediction = batch_predictions[i]
                
                # Create visualization
                from src.utils.visualization import visualize_detection
                vis_image = visualize_detection(
                    image=image,
                    boxes=prediction["boxes"],
                    labels=prediction["labels"],
                    scores=prediction["scores"],
                    class_names=predictor.class_names,
                    score_threshold=predictor.confidence_threshold,
                    save_path=os.path.join(vis_dir, f"detection_{vis_count}.jpg"),
                    show=False
                )
                
                vis_count += 1
    
    # Calculate metrics
    map_50 = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
    map_range = calculate_map_range(all_predictions, all_targets)
    
    # Prepare metrics dictionary
    metrics = {
        "mAP@0.5": float(map_50),
        "mAP@[0.5:0.95]": float(map_range),
        "num_samples": len(all_targets)
    }
    
    # Save metrics to JSON file
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    return metrics


def main():
    """Main evaluation function."""
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
    
    # Create data processor
    data_processor = DataProcessor(
        data_dir=config["data_dir"],
        img_size=tuple(config["img_size"]),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        augment=False
    )
    
    # Prepare dataset
    dataloaders = data_processor.prepare_dataset()
    
    # Create predictor
    predictor = Predictor(
        model_path=config["model_path"],
        device=config["device"],
        confidence_threshold=config["confidence_threshold"]
    )
    
    # Evaluate model
    if "test" in dataloaders:
        metrics = evaluate_model(
            predictor=predictor,
            dataloader=dataloaders["test"],
            output_dir=config["output_dir"],
            visualize=config.get("visualize", False),
            max_visualizations=config.get("max_visualizations", 10)
        )
        
        # Print metrics
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    else:
        print("No test data found. Please check your data directory.")
    
    # Plot training metrics if available
    checkpoint = torch.load(config["model_path"], map_location="cpu")
    if "train_metrics" in checkpoint and "val_metrics" in checkpoint:
        metrics = {
            "train": checkpoint["train_metrics"],
            "val": checkpoint["val_metrics"]
        }
        
        plot_training_metrics(
            metrics=metrics,
            save_path=os.path.join(config["output_dir"], "training_metrics.png"),
            show=False
        )
        
        print(f"Training metrics plot saved to {os.path.join(config['output_dir'], 'training_metrics.png')}")


if __name__ == "__main__":
    main()
