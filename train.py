"""
Main training script for VisionDetect.

This script provides a command-line interface for training object detection models
with various configurations and datasets.
"""

import os
import argparse
import yaml
import torch
from pathlib import Path

from src.data.preprocessing import DataProcessor
from src.models.trainer import ModelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train object detection model")
    
    # Dataset arguments
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing the dataset")
    parser.add_argument("--img-size", type=int, nargs=2, default=[640, 640],
                        help="Image size (height, width)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset if not available")
    
    # Model arguments
    parser.add_argument("--model-type", type=str, default="faster_rcnn",
                        choices=["faster_rcnn"],
                        help="Type of object detection model")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2", "efficientnet_b0"],
                        help="Backbone network for the model")
    parser.add_argument("--num-classes", type=int, default=91,
                        help="Number of classes (including background)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum for SGD optimizer")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration file")
    
    # Other arguments
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
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
    
    # Set random seed
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    
    # Create checkpoint directory
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # Create data processor
    data_processor = DataProcessor(
        data_dir=config["data_dir"],
        img_size=tuple(config["img_size"]),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        augment=True
    )
    
    # Prepare dataset
    dataloaders = data_processor.prepare_dataset(download=config.get("download", False))
    
    # Create trainer
    trainer = ModelTrainer(
        model_type=config["model_type"],
        backbone_name=config["backbone"],
        num_classes=config["num_classes"],
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"],
        batch_size=config["batch_size"],
        num_epochs=config["epochs"],
        checkpoint_dir=config["checkpoint_dir"],
        device=config["device"]
    )
    
    # Resume training if checkpoint is provided
    if config.get("resume"):
        trainer.load_checkpoint(config["resume"])
    
    # Train model
    if "train" in dataloaders:
        trainer.train(
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders.get("val")
        )
    else:
        print("No training data found. Please check your data directory.")


if __name__ == "__main__":
    main()
