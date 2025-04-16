"""
Main module for VisionDetect.

This module provides a simple interface to the VisionDetect framework.
"""

import os
from pathlib import Path

from src.data.preprocessing import DataProcessor
from src.models.architecture import ObjectDetectionModel
from src.models.trainer import ModelTrainer
from src.models.predictor import Predictor
from src.utils.logging import logger


class VisionDetect:
    """
    Main class for the VisionDetect framework.
    
    This class provides a simplified interface to the VisionDetect framework,
    allowing users to easily train models and make predictions.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the VisionDetect framework.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        logger.info("Initializing VisionDetect framework")
        
        # Create default directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    def train(self, data_dir="data", model_type="faster_rcnn", backbone="resnet50", 
              num_classes=91, epochs=50, batch_size=8, learning_rate=0.001):
        """
        Train an object detection model.
        
        Args:
            data_dir: Directory containing the dataset
            model_type: Type of model ('faster_rcnn', etc.)
            backbone: Backbone network ('resnet50', 'mobilenet_v2', etc.)
            num_classes: Number of classes (including background)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            
        Returns:
            Trained model and training metrics
        """
        logger.info(f"Training {model_type} model with {backbone} backbone")
        
        # Create data processor
        processor = DataProcessor(
            data_dir=data_dir,
            batch_size=batch_size,
            augment=True
        )
        
        # Prepare dataset
        dataloaders = processor.prepare_dataset()
        
        # Create trainer
        trainer = ModelTrainer(
            model_type=model_type,
            backbone_name=backbone,
            num_classes=num_classes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=epochs
        )
        
        # Train model
        if 'train' in dataloaders:
            metrics = trainer.train(
                train_dataloader=dataloaders['train'],
                val_dataloader=dataloaders.get('val')
            )
            
            logger.info("Training completed")
            return trainer, metrics
        else:
            logger.error("No training data found")
            return None, None
    
    def evaluate(self, model_path, data_dir="data", batch_size=8, visualize=False):
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing the dataset
            batch_size: Batch size for evaluation
            visualize: Whether to generate visualizations
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model from {model_path}")
        
        # Create data processor
        processor = DataProcessor(
            data_dir=data_dir,
            batch_size=batch_size,
            augment=False
        )
        
        # Prepare dataset
        dataloaders = processor.prepare_dataset()
        
        # Create predictor
        predictor = Predictor(
            model_path=model_path,
            confidence_threshold=0.5
        )
        
        # Evaluate model
        if 'test' in dataloaders:
            from src.utils.metrics import calculate_map, calculate_map_range
            
            # Initialize metrics
            all_predictions = []
            all_targets = []
            
            # Process each batch
            for batch in dataloaders['test']:
                # Get images and targets
                images = batch["images"].cpu().numpy()
                targets = batch["targets"]
                
                # Make predictions
                batch_predictions = []
                for i in range(len(images)):
                    # Convert image from CHW to HWC format
                    import numpy as np
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
            
            # Calculate metrics
            map_50 = calculate_map(all_predictions, all_targets, iou_threshold=0.5)
            map_range = calculate_map_range(all_predictions, all_targets)
            
            metrics = {
                "mAP@0.5": float(map_50),
                "mAP@[0.5:0.95]": float(map_range),
                "num_samples": len(all_targets)
            }
            
            logger.info(f"Evaluation results: mAP@0.5={map_50:.4f}, mAP@[0.5:0.95]={map_range:.4f}")
            return metrics
        else:
            logger.error("No test data found")
            return None
    
    def predict(self, image_path, model_path, confidence_threshold=0.5, visualize=True):
        """
        Make a prediction on an image.
        
        Args:
            image_path: Path to image
            model_path: Path to trained model checkpoint
            confidence_threshold: Confidence threshold for filtering predictions
            visualize: Whether to return visualization
            
        Returns:
            Prediction results
        """
        logger.info(f"Making prediction on {image_path}")
        
        # Create predictor
        predictor = Predictor(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        # Make prediction
        result = predictor.predict(image_path, return_visualization=visualize)
        
        return result


# Example usage
if __name__ == "__main__":
    # Create VisionDetect instance
    vd = VisionDetect()
    
    # Train model
    trainer, metrics = vd.train(
        data_dir="data",
        model_type="faster_rcnn",
        backbone="resnet50",
        num_classes=91,
        epochs=50,
        batch_size=8,
        learning_rate=0.001
    )
    
    # Evaluate model
    eval_metrics = vd.evaluate(
        model_path="checkpoints/best_model.pth",
        data_dir="data",
        batch_size=8,
        visualize=True
    )
    
    # Make prediction
    result = vd.predict(
        image_path="data/samples/image.jpg",
        model_path="checkpoints/best_model.pth",
        confidence_threshold=0.5,
        visualize=True
    )
