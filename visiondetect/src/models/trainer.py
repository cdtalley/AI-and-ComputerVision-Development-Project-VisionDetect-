"""
Model training module for VisionDetect.

This module contains the training logic for object detection models,
including training loops, optimization, and model checkpointing.
"""

import os
import time
import datetime
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.models.architecture import ObjectDetectionModel
from src.utils.metrics import calculate_map


class ModelTrainer:
    """
    Trainer class for object detection models.
    
    This class handles the training process, including:
    - Setting up optimizers and learning rate schedulers
    - Training loops
    - Validation
    - Model checkpointing
    - Training metrics tracking
    """
    
    def __init__(
        self,
        model_type: str = "faster_rcnn",
        backbone_name: str = "resnet50",
        num_classes: int = 91,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0005,
        momentum: float = 0.9,
        batch_size: int = 8,
        num_epochs: int = 50,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        device: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model ('faster_rcnn', 'retinanet', etc.)
            backbone_name: Name of the backbone network
            num_classes: Number of classes (including background)
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            momentum: Momentum for SGD optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save model checkpoints
            device: Device to use for training ('cuda', 'cpu')
            config_path: Path to configuration file
        """
        # Load config if provided
        if config_path:
            self._load_config(config_path)
        else:
            self.model_type = model_type
            self.backbone_name = backbone_name
            self.num_classes = num_classes
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.momentum = momentum
            self.batch_size = batch_size
            self.num_epochs = num_epochs
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize training metrics
        self.train_metrics = {
            'loss': [],
            'learning_rate': []
        }
        
        self.val_metrics = {
            'loss': [],
            'mAP': []
        }
        
        # Best validation metrics
        self.best_map = 0.0
        self.best_epoch = 0
    
    def _load_config(self, config_path: Union[str, Path]):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set attributes from config
        self.model_type = config.get('model_type', 'faster_rcnn')
        self.backbone_name = config.get('backbone_name', 'resnet50')
        self.num_classes = config.get('num_classes', 91)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0005)
        self.momentum = config.get('momentum', 0.9)
        self.batch_size = config.get('batch_size', 8)
        self.num_epochs = config.get('num_epochs', 50)
    
    def _create_model(self) -> nn.Module:
        """
        Create the model based on configuration.
        
        Returns:
            Configured model
        """
        return ObjectDetectionModel.create_model(
            model_type=self.model_type,
            num_classes=self.num_classes,
            backbone_name=self.backbone_name,
            pretrained=True
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        Create optimizer for training.
        
        Returns:
            Configured optimizer
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler.
        
        Returns:
            Configured learning rate scheduler
        """
        return optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1
        )
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move data to device
            images = batch["images"].to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in batch["targets"]]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            losses.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += losses.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": losses.item()})
        
        # Calculate average loss
        avg_loss = epoch_loss / len(dataloader)
        
        return {"loss": avg_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation")
            for batch in progress_bar:
                # Move data to device
                images = batch["images"].to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in batch["targets"]]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                
                # Calculate total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Update metrics
                val_loss += losses.item()
                
                # Get predictions
                predictions = self.model(images)
                
                # Store predictions and targets for mAP calculation
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # Update progress bar
                progress_bar.set_postfix({"loss": losses.item()})
        
        # Calculate average loss
        avg_loss = val_loss / len(dataloader)
        
        # Calculate mAP
        map_value = calculate_map(all_predictions, all_targets)
        
        return {
            "loss": avg_loss,
            "mAP": map_value
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Dictionary of training and validation metrics
        """
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model_type} with {self.backbone_name} backbone")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Store training metrics
            self.train_metrics['loss'].append(train_metrics['loss'])
            self.train_metrics['learning_rate'].append(current_lr)
            
            # Validate if validation dataloader is provided
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
                
                # Store validation metrics
                self.val_metrics['loss'].append(val_metrics['loss'])
                self.val_metrics['mAP'].append(val_metrics['mAP'])
                
                # Print metrics
                print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val mAP: {val_metrics['mAP']:.4f}, LR: {current_lr:.6f}")
                
                # Save best model
                if val_metrics['mAP'] > self.best_map:
                    self.best_map = val_metrics['mAP']
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(f"best_model.pth")
                    print(f"New best model saved with mAP: {self.best_map:.4f}")
            else:
                # Print metrics
                print(f"Train Loss: {train_metrics['loss']:.4f}, LR: {current_lr:.6f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"model_epoch_{epoch+1}.pth")
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        
        # Calculate training time
        total_time = time.time() - start_time
        print(f"\nTraining completed in {datetime.timedelta(seconds=int(total_time))}")
        
        if val_dataloader:
            print(f"Best mAP: {self.best_map:.4f} at epoch {self.best_epoch}")
        
        # Save training metrics
        self.save_metrics()
        
        return {
            'train': self.train_metrics,
            'val': self.val_metrics
        }
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_type': self.model_type,
            'backbone_name': self.backbone_name,
            'num_classes': self.num_classes,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_map': self.best_map,
            'best_epoch': self.best_epoch
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model parameters
        self.model_type = checkpoint.get('model_type', self.model_type)
        self.backbone_name = checkpoint.get('backbone_name', self.backbone_name)
        self.num_classes = checkpoint.get('num_classes', self.num_classes)
        
        # Recreate model if needed
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
        
        # Load optimizer and scheduler states
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load metrics
        self.train_metrics = checkpoint.get('train_metrics', self.train_metrics)
        self.val_metrics = checkpoint.get('val_metrics', self.val_metrics)
        self.best_map = checkpoint.get('best_map', self.best_map)
        self.best_epoch = checkpoint.get('best_epoch', self.best_epoch)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_metrics(self):
        """Save training and validation metrics to JSON file."""
        metrics_path = self.checkpoint_dir / "training_metrics.json"
        
        metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'best_map': self.best_map,
            'best_epoch': self.best_epoch
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)


# Example usage
if __name__ == "__main__":
    from src.data.preprocessing import DataProcessor
    
    # Create data processor
    processor = DataProcessor(data_dir="data")
    dataloaders = processor.prepare_dataset(download=True)
    
    # Create trainer
    trainer = ModelTrainer(
        model_type="faster_rcnn",
        backbone_name="resnet50",
        num_classes=91,
        learning_rate=0.001,
        batch_size=2,
        num_epochs=20
    )
    
    # Train model
    if 'train' in dataloaders and 'val' in dataloaders:
        trainer.train(
            train_dataloader=dataloaders['train'],
            val_dataloader=dataloaders['val']
        )
    elif 'train' in dataloaders:
        trainer.train(
            train_dataloader=dataloaders['train']
        )
