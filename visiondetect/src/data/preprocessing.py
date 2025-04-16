"""
Data preprocessing module for VisionDetect.

This module handles image preprocessing tasks including:
- Loading and parsing image data
- Resizing and normalization
- Data augmentation
- Dataset splitting
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    """
    Dataset class for loading and preprocessing images for object detection.
    
    Attributes:
        image_paths (List[str]): List of paths to images
        labels (List[Dict]): List of label dictionaries containing bounding boxes and classes
        transform (A.Compose): Albumentations transformations to apply
        class_map (Dict[int, str]): Mapping from class IDs to class names
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[Dict],
        transform: Optional[A.Compose] = None,
        class_map: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of label dictionaries containing bounding boxes and classes
            transform: Albumentations transformations to apply
            class_map: Mapping from class IDs to class names
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_map = class_map or {}
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dict containing image tensor and target information
        """
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get labels
        label = self.labels[idx]
        boxes = label.get("boxes", [])
        class_ids = label.get("class_ids", [])
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_ids=class_ids
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            class_ids = transformed["class_ids"]
        
        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        # Prepare target
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(class_ids, dtype=torch.int64) if class_ids else torch.zeros(0, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32) if boxes else torch.zeros(0, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64) if boxes else torch.zeros(0, dtype=torch.int64)
        }
        
        return {
            "image": image,
            "target": target,
            "image_path": image_path
        }


class DataProcessor:
    """
    Handles data processing operations for object detection.
    
    This class provides methods for:
    - Loading datasets from various formats
    - Preprocessing images
    - Creating data splits (train/val/test)
    - Setting up data augmentation pipelines
    - Creating PyTorch DataLoaders
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        img_size: Tuple[int, int] = (640, 640),
        batch_size: int = 16,
        num_workers: int = 4,
        augment: bool = True
    ):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the dataset
            img_size: Target image size (height, width)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.class_map = {}
        
    def get_transforms(self, train: bool = False) -> A.Compose:
        """
        Get image transformations.
        
        Args:
            train: Whether to include training augmentations
            
        Returns:
            Albumentations composition of transforms
        """
        if train and self.augment:
            return A.Compose(
                [
                    A.RandomResizedCrop(height=self.img_size[0], width=self.img_size[1], scale=(0.8, 1.0)),
                    A.RandomRotate90(),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'])
            )
        else:
            return A.Compose(
                [
                    A.Resize(height=self.img_size[0], width=self.img_size[1]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'])
            )
    
    def load_dataset(self, split: str = 'train') -> Tuple[List[str], List[Dict]]:
        """
        Load dataset from disk.
        
        This is a placeholder method that should be implemented based on the specific
        dataset format you're working with (COCO, Pascal VOC, custom format, etc.)
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            
        Returns:
            Tuple of (image_paths, labels)
        """
        # Placeholder implementation - replace with actual dataset loading logic
        split_dir = self.data_dir / split
        image_paths = list(split_dir.glob('images/*.jpg'))
        
        # Placeholder for labels - in a real implementation, you would load these from annotation files
        labels = [{"boxes": [], "class_ids": []} for _ in image_paths]
        
        return image_paths, labels
    
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for training, validation, and testing.
        
        Returns:
            Dictionary of DataLoaders for each split
        """
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            try:
                image_paths, labels = self.load_dataset(split)
                
                if not image_paths:
                    continue
                    
                dataset = ImageDataset(
                    image_paths=image_paths,
                    labels=labels,
                    transform=self.get_transforms(train=(split == 'train')),
                    class_map=self.class_map
                )
                
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=self.num_workers,
                    collate_fn=self._collate_fn
                )
            except Exception as e:
                print(f"Error creating {split} dataloader: {e}")
        
        return dataloaders
    
    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Custom collate function for DataLoader.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Batched samples
        """
        images = [item["image"] for item in batch]
        targets = [item["target"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        
        return {
            "images": torch.stack(images),
            "targets": targets,
            "image_paths": image_paths
        }
    
    def prepare_dataset(self, download: bool = False) -> Dict[str, DataLoader]:
        """
        Prepare the dataset for training.
        
        Args:
            download: Whether to download the dataset if not available locally
            
        Returns:
            Dictionary of DataLoaders
        """
        # Create directories if they don't exist
        os.makedirs(self.data_dir / 'train' / 'images', exist_ok=True)
        os.makedirs(self.data_dir / 'val' / 'images', exist_ok=True)
        os.makedirs(self.data_dir / 'test' / 'images', exist_ok=True)
        
        # Download dataset if requested and not available
        if download and not list(self.data_dir.glob('**/*.jpg')):
            self._download_dataset()
        
        # Load class map
        self._load_class_map()
        
        # Create and return dataloaders
        return self.create_dataloaders()
    
    def _download_dataset(self):
        """
        Download dataset from a remote source.
        
        This is a placeholder method that should be implemented based on the specific
        dataset you want to use.
        """
        # Placeholder - implement dataset download logic
        pass
    
    def _load_class_map(self):
        """
        Load class map from dataset.
        
        This is a placeholder method that should be implemented based on the specific
        dataset you're working with.
        """
        # Placeholder - implement class map loading logic
        # Example:
        self.class_map = {
            0: "background",
            1: "person",
            2: "car",
            3: "bicycle",
            # Add more classes as needed
        }


# Example usage
if __name__ == "__main__":
    processor = DataProcessor(data_dir="data")
    dataloaders = processor.prepare_dataset(download=True)
    
    # Print dataset information
    for split, dataloader in dataloaders.items():
        print(f"{split} dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
