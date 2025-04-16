"""
Model inference and prediction module for VisionDetect.

This module contains functionality for making predictions with trained models,
including loading models, preprocessing inputs, and postprocessing outputs.
"""

import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

from src.models.architecture import ObjectDetectionModel


class Predictor:
    """
    Class for making predictions with trained object detection models.
    
    This class handles:
    - Loading trained models
    - Preprocessing input images
    - Running inference
    - Postprocessing predictions
    - Visualizing results
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use for inference ('cuda', 'cpu')
            confidence_threshold: Threshold for filtering predictions
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model, self.model_info = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Get class names
        self.class_names = self.model_info.get('class_names', {})
    
    def _load_model(self) -> Tuple[torch.nn.Module, Dict]:
        """
        Load model from checkpoint.
        
        Returns:
            Tuple of (model, model_info)
        """
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model parameters
        model_type = checkpoint.get('model_type', 'faster_rcnn')
        backbone_name = checkpoint.get('backbone_name', 'resnet50')
        num_classes = checkpoint.get('num_classes', 91)
        
        # Create model
        model = ObjectDetectionModel.create_model(
            model_type=model_type,
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract model info
        model_info = {
            'model_type': model_type,
            'backbone_name': backbone_name,
            'num_classes': num_classes,
            'class_names': checkpoint.get('class_names', {})
        }
        
        return model, model_info
    
    def preprocess_image(self, image_path: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to image or image array
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
            
            # Convert BGR to RGB if needed
            if image.shape[2] == 3 and not isinstance(image_path, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image
    
    def predict(
        self,
        image_path: Union[str, Path, np.ndarray],
        return_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Make a prediction on an image.
        
        Args:
            image_path: Path to image or image array
            return_visualization: Whether to return a visualization of the prediction
            
        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        if isinstance(image_path, (str, Path)):
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_path.copy()
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        result = self._process_predictions(predictions[0], original_image)
        
        # Create visualization if requested
        if return_visualization:
            visualization = self._visualize_prediction(original_image, result)
            result['visualization'] = visualization
        
        return result
    
    def _process_predictions(self, prediction: Dict[str, torch.Tensor], image: np.ndarray) -> Dict[str, Any]:
        """
        Process raw model predictions.
        
        Args:
            prediction: Raw prediction from model
            image: Original image
            
        Returns:
            Processed prediction results
        """
        # Extract predictions
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Convert class IDs to names if available
        class_names = []
        for label in labels:
            class_name = self.class_names.get(int(label), f"Class {label}")
            class_names.append(class_name)
        
        # Prepare result
        result = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'image_shape': image.shape[:2]
        }
        
        return result
    
    def _visualize_prediction(self, image: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray:
        """
        Create a visualization of the prediction.
        
        Args:
            image: Original image
            prediction: Processed prediction
            
        Returns:
            Visualization image
        """
        # Make a copy of the image
        visualization = image.copy()
        
        # Draw boxes
        for box, score, class_name in zip(prediction['boxes'], prediction['scores'], prediction['class_names']):
            # Convert box coordinates to integers
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw box
            cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(visualization, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return visualization
    
    def batch_predict(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = Predictor(
        model_path="checkpoints/best_model.pth",
        confidence_threshold=0.5
    )
    
    # Make prediction
    result = predictor.predict(
        image_path="data/test/images/test_image.jpg",
        return_visualization=True
    )
    
    # Print results
    print(f"Found {len(result['boxes'])} objects")
    
    for i, (box, score, class_name) in enumerate(zip(result['boxes'], result['scores'], result['class_names'])):
        print(f"Object {i+1}: {class_name}, Score: {score:.2f}, Box: {box}")
    
    # Save visualization
    if 'visualization' in result:
        visualization = result['visualization']
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite("prediction_visualization.jpg", visualization)
