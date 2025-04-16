"""
Utility functions for visualization and plotting.

This module contains functions for visualizing model predictions,
training metrics, and other useful visualizations for object detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any


def plot_training_metrics(
    metrics: Dict[str, Dict[str, List[float]]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> None:
    """
    Plot training and validation metrics.
    
    Args:
        metrics: Dictionary containing training and validation metrics
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot training and validation loss
    axs[0].plot(metrics['train']['loss'], label='Train Loss')
    if 'val' in metrics and 'loss' in metrics['val']:
        axs[0].plot(metrics['val']['loss'], label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot learning rate
    if 'learning_rate' in metrics['train']:
        ax_lr = axs[0].twinx()
        ax_lr.plot(metrics['train']['learning_rate'], 'r--', label='Learning Rate')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.legend(loc='upper right')
    
    # Plot validation mAP
    if 'val' in metrics and 'mAP' in metrics['val']:
        axs[1].plot(metrics['val']['mAP'], label='Validation mAP')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('mAP')
        axs[1].set_title('Validation mAP')
        axs[1].legend()
        axs[1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path)
    
    # Show plot if requested
    if show:
        plt.show()
    else:
        plt.close()


def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: Dict[int, str] = None,
    score_threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
) -> np.ndarray:
    """
    Visualize object detection predictions on an image.
    
    Args:
        image: Input image (RGB)
        boxes: Bounding boxes in format [x1, y1, x2, y2]
        labels: Class labels
        scores: Confidence scores
        class_names: Dictionary mapping class IDs to class names
        score_threshold: Threshold for filtering predictions by score
        save_path: Path to save the visualization
        show: Whether to display the visualization
        
    Returns:
        Visualization image
    """
    # Make a copy of the image
    vis_image = image.copy()
    
    # Filter by score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Define colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    
    # Draw boxes and labels
    for box, label, score in zip(boxes, labels, scores):
        # Convert box coordinates to integers
        x1, y1, x2, y2 = box.astype(int)
        
        # Get class name
        if class_names and label in class_names:
            class_name = class_names[label]
        else:
            class_name = f"Class {label}"
        
        # Get color for this class
        color = colors[label % len(colors)]
        color = [int(c * 255) for c in color[:3]]
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{class_name}: {score:.2f}"
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(vis_image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save visualization if path is provided
    if save_path:
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    # Show visualization if requested
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.show()
    
    return vis_image


def visualize_batch(
    images: np.ndarray,
    predictions: List[Dict[str, np.ndarray]],
    class_names: Dict[int, str] = None,
    max_images: int = 4,
    score_threshold: float = 0.5,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = True
) -> List[np.ndarray]:
    """
    Visualize object detection predictions for a batch of images.
    
    Args:
        images: Batch of images (RGB)
        predictions: List of prediction dictionaries
        class_names: Dictionary mapping class IDs to class names
        max_images: Maximum number of images to visualize
        score_threshold: Threshold for filtering predictions by score
        save_dir: Directory to save visualizations
        show: Whether to display visualizations
        
    Returns:
        List of visualization images
    """
    # Limit number of images
    num_images = min(len(images), max_images)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create figure
    if show:
        fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
        if num_images == 1:
            axs = [axs]
    
    visualizations = []
    
    # Process each image
    for i in range(num_images):
        image = images[i]
        prediction = predictions[i]
        
        # Visualize detection
        save_path = Path(save_dir) / f"detection_{i}.jpg" if save_dir else None
        vis_image = visualize_detection(
            image=image,
            boxes=prediction['boxes'],
            labels=prediction['labels'],
            scores=prediction['scores'],
            class_names=class_names,
            score_threshold=score_threshold,
            save_path=save_path,
            show=False
        )
        
        visualizations.append(vis_image)
        
        # Add to figure
        if show:
            axs[i].imshow(vis_image)
            axs[i].axis('off')
    
    # Show figure
    if show:
        plt.tight_layout()
        plt.show()
    
    return visualizations


# Example usage
if __name__ == "__main__":
    # Example training metrics
    metrics = {
        'train': {
            'loss': [2.5, 2.0, 1.5, 1.2, 1.0],
            'learning_rate': [0.001, 0.001, 0.0001, 0.0001, 0.0001]
        },
        'val': {
            'loss': [2.7, 2.1, 1.6, 1.3, 1.1],
            'mAP': [0.3, 0.4, 0.5, 0.55, 0.6]
        }
    }
    
    # Plot training metrics
    plot_training_metrics(metrics, save_path="training_metrics.png")
    
    # Example detection visualization
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    labels = np.array([1, 2])
    scores = np.array([0.9, 0.8])
    class_names = {1: "Person", 2: "Car"}
    
    visualize_detection(
        image=image,
        boxes=boxes,
        labels=labels,
        scores=scores,
        class_names=class_names,
        save_path="detection_example.jpg"
    )
