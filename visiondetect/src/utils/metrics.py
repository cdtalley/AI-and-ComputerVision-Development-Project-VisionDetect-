"""
Utility functions for metrics calculation and evaluation.

This module contains functions for calculating various metrics
for object detection models, including mAP, precision, and recall.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional, Any


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box in format [x1, y1, x2, y2]
        box2: Second bounding box in format [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou


def calculate_precision_recall(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, int]]:
    """
    Calculate precision and recall for object detection predictions.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Tuple of (precisions, recalls, class_counts)
    """
    # Initialize dictionaries to store precision and recall values for each class
    precisions = {}
    recalls = {}
    class_counts = {}
    
    # Process each image
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()
        
        # Sort predictions by score
        indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[indices]
        pred_labels = pred_labels[indices]
        pred_scores = pred_scores[indices]
        
        # Process each class
        unique_classes = np.unique(np.concatenate([pred_labels, target_labels]))
        
        for cls in unique_classes:
            # Initialize if class not seen before
            if cls not in precisions:
                precisions[cls] = []
                recalls[cls] = []
                class_counts[cls] = 0
            
            # Get predictions and targets for this class
            cls_pred_indices = np.where(pred_labels == cls)[0]
            cls_target_indices = np.where(target_labels == cls)[0]
            
            cls_pred_boxes = pred_boxes[cls_pred_indices]
            cls_target_boxes = target_boxes[cls_target_indices]
            
            # Update class count
            class_counts[cls] += len(cls_target_indices)
            
            # If no targets for this class, all predictions are false positives
            if len(cls_target_indices) == 0:
                if len(cls_pred_indices) > 0:
                    precisions[cls].extend([0.0] * len(cls_pred_indices))
                    recalls[cls].extend([0.0] * len(cls_pred_indices))
                continue
            
            # If no predictions for this class, recall is 0
            if len(cls_pred_indices) == 0:
                recalls[cls].append(0.0)
                continue
            
            # Track which targets have been matched
            target_matched = np.zeros(len(cls_target_boxes), dtype=bool)
            
            # Calculate precision and recall at each detection
            true_positives = 0
            false_positives = 0
            
            for pred_box in cls_pred_boxes:
                # Find best matching target
                best_iou = 0
                best_target_idx = -1
                
                for i, target_box in enumerate(cls_target_boxes):
                    if target_matched[i]:
                        continue
                    
                    iou = calculate_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = i
                
                # Check if match is good enough
                if best_target_idx >= 0 and best_iou >= iou_threshold:
                    true_positives += 1
                    target_matched[best_target_idx] = True
                else:
                    false_positives += 1
                
                # Calculate precision and recall at this point
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / len(cls_target_boxes)
                
                precisions[cls].append(precision)
                recalls[cls].append(recall)
    
    return precisions, recalls, class_counts


def calculate_average_precision(
    precisions: List[float],
    recalls: List[float]
) -> float:
    """
    Calculate Average Precision (AP) from precision-recall values.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        
    Returns:
        Average Precision value
    """
    if not precisions or not recalls:
        return 0.0
    
    # Sort by recall
    indices = np.argsort(recalls)
    recalls = np.array(recalls)[indices]
    precisions = np.array(precisions)[indices]
    
    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate area under PR curve
    ap = 0.0
    for i in range(len(recalls) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
    
    return ap


def calculate_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate mean Average Precision (mAP) for object detection predictions.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        mAP value
    """
    # Calculate precision and recall for each class
    precisions, recalls, class_counts = calculate_precision_recall(
        predictions, targets, iou_threshold
    )
    
    # Calculate AP for each class
    aps = {}
    for cls in precisions:
        if class_counts[cls] > 0:
            aps[cls] = calculate_average_precision(precisions[cls], recalls[cls])
        else:
            aps[cls] = 0.0
    
    # Calculate mAP
    if not aps:
        return 0.0
    
    return sum(aps.values()) / len(aps)


def calculate_map_range(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
) -> float:
    """
    Calculate mAP over a range of IoU thresholds (COCO-style).
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_thresholds: List of IoU thresholds
        
    Returns:
        mAP value averaged over IoU thresholds
    """
    maps = []
    
    for iou_threshold in iou_thresholds:
        map_value = calculate_map(predictions, targets, iou_threshold)
        maps.append(map_value)
    
    return sum(maps) / len(maps)


# Example usage
if __name__ == "__main__":
    # Create dummy predictions and targets
    predictions = [
        {
            'boxes': torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]]),
            'scores': torch.tensor([0.9, 0.8]),
            'labels': torch.tensor([1, 2])
        }
    ]
    
    targets = [
        {
            'boxes': torch.tensor([[11, 11, 21, 21], [31, 31, 41, 41]]),
            'labels': torch.tensor([1, 2])
        }
    ]
    
    # Calculate mAP
    map_value = calculate_map(predictions, targets)
    print(f"mAP@0.5: {map_value:.4f}")
    
    # Calculate mAP over range of IoU thresholds
    map_range = calculate_map_range(predictions, targets)
    print(f"mAP@[0.5:0.95]: {map_range:.4f}")
