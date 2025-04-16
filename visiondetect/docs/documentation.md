# VisionDetect Documentation

## Overview

VisionDetect is a comprehensive computer vision framework for object detection and classification using state-of-the-art deep learning techniques. This documentation provides detailed information about the project's architecture, components, and usage.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Inference](#inference)
7. [Configuration](#configuration)
8. [Examples](#examples)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/visiondetect.git
cd visiondetect

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

The project follows a modular structure to ensure clean separation of concerns:

```
visiondetect/
├── config/               # Configuration files
├── data/                 # Data storage (gitignored)
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks for exploration and demos
├── src/                  # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # Model implementations
│   ├── utils/            # Utility functions
│   └── api/              # API for model serving
├── tests/                # Unit and integration tests
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── infer.py              # Inference script
├── .gitignore            # Git ignore file
├── LICENSE               # License file
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Core Components

### Data Processing

The data processing module (`src/data/preprocessing.py`) handles:

- Loading and parsing image data
- Resizing and normalization
- Data augmentation
- Dataset splitting

Key classes:
- `ImageDataset`: PyTorch dataset for object detection
- `DataProcessor`: Handles data processing operations

### Model Architecture

The model architecture module (`src/models/architecture.py`) provides:

- Implementation of object detection models
- Support for different backbones (ResNet50, MobileNetV2, EfficientNet)
- Factory classes for creating models with PyTorch and TensorFlow

Key classes:
- `ObjectDetectionModel`: Factory for PyTorch models
- `TensorFlowObjectDetectionModel`: Factory for TensorFlow models

### Training

The training module (`src/models/trainer.py`) includes:

- Training loops with validation
- Optimization and learning rate scheduling
- Model checkpointing
- Training metrics tracking

Key classes:
- `ModelTrainer`: Handles the training process

### Prediction

The prediction module (`src/models/predictor.py`) provides:

- Loading trained models
- Preprocessing input images
- Running inference
- Postprocessing predictions

Key classes:
- `Predictor`: Makes predictions with trained models

### Utilities

Utility modules include:

- `src/utils/metrics.py`: Evaluation metrics (mAP, precision, recall)
- `src/utils/visualization.py`: Visualization tools for predictions and metrics

## Training Pipeline

The training pipeline is implemented in `train.py`, which provides a command-line interface for training object detection models.

### Basic Usage

```bash
python train.py --data-dir data --model-type faster_rcnn --backbone resnet50 --epochs 50
```

### Advanced Usage

```bash
python train.py --config config/default_config.yaml --data-dir custom_data --batch-size 16 --lr 0.0005
```

### Command-Line Arguments

- `--data-dir`: Directory containing the dataset
- `--img-size`: Image size (height, width)
- `--batch-size`: Batch size for training
- `--num-workers`: Number of workers for data loading
- `--download`: Download dataset if not available
- `--model-type`: Type of object detection model
- `--backbone`: Backbone network for the model
- `--num-classes`: Number of classes (including background)
- `--epochs`: Number of training epochs
- `--lr`: Initial learning rate
- `--weight-decay`: Weight decay for optimizer
- `--momentum`: Momentum for SGD optimizer
- `--checkpoint-dir`: Directory to save model checkpoints
- `--resume`: Path to checkpoint to resume training from
- `--config`: Path to configuration file
- `--device`: Device to use for training (cuda, cpu)
- `--seed`: Random seed for reproducibility

## Evaluation

The evaluation pipeline is implemented in `evaluate.py`, which provides a command-line interface for evaluating trained models.

### Basic Usage

```bash
python evaluate.py --model-path checkpoints/best_model.pth --data-dir data
```

### Advanced Usage

```bash
python evaluate.py --model-path checkpoints/best_model.pth --data-dir data --visualize --max-visualizations 20
```

### Command-Line Arguments

- `--data-dir`: Directory containing the dataset
- `--img-size`: Image size (height, width)
- `--batch-size`: Batch size for evaluation
- `--num-workers`: Number of workers for data loading
- `--model-path`: Path to trained model checkpoint
- `--confidence-threshold`: Confidence threshold for filtering predictions
- `--output-dir`: Directory to save evaluation results
- `--visualize`: Generate visualizations of predictions
- `--max-visualizations`: Maximum number of visualizations to generate
- `--device`: Device to use for evaluation (cuda, cpu)
- `--config`: Path to configuration file

## Inference

The inference pipeline is implemented in `infer.py`, which provides a command-line interface for running inference on images or videos.

### Basic Usage

```bash
python infer.py --model-path checkpoints/best_model.pth --input path/to/image.jpg
```

### Advanced Usage

```bash
python infer.py --model-path checkpoints/best_model.pth --input path/to/video.mp4 --video --save-video --show
```

### Command-Line Arguments

- `--input`: Path to input image, directory of images, or video file
- `--output-dir`: Directory to save results
- `--model-path`: Path to trained model checkpoint
- `--confidence-threshold`: Confidence threshold for filtering predictions
- `--device`: Device to use for inference (cuda, cpu)
- `--config`: Path to configuration file
- `--video`: Process input as video
- `--save-video`: Save processed video
- `--show`: Show results in window

## Configuration

VisionDetect uses YAML configuration files for flexible configuration. The default configuration file is located at `config/default_config.yaml`.

### Example Configuration

```yaml
data:
  train_config:
    data_dir: "data"
    img_size: [640, 640]
    batch_size: 8
    num_workers: 4
    augment: true
    download: false

model:
  model_type: "faster_rcnn"
  backbone: "resnet50"
  num_classes: 91
  pretrained: true

training:
  epochs: 50
  lr: 0.001
  weight_decay: 0.0005
  momentum: 0.9
  checkpoint_dir: "checkpoints"
  
evaluation:
  confidence_threshold: 0.5
  output_dir: "evaluation"
  visualize: true
  max_visualizations: 10
  
inference:
  confidence_threshold: 0.5
  output_dir: "results"
  show: false
```

## Examples

### Training a Model

```python
from src.data.preprocessing import DataProcessor
from src.models.trainer import ModelTrainer

# Create data processor
processor = DataProcessor(data_dir="data")
dataloaders = processor.prepare_dataset(download=True)

# Create trainer
trainer = ModelTrainer(
    model_type="faster_rcnn",
    backbone_name="resnet50",
    num_classes=91,
    learning_rate=0.001,
    batch_size=8,
    num_epochs=50
)

# Train model
trainer.train(
    train_dataloader=dataloaders['train'],
    val_dataloader=dataloaders['val']
)
```

### Making Predictions

```python
from src.models.predictor import Predictor
import cv2

# Create predictor
predictor = Predictor(
    model_path="checkpoints/best_model.pth",
    confidence_threshold=0.5
)

# Make prediction on an image
result = predictor.predict("path/to/image.jpg", return_visualization=True)

# Display result
cv2.imshow("Detection", cv2.cvtColor(result["visualization"], cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```

## API Reference

### Data Module

#### `ImageDataset`

```python
ImageDataset(image_paths, labels, transform=None, class_map=None)
```

PyTorch dataset for loading and preprocessing images for object detection.

#### `DataProcessor`

```python
DataProcessor(data_dir, img_size=(640, 640), batch_size=16, num_workers=4, augment=True)
```

Handles data processing operations for object detection.

### Models Module

#### `ObjectDetectionModel`

```python
ObjectDetectionModel.create_model(model_type, num_classes, backbone_name="resnet50", pretrained=True, **kwargs)
```

Factory class for creating object detection models.

#### `ModelTrainer`

```python
ModelTrainer(model_type="faster_rcnn", backbone_name="resnet50", num_classes=91, learning_rate=0.001, weight_decay=0.0005, momentum=0.9, batch_size=8, num_epochs=50, checkpoint_dir="checkpoints", device=None, config_path=None)
```

Trainer class for object detection models.

#### `Predictor`

```python
Predictor(model_path, device=None, confidence_threshold=0.5)
```

Class for making predictions with trained object detection models.

### Utils Module

#### `calculate_map`

```python
calculate_map(predictions, targets, iou_threshold=0.5)
```

Calculate mean Average Precision (mAP) for object detection predictions.

#### `visualize_detection`

```python
visualize_detection(image, boxes, labels, scores, class_names=None, score_threshold=0.5, save_path=None, show=True)
```

Visualize object detection predictions on an image.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
