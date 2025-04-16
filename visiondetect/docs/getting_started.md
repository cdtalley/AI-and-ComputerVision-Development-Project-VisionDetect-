# Getting Started with VisionDetect

This guide will help you get started with the VisionDetect framework for object detection using deep learning.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/visiondetect.git
cd visiondetect

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

### Training a Model

```python
from src import VisionDetect

# Create VisionDetect instance
vd = VisionDetect()

# Train model
trainer, metrics = vd.train(
    data_dir="path/to/data",
    model_type="faster_rcnn",
    backbone="resnet50",
    num_classes=91,  # COCO dataset has 90 classes + background
    epochs=50,
    batch_size=8,
    learning_rate=0.001
)
```

### Evaluating a Model

```python
# Evaluate model
eval_metrics = vd.evaluate(
    model_path="checkpoints/best_model.pth",
    data_dir="path/to/data",
    batch_size=8,
    visualize=True
)

print(f"mAP@0.5: {eval_metrics['mAP@0.5']:.4f}")
print(f"mAP@[0.5:0.95]: {eval_metrics['mAP@[0.5:0.95]']:.4f}")
```

### Making Predictions

```python
# Make prediction on a single image
result = vd.predict(
    image_path="path/to/image.jpg",
    model_path="checkpoints/best_model.pth",
    confidence_threshold=0.5,
    visualize=True
)

# Display results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(result["visualization"])
plt.axis('off')
plt.show()

# Print detection results
for i, (box, score, class_name) in enumerate(zip(result['boxes'], result['scores'], result['class_names'])):
    print(f"Object {i+1}: {class_name}, Score: {score:.2f}, Box: {box}")
```

## Using the Command-Line Interface

VisionDetect provides command-line scripts for training, evaluation, and inference.

### Training

```bash
python train.py --data-dir data --model-type faster_rcnn --backbone resnet50 --epochs 50
```

### Evaluation

```bash
python evaluate.py --model-path checkpoints/best_model.pth --data-dir data --visualize
```

### Inference

```bash
python infer.py --model-path checkpoints/best_model.pth --input path/to/image.jpg
```

## Next Steps

- Check out the [documentation](docs/documentation.md) for detailed information about the framework
- Explore the [tutorial notebook](notebooks/object_detection_tutorial.ipynb) for a step-by-step guide
- Read the [code style guide](docs/code_style_guide.md) if you want to contribute to the project

## Getting Help

If you encounter any issues or have questions, please:

1. Check the [documentation](docs/documentation.md)
2. Look for similar issues in the GitHub repository
3. Open a new issue if needed

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
