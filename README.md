# Wildfire Detection Using Deep Learning

## Overview

This project implements a deep learning model for detecting wildfires from images using convolutional neural networks (CNNs). The model classifies images into two categories: "wildfire" and "nowildfire" with high accuracy, making it suitable for early wildfire detection systems.

## Features

### Model Architecture
- **Deep CNN Structure**: 3 convolutional blocks with increasing filter sizes (32, 64, 128)
- **Regularization Techniques**: L2 regularization (weight decay: 1e-4) to prevent overfitting
- **Normalization**: Batch normalization after each convolutional layer
- **Dropout Layers**: Multiple dropout layers (0.25-0.5 rates) for regularization
- **Dense Layers**: Two fully connected layers (512 and 256 units) before final classification

### Data Preprocessing & Augmentation
- **Image Resizing**: Standardized to 224×224 pixels
- **Data Normalization**: Pixel values scaled to [0, 1] range
- **Augmentation Techniques**:
  - Rotation (20° range)
  - Width and height shifting (20% range)
  - Horizontal flipping
  - Zooming (20% range)
  - Shearing (20% range)
  - Nearest neighbor filling

### Training Features
- **Class Weight Balancing**: Automatic computation to handle imbalanced datasets
- **Early Stopping**: Patience of 15 epochs to prevent overfitting
- **Model Checkpointing**: Saves best model based on validation accuracy
- **Adam Optimizer**: Learning rate of 0.0001
- **Comprehensive Metrics**: Accuracy, AUC, Precision, and Recall monitoring

### Evaluation Metrics
- **Test Loss**: Categorical cross-entropy
- **Test Accuracy**: Overall classification accuracy
- **AUC Score**: Area Under ROC Curve
- **Precision & Recall**: For both wildfire and non-wildfire classes

## Model Performance

The model achieves:
- High classification accuracy on test data
- Excellent AUC scores indicating strong discriminatory power
- Balanced precision and recall across both classes
- Robust performance through extensive regularization

## Technical Specifications

- **Framework**: TensorFlow/Keras
- **Input Size**: 224×224×3 (RGB images)
- **Output**: 2-class softmax classification
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Regularization**: L2 weight decay + dropout
- **Optimizer**: Adam with custom learning rate

## Usage

### Training
```python
# The model automatically handles:
# - Data loading from train/validation/test directories
# - Class weight computation
# - Augmentation during training
# - Best model saving
```

### Prediction
```python
from model_utils import predict_wildfire

result, confidence = predict_wildfire('image_path.jpg', model)
print(f"Prediction: {result} (Confidence: {confidence:.2f})")
```

## File Structure
```
wildfireprediction2.ipynb - Main training and evaluation notebook
wildfire_model_best.h5 - Best model weights (saved during training)
wildfire_detection_model_final.h5 - Final trained model
dataset/ - Contains train, validation, and test subdirectories
```

## Applications

- Early wildfire detection systems
- Satellite image analysis
- Drone-based monitoring
- Forest management and conservation
- Emergency response systems

## Requirements

- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- PIL/Pillow

This model provides a robust foundation for wildfire detection applications with careful attention to regularization, class balancing, and comprehensive performance evaluation.
