# MNIST Digit Classification

## Overview
This project implements digit classification on the MNIST dataset using both Fully Connected Neural Networks (FCNN) and Convolutional Neural Networks (CNN). The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9).

## Dataset Information
- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Image dimensions**: 28x28 pixels (grayscale)

## Implementation Details

### Fully Connected Neural Network (FCNN)
- **Preprocessing**: Flattened images to 784 features (28×28) and normalized pixel values
- **Architecture**:
  - Input layer: 784 neurons
  - Hidden layer 1: 128 neurons with ReLU activation
  - Hidden layer 2: 64 neurons with ReLU activation
  - Output layer: 10 neurons with Softmax activation
- **Training**: Adam optimizer, categorical cross-entropy loss

### Convolutional Neural Network (CNN)
- **Preprocessing**: Reshaped images to (28, 28, 1) and normalized pixel values
- **Architecture**:
  - Conv2D layer: 32 filters, 3×3 kernel, ReLU activation
  - MaxPooling2D layer: 2×2 pool size
  - Conv2D layer: 64 filters, 3×3 kernel, ReLU activation
  - MaxPooling2D layer: 2×2 pool size
  - Flatten layer
  - Dense layer: 64 neurons with ReLU activation
  - Output layer: 10 neurons with Softmax activation
- **Training**: Adam optimizer, categorical cross-entropy loss

## Results
- **FCNN Test Accuracy**: 97.38%
- **CNN Test Accuracy**: 98.59%
- **Improvement with CNN**: 1.21%

## Running the Code
To run the MNIST classification:

```bash
python mnist_classifier.py
```

## Conclusion
The CNN model outperforms the FCNN model by achieving higher accuracy. This demonstrates the effectiveness of convolutional layers in capturing spatial patterns in image data, which is crucial for tasks like digit recognition.