# MNIST Dataset Loading Examples
# This script demonstrates how to load the MNIST dataset using both TensorFlow/Keras and PyTorch
# and displays sample images from both implementations side by side for comparison.

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from tensorflow.keras.datasets import mnist

# Set up the figure for displaying images
plt.figure(figsize=(15, 8))

# ===== TensorFlow/Keras MNIST Loading =====
print("Loading MNIST dataset using TensorFlow/Keras...")

# Load the MNIST dataset using Keras
(X_train_keras, y_train_keras), (_, _) = mnist.load_data()

# Display 4 images from Keras dataset in the top row
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_train_keras[i], cmap='gray')
    plt.title(f"Keras - Label: {y_train_keras[i]}")
    plt.axis('off')

# ===== PyTorch MNIST Loading =====
print("\nLoading MNIST dataset using PyTorch...")

# Define the transformation to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
mnist_pytorch = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader to load the dataset in batches
train_loader_pytorch = torch.utils.data.DataLoader(mnist_pytorch, batch_size=1, shuffle=False)

# Display 4 images from PyTorch dataset in the bottom row
for i, (image, label) in enumerate(train_loader_pytorch):
    if i < 4:  # Print the first 4 samples
        plt.subplot(2, 4, i+5)
        plt.imshow(image[0].squeeze(), cmap='gray')
        plt.title(f"PyTorch - Label: {label.item()}")
        plt.axis('off')
    else:
        break  # Exit the loop after printing 4 samples

plt.tight_layout()
plt.suptitle("MNIST Dataset Comparison: Keras vs PyTorch", fontsize=16)
plt.subplots_adjust(top=0.9)

# ===== Significance of MNIST in Machine Learning =====
print("\nSignificance of MNIST in Machine Learning:")
print("1. Benchmarking: MNIST provides a straightforward dataset to test and benchmark")
print("   machine learning models, particularly in image recognition algorithms.")
print("2. Learning Tool: Due to its simplicity and small size, MNIST is an excellent")
print("   dataset for beginners to learn the basics of machine learning and pattern recognition.")
print("3. Research: It continues to be a reference dataset for evaluating new machine")
print("   learning techniques.")
print("4. Dataset Properties:")
print("   - Training set: 60,000 images")
print("   - Test set: 10,000 images")
print("   - Image dimensions: 28x28 pixels (grayscale)")
print("   - 10 classes (digits 0-9)")

# Save the figure
plt.savefig('mnist_comparison.png')

# Show the plot
plt.show()

print("\nComparison completed and visualization displayed.")