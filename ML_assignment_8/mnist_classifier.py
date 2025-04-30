# MNIST Classification using Neural Networks and CNN
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

print("\n===== MNIST Classification using Neural Networks and CNN =====\n")

# Load MNIST dataset
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Print shapes
print('Training Data Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Data Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Display sample information
print(f"\nSample image label: {y_train[0]}")
print("(Image display not available in console mode)")

print("\n===== Fully Connected Neural Network (FCNN) =====\n")

# Preprocess data for FCNN
print("Preprocessing data for FCNN...")
X_train_fcnn = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_fcnn = X_test.reshape(X_test.shape[0], -1) / 255.0

# One-hot encode labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# Create FCNN model
print("Creating and training FCNN model...")
fcnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
fcnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train model
fcnn_history = fcnn_model.fit(X_train_fcnn, y_train_encoded,
                             batch_size=128,
                             epochs=5,  # Reduced epochs for faster execution
                             validation_split=0.2,
                             verbose=1)

# Evaluate FCNN
fcnn_test_loss, fcnn_test_acc = fcnn_model.evaluate(X_test_fcnn, y_test_encoded)
print(f'FCNN Test accuracy: {fcnn_test_acc:.4f}')

print("\n===== Convolutional Neural Network (CNN) =====\n")

# Preprocess data for CNN
print("Preprocessing data for CNN...")
X_train_cnn = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test_cnn = X_test.reshape(-1, 28, 28, 1) / 255.0

# Create CNN model
print("Creating and training CNN model...")
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
cnn_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Train model
cnn_history = cnn_model.fit(X_train_cnn, y_train_encoded,
                           batch_size=128,
                           epochs=5,  # Reduced epochs for faster execution
                           validation_split=0.2,
                           verbose=1)

# Evaluate CNN
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test_encoded)
print(f'CNN Test accuracy: {cnn_test_acc:.4f}')

# Compare results
print('\nModel Comparison:')
print(f'FCNN Test Accuracy: {fcnn_test_acc:.4f}')
print(f'CNN Test Accuracy: {cnn_test_acc:.4f}')
print(f'Improvement with CNN: {(cnn_test_acc - fcnn_test_acc) * 100:.2f}%')

print("\n===== MNIST Classification Complete =====\n")