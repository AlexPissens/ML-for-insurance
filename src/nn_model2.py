# -*- coding: utf-8 -*-
"""
Created on Wed May  7 16:33:00 2025

@author: apissens
"""

import nn_data as nd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from glm_baseline import mean_poisson_deviance

# Load preprocessed data from nn_data.py
X_train = nd.X_train_np
X_val = nd.X_val_np
y_train_dev = nd.y_train_dev  # [frequency, Exposure]
y_val_dev = nd.y_val_dev      # [frequency, Exposure]

# Define the custom Poisson loss function
def custom_poisson_loss(y_true, y_pred):
    """
    Custom Poisson loss for claim counts.
    y_true: tensor with [frequency, Exposure]
    y_pred: predicted rate (lambda)
    """
    frequency = y_true[:, 0]
    exposure = y_true[:, 1]
    y = frequency * exposure  # Approximate ClaimNb
    lambda_pred = y_pred      # Predicted rate
    expected_count = lambda_pred * exposure
    loss = expected_count - y * tf.math.log(expected_count + 1e-10)
    return tf.reduce_mean(loss)

# Build the Neural Network (Deeper with More Neurons)
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),  # First hidden layer with 64 units
    layers.Dense(32, activation="relu"),                                   # Second hidden layer with 32 units
    layers.Dense(1, activation="exponential")                              # Output layer for positive rate predictions
])

# Compile the model with the custom Poisson loss
model.compile(
    optimizer="adam",
    loss=custom_poisson_loss,
    metrics=["mean_squared_error"]  # Additional metric for monitoring
)

# Train the model
history = model.fit(
    X_train, y_train_dev,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val_dev),
    verbose=1
)

# Evaluate the model
y_pred = model.predict(X_val)  # Predicted rates

# Compute true counts and predicted counts
true_counts = y_val_dev[:, 0] * y_val_dev[:, 1]  # frequency * Exposure ≈ ClaimNb
pred_counts = y_pred * y_val_dev[:, 1]           # predicted rate * Exposure

# Define mean Poisson deviance for evaluation

   

# Calculate and print the mean Poisson deviance per observation
deviance=mean_poisson_deviance(true_counts, pred_counts,y_val_dev[:,1])
print(f"Validation Poisson Deviance (per observation): {deviance / len(true_counts):.4f}")

# Plot training history
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Custom Poisson Loss")
plt.legend()
plt.savefig("nn_training_loss_adjusted.png")