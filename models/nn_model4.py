from tensorflow import keras
from tensorflow.keras import layers
from utils.model_utils import custom_poisson_loss, mean_poisson_deviance
from scripts import nn_data as nd  

def run_nn_model4():
    X_train = nd.X_train_np
    X_val = nd.X_val_np
    y_train_dev = nd.y_train_dev
    y_val_dev = nd.y_val_dev


# Build the Enhanced Neural Network
    model = keras.Sequential([
    # First hidden layer: Keep 64 units, add L2 regularization
    layers.Dense(
        64, 
        activation="relu", 
        input_shape=(X_train.shape[1],),
        kernel_regularizer=keras.regularizers.l2(0.005)  # L2 regularization with lambda=0.005
    ),
    layers.BatchNormalization(),  # Keep batch normalization to stabilize training
    layers.Dropout(0.3),          # Add dropout (30%) to prevent overfitting
    
    # Second hidden layer: Keep 32 units, add L2 regularization and batch normalization
    layers.Dense(
        32, 
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.005)  # L2 regularization
    ),
    layers.BatchNormalization(),  # Add batch normalization for stability
    layers.Dropout(0.2),          # Add dropout (20%) for regularization
    
    # Third hidden layer: Small layer for refined feature extraction
    layers.Dense(
        16, 
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(0.005)  # L2 regularization
    ),
    
    # Output layer: Exponential activation for positive rate predictions
    layers.Dense(1, activation="exponential")  # Output layer for positive rate predictions
    ])


    model.compile(optimizer="adam", loss=custom_poisson_loss, metrics=["mean_squared_error"])
    model.fit(X_train, y_train_dev, epochs=50, batch_size=32, validation_data=(X_val, y_val_dev), verbose=1)

    y_pred = model.predict(X_val, verbose=0)
    true_counts = y_val_dev[:, 0] * y_val_dev[:, 1]
    pred_counts = y_pred.flatten() * y_val_dev[:, 1]
    dev_val = mean_poisson_deviance(true_counts, pred_counts, y_val_dev[:, 1])

    print(f"NN Model 1 - Val Deviance: {dev_val:.4f}")
    return dev_val

if __name__ == "__main__":
    run_nn_model4()