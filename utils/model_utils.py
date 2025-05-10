import tensorflow as tf
import numpy as np

def custom_poisson_loss(y_true, y_pred):
    """
    Custom Poisson loss for claim counts.
    y_true: tensor with [frequency, Exposure]
    y_pred: predicted rate (lambda)
    """
    frequency = y_true[:, 0]
    exposure = y_true[:, 1]
    y = frequency * exposure
    lambda_pred = y_pred
    expected_count = lambda_pred * exposure
    loss = expected_count - y * tf.math.log(expected_count + 1e-10)
    return tf.reduce_mean(loss)

def mean_poisson_deviance(y_true, y_pred, exposure):
    """
    Calculate mean Poisson deviance.
    y_true: observed counts
    y_pred: predicted counts
    exposure: exposure values
    """
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, None)
    y_true = np.clip(y_true, epsilon, None)
    dev = 2 * (y_true * np.log(y_true / y_pred) - (y_true - y_pred))
    total = np.sum(exposure * dev)
    mean_dev = total / np.sum(exposure)
    return mean_dev