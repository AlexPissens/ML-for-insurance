# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:46:34 2025

@author: apissens
"""

import tensorflow as tf

def poisson_dev_loss(y_true, y_pred):
    """
    Mean Poisson deviance (count‐scale) as in the course slides.
    y_true: [y_freq, exposure]
    y_pred: predicted y_freq (per-year)
    """
    # split the two columns
    y_freq = y_true[:, 0]               # ClaimNb / Exposure
    expo   = y_true[:, 1]               # Exposure

    # ensure positivity
    eps    = tf.keras.backend.epsilon()
    y_freq = tf.maximum(y_freq, eps)
    y_pred = tf.maximum(y_pred, eps)

    # per‐record deviance term: 2 * e * [ y log(y/ŷ) - y + ŷ ]
    dev = 2.0 * expo * (
        y_freq * tf.math.log(y_freq / y_pred)
        - y_freq
        + y_pred
    )
    return tf.reduce_mean(dev)


def calculate_poisson_deviance(true_counts, pred_counts):
    """
    Calculate mean Poisson deviance for observed and predicted counts.
    Args:
        true_counts: Array of observed claim counts
        pred_counts: Array of predicted claim counts
    Returns:
        Mean Poisson deviance
    """
    import numpy as np
    
    # Avoid log(0) or division by 0
    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)
    pred_counts = np.clip(pred_counts, 1e-10, None)  # Avoid zero predictions
    
    # Calculate deviance per observation
    deviance = 2 * (true_counts * np.log(true_counts / pred_counts) - (true_counts - pred_counts))
    
    # Handle cases where true_counts = 0 (log term becomes 0)
    deviance = np.where(true_counts == 0, 2 * pred_counts, deviance)
    
    # Return mean deviance
    return np.mean(deviance)