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