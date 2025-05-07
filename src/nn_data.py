# -*- coding: utf-8 -*-

"""
Utility module for neural‑network experiments on the motor‑claim dataset.
It provides:
    • X_train_np, X_val_np  – float32 design matrices (annual‑frequency scale)
    • y_train, y_val        – ClaimFreq targets (count / exposure)
    • exp_train, exp_val    – exposure vectors (float32)
    • poisson_freq_loss     – numerically stable Poisson‑deviance loss

The module *only* loads data and performs deterministic preprocessing.  All
networks can therefore `import nn_data as nd` and reuse the same tensors.
"""
from pathlib import Path
import joblib, numpy as np, pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# 1. Locate project root and frozen split indices
# ---------------------------------------------------------------------
PROJ = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJ / "data" / "splits"

idx_train = joblib.load(SPLIT_DIR / "train_idx.pkl")
idx_val   = joblib.load(SPLIT_DIR / "val_idx.pkl")

# ---------------------------------------------------------------------
# 2. Load the cleaned DataFrame from your earlier ETL step
# ---------------------------------------------------------------------
from data_prep import load_clean_df    # noqa – local module

df       = load_clean_df()
train_df = df.loc[idx_train].copy()
val_df   = df.loc[idx_val].copy()

# ---------------------------------------------------------------------
# 3. Feature lists (keep in one place to avoid drift)
# ---------------------------------------------------------------------
FEATURES_CAT = ["Area", "VehBrand", "VehGas"]
FEATURES_NUM = [
    "VehPower", "VehAge", "DrivAge",
    "log_Density", "log_BonusMalus",
]

# ---------------------------------------------------------------------
# 4. Design matrix: one‑hot categoricals + scaled numeric columns
#    Target:  ClaimFreq  = ClaimNb / Exposure  (annualised frequency)
# ---------------------------------------------------------------------
X_train = pd.get_dummies(
    train_df[FEATURES_CAT + FEATURES_NUM], drop_first=True
)
X_val = pd.get_dummies(
    val_df[FEATURES_CAT + FEATURES_NUM], drop_first=True
).reindex(columns=X_train.columns, fill_value=0)

# Standard‑scale the numeric columns only (leave dummies 0/1)
scaler = StandardScaler().fit(X_train[FEATURES_NUM])
X_train[FEATURES_NUM] = scaler.transform(X_train[FEATURES_NUM])
X_val[FEATURES_NUM]   = scaler.transform(X_val[FEATURES_NUM])

# Convert to contiguous float32 NumPy for TensorFlow
X_train_np = X_train.to_numpy(dtype="float32")
X_val_np   = X_val.to_numpy(dtype="float32")

# ---------------------------------------------------------------------
# 5. Targets and exposure vectors
# ---------------------------------------------------------------------
y_train = (train_df["ClaimNb"] / train_df["Exposure"]).values.astype("float32")
y_val   = (val_df["ClaimNb"]   / val_df["Exposure"]).values.astype("float32")

exp_train = train_df["Exposure"].values.astype("float32")
exp_val   = val_df["Exposure"].values.astype("float32")

# ---------------------------------------------------------------------
# 6. Numerically‑safe Poisson deviance on frequency scale  (TensorFlow op)
# ---------------------------------------------------------------------

def poisson_freq_loss(y_true, y_pred, eps=tf.keras.backend.epsilon()):
    """Mean Poisson deviance between annual frequencies.

    y_true : tensor, ClaimFreq (>=0)
    y_pred : tensor, positive predictions (softplus / exp output)
    """
    y_true = tf.maximum(y_true, 0.0)
    y_pred = tf.maximum(y_pred, eps)  # ensure strictly > 0

    log_term = tf.where(
        y_true > 0.0,
        y_true * tf.math.log(y_true / y_pred),
        tf.zeros_like(y_true)
    )
    dev = 2.0 * (y_pred - y_true + log_term)
    return tf.reduce_mean(dev)

y_train_dev = np.stack([y_train, exp_train], axis=1).astype("float32")
y_val_dev   = np.stack([y_val,   exp_val  ], axis=1).astype("float32")

# ---------------------------------------------------------------------
# 7. Quick integrity check when module is run directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("nn_data sanity check →", X_train_np.shape, "train rows",
          "freq mean", y_train.mean().round(4))
