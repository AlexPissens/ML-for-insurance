# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:14:25 2025

@author: apissens
"""

from pathlib import Path
import joblib, pandas as pd, numpy as np
import statsmodels.api as sm
from data_prep import load_clean_df
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

# ---------- 1. load data & splits ------------------------------------
df = load_clean_df()
idx_train = joblib.load(SPLIT_DIR / "train_idx.pkl")
idx_val   = joblib.load(SPLIT_DIR / "val_idx.pkl")

train = df.loc[idx_train].copy()
val   = df.loc[idx_val].copy()

# ---------- 2. choose predictors -------------------------------------
FEATURES_CAT = ["Area", "VehBrand", "VehGas"]
FEATURES_NUM = ["VehPower", "VehAge", "DrivAge",
                "log_Density", "log_BonusMalus"]  # after log1p
FEATURES = FEATURES_CAT + FEATURES_NUM

# one‑hot encode categoricals (drop first to avoid dummy trap)
X_train = pd.get_dummies(train[FEATURES], drop_first=True)
X_val   = pd.get_dummies(val[FEATURES],   drop_first=True)

# statsmodels needs same columns in both sets
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

# add intercept manually if you want (sm adds if we call add_constant)
X_train = sm.add_constant(X_train)
X_val   = sm.add_constant(X_val, has_constant='add')

y_train = train["ClaimNb"].values
y_val   = val["ClaimNb"].values

X_train = X_train.astype(float)      # turns bool → 0.0 / 1.0
X_val   = X_val.astype(float)

# (optional) also coerce y and offset for consistency
y_train = y_train.astype(float)
y_val   = y_val.astype(float)
train_offset = train["offset"].astype(float)
val_offset   = val["offset"].astype(float)

# ---------- 3. fit Poisson GLM with offset ---------------------------

glm_pois = sm.GLM(
    y_train,
    X_train,
    family=sm.families.Poisson(),
    offset=train["offset"]           # log(Exposure)
)
result = glm_pois.fit()
print(result.summary())              # <- full coefficient table

# ---------- 4. evaluate deviance -------------------------------------
def mean_poisson_deviance(y_true, y_pred):
    """Vectorised mean Poisson deviance."""
    y_true = y_true.astype(float)
    return np.mean(
        2 * (y_pred - y_true + y_true * np.where(y_true == 0, 0, np.log(y_true / y_pred)))
    )

mu_train = result.predict(X_train, offset=train["offset"])
mu_val   = result.predict(X_val,   offset=val["offset"])

dev_train = mean_poisson_deviance(y_train, mu_train)
dev_val   = mean_poisson_deviance(y_val,   mu_val)

print(f"\nMean Poisson deviance  –  train: {dev_train:,.4f}   |   val: {dev_val:,.4f}")

# ---------- 5. save predictions for later comparison -----------------
out_dir = PROJECT_ROOT / "reports" / "predictions"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({
    "PolicyID": val.index,
    "Obs_ClaimNb": y_val,
    "Pred_ClaimNb": mu_val
}).to_csv(out_dir / "glm_poisson_val_preds.csv", index=False)
print(f"Saved validation predictions to {out_dir}")


val_preds = pd.DataFrame({
    "ClaimNb_obs": y_val,
    "ClaimNb_pred": mu_val,
    "Exposure": val["Exposure"],
    "ClaimFreq_obs": y_val / val["Exposure"],
    "ClaimFreq_pred": mu_val / val["Exposure"]
})

val_preds["decile"] = pd.qcut(val_preds["ClaimFreq_pred"], 10, labels=False)

calib = (val_preds
         .groupby("decile")
         .apply(lambda g: pd.Series({
             "obs_freq": g.ClaimNb_obs.sum() / g.Exposure.sum(),
             "pred_freq": g.ClaimNb_pred.sum() / g.Exposure.sum()
         })))
calib.plot(y=["obs_freq","pred_freq"], kind="bar")
plt.ylabel("Claim frequency")
plt.title("GLM calibration by predicted‑risk decile")
plt.show()