# ---------------------------------------------------------------------
# src/nn_model1.py   (frequency‑target, safe & simple)
# ---------------------------------------------------------------------
import numpy as np, pandas as pd, joblib, tensorflow as tf
from pathlib import Path
from nn_data import (X_train_np, X_val_np,            # frequency inputs
                     y_train as y_train_freq,         # ClaimFreq target
                     y_val   as y_val_freq,
                     exp_train, exp_val)               # safe loss
from glm_baseline import mean_poisson_deviance        # count dev metric
from nn_data import X_train_np, X_val_np, y_train_dev, y_val_dev  # notice the new variables
from nn_data import exp_val  # if you still need it for evaluation
from nn_model_poisson_loss import  poisson_dev_loss

# 1. Model architecture
inputs = tf.keras.Input(shape=(X_train_np.shape[1],))
x = tf.keras.layers.BatchNormalization()(inputs)   # NEW
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
logits = tf.keras.layers.Dense(1)(x)
logits = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -4.0, 4.0))(logits)
outputs = tf.keras.layers.Lambda(lambda t: tf.exp(t) + 1e-6)(logits)


model   = tf.keras.Model(inputs, outputs)

opt = tf.keras.optimizers.Adam(1e-4, clipnorm=1.0)

from tensorflow.keras.losses import Poisson
model.compile(optimizer=opt, loss=poisson_dev_loss, metrics=[poisson_dev_loss])
model.summary()

# 2. Train
history = model.fit(
    X_train_np, y_train_dev,
    validation_data=(X_val_np, y_val_dev),
    epochs=100,
    batch_size=2048,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_poisson_dev_loss",  # your custom loss metric
            patience=10,                     # wait this many epochs
            min_delta=1e-4,                  # require at least this improvement
            mode="min",                      # looking for a minimum
            restore_best_weights=True
        )
    ],
    verbose=2
)

# 3. Predict frequency, convert to counts
freq_pred_val  = model.predict(X_val_np, batch_size=4096).flatten()
count_pred_val = freq_pred_val * exp_val                  # μ̂ (counts)

# 4. Evaluate count‑scale deviance for fair comparison with GLM
freq_pred_val  = model.predict(X_val_np).flatten()
count_pred_val = freq_pred_val * exp_val
y_val_counts   = y_val_dev[:,0] * y_val_dev[:,1]  # freq × exposure

from glm_baseline import mean_poisson_deviance
dev_val = mean_poisson_deviance(y_val_counts, count_pred_val)

print(f"Validation mean Poisson deviance (count scale): {dev_val:.4f}")

#y_val_counts = (y_val_freq * exp_val).astype("float32")   # ClaimNb
#dev_val_nn1  = mean_poisson_deviance(y_val_counts, count_pred_val)

#print("\nValidation mean Poisson deviance (count scale)")
#print("GLM  :", 0.2510)          # replace with your exact GLM value
#print("NN‑1 :", f"{dev_val_nn1:0.4f}")

# 5. Save predictions for plots
proj = Path(__file__).resolve().parents[1]
outf = proj / "reports" / "predictions" / "nn1_val_preds.csv"
outf.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "PolicyID":  np.arange(len(y_val_counts)),
    "Obs_Count": y_val_counts,
    "Pred_Count": count_pred_val,
    "Exposure": exp_val
}).to_csv(outf, index=False)

print("Predictions saved to", outf)
# ---------------------------------------------------------------------
