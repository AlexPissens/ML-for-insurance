import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scripts.data_prep import load_clean_df

# Define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
PRED_DIR = PROJECT_ROOT / "reports" / "predictions"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
subgroup_col='Area'

"""
Adjust the bias for each subgroup and compare means (observed, original predicted, adjusted predicted).

Parameters:
- subgroup_col: Column name to define subgroups (e.g., 'Area').
"""
# Load validation data
idx_val = joblib.load(SPLIT_DIR / "val_idx.pkl")
df = load_clean_df()
val_df = df.loc[idx_val].copy()

# Load predictions from GLM and NN
glm_preds = pd.read_csv(PRED_DIR / "glm_val_preds.csv", index_col='PolicyID')
nn1_preds = pd.read_csv(PRED_DIR / "nn1_val_preds.csv", index_col='PolicyID')

# Merge predictions into val_df
val_df = val_df.join(glm_preds['Pred_ClaimNb'].rename('predicted_claims_glm'))
val_df = val_df.join(nn1_preds['Pred_ClaimNb'].rename('predicted_claims_nn'))

# Define subgroups
subgroups = val_df[subgroup_col].unique()

# Results for comparison
results_glm = {}
results_nn = {}

# Adjust bias for GLM predictions
for subgroup in subgroups:
    subgroup_df = val_df[val_df[subgroup_col] == subgroup].copy()
    exposure = subgroup_df['Exposure'].values

    # Calculate observed and predicted frequencies (weighted by exposure)
    observed_freq = (subgroup_df['ClaimNb'] / subgroup_df['Exposure']).values
    predicted_freq_glm = (subgroup_df['predicted_claims_glm'] / subgroup_df['Exposure']).values

    # Weighted means
    observed_mean = np.sum(observed_freq * exposure) / np.sum(exposure)
    predicted_mean_glm = np.sum(predicted_freq_glm * exposure) / np.sum(exposure)

    # Adjustment factor
    adjustment_factor_glm = observed_mean / predicted_mean_glm if predicted_mean_glm != 0 else 1.0

    # Debug: Print intermediate values
    print(f"\nSubgroup: {subgroup} (GLM)")
    print(f"Adjustment Factor (GLM): {adjustment_factor_glm:.4f}")

    # Adjust predictions (apply to the predicted claims, not frequency)
    adjusted_predicted_glm = subgroup_df['predicted_claims_glm'] * adjustment_factor_glm

    # Calculate adjusted predicted frequency (weighted by exposure)
    adjusted_pred_freq_glm = adjusted_predicted_glm / subgroup_df['Exposure']
    adjusted_predicted_mean_glm = np.sum(adjusted_pred_freq_glm * exposure) / np.sum(exposure)

    results_glm[subgroup] = {
        'Observed Frequency': observed_mean,
        'Original Predicted Frequency': predicted_mean_glm,
        'Adjusted Predicted Frequency': adjusted_predicted_mean_glm
    }

# Adjust bias for NN predictions
for subgroup in subgroups:
    subgroup_df = val_df[val_df[subgroup_col] == subgroup].copy()
    exposure = subgroup_df['Exposure'].values

    # Calculate observed and predicted frequencies (weighted by exposure)
    observed_freq = (subgroup_df['ClaimNb'] / subgroup_df['Exposure']).values
    predicted_freq_nn = (subgroup_df['predicted_claims_nn'] / subgroup_df['Exposure']).values

    # Weighted means
    observed_mean = np.sum(observed_freq * exposure) / np.sum(exposure)
    predicted_mean_nn = np.sum(predicted_freq_nn * exposure) / np.sum(exposure)

    # Adjustment factor
    adjustment_factor_nn = observed_mean / predicted_mean_nn if predicted_mean_nn != 0 else 1.0

    # Debug: Print intermediate values
    print(f"Subgroup: {subgroup} (NN)")
    print(f"Adjustment Factor (NN): {adjustment_factor_nn:.4f}")

    # Adjust predictions (apply to the predicted claims, not frequency)
    adjusted_predicted_nn = subgroup_df['predicted_claims_nn'] * adjustment_factor_nn

    # Calculate adjusted predicted frequency (weighted by exposure)
    adjusted_pred_freq_nn = adjusted_predicted_nn / subgroup_df['Exposure']
    adjusted_predicted_mean_nn = np.sum(adjusted_pred_freq_nn * exposure) / np.sum(exposure)

    results_nn[subgroup] = {
        'Observed Frequency': observed_mean,
        'Original Predicted Frequency': predicted_mean_nn,
        'Adjusted Predicted Frequency': adjusted_predicted_mean_nn
    }

# Convert results to DataFrames and save
glm_results_df = pd.DataFrame.from_dict(results_glm, orient='index')
glm_results_df.index.name = subgroup_col
nn_results_df = pd.DataFrame.from_dict(results_nn, orient='index')
nn_results_df.index.name = subgroup_col

glm_results_df.to_csv(TABLE_DIR / f"glm_bias_adjustment_{subgroup_col}.csv")
nn_results_df.to_csv(TABLE_DIR / f"nn_bias_adjustment_{subgroup_col}.csv")

# Print results
print(f"\nGLM Bias Adjustment Results for {subgroup_col}:")
print(glm_results_df)
print(f"\nNN Bias Adjustment Results for {subgroup_col}:")
print(nn_results_df)

