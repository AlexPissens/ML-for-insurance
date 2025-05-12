import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from scripts.data_prep import load_clean_df

# Define project root and directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
PRED_DIR = PROJECT_ROOT / "reports" / "predictions"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

# Ensure directories exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def run_subgroup_analysis():
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

    # Add binned columns for vehicle power and driver age
    val_df['VehPower_cat'] = pd.cut(val_df['VehPower'], bins=[0, 6, 9, 12, 15], 
                                    labels=['Low', 'Medium', 'High', 'Very High'])
    val_df['DrivAge_cat'] = pd.cut(val_df['DrivAge'], bins=[17, 25, 35, 45, 55, 65, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+'])

    # Function to calculate frequencies
    def calculate_frequencies(df, group_by=None):
        if group_by is None:
            observed_freq = df['ClaimNb'].sum() / df['Exposure'].sum()
            predicted_freq_glm = df['predicted_claims_glm'].sum() / df['Exposure'].sum()
            predicted_freq_nn = df['predicted_claims_nn'].sum() / df['Exposure'].sum()
            return pd.DataFrame({
                'Subgroup': ['Global'],
                'Observed': [observed_freq],
                'GLM': [predicted_freq_glm],
                'NN': [predicted_freq_nn]
            })
        else:
            grouped = df.groupby(group_by)
            result = []
            for name, group in grouped:
                observed_freq = group['ClaimNb'].sum() / group['Exposure'].sum()
                predicted_freq_glm = group['predicted_claims_glm'].sum() / group['Exposure'].sum()
                predicted_freq_nn = group['predicted_claims_nn'].sum() / group['Exposure'].sum()
                result.append({
                    'Subgroup': name,
                    'Observed': observed_freq,
                    'GLM': predicted_freq_glm,
                    'NN': predicted_freq_nn
                })
            return pd.DataFrame(result)

    # Calculate frequencies for subgroups
    global_freq = calculate_frequencies(val_df)
    area_freq = calculate_frequencies(val_df, 'Area')
    vehpower_freq = calculate_frequencies(val_df, 'VehPower_cat')
    drivage_freq = calculate_frequencies(val_df, 'DrivAge_cat')

    # Save tables to CSV
    global_freq.to_csv(TABLE_DIR / "global_freq.csv", index=False)
    area_freq.to_csv(TABLE_DIR / "area_freq.csv", index=False)
    vehpower_freq.to_csv(TABLE_DIR / "vehpower_freq.csv", index=False)
    drivage_freq.to_csv(TABLE_DIR / "drivage_freq.csv", index=False)

    # Function to plot frequencies
    def plot_frequencies(df, subgroup_type):
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(x='Subgroup', y=['Observed', 'GLM', 'NN'], kind='bar', ax=ax)
        ax.set_title(f'Claim Frequencies by {subgroup_type}')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f'frequencies_{subgroup_type}.png')
        plt.close()
 # Generate plots
    plot_frequencies(area_freq, 'Area')
    plot_frequencies(vehpower_freq, 'VehiclePower')
    plot_frequencies(drivage_freq, 'DriverAge')
    plot_frequencies(global_freq, 'Global')

    print("Subgroup analysis complete. Tables and plots saved.")

if __name__ == "__main__":
    run_subgroup_analysis()