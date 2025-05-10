import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from scripts.data_prep import load_clean_df
from utils.model_utils import mean_poisson_deviance
import joblib
from sklearn.linear_model import PoissonRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

def run_glm():
    df = load_clean_df()
    idx_train = joblib.load(SPLIT_DIR / "train_idx.pkl")
    idx_val = joblib.load(SPLIT_DIR / "val_idx.pkl")
    train = df.loc[idx_train].copy()
    val = df.loc[idx_val].copy()

    FEATURES_CAT = ["Area", "VehBrand", "VehGas"]
    FEATURES_NUM = ["VehPower", "VehAge", "DrivAge", "Density", "BonusMalus"]
    FEATURES = FEATURES_CAT + FEATURES_NUM

    X_train = pd.get_dummies(train[FEATURES], drop_first=True)
    X_val = pd.get_dummies(val[FEATURES], drop_first=True).reindex(columns=X_train.columns, fill_value=0)
    X_train = sm.add_constant(X_train)
    X_val = sm.add_constant(X_val)
    
    y_train = train["ClaimNb"].values
    y_val = val["ClaimNb"].values
    
    X_train = X_train.astype(float)      # turns bool → 0.0 / 1.0
    X_val   = X_val.astype(float)
    
    train_offset = train["offset"].values.astype(float)
    val_offset = val["offset"].values.astype(float)

    glm = PoissonRegressor(alpha=0.0)

    glm.fit(X_train, y_train)


    #glm_pois = sm.GLM(y_train, X_train, family=sm.families.Poisson(), offset=train_offset)
    #result = glm_pois.fit()

    mu_train = glm.predict(X_train)
    mu_val = glm.predict(X_val)

    dev_train = mean_poisson_deviance(y_train, mu_train, train["Exposure"])
    dev_val = mean_poisson_deviance(y_val, mu_val, val["Exposure"])

    print(f"GLM - Train Deviance: {dev_train:.4f}, Val Deviance: {dev_val:.4f}")
    return dev_val

if __name__ == "__main__":
    run_glm()