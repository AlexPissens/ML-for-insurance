# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:18:02 2025

@author: apissens
"""
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from data_prep import load_clean_df

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SPLIT_DIR.mkdir(exist_ok=True)

def create_train_val(test_size=0.20, seed=42):
    # ---------- 1. load the cleaned dataframe -------------------
    df = load_clean_df()
    # ---------- 2. create binary flag for stratification --------
    df["has_claim"] = (df.ClaimNb > 0).astype(int)
    # ---------- 3. split ---------------------------------------
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["has_claim"])
    # ---------- 4. drop helper column (optional) ---------------
    train_df = train_df.drop(columns=["has_claim"])
    val_df   = val_df.drop(columns=["has_claim"])
    
    # ---------- 5. persist indices so every script uses same split
    joblib.dump(train_df.index, SPLIT_DIR / "train_idx.pkl")
    joblib.dump(val_df.index,   SPLIT_DIR / "val_idx.pkl")

    print("Saved indices:", SPLIT_DIR)
    print(f"Train shape : {train_df.shape}")
    print(f"Valid shape : {val_df.shape}")
    



if __name__ == "__main__":
    create_train_val()

