# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:53:42 2025

@author: apissens
"""
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_clean_df(csv_path: Path = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "dataset.csv"
    # ---------- 1. read with correct delimiter ------------------
    df = pd.read_csv(csv_path, sep=';', decimal=',', engine='python')
    df.columns = df.columns.str.strip()

    # ---------- 2. derive ClaimFreq -----------------------------
    df["ClaimFreq"] = df["ClaimNb"] / df["Exposure"]

    # ---------- 3. identify feature groups ----------------------
    CAT_COLS   = ["Area", "VehBrand", "VehGas"]       # low/medium cardinality
    NUM_COLS   = ["VehPower", "VehAge", "DrivAge",    # numeric predictors
                  "BonusMalus", "Density"]
    OFFSET_COL = "Exposure"                           # keep as float
    
    # ---------- 4. cast dtypes ----------------------------------
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    
    # numeric (ensure float, even if integer‑like)
    for col in NUM_COLS + [OFFSET_COL, "ClaimNb", "ClaimFreq"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # ---------- 5. basic missing‑value fix ----------------------
    # simple rule‑of‑thumb: numeric → median, categorical → 'Missing'
    for col in NUM_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    for col in CAT_COLS:
        if df[col].isna().any():
            df[col] = df[col].cat.add_categories("Missing").fillna("Missing")
    
    # ---------- 6. tame skewed numeric distributions -----------
    # log‑transform heavily right‑skewed vars (example: Density)
    skewed = ["Density", "BonusMalus"]       # pick after inspecting histograms
    
    for col in skewed:
        df[f"log_{col}"] = np.log1p(df[col])     # log(1+x) avoids log(0)
    
    # (optionally drop original or keep both)
    # df.drop(columns=skewed, inplace=True)
    
    # ---------- 7. confirm types --------------------------------
    print(df.dtypes)
    
    if (df["Exposure"] == 0).any():
        raise ValueError("Zero exposure rows found – drop or fix before log.")
    
    df["offset"] = np.log(df["Exposure"])

    # add offset
    df = df[df.Exposure > 0]
    df["offset"] = np.log(df.Exposure)

    return df

