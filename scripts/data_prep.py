from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_clean_df(csv_path: Path = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data" / "dataset.csv"
    df = pd.read_csv(csv_path, sep=';', decimal=',', engine='python')
    df.columns = df.columns.str.strip()
    # Add your data cleaning steps here (e.g., handling missing values)
    df["offset"] = np.log(df["Exposure"])
    return df

