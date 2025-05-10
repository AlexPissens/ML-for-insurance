from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from scripts.data_prep import load_clean_df

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
SPLIT_DIR.mkdir(exist_ok=True)

def create_train_val(test_size=0.20, seed=42):
    df = load_clean_df()
    df["has_claim"] = (df["ClaimNb"] > 0).astype(int)
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["has_claim"]
    )
    train_df = train_df.drop(columns=["has_claim"])
    val_df = val_df.drop(columns=["has_claim"])
    joblib.dump(train_df.index, SPLIT_DIR / "train_idx.pkl")
    joblib.dump(val_df.index, SPLIT_DIR / "val_idx.pkl")
    print("Saved indices:", SPLIT_DIR)


if __name__ == "__main__":
    create_train_val()