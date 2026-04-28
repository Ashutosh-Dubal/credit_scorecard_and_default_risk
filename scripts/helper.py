import os
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_APP_TRAIN = "data/raw/application_train.csv"
DATA_CLEAN_PATH = "data/clean"
VISUALS_EDA = "visuals/EDA"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)
# ── Audit ─────────────────────────────────────────────────────────────────────
def data_audit(df):
    print("\nData Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nNumeric Summary:")
    print(df.describe())

    print("\nDuplicates:", df.duplicated().sum())

    for col in df.select_dtypes(include='object'):
        print(f"\nUnique values in {col}:", df[col].dropna().unique())


def save_model(model, name):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"[helper] Model saved → {path}")
    return path

def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    return joblib.load(path)
