"""
fetch_data.py — Download the Lending Club dataset.

The public Lending Club accepted-loans CSV is available on Kaggle:
  https://www.kaggle.com/datasets/wordsforthewise/lending-club

You need a Kaggle API token (~/.kaggle/kaggle.json) to use the automatic
download below. Alternatively, download the CSV manually and place it at:
  data/raw/lending_club.csv
"""

import os
import sys
import zipfile

try:
    import kaggle  # pip install kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.helper import DATA_RAW

DATASET    = "wordsforthewise/lending-club"
TARGET_CSV = os.path.join(DATA_RAW, "lending_club.csv")
TARGET_ZIP = os.path.join(DATA_RAW, "lending_club.zip")


def fetch_via_kaggle():
    print(f"[fetch] Downloading '{DATASET}' from Kaggle …")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=DATA_RAW, unzip=False)
    # Kaggle names the zip after the dataset slug
    zip_path = os.path.join(DATA_RAW, "lending-club.zip")
    if os.path.exists(zip_path):
        print("[fetch] Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_RAW)
        os.remove(zip_path)
        print("[fetch] Done.")
    else:
        print("[fetch] Zip not found — check Kaggle credentials or dataset slug.")


def main():
    if os.path.exists(TARGET_CSV):
        print(f"[fetch] Raw CSV already exists at {TARGET_CSV}. Skipping download.")
        return

    if not KAGGLE_AVAILABLE:
        print(
            "[fetch] 'kaggle' package not found. Install it with:\n"
            "  pip install kaggle\n"
            "Then place your ~/.kaggle/kaggle.json API token and re-run.\n"
            f"Or download the CSV manually to: {TARGET_CSV}"
        )
        return

    fetch_via_kaggle()


if __name__ == "__main__":
    main()
