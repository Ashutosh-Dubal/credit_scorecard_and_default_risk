import kagglehub
import shutil
import os

target_path = "/Users/ashutoshdubal/PythonProjects/credit_scorecard_and_default_risk/data/raw"
target_file = os.path.join(target_path, "application_train.csv")

if os.path.exists(target_file):
    print("Dataset already exists in target location. Skipping download and copy.")
else:
    print("Dataset not found locally. Downloading...")

    source_path = kagglehub.competition_download('home-credit-default-risk')

    os.makedirs(target_path, exist_ok=True)

    for file_name in os.listdir(source_path):
        src_file = os.path.join(source_path, file_name)
        dst_file = os.path.join(target_path, file_name)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

    print("Files copied to:", target_path)