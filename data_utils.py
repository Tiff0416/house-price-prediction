import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def get_data():
    os.makedirs("data", exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Download and unzip
    api.competition_download_files("house-prices-advanced-regression-techniques", path="data")

    with zipfile.ZipFile("data/house-prices-advanced-regression-techniques.zip", "r") as zip_ref:
        zip_ref.extractall("data")
    print("Data downloaded and extracted.")
    
get_data()

