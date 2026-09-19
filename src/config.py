from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"


TARGET = "Churn"
ID_COL = "customerID"
