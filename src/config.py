from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
PARAMS_PATH = ROOT_DIR / "params.yaml"
MODEL_PATH = MODELS_DIR / "model.joblib"


TARGET = "Churn"
ID_COL = "customerID"
