"""Cleaning, splitting and the encode + scale pipeline."""
import json

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.config import (
    ID_COL, PARAMS_PATH, PROCESSED_DIR, RAW_DATA_PATH, REPORTS_DIR,
    TARGET, TEST_PATH, TRAIN_PATH,
)
from src.data.validate import validate_clean

# ---------------------------------------------------------------- column groups
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ONEHOT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaymentMethod",
]
ORDINAL_COLS = ["Contract"]
CONTRACT_ORDER = [["Month-to-month", "One year", "Two year"]]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "n_addon_services"]
PASSTHROUGH_COLS = ["SeniorCitizen"]  # already 0/1
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


# ---------------------------------------------------------------- 1) cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """MINIMAL default cleaning. Replace/extend with your own; the only
    requirement is that the result passes validate_clean()."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- 2) features
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived feature: how many add-on services the customer has."""
    df = df.copy()
    df["n_addon_services"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    return df


# ---------------------------------------------------------------- 3) encode+scale
def build_preprocessor() -> Pipeline:
    """UNFITTED preprocessing pipeline: features -> encode -> scale."""
    encode_scale = ColumnTransformer(
        transformers=[
            ("binary", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                     sparse_output=False), BINARY_COLS),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False), ONEHOT_COLS),
            ("ordinal", OrdinalEncoder(categories=CONTRACT_ORDER,
                                       handle_unknown="use_encoded_value",
                                       unknown_value=-1), ORDINAL_COLS),
            ("scale", StandardScaler(), NUMERIC_COLS),
            ("pass", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("encode_scale", encode_scale),
    ])
    return pipe.set_output(transform="pandas")


# ---------------------------------------------------------------- 4) split
def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    """Stratified split so train and test keep the same churn rate."""
    return train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET]
    )


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())["data"]

    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_data(df)
    df = validate_clean(df)                    # gate: is the cleaning good enough?
    df = df.drop(columns=[ID_COL])             # identifier is not a feature
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    train_df, test_df = split_data(df, params["test_size"], params["random_state"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "train_rows": len(train_df), "test_rows": len(test_df),
        "train_churn_rate": round(float(train_df[TARGET].mean()), 4),
        "test_churn_rate": round(float(test_df[TARGET].mean()), 4),
    }
    (REPORTS_DIR / "preprocess.json").write_text(json.dumps(report, indent=2))
    print(f"[preprocess] done -> {report}")


if __name__ == "__main__":
    main()
