"""Pandera schemas describing what "good" Telco churn data looks like."""
import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

YES_NO = ["Yes", "No"]
YES_NO_NOINTERNET = ["Yes", "No", "No internet service"]


def _blank_or_number(s: pd.Series) -> pd.Series:
    """True where a text value is blank/whitespace OR parses as a number."""
    stripped = s.astype(str).str.strip()
    return (stripped == "") | pd.to_numeric(stripped, errors="coerce").notna()


def _no_duplicate_ids(df: pd.DataFrame) -> bool:
    return not df["customerID"].duplicated().any()


def _churn_rate_sane(df: pd.DataFrame) -> bool:
    rate = (df["Churn"] == "Yes").mean()
    return 0.10 <= rate <= 0.50


SHARED_COLUMNS = {
    "customerID": Column(str, nullable=False),
    "gender": Column(str, Check.isin(["Male", "Female"])),
    "SeniorCitizen": Column(int, Check.isin([0, 1])),
    "Partner": Column(str, Check.isin(YES_NO)),
    "Dependents": Column(str, Check.isin(YES_NO)),
    "tenure": Column(int, Check.in_range(0, 100)),
    "PhoneService": Column(str, Check.isin(YES_NO)),
    "MultipleLines": Column(str, Check.isin(YES_NO + ["No phone service"])),
    "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
    "OnlineSecurity": Column(str, Check.isin(YES_NO_NOINTERNET)),
    "OnlineBackup": Column(str, Check.isin(YES_NO_NOINTERNET)),
    "DeviceProtection": Column(str, Check.isin(YES_NO_NOINTERNET)),
    "TechSupport": Column(str, Check.isin(YES_NO_NOINTERNET)),
    "StreamingTV": Column(str, Check.isin(YES_NO_NOINTERNET)),
    "StreamingMovies": Column(str, Check.isin(YES_NO_NOINTERNET)),
    "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
    "PaperlessBilling": Column(str, Check.isin(YES_NO)),
    "PaymentMethod": Column(
        str,
        Check.isin([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]),
    ),
    "MonthlyCharges": Column(float, Check.in_range(0, 500)),
    "Churn": Column(str, Check.isin(YES_NO)),
}

TABLE_CHECKS = [
    Check(_no_duplicate_ids, error="customerID contains duplicates"),
    Check(_churn_rate_sane, error="Churn rate outside 10%-50% (label problem?)"),
    Check(lambda df: len(df) >= 1000, error="Fewer than 1000 rows"),
]

# 1) RAW: the file exactly as it arrives. TotalCharges is still TEXT here
#    because the source has blank strings in it.
RAW_SCHEMA = DataFrameSchema(
    {
        **SHARED_COLUMNS,
        "TotalCharges": Column(
            str,
            Check(_blank_or_number, error="TotalCharges must be a number or blank"),
        ),
    },
    checks=TABLE_CHECKS,
    strict=True,  # unexpected extra columns => fail
)

# 2) CLEAN: after YOUR cleaning step (Part 3). TotalCharges must now be a
#    non-negative float with no nulls anywhere.
CLEAN_SCHEMA = DataFrameSchema(
    {
        **SHARED_COLUMNS,
        "TotalCharges": Column(float, Check.ge(0), nullable=False),
    },
    checks=TABLE_CHECKS,
    strict=True,
)
