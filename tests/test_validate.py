import pandas as pd
import pandera.pandas as pa
import pytest

from src.config import RAW_DATA_PATH
from src.data.validate import validate, validate_clean


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(RAW_DATA_PATH)


def test_raw_data_passes(raw_df):
    validate(raw_df)


def test_unknown_category_fails(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Contract"] = "Five year"
    with pytest.raises(pa.errors.SchemaErrors):
        validate(bad)


def test_negative_tenure_fails(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "tenure"] = -5
    with pytest.raises(pa.errors.SchemaErrors):
        validate(bad)


def test_missing_column_fails(raw_df):
    with pytest.raises(pa.errors.SchemaErrors):
        validate(raw_df.drop(columns=["Contract"]))


def test_duplicate_ids_fail(raw_df):
    bad = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(pa.errors.SchemaErrors):
        validate(bad)


def test_raw_data_fails_clean_schema(raw_df):
    # TotalCharges is still text with blanks, so the CLEAN schema must reject it
    with pytest.raises(pa.errors.SchemaErrors):
        validate_clean(raw_df)


def test_cleaned_data_passes_clean_schema(raw_df):
    df = raw_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    validate_clean(df)
