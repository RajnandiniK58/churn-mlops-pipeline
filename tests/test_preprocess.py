import pandas as pd
import pytest

from src.config import RAW_DATA_PATH, TARGET, ID_COL
from src.data.validate import validate_clean
from src.features.preprocess import (
    build_preprocessor, clean_data, split_data,
)


@pytest.fixture(scope="module")
def clean_df():
    return clean_data(pd.read_csv(RAW_DATA_PATH))


def test_clean_data_passes_clean_schema(clean_df):
    validate_clean(clean_df)


def test_clean_data_has_no_nulls(clean_df):
    assert clean_df.isna().sum().sum() == 0


@pytest.fixture(scope="module")
def split(clean_df):
    df = clean_df.drop(columns=[ID_COL])
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return split_data(df, test_size=0.2, random_state=42)


def test_split_is_stratified(split):
    train, test = split
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.01


def test_split_has_no_overlap(split):
    train, test = split
    assert set(train.index).isdisjoint(test.index)


def test_preprocessor_output_is_numeric_and_complete(split):
    train, test = split
    pre = build_preprocessor()
    Xtr = pre.fit_transform(train.drop(columns=[TARGET]))
    Xte = pre.transform(test.drop(columns=[TARGET]))
    assert Xtr.isna().sum().sum() == 0 and Xte.isna().sum().sum() == 0
    assert list(Xtr.columns) == list(Xte.columns)
    assert all(pd.api.types.is_numeric_dtype(t) for t in Xtr.dtypes)


def test_scaler_is_fit_on_train_only(split):
    train, test = split
    pre = build_preprocessor()
    Xtr = pre.fit_transform(train.drop(columns=[TARGET]))
    Xte = pre.transform(test.drop(columns=[TARGET]))
    assert abs(Xtr["tenure"].mean()) < 1e-6      # train mean is exactly ~0
    assert abs(Xte["tenure"].mean()) > 1e-6      # test mean is NOT forced to 0


def test_unseen_category_does_not_crash(split):
    train, test = split
    pre = build_preprocessor()
    pre.fit(train.drop(columns=[TARGET]))
    row = test.drop(columns=[TARGET]).head(1).copy()
    row["PaymentMethod"] = "Crypto"
    row["Contract"] = "Five year"
    out = pre.transform(row)
    assert out.shape[0] == 1
