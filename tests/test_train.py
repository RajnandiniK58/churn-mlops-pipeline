import joblib
import pandas as pd
import pytest
import yaml

from src.config import MODEL_PATH, PARAMS_PATH, TARGET, TEST_PATH, TRAIN_PATH
from src.models.train import (
    compute_scale_pos_weight, evaluate, load_xy, train_model,
)


@pytest.fixture(scope="module")
def trained():
    p = yaml.safe_load(PARAMS_PATH.read_text())
    X, y = load_xy(TRAIN_PATH)
    pipeline, info = train_model(X, y, p["train"], p["data"]["random_state"])
    return pipeline, info


def test_scale_pos_weight_matches_class_ratio():
    _, y = load_xy(TRAIN_PATH)
    assert 2.5 < compute_scale_pos_weight(y) < 3.0


def test_early_stopping_uses_fewer_trees_than_max(trained):
    _, info = trained
    assert 1 <= info["n_trees"] < 1000


def test_pipeline_predicts_from_raw_records(trained):
    pipeline, _ = trained
    X_test, _ = load_xy(TEST_PATH)
    proba = pipeline.predict_proba(X_test.head(5))[:, 1]
    assert proba.shape == (5,) and ((proba >= 0) & (proba <= 1)).all()


def test_model_beats_random_on_test_set(trained):
    pipeline, _ = trained
    X_test, y_test = load_xy(TEST_PATH)
    assert evaluate(pipeline, X_test, y_test)["roc_auc"] > 0.78


def test_unseen_category_still_predicts(trained):
    pipeline, _ = trained
    X_test, _ = load_xy(TEST_PATH)
    row = X_test.head(1).copy()
    row["PaymentMethod"] = "Crypto"
    assert len(pipeline.predict_proba(row)) == 1


def test_saved_model_reloads_and_matches():
    # requires `python -m src.models.train` (or dvc repro) to have run
    saved = joblib.load(MODEL_PATH)
    X_test, y_test = load_xy(TEST_PATH)
    assert evaluate(saved, X_test, y_test)["roc_auc"] > 0.78
