"""Train the XGBoost churn model (preprocessing + model saved as ONE pipeline)."""
import json

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    MODEL_PATH, MODELS_DIR, PARAMS_PATH, REPORTS_DIR, TARGET, TEST_PATH, TRAIN_PATH,
)
from src.features.preprocess import build_preprocessor


def load_xy(path):
    df = pd.read_csv(path)
    return df.drop(columns=[TARGET]), df[TARGET]


def compute_scale_pos_weight(y: pd.Series) -> float:
    """(# negatives) / (# positives): tells XGBoost how much to up-weight churners."""
    return float((y == 0).sum() / (y == 1).sum())


def make_classifier(p: dict, n_estimators: int, scale_pos_weight: float,
                    random_state: int, early_stopping_rounds=None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        min_child_weight=p["min_child_weight"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
    )


def find_best_n_estimators(X, y, p: dict, spw: float, random_state: int) -> int:
    """PHASE 1 - early stopping on a validation slice of the TRAINING data.
    Tells us how many trees are enough before the model starts overfitting."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=p["val_size"], stratify=y, random_state=random_state
    )
    pre = build_preprocessor()
    Xf = pre.fit_transform(X_fit)          # fit on the fitting slice only
    Xv = pre.transform(X_val)

    clf = make_classifier(p, p["n_estimators_max"], spw, random_state,
                          early_stopping_rounds=p["early_stopping_rounds"])
    clf.fit(Xf, y_fit, eval_set=[(Xv, y_val)], verbose=False)
    return int(clf.best_iteration) + 1


def train_model(X, y, p: dict, random_state: int):
    """PHASE 2 - refit ONE clean Pipeline on ALL training data with the
    tree count found in phase 1."""
    spw = compute_scale_pos_weight(y)
    best_n = find_best_n_estimators(X, y, p, spw, random_state)
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", make_classifier(p, best_n, spw, random_state)),
    ])
    pipeline.fit(X, y)
    return pipeline, {"n_trees": best_n, "scale_pos_weight": round(spw, 3)}


def evaluate(pipeline, X, y, threshold: float = 0.5) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
    }


def feature_importance(pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    return (pd.DataFrame({"feature": model.feature_names_in_,
                          "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    p, rs = params["train"], params["data"]["random_state"]

    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)

    pipeline, info = train_model(X_train, y_train, p, rs)
    print(f"[train] trees used: {info['n_trees']} | scale_pos_weight: {info['scale_pos_weight']}")

    baseline_acc = round(float((y_test == 0).mean()), 4)   # "always predict no churn"
    metrics = {**info, "baseline_accuracy": baseline_acc,
               **{f"test_{k}": v for k, v in evaluate(pipeline, X_test, y_test).items()}}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    (REPORTS_DIR / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    fi = feature_importance(pipeline)
    fi.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    print(f"[train] metrics: {metrics}")
    print("[train] top 10 features:\n", fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
