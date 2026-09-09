import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.evaluate import (
    check_threshold,
    compute_metrics,
    evaluate_model,
    save_metrics,
)

TITANIC_FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked_Q", "embarked_S"]


@pytest.fixture
def fitted_titanic_model():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "pclass":      rng.integers(1, 4, n),
            "sex":         rng.integers(0, 2, n),
            "age":         rng.uniform(1, 80, n),
            "sibsp":       rng.integers(0, 5, n),
            "parch":       rng.integers(0, 4, n),
            "fare":        rng.uniform(5, 300, n),
            "embarked_Q":  rng.integers(0, 2, n),
            "embarked_S":  rng.integers(0, 2, n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n), name="survived")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:160], y[:160])
    return model, X[160:].reset_index(drop=True), y[160:].reset_index(drop=True)


def test_compute_metrics_has_all_keys():
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_pred =           [0, 1, 1, 1, 0, 0]
    y_prob =           [0.1, 0.9, 0.6, 0.8, 0.3, 0.2]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    for key in ["accuracy", "f1", "precision", "recall", "roc_auc", "confusion_matrix"]:
        assert key in metrics


def test_compute_metrics_values_in_range():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred =           [0, 1, 0, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_model_returns_dict(fitted_titanic_model):
    model, X_test, y_test = fitted_titanic_model
    metrics = evaluate_model(model, X_test, y_test)
    assert isinstance(metrics, dict)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "roc_auc" in metrics


def test_save_metrics_json_serializable(fitted_titanic_model, tmp_path):
    model, X_test, y_test = fitted_titanic_model
    metrics = evaluate_model(model, X_test, y_test)
    out = str(tmp_path / "metrics" / "metrics.json")
    save_metrics(metrics, out)
    assert os.path.exists(out)
    with open(out) as f:
        loaded = json.load(f)
    assert "accuracy" in loaded
    assert "confusion_matrix" in loaded


def test_check_threshold_pass():
    assert check_threshold({"f1": 0.80}, metric="f1", threshold=0.75)


def test_check_threshold_fail():
    assert not check_threshold({"f1": 0.60}, metric="f1", threshold=0.75)
