import json
import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.evaluate import (
    check_threshold,
    compute_metrics,
    evaluate_model,
    save_metrics,
)


@pytest.fixture
def fitted_model_and_data():
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    y_s = pd.Series(y)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df[:160], y_s[:160])
    return model, X_df[160:], y_s[160:]


def test_compute_metrics_keys():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = [0, 1, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics


def test_evaluate_model_returns_dict(fitted_model_and_data):
    model, X_test, y_test = fitted_model_and_data
    metrics = evaluate_model(model, X_test, y_test)
    assert isinstance(metrics, dict)
    assert metrics["accuracy"] > 0


def test_save_metrics(fitted_model_and_data, tmp_path):
    model, X_test, y_test = fitted_model_and_data
    metrics = evaluate_model(model, X_test, y_test)
    out = str(tmp_path / "metrics" / "metrics.json")
    save_metrics(metrics, out)
    assert os.path.exists(out)
    with open(out) as f:
        loaded = json.load(f)
    assert "accuracy" in loaded


def test_check_threshold_pass(fitted_model_and_data):
    model, X_test, y_test = fitted_model_and_data
    metrics = evaluate_model(model, X_test, y_test)
    assert check_threshold(metrics, metric="accuracy", threshold=0.5)


def test_check_threshold_fail():
    metrics = {"accuracy": 0.3}
    assert not check_threshold(metrics, metric="accuracy", threshold=0.9)
