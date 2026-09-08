import os

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.training.train import build_model, load_model, save_model, train


@pytest.fixture
def training_data():
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    return (
        pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]),
        pd.Series(y),
    )


def test_build_model():
    model = build_model({"n_estimators": 5, "random_state": 0, "n_jobs": 1})
    assert model.n_estimators == 5


def test_train_returns_fitted_model(training_data):
    X, y = training_data
    model = build_model({"n_estimators": 5, "random_state": 0, "n_jobs": 1})
    fitted = train(model, X, y)
    assert hasattr(fitted, "estimators_")


def test_save_and_load_model(training_data, tmp_path):
    X, y = training_data
    model = train(
        build_model({"n_estimators": 5, "random_state": 0, "n_jobs": 1}), X, y
    )
    path = str(tmp_path / "models" / "test_model.pkl")
    save_model(model, path)
    assert os.path.exists(path)
    loaded = load_model(path)
    preds_original = model.predict(X)
    preds_loaded = loaded.predict(X)
    assert (preds_original == preds_loaded).all()
