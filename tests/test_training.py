import os

import pandas as pd
import pytest

from src.training.train import build_model, load_model, save_model, train

# Features resultantes del preprocessing de Titanic
TITANIC_FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked_Q", "embarked_S"]


@pytest.fixture
def titanic_train_data():
    """Dataset sintético con la misma estructura que el Titanic preprocesado."""
    n = 80
    import numpy as np
    rng = np.random.default_rng(42)
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
    return X, y


def test_build_model_default_params():
    model = build_model()
    assert model.n_estimators == 200
    assert model.max_depth == 8


def test_build_model_custom_params():
    model = build_model({"n_estimators": 10, "random_state": 0, "n_jobs": 1})
    assert model.n_estimators == 10


def test_train_fits_model(titanic_train_data):
    X, y = titanic_train_data
    model = build_model({"n_estimators": 5, "random_state": 0, "n_jobs": 1})
    fitted = train(model, X, y)
    assert hasattr(fitted, "estimators_")
    assert len(fitted.feature_importances_) == len(TITANIC_FEATURES)


def test_save_and_load_model_consistency(titanic_train_data, tmp_path):
    X, y = titanic_train_data
    model = train(
        build_model({"n_estimators": 5, "random_state": 0, "n_jobs": 1}), X, y
    )
    path = str(tmp_path / "models" / "titanic_rf.pkl")
    save_model(model, path)
    assert os.path.exists(path)
    loaded = load_model(path)
    assert (model.predict(X) == loaded.predict(X)).all()
