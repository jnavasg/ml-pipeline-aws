"""
Training stage — entrena un RandomForestClassifier con los datos de Titanic.

Hiperparámetros elegidos tras una búsqueda manual básica; para tuning
automático considera GridSearchCV o SageMaker HPO.

Entrada : data/processed/X_train.parquet, y_train.parquet
Salida  : models/titanic_rf.pkl
"""

import logging
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/titanic_rf.pkl"

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


def load_processed(processed_dir: str = PROCESSED_DIR) -> tuple:
    """Carga X_train e y_train desde data/processed/."""
    X_train = pd.read_parquet(f"{processed_dir}/X_train.parquet")
    y_train = pd.read_parquet(f"{processed_dir}/y_train.parquet").squeeze()
    logger.info("Train cargado: %s filas, %s features", *X_train.shape)
    return X_train, y_train


def build_model(params: dict | None = None) -> RandomForestClassifier:
    """Instancia el RandomForestClassifier con los hiperparámetros dados."""
    params = params or DEFAULT_PARAMS
    model = RandomForestClassifier(**params)
    logger.info("Modelo: %s", model)
    return model


def train(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Entrena el modelo e imprime la importancia de features."""
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    logger.info(
        "Feature importances:\n%s",
        importances.sort_values(ascending=False).to_string(),
    )
    return model


def save_model(model: RandomForestClassifier, output_path: str = MODEL_PATH) -> None:
    """Serializa el modelo con joblib en models/."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Modelo guardado en %s", output_path)


def load_model(model_path: str = MODEL_PATH) -> RandomForestClassifier:
    """Carga un modelo serializado desde disco."""
    model = joblib.load(model_path)
    logger.info("Modelo cargado desde %s", model_path)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X_train, y_train = load_processed()
    model = build_model()
    model = train(model, X_train, y_train)
    save_model(model)
