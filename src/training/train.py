"""
Training stage — entrena el modelo con los datos procesados.

Responsabilidades:
  - Instanciar y configurar el modelo (hiperparámetros).
  - Ejecutar el entrenamiento.
  - Loggear métricas de entrenamiento.
  - Persistir el modelo entrenado en models/.

Para experimentos a escala considera integrar MLflow o AWS SageMaker Experiments.
"""

import logging
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # TODO: cambia según tu tarea

logger = logging.getLogger(__name__)

# -- Hiperparámetros por defecto -------------------------------------------------
# TODO: externalizar en un config.yaml o recibirlos via CLI / SageMaker HPO
DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42,
    "n_jobs": -1,
}
# -------------------------------------------------------------------------------


def build_model(params: dict | None = None) -> RandomForestClassifier:
    """
    Instancia el modelo con los hiperparámetros dados.

    Args:
        params: Dict de hiperparámetros. Si None usa DEFAULT_PARAMS.

    Returns:
        Modelo sin entrenar.
    """
    params = params or DEFAULT_PARAMS
    model = RandomForestClassifier(**params)
    logger.info("Modelo creado: %s", model)
    return model


def train(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Entrena el modelo.

    TODO:
      - Añadir callbacks o early stopping si cambias a un modelo gradient boosting.
      - Integrar MLflow: mlflow.sklearn.log_model(model, "model")
    """
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado.")
    return model


def save_model(model: RandomForestClassifier, output_path: str) -> None:
    """
    Serializa el modelo en models/ usando joblib.

    Args:
        model: Modelo entrenado.
        output_path: Ruta de destino (ej. models/random_forest_v1.pkl).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Modelo guardado en %s", output_path)


def load_model(model_path: str) -> RandomForestClassifier:
    """Carga un modelo serializado desde disco o S3 (previa descarga)."""
    model = joblib.load(model_path)
    logger.info("Modelo cargado desde %s", model_path)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # TODO: reemplazar con los datos reales de preprocessing
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y_s = pd.Series(y)

    model = build_model()
    model = train(model, X_df, y_s)
    save_model(model, "models/random_forest_v1.pkl")
