"""
Evaluation stage — mide la calidad del modelo sobre el conjunto de test.

Métricas calculadas (clasificación por defecto):
  - Accuracy, Precision, Recall, F1
  - ROC-AUC
  - Matriz de confusión

Para regresión reemplaza con MAE, RMSE, R².
Los resultados se loggean y opcionalmente se suben a S3 / CloudWatch.
"""

import json
import logging
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred, y_prob=None) -> dict:
    """
    Calcula métricas de clasificación binaria.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones del modelo.
        y_prob: Probabilidades de clase positiva (para AUC).

    Returns:
        Diccionario con todas las métricas.

    TODO:
      - Añadir métricas de negocio (ej. revenue lift, coste de error).
      - Soportar multiclase con average='macro'/'weighted'.
    """
    metrics: dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    if "roc_auc" in metrics:
        logger.info("ROC-AUC: %.4f", metrics["roc_auc"])
    return metrics


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Pipeline completo de evaluación: predice y calcula métricas.

    Args:
        model: Modelo entrenado con interfaz sklearn.
        X_test: Features de test.
        y_test: Etiquetas de test.

    Returns:
        Diccionario de métricas.
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    return compute_metrics(y_test, y_pred, y_prob)


def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Guarda las métricas en un JSON.

    TODO:
      - Subir a S3 para trazabilidad entre experimentos.
      - Publicar en CloudWatch como métricas custom.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métricas guardadas en %s", output_path)


def check_threshold(metrics: dict, metric: str = "accuracy", threshold: float = 0.8) -> bool:
    """
    Valida si el modelo supera el umbral mínimo de calidad.

    Úsalo como gate antes de promover el modelo a producción.
    """
    value = metrics.get(metric, 0.0)
    passed = value >= threshold
    logger.info(
        "Threshold check [%s >= %.2f]: %s (valor=%.4f)",
        metric, threshold, "PASS" if passed else "FAIL", value,
    )
    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # TODO: reemplazar con datos y modelo reales
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y_s = pd.Series(y)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df[:160], y_s[:160])

    metrics = evaluate_model(model, X_df[160:], y_s[160:])
    save_metrics(metrics, "models/metrics.json")
    check_threshold(metrics, metric="accuracy", threshold=0.8)
