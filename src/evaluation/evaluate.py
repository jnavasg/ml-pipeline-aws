"""
Evaluation stage — evalúa el modelo de Titanic en el conjunto de test.

Métricas calculadas:
  Accuracy, Precision, Recall, F1 (clase 'survived=1'), ROC-AUC
  + Matriz de confusión y classification_report completo.

Threshold de producción: F1 >= 0.75
  Un F1 bajo en la clase positiva (superviviente) penaliza tanto
  falsos negativos como falsos positivos, relevante en este dominio.

Entrada : data/processed/X_test.parquet, y_test.parquet
          models/titanic_rf.pkl
Salida  : models/metrics.json
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/titanic_rf.pkl"
METRICS_PATH = "models/metrics.json"


class _NumpyEncoder(json.JSONEncoder):
    """Convierte tipos numpy a tipos Python nativos para serialización JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_test_data(processed_dir: str = PROCESSED_DIR) -> tuple:
    """Carga X_test e y_test desde data/processed/."""
    X_test = pd.read_parquet(f"{processed_dir}/X_test.parquet")
    y_test = pd.read_parquet(f"{processed_dir}/y_test.parquet").squeeze()
    logger.info("Test cargado: %s filas", len(X_test))
    return X_test, y_test


def load_model(model_path: str = MODEL_PATH):
    """Carga el modelo serializado."""
    import joblib
    model = joblib.load(model_path)
    logger.info("Modelo cargado desde %s", model_path)
    return model


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Calcula métricas de clasificación binaria para Titanic.

    Args:
        y_true : etiquetas reales (survived 0/1).
        y_pred : predicciones del modelo.
        y_prob : probabilidad de clase positiva (survived=1) para ROC-AUC.

    Returns:
        dict con accuracy, f1, precision, recall, roc_auc,
        confusion_matrix y classification_report.
    """
    metrics: dict = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall":    round(float(recall_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["no survived", "survived"],
            output_dict=True,
        ),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)

    _log_summary(metrics)
    return metrics


def _log_summary(metrics: dict) -> None:
    logger.info("=" * 40)
    for key in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        if key in metrics:
            logger.info("  %-12s %.4f", key.upper() + ":", metrics[key])
    cm = metrics["confusion_matrix"]
    logger.info("  Confusion matrix:")
    logger.info("    TN=%-4d FP=%d", cm[0][0], cm[0][1])
    logger.info("    FN=%-4d TP=%d", cm[1][0], cm[1][1])
    logger.info("=" * 40)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Predice sobre X_test y calcula todas las métricas."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    return compute_metrics(y_test, y_pred, y_prob)


def save_metrics(metrics: dict, output_path: str = METRICS_PATH) -> None:
    """Persiste el diccionario de métricas en JSON (compatible con S3/CloudWatch)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, cls=_NumpyEncoder)
    logger.info("Métricas guardadas en %s", output_path)


def check_threshold(
    metrics: dict,
    metric: str = "f1",
    threshold: float = 0.75,
) -> bool:
    """
    Gate de calidad antes de promover el modelo a producción.
    Por defecto exige F1 >= 0.75 sobre la clase 'survived'.
    """
    value = metrics.get(metric, 0.0)
    passed = value >= threshold
    status = "PASS ✓" if passed else "FAIL ✗"
    logger.info("Threshold [%s=%.4f >= %.2f]: %s", metric, value, threshold, status)
    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = load_model()
    X_test, y_test = load_test_data()
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics)
    check_threshold(metrics, metric="f1", threshold=0.75)
