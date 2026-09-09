"""
Orquestador del pipeline completo de Titanic:
  1. Ingestion   → data/raw/titanic.csv
  2. Preprocessing → data/processed/{X,y}_{train,test}.parquet
  3. Training    → models/titanic_rf.pkl
  4. Evaluation  → models/metrics.json

Ejecutar desde la raíz del proyecto:
  python pipeline.py
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

from src.ingestion.ingestion import load_titanic, save_raw
from src.preprocessing.preprocessing import (
    clean_titanic,
    encode_titanic,
    load_raw,
    save_processed,
    split,
)
from src.training.train import build_model, save_model, train
from src.training.train import load_processed
from src.evaluation.evaluate import (
    check_threshold,
    evaluate_model,
    load_model,
    load_test_data,
    save_metrics,
)

RAW_PATH = "data/raw/titanic.csv"
METRICS_PATH = "models/metrics.json"
MODEL_PATH = "models/titanic_rf.pkl"


def run():
    # ── 1. Ingestion ────────────────────────────────────────────────────────────
    logger.info("━━━ ETAPA 1: INGESTION ━━━")
    df_raw = load_titanic()
    save_raw(df_raw, RAW_PATH)

    # ── 2. Preprocessing ────────────────────────────────────────────────────────
    logger.info("━━━ ETAPA 2: PREPROCESSING ━━━")
    df = load_raw(RAW_PATH)
    df = clean_titanic(df)
    df = encode_titanic(df)
    X_train, X_test, y_train, y_test = split(df)
    save_processed(X_train, X_test, y_train, y_test)

    # ── 3. Training ─────────────────────────────────────────────────────────────
    logger.info("━━━ ETAPA 3: TRAINING ━━━")
    X_train_p, y_train_p = load_processed()
    model = build_model()
    model = train(model, X_train_p, y_train_p)
    save_model(model, MODEL_PATH)

    # ── 4. Evaluation ───────────────────────────────────────────────────────────
    logger.info("━━━ ETAPA 4: EVALUATION ━━━")
    model = load_model(MODEL_PATH)
    X_test_p, y_test_p = load_test_data()
    metrics = evaluate_model(model, X_test_p, y_test_p)
    save_metrics(metrics, METRICS_PATH)
    passed = check_threshold(metrics, metric="f1", threshold=0.75)

    # ── Resumen final ───────────────────────────────────────────────────────────
    logger.info("━━━ RESULTADOS FINALES ━━━")
    for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
        if k in metrics:
            logger.info("  %-12s %.4f", k.upper() + ":", metrics[k])
    cm = metrics["confusion_matrix"]
    logger.info("  Confusion matrix  [[TN=%d, FP=%d], [FN=%d, TP=%d]]",
                cm[0][0], cm[0][1], cm[1][0], cm[1][1])
    logger.info("  Gate de calidad (F1 >= 0.75): %s", "PASS ✓" if passed else "FAIL ✗")
    return metrics


if __name__ == "__main__":
    run()
