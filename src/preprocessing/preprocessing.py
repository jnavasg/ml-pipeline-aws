"""
Preprocessing stage — transforma los datos crudos en features listos para entrenar.

Tareas típicas:
  - Limpieza (nulos, duplicados, outliers)
  - Encoding de variables categóricas
  - Escalado / normalización
  - Feature engineering
  - Split train/validation/test

Los datos procesados se guardan en data/processed/.
"""

import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_raw(input_path: str) -> pd.DataFrame:
    """Carga los datos crudos desde data/raw/."""
    return pd.read_parquet(input_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con nulos y duplicados.

    TODO:
      - Definir estrategia por columna (imputar vs. eliminar).
      - Tratar outliers con IQR o z-score según el dominio.
    """
    df = df.drop_duplicates()
    df = df.dropna()
    logger.info("Tras limpieza: %s filas", len(df))
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """
    One-hot encoding de columnas categóricas.

    TODO:
      - Evaluar OrdinalEncoder para variables con orden natural.
      - Persistir el encoder para reproducirlo en inferencia.
    """
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def scale_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Escala features numéricas ajustando solo sobre train.

    TODO: persistir el scaler (joblib) para usarlo en el endpoint de inferencia.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_val_scaled, X_test_scaled


def split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Divide en train / validation / test respetando la proporción indicada."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), random_state=random_state
    )
    logger.info(
        "Split: train=%s, val=%s, test=%s", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """Guarda el dataset procesado en data/processed/."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Datos procesados guardados en %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # TODO: conectar con la salida real de ingestion
    df = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, None, 4.0],
            "feature_2": ["a", "b", "a", "b"],
            "target": [0, 1, 0, 1],
        }
    )
    df = clean(df)
    df = encode_categoricals(df, cat_cols=["feature_2"])
    save_processed(df, "data/processed/sample_processed.parquet")
