"""
Preprocessing stage — transforma el CSV crudo de Titanic en features listos para entrenar.

Columnas de entrada (data/raw/titanic.csv):
  survived, pclass, sex, age, sibsp, parch, fare, embarked,
  class, who, adult_male, deck, embark_town, alive, alone

Columnas eliminadas:
  - class, embark_town : redundantes con pclass / embarked
  - who, adult_male    : derivables de sex + age
  - deck               : >75 % nulos
  - alive              : fuga del target (es survived en texto)
  - alone              : derivable de sibsp + parch

Columnas de salida (data/processed/):
  pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S
  → target: survived
"""

import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/titanic.csv"
PROCESSED_DIR = "data/processed"
TARGET_COL = "survived"

_DROP_COLS = ["class", "who", "adult_male", "deck", "embark_town", "alive", "alone"]


def load_raw(input_path: str = RAW_PATH) -> pd.DataFrame:
    """Carga el CSV crudo producido por ingestion."""
    return pd.read_csv(input_path)


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza específica para Titanic:
      - Elimina columnas redundantes / con alta tasa de nulos.
      - Imputa 'age' con la mediana del conjunto completo.
      - Imputa los 2 nulos de 'embarked' con la moda.
      - Elimina duplicados exactos.
    """
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])
    df = df.copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df = df.drop_duplicates()
    logger.info("Tras limpieza: %s filas, columnas: %s", len(df), list(df.columns))
    return df


def encode_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding para Titanic:
      - 'sex'      → binario (0 = female, 1 = male).
      - 'embarked' → one-hot eliminando la categoría de referencia 'C'.
                     Genera: embarked_Q, embarked_S.

    No se aplica StandardScaler porque RandomForest no lo necesita.
    Si cambias a regresión logística o SVM, añade el escalado aquí.
    """
    df = df.copy()
    df["sex"] = (df["sex"] == "male").astype(int)
    embarked_dummies = (
        pd.get_dummies(df["embarked"], prefix="embarked", drop_first=True)
        .astype(int)
    )
    df = pd.concat([df.drop(columns=["embarked"]), embarked_dummies], axis=1)
    logger.info("Tras encoding: %s columnas", df.shape[1])
    return df


def split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Divide en train / test con stratify sobre el target para preservar
    la proporción de supervivientes (~38 %) en ambos conjuntos.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Split — train: %s, test: %s", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: str = PROCESSED_DIR,
) -> None:
    """
    Guarda los cuatro conjuntos como parquet en data/processed/.
    Archivos: X_train, X_test, y_train, y_test.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_parquet(f"{out_dir}/X_train.parquet", index=False)
    X_test.to_parquet(f"{out_dir}/X_test.parquet", index=False)
    y_train.to_frame().to_parquet(f"{out_dir}/y_train.parquet", index=False)
    y_test.to_frame().to_parquet(f"{out_dir}/y_test.parquet", index=False)
    logger.info(
        "Procesados guardados en %s — X_train %s, X_test %s",
        out_dir, X_train.shape, X_test.shape,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_raw()
    df = clean_titanic(df)
    df = encode_titanic(df)
    X_train, X_test, y_train, y_test = split(df)
    save_processed(X_train, X_test, y_train, y_test)
