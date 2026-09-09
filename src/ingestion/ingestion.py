"""
Ingestion stage — carga el dataset de Titanic y lo persiste en data/raw/.

En producción (AWS) se reemplazaría load_titanic() por load_from_s3(),
apuntando al bucket que almacena el CSV original.
"""

import logging
import os

import boto3
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

RAW_PATH = "data/raw/titanic.csv"


def load_titanic() -> pd.DataFrame:
    """
    Descarga el dataset de Titanic via seaborn (fuente: GitHub/mwaskom).
    Devuelve las 891 filas originales con todas las columnas crudas.
    """
    df = sns.load_dataset("titanic")
    logger.info("Titanic cargado: %s filas, %s columnas", *df.shape)
    return df


def load_from_s3(bucket: str, key: str, local_path: str) -> str:
    """
    Descarga un objeto desde S3 y lo guarda localmente.
    Configura credenciales via variables de entorno o IAM Role (en EC2/ECS/Lambda).
    """
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)
    logger.info("Descargado s3://%s/%s -> %s", bucket, key, local_path)
    return local_path


def load_from_csv(file_path: str) -> pd.DataFrame:
    """Carga datos desde un CSV local — útil en desarrollo o tras descargar de S3."""
    df = pd.read_csv(file_path)
    logger.info("Cargado %s filas desde %s", len(df), file_path)
    return df


def save_raw(df: pd.DataFrame, output_path: str) -> None:
    """
    Persiste el DataFrame crudo en data/raw/ como CSV.
    Se guarda sin transformaciones para mantener trazabilidad del dato original.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Datos crudos guardados en %s (%s filas)", output_path, len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_titanic()
    save_raw(df, RAW_PATH)
