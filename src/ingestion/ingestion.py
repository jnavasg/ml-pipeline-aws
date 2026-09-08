"""
Ingestion stage — responsable de cargar datos desde la fuente de origen.

Fuentes típicas:
  - S3 (via boto3)
  - Bases de datos SQL / NoSQL
  - APIs externas
  - Archivos locales para desarrollo

Los datos crudos se guardan en data/raw/ sin transformaciones.
"""

import logging
import os

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


def load_from_s3(bucket: str, key: str, local_path: str) -> str:
    """
    Descarga un archivo desde S3 y lo guarda localmente.

    Args:
        bucket: Nombre del bucket S3.
        key: Ruta del objeto dentro del bucket.
        local_path: Ruta local donde guardar el archivo.

    Returns:
        Ruta local del archivo descargado.
    """
    # TODO: configurar credenciales via variables de entorno o IAM role
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)
    logger.info("Descargado s3://%s/%s -> %s", bucket, key, local_path)
    return local_path


def load_from_csv(file_path: str) -> pd.DataFrame:
    """
    Carga datos desde un CSV local (útil en desarrollo).

    Args:
        file_path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos crudos.
    """
    # TODO: ajustar sep, encoding y dtypes según el dataset real
    df = pd.read_csv(file_path)
    logger.info("Cargado %s filas desde %s", len(df), file_path)
    return df


def save_raw(df: pd.DataFrame, output_path: str) -> None:
    """
    Persiste el DataFrame crudo en data/raw/ en formato parquet.

    Args:
        df: DataFrame a guardar.
        output_path: Ruta de destino (recomendado .parquet).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Datos crudos guardados en %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # TODO: reemplazar con la fuente real
    sample_df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    save_raw(sample_df, "data/raw/sample.parquet")
