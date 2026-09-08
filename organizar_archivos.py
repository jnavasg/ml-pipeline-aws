# -*- coding: utf-8 -*-
"""Organiza archivos de una carpeta por extension, sin entrar en subcarpetas."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CATEGORIAS = {
    "Fotos": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tiff", ".svg", ".raw"},
    "Documentos": {".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".md"},
    "Videos": {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v"},
    "Audio": {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma"},
    "Comprimidos": {".zip", ".rar", ".7z", ".tar", ".gz"},
    "Instaladores": {".exe", ".msi", ".bat"},
}

CATEGORIA_OTROS = "Otros"
NOMBRE_LOG = "organizar_archivos_log.txt"
EXTENSION_A_CATEGORIA = {
    extension: categoria
    for categoria, extensiones in CATEGORIAS.items()
    for extension in extensiones
}


@dataclass(frozen=True)
class Movimiento:
    """Representa una operacion planeada antes de mover archivos."""

    origen: Path
    categoria: str


def obtener_categoria(extension: str) -> str:
    """Obtiene la categoria en tiempo constante mediante un indice."""
    return EXTENSION_A_CATEGORIA.get(extension.lower(), CATEGORIA_OTROS)


def nombre_disponible(destino: Path) -> Path:
    """Devuelve un nombre libre sin sobrescribir un archivo existente."""
    if not destino.exists():
        return destino

    contador = 1
    while True:
        candidato = destino.with_name(
            f"{destino.stem} ({contador}){destino.suffix}"
        )
        if not candidato.exists():
            return candidato
        contador += 1


def escanear_carpeta(carpeta: Path) -> list[Movimiento]:
    """Crea un plan con archivos regulares del primer nivel solamente."""
    plan = []
    for item in carpeta.iterdir():
        if (
            not item.is_file()
            or item.is_symlink()
            or item.name.startswith(".")
            or item.name == NOMBRE_LOG
        ):
            continue

        plan.append(Movimiento(item, obtener_categoria(item.suffix)))

    return sorted(plan, key=lambda movimiento: movimiento.origen.name.casefold())


def mostrar_simulacion(plan: list[Movimiento]) -> None:
    if not plan:
        print("\nNo se encontraron archivos para organizar en esa carpeta.")
        return

    resumen = Counter(movimiento.categoria for movimiento in plan)
    print(f"\nSe encontraron {len(plan)} archivo(s). Asi quedarian organizados:\n")
    for categoria, cantidad in sorted(resumen.items()):
        print(f"  [carpeta] {categoria}: {cantidad} archivo(s)")

    print("\nDetalle:")
    for movimiento in plan:
        print(
            f"  {movimiento.origen.name} -> "
            f"{movimiento.categoria}/{movimiento.origen.name}"
        )


def ejecutar_movimientos(carpeta: Path, plan: list[Movimiento]) -> None:
    log_path = carpeta / NOMBRE_LOG
    movidos = 0
    errores = 0
    carpetas_creadas: dict[str, Path] = {}

    with log_path.open("a", encoding="utf-8") as log:
        for movimiento in plan:
            carpeta_destino = carpetas_creadas.setdefault(
                movimiento.categoria, carpeta / movimiento.categoria
            )
            carpeta_destino.mkdir(exist_ok=True)
            destino = nombre_disponible(
                carpeta_destino / movimiento.origen.name
            )

            try:
                shutil.move(movimiento.origen, destino)
            except OSError as error:
                errores += 1
                linea = f"ERROR moviendo {movimiento.origen.name}: {error}"
                print(f"  [!] {linea}")
            else:
                movidos += 1
                linea = (
                    f"Movido: {movimiento.origen.name} -> "
                    f"{movimiento.categoria}/{destino.name}"
                )
                print(f"  [ok] {linea}")

            log.write(linea + "\n")

    print(f"\nListo. Se movieron {movidos} de {len(plan)} archivo(s).")
    if errores:
        print(f"No se pudieron mover {errores} archivo(s).")
    print(f"Registro guardado en: {log_path}")


def main() -> None:
    print("=" * 60)
    print("ORGANIZADOR DE ARCHIVOS POR TIPO")
    print("=" * 60)

    ruta_texto = input(
        "\nEscribe la ruta de la carpeta a organizar\n"
        "(ejemplo: C:\\Users\\TuUsuario\\Downloads): "
    ).strip().strip('"')
    carpeta = Path(ruta_texto).expanduser()

    if not carpeta.is_dir():
        print(f"\nLa ruta '{ruta_texto}' no existe o no es una carpeta valida.")
        return

    plan = escanear_carpeta(carpeta)
    mostrar_simulacion(plan)
    if not plan:
        return

    respuesta = input(
        "\nQuieres proceder a mover estos archivos ahora? (s/n): "
    ).strip().lower()
    if respuesta != "s":
        print("\nOperacion cancelada. No se movio ningun archivo.")
        return

    ejecutar_movimientos(carpeta, plan)


if __name__ == "__main__":
    main()
