"""Descarga el dataset Rest-Mex 2025 (reseñas turísticas de México en español).

Fuente original: Rest-Mex 2025, IberLEF (organizado por CIMAT).
Copia usada: https://github.com/lahernandezguevara/scala-spark-reviews
"""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/lahernandezguevara/scala-spark-reviews/main/Rest-Mex_2025_train.csv"

# Carpeta data/raw, calculada desde la ubicación de este archivo
CARPETA_RAW = Path(__file__).resolve().parent / "raw"
ARCHIVO_DESTINO = CARPETA_RAW / "Rest-Mex_2025_train.csv"


def descargar():
    CARPETA_RAW.mkdir(parents=True, exist_ok=True)

    if ARCHIVO_DESTINO.exists():
        print(f"El archivo ya existe: {ARCHIVO_DESTINO}")
        return

    print("Descargando dataset (aprox. 89 MB), espera un momento...")
    urllib.request.urlretrieve(URL, ARCHIVO_DESTINO)
    print(f"Descarga completa: {ARCHIVO_DESTINO}")


if __name__ == "__main__":
    descargar()