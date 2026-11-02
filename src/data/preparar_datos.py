"""Preparación del dataset de reseñas de restaurantes.

Lee el dataset crudo Rest-Mex 2025 y genera el CSV limpio que usa el modelado
(src/models/data.py), con dos columnas: texto y sentimiento.

Uso, desde la raíz del proyecto:
    python -m src.data.preparar_datos
"""

from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, PARAMS

RUTA_CRUDA = BASE_DIR / "data" / "raw" / "Rest-Mex_2025_train.csv"


def construir_texto(titulo, resena) -> str:
    """Une título y reseña. Si no hay título, devuelve solo la reseña."""
    titulo = "" if pd.isna(titulo) else str(titulo).strip()
    resena = "" if pd.isna(resena) else str(resena).strip()
    if titulo:
        return f"{titulo}. {resena}"
    return resena


def preparar_datos(ruta_cruda: Path = RUTA_CRUDA, ruta_salida: Path | None = None) -> pd.DataFrame:
    """Limpia el dataset crudo y guarda el CSV con las columnas texto y sentimiento."""
    if ruta_salida is None:
        ruta_salida = BASE_DIR / PARAMS["data"]["raw_path"]

    if not ruta_cruda.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_cruda}. Ejecuta primero: python data/descargar_datos.py"
        )

    df = pd.read_csv(ruta_cruda)
    total_inicial = len(df)

    # 1. Solo reseñas de restaurantes
    df = df[df["Type"] == "Restaurant"].copy()
    n_restaurantes = len(df)

    # 2. Quitar las de 3 estrellas (ambiguas)
    df = df[df["Polarity"] != 3].copy()

    # 3. Etiqueta: 1-2 estrellas = negativo, 4-5 = positivo
    df["sentimiento"] = df["Polarity"].apply(lambda p: "positivo" if p >= 4 else "negativo")

    # 4. Texto = título + ". " + reseña (con strip; sin título, solo la reseña)
    df["texto"] = [construir_texto(t, r) for t, r in zip(df["Title"], df["Review"])]

    # 5. Quitar textos vacíos y duplicados
    df = df[df["texto"] != ""]
    df = df.drop_duplicates(subset="texto")

    resultado = df[["texto", "sentimiento"]].reset_index(drop=True)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(ruta_salida, index=False, encoding="utf-8")

    print(f"Reseñas en el dataset crudo:  {total_inicial:,}")
    print(f"Reseñas de restaurantes:      {n_restaurantes:,}")
    print(f"Reseñas finales (limpias):    {len(resultado):,}")
    print(resultado["sentimiento"].value_counts().to_string())
    print(f"Guardado en: {ruta_salida}")

    return resultado


if __name__ == "__main__":
    preparar_datos()