"""Carga, limpieza y división de los datos para el modelado."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, PARAMS


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV de restaurantes y quita textos vacíos y duplicados."""
    ruta = BASE_DIR / PARAMS["data"]["raw_path"]
    df = pd.read_csv(ruta)
    df = df.dropna(subset=["texto", "sentimiento"])
    df = df[df["texto"].str.strip() != ""]
    df = df.drop_duplicates(subset="texto")
    return df.reset_index(drop=True)


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """División estratificada train/test con semilla fija (ambos de params.yaml)."""
    train, test = train_test_split(
        df,
        test_size=PARAMS["data"]["test_size"],
        stratify=df["sentimiento"],
        random_state=PARAMS["seed"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)