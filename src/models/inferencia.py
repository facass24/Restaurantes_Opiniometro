"""Inferencia por lote: clasifica muchas reseñas con una sola llamada al modelo."""

from typing import Iterable

import numpy as np
import pandas as pd

CLASE_NEGATIVA = "negativo"
CLASE_POSITIVA = "positivo"
COLUMNAS = ["sentimiento", "certeza", "p_pos", "p_neg"]


def _es_vacio(texto) -> bool:
    """True si el texto es None, NaN o solo espacios."""
    if texto is None:
        return True
    if isinstance(texto, float) and np.isnan(texto):
        return True
    return str(texto).strip() == ""


def clasificar_lote(pipeline, textos: Iterable) -> pd.DataFrame:
    """Clasifica una lista de textos con UNA sola llamada a predict_proba.

    Devuelve un DataFrame con una fila por texto, en el mismo orden:
      - sentimiento: "positivo" o "negativo" (vacío si el texto está vacío)
      - certeza: probabilidad de la clase elegida, entre 0.5 y 1
      - p_pos, p_neg: probabilidades de cada clase (suman 1)
    Las filas vacías no se envían al modelo y quedan con valores vacíos.
    """
    textos = list(textos)
    resultado = pd.DataFrame(index=range(len(textos)), columns=COLUMNAS, dtype=object)
    if not textos:
        return resultado

    validos = [i for i, t in enumerate(textos) if not _es_vacio(t)]
    if not validos:
        return resultado

    clases = list(pipeline.classes_)
    i_neg = clases.index(CLASE_NEGATIVA)
    i_pos = clases.index(CLASE_POSITIVA)

    # Una sola llamada al modelo para todos los textos válidos
    probas = pipeline.predict_proba([str(textos[i]) for i in validos])
    p_neg = probas[:, i_neg]
    p_pos = probas[:, i_pos]

    resultado.loc[validos, "p_pos"] = p_pos
    resultado.loc[validos, "p_neg"] = p_neg
    resultado.loc[validos, "sentimiento"] = np.where(p_pos > p_neg, CLASE_POSITIVA, CLASE_NEGATIVA)
    resultado.loc[validos, "certeza"] = np.maximum(p_pos, p_neg)
    return resultado