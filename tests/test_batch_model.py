"""Pruebas de la inferencia por lote (src/models/inferencia.py)."""

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH
from src.models.inferencia import clasificar_lote

TEXTOS = ["muy rico todo", "pesimo servicio", "excelente comida", "no me gusto", "rico pero lento"]


@pytest.fixture(scope="module")
def pipeline():
    """Modelo pequeño de prueba: rápido y no depende de haber entrenado."""
    X = ["muy rico todo", "pesimo servicio", "excelente comida", "horrible y caro",
         "rico y barato", "malo malo", "me encanto", "no me gusto"]
    y = ["positivo", "negativo", "positivo", "negativo",
         "positivo", "negativo", "positivo", "negativo"]
    return Pipeline([("tfidf", TfidfVectorizer()), ("model", LogisticRegression())]).fit(X, y)


def test_lote_igual_a_fila_por_fila(pipeline):
    lote = clasificar_lote(pipeline, TEXTOS)
    uno_por_uno = np.array([pipeline.predict_proba([t])[0] for t in TEXTOS])  # [p_neg, p_pos]
    assert np.allclose(lote["p_neg"].astype(float), uno_por_uno[:, 0])
    assert np.allclose(lote["p_pos"].astype(float), uno_por_uno[:, 1])
    assert list(lote["sentimiento"]) == list(pipeline.predict(TEXTOS))


def test_probabilidades_suman_uno(pipeline):
    lote = clasificar_lote(pipeline, TEXTOS)
    assert np.allclose(lote["p_pos"].astype(float) + lote["p_neg"].astype(float), 1.0)


def test_certeza_entre_05_y_1(pipeline):
    certeza = clasificar_lote(pipeline, TEXTOS)["certeza"].astype(float)
    assert ((certeza >= 0.5) & (certeza <= 1.0)).all()


def test_lote_con_filas_vacias(pipeline):
    textos = ["muy rico todo", "", None, "   ", float("nan"), "pesimo servicio"]
    lote = clasificar_lote(pipeline, textos)
    assert len(lote) == len(textos)                      # conserva una fila por texto
    assert lote.loc[[1, 2, 3, 4], "sentimiento"].isna().all()
    assert lote.loc[[0, 5], "sentimiento"].notna().all()


def test_lote_vacio(pipeline):
    lote = clasificar_lote(pipeline, [])
    assert len(lote) == 0
    assert list(lote.columns) == ["sentimiento", "certeza", "p_pos", "p_neg"]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Falta entrenar: python -m src.models.train")
def test_clases_del_modelo_real():
    modelo = joblib.load(MODEL_PATH)
    assert list(modelo.classes_) == ["negativo", "positivo"]