"""Pruebas unitarias para el Pipeline de Preprocesamiento NLP."""

import pytest
from src.nlp.pipeline import PreprocesadorTexto


def test_paso_a_paso_genera_todas_las_etapas(preprocesador: PreprocesadorTexto):
    """Verifica que el método paso_a_paso construya la traza completa (CU3)."""
    texto = "¡La comida estuvo riquísimooo pero muy cara!"
    traza = preprocesador.paso_a_paso(texto)

    assert traza.original == texto
    assert isinstance(traza.normalizado, str)
    assert isinstance(traza.tokens, list)
    assert isinstance(traza.sin_stopwords, list)
    assert isinstance(traza.reducidos, list)

    # Verifica reducción de repeticiones ("riquísimooo" -> "riquísimo")
    assert "riquísimooo" not in traza.normalizado

    # Verifica que 'pero' esté en los tokens limpios
    assert "pero" in traza.sin_stopwords


def test_transform_scikit_learn_compatible(preprocesador: PreprocesadorTexto):
    """Verifica que transform retorne una lista de cadenas apta para TfidfVectorizer."""
    textos = [
        "Excelente restaurante mexicano.",
        "Pésimo servicio y comida fría.",
    ]
    limpios = preprocesador.transform(textos)

    assert len(limpios) == 2
    assert all(isinstance(t, str) for t in limpios)

