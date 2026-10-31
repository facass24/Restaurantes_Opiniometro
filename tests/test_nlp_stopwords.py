"""Pruebas unitarias para el Filtro de Stop Words en español (Parte individual Integrante C).

Evalúa la justificación lingüística del 'no', negaciones y conjunciones adversativas.
"""

import pytest
from src.nlp.stopwords import FiltroStopWords


def test_conservacion_de_negaciones_fundamentales(filtro_stopwords: FiltroStopWords):
    """Verifica que palabras clave de negación ('no', 'nunca', 'jamás', etc.) NO sean eliminadas."""
    tokens = ["el", "servicio", "no", "fue", "bueno", "nunca", "volveria", "jamas"]
    filtrados = filtro_stopwords.filtrar(tokens)

    # 'el' y 'fue' son stop words estándar y deben removerse
    assert "el" not in filtrados
    assert "fue" not in filtrados

    # Negaciones críticas para el análisis de sentimiento deben conservarse
    assert "no" in filtrados
    assert "nunca" in filtrados
    assert "jamas" in filtrados


def test_conservacion_de_conjuncion_pero(filtro_stopwords: FiltroStopWords):
    """Verifica que la conjunción 'pero' se preserve para detectar reseñas mixtas."""
    tokens = ["comida", "excelente", "pero", "la", "atencion", "pesima"]
    filtrados = filtro_stopwords.filtrar(tokens)

    assert "pero" in filtrados
    assert "la" not in filtrados
    assert "excelente" in filtrados
    assert "pesima" in filtrados


def test_eliminacion_de_stopwords_estandar(filtro_stopwords: FiltroStopWords):
    """Verifica que las palabras vacías comunes de español sí sean eliminadas."""
    tokens_vacio = ["de", "la", "que", "el", "en", "y", "a", "los", "del", "se"]
    filtrados = filtro_stopwords.filtrar(tokens_vacio)

    # Ninguna de estas palabras contiene información semántica de polaridad
    assert len(filtrados) == 0


def test_stop_words_de_dominio_opcionales():
    """Verifica que al activar el modo dominio, palabras como 'restaurante' o 'comida' se filtren."""
    filtro_estandar = FiltroStopWords(incluir_dominio=False)
    filtro_dominio = FiltroStopWords(incluir_dominio=True)

    tokens = ["el", "restaurante", "tiene", "buena", "comida"]

    # En estándar, 'restaurante' y 'comida' no son stop words
    res_estandar = filtro_estandar.filtrar(tokens)
    assert "restaurante" in res_estandar
    assert "comida" in res_estandar

    # En modo dominio, se eliminan por ser términos hiperfrecuentes no informativos
    res_dominio = filtro_dominio.filtrar(tokens)
    assert "restaurante" not in res_dominio
    assert "comida" not in res_dominio
    assert "buena" in res_dominio

