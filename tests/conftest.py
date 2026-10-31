"""Configuración y fixtures compartidos para pytest."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.nlp.stopwords import FiltroStopWords
from src.nlp.pipeline import PreprocesadorTexto


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Fixture que provee un cliente HTTP de prueba para la API FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def filtro_stopwords() -> FiltroStopWords:
    """Fixture para pruebas unitarias del filtro de stop words."""
    return FiltroStopWords()


@pytest.fixture
def preprocesador() -> PreprocesadorTexto:
    """Fixture para pruebas unitarias del preprocesador completo."""
    return PreprocesadorTexto()

