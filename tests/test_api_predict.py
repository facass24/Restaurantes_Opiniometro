"""Pruebas de predicción individual y validaciones (CU1, CU3 y CU4)."""

import pytest
from fastapi.testclient import TestClient


def test_prediccion_resena_positiva(client: TestClient):
    """Verifica que una reseña claramente positiva se clasifique adecuadamente."""
    payload = {"texto": "La comida estuvo riquísima y la atención del personal fue excelente."}
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["clase"] == "positivo"
    assert 0.5 <= data["probabilidad"] <= 1.0
    assert "preprocesamiento" in data
    assert data["preprocesamiento"]["original"] == payload["texto"]
    assert len(data["preprocesamiento"]["tokens"]) > 0


def test_prediccion_resena_negativa_con_negacion(client: TestClient):
    """Verifica que 'no me gustó' clasifique como negativo gracias al filtro de stop words."""
    payload = {"texto": "No me gustó nada la comida, estaba fría y tardó una hora."}
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["clase"] == "negativo"
    assert data["probabilidad"] >= 0.5


def test_validacion_texto_vacio_falla_con_422(client: TestClient):
    """CU4: Verifica que enviar texto vacío devuelva HTTP 422 Unprocessable Entity."""
    payload = {"texto": "   "}
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


def test_validacion_texto_muy_corto_falla_con_422(client: TestClient):
    """CU4: Verifica que textos menores a 3 caracteres sean rechazados."""
    payload = {"texto": "ok"}
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422


def test_validacion_solo_numeros_o_simbolos_falla_con_422(client: TestClient):
    """CU4: Verifica que cadenas sin caracteres alfabéticos válidos sean rechazadas."""
    payload = {"texto": "1234567890 !@#$"}
    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422

