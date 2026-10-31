"""Pruebas para el endpoint de monitoreo y salud (CU7)."""

from fastapi.testclient import TestClient


def test_consultar_salud_retorna_200_ok(client: TestClient):
    """Verifica que el endpoint /health responda con status 200 y formato válido."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["estado"] in ("ok", "degraded")
    assert "modelo_cargado" in data
    assert "version" in data
    assert "detalles" in data

