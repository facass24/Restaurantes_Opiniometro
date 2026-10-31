"""Pruebas para consulta y comparación de métricas (CU5 y CU6)."""

from fastapi.testclient import TestClient


def test_obtener_metricas_produccion(client: TestClient):
    """CU5: Verifica la consulta de métricas del modelo activo."""
    response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    data = response.json()
    assert "modelo_produccion" in data
    assert "modelos" in data
    assert len(data["modelos"]) >= 1

    primer_modelo = data["modelos"][0]
    metricas = primer_modelo["metricas"]
    assert "accuracy" in metricas
    assert "f1_macro" in metricas
    assert "matriz_confusion" in metricas


def test_comparar_modelos(client: TestClient):
    """CU6: Verifica la entrega de la comparativa de modelos (Naive Bayes vs Reg. Logística)."""
    response = client.get("/api/v1/metrics/compare")

    assert response.status_code == 200
    modelos = response.json()
    assert isinstance(modelos, list)
    assert len(modelos) >= 2

    nombres = [m["nombre"] for m in modelos]
    assert any("Regresión Logística" in n for n in nombres)
    assert any("Naive Bayes" in n for n in nombres)

