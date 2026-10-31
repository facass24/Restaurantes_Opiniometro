"""Pruebas para el procesamiento y exportación de archivos CSV (CU2)."""

import io
from fastapi.testclient import TestClient


def test_prediccion_csv_exitoso(client: TestClient):
    """Verifica que un archivo CSV con reseñas sea procesado y consolidado adecuadamente."""
    contenido_csv = (
        "id,reseña\n"
        "1,La comida estuvo deliciosa y el ambiente muy agradable.\n"
        "2,Pésimo servicio y el mesero fue muy grosero.\n"
        "3,Excelente lugar para compartir en familia.\n"
    )

    archivos = {
        "file": ("resenas.csv", io.BytesIO(contenido_csv.encode("utf-8")), "text/csv")
    }

    response = client.post("/api/v1/predict/csv", files=archivos)

    assert response.status_code == 200
    data = response.json()
    assert "resumen" in data
    assert data["resumen"]["total_procesadas"] == 3
    assert data["resumen"]["positivas"] >= 1
    assert data["resumen"]["negativas"] >= 1
    assert len(data["resultados"]) == 3


def test_prediccion_csv_archivo_no_csv_falla_con_400(client: TestClient):
    """Verifica que archivos que no terminen en .csv sean rechazados con 400 Bad Request."""
    archivos = {
        "file": ("datos.txt", io.BytesIO(b"texto de prueba"), "text/plain")
    }
    response = client.post("/api/v1/predict/csv", files=archivos)

    assert response.status_code == 400


def test_prediccion_csv_archivo_vacio_falla_con_400(client: TestClient):
    """Verifica que un CSV vacío sea rechazado con 400 Bad Request."""
    archivos = {
        "file": ("vacio.csv", io.BytesIO(b""), "text/csv")
    }
    response = client.post("/api/v1/predict/csv", files=archivos)

    assert response.status_code == 400


def test_exportar_csv_con_predicciones(client: TestClient):
    """Verifica la exportación del archivo CSV procesado con las nuevas columnas añadidas."""
    contenido_csv = (
        "id,reseña\n"
        "1,Comida deliciosa.\n"
        "2,Servicio horrible.\n"
    )

    archivos = {
        "file": ("opiniones.csv", io.BytesIO(contenido_csv.encode("utf-8")), "text/csv")
    }

    response = client.post("/api/v1/predict/csv/export", files=archivos)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert b"prediccion_sentimiento" in response.content
    assert b"probabilidad" in response.content

