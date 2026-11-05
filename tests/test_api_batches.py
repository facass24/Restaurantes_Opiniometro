"""Suite de pruebas automatizadas para los endpoints de archivos y lotes (Batches).

Verifica:
- POST /api/v1/predict/file (CSV y Excel .xlsx)
- GET /api/v1/batches/{id}
- GET /api/v1/batches/{id}/download
- GET /api/v1/batches/{id}/alerts
"""

import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def test_subir_archivo_csv_valido_retorna_200_completado(client: TestClient):
    """Verifica que un archivo CSV con reseñas y sucursales se procese síncronamente."""
    csv_content = (
        "reseña,sucursal,fecha\n"
        "La comida estuvo verdaderamente deliciosa y el servicio de diez,Centro,2026-10-01\n"
        "Pésima experiencia la comida tardó una hora y llegó helada,Norte,2026-10-02\n"
        "Excelente atención y los tacos al pastor exquisitos,Centro,2026-10-03\n"
    )

    archivos = {
        "file": ("resenas_test.csv", csv_content.encode("utf-8"), "text/csv")
    }

    resp = client.post("/api/v1/predict/file", files=archivos)
    assert resp.status_code == 200

    datos = resp.json()
    assert datos["estado"] == "COMPLETED"
    assert datos["total_filas"] == 3
    assert datos["filas_procesadas"] == 3
    assert datos["archivo_descargable"] is True
    assert "batch_id" in datos
    assert datos["resumen"]["total_procesadas"] == 3
    assert datos["resumen"]["positivas"] >= 1
    assert datos["resumen"]["negativas"] >= 1
    assert datos["resumen"]["tiene_sucursales"] is True
    assert datos["resumen"]["tiene_fechas"] is True

    # Verificar estructura de los resultados
    item = datos["resultados"][0]
    assert "reseña_original" in item
    assert item["sentimiento"] in ("Positivo", "Negativo")
    assert isinstance(item["certeza"], (int, float))
    assert 0.0 <= item["probabilidad_positiva"] <= 1.0
    assert 0.0 <= item["probabilidad_negativa"] <= 1.0
    assert item["sucursal"] == "Centro"
    assert item["fecha"] == "2026-10-01"


def test_subir_archivo_excel_xlsx_retorna_200(client: TestClient):
    """Verifica la recepción y procesamiento de archivos Excel modernos (.xlsx)."""
    df = pd.DataFrame({
        "comentario": [
            "Delicioso el mole y la salsa muy fresca",
            "El servicio fue muy lento y la carne estaba quemada",
            "Ambiente agradable y comida rica",
        ],
        "sede": ["Sucursal 1", "Sucursal 2", "Sucursal 1"],
        "dia": ["2026-10-10", "2026-10-11", "2026-10-12"],
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Opiniones")
    excel_bytes = buffer.getvalue()

    archivos = {
        "file": (
            "resenas.xlsx",
            excel_bytes,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }

    resp = client.post("/api/v1/predict/file", files=archivos)
    assert resp.status_code == 200

    datos = resp.json()
    assert datos["estado"] == "COMPLETED"
    assert datos["total_filas"] == 3
    assert datos["resumen"]["tiene_sucursales"] is True


def test_consultar_lote_por_id_y_descargar_excel(client: TestClient):
    """Verifica la consulta de estado y la descarga del Excel con las nuevas columnas."""
    csv_content = (
        "opinion,sucursal\n"
        "Excelente restaurante comida muy rica,Sur\n"
        "Terrible atención mesero grosero,Sur\n"
    )
    archivos = {"file": ("test_lote.csv", csv_content.encode("utf-8"), "text/csv")}
    resp_post = client.post("/api/v1/predict/file", files=archivos)
    assert resp_post.status_code == 200
    batch_id = resp_post.json()["batch_id"]

    # 1. Consultar GET /api/v1/batches/{id}
    resp_get = client.get(f"/api/v1/batches/{batch_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["batch_id"] == batch_id
    assert resp_get.json()["estado"] == "COMPLETED"

    # 2. Descargar GET /api/v1/batches/{id}/download
    resp_down = client.get(f"/api/v1/batches/{batch_id}/download")
    assert resp_down.status_code == 200
    assert (
        resp_down.headers["content-type"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Verificar que el contenido descargado es un Excel válido con las 4 nuevas columnas
    df_descargado = pd.read_excel(io.BytesIO(resp_down.content), engine="openpyxl")
    assert "opinion" in df_descargado.columns
    assert "sucursal" in df_descargado.columns
    assert "sentimiento" in df_descargado.columns
    assert "certeza" in df_descargado.columns
    assert "probabilidad_positiva" in df_descargado.columns
    assert "probabilidad_negativa" in df_descargado.columns

    # Verificar que la certeza es un número decimal
    assert pd.api.types.is_numeric_dtype(df_descargado["certeza"])


def test_consultar_alertas_por_sucursal(client: TestClient):
    """Verifica que el cálculo de alertas clasifique niveles (NORMAL, ATENCION, CRITICO)."""
    # Sucursal 'Centro' tendrá quejas concentradas para activar alerta crítica o de atención
    csv_content = (
        "reseña,sucursal\n"
        "Comida fría pésimo servicio muy tardo,Norte\n"
        "Carne cruda y mala atención,Norte\n"
        "Deliciosa comida todo excelente,Sur\n"
        "Maravilloso lugar muy rico,Sur\n"
    )
    archivos = {"file": ("alertas_test.csv", csv_content.encode("utf-8"), "text/csv")}
    resp_post = client.post("/api/v1/predict/file", files=archivos)
    assert resp_post.status_code == 200
    batch_id = resp_post.json()["batch_id"]

    # Consultar alertas
    resp_alerts = client.get(f"/api/v1/batches/{batch_id}/alerts")
    assert resp_alerts.status_code == 200

    datos = resp_alerts.json()
    assert datos["batch_id"] == batch_id
    assert datos["total_sucursales"] == 2
    assert len(datos["alertas"]) == 2

    # Verificar que la sucursal con quejas tenga nivel de atención o crítico
    sedes = {a["sucursal"]: a for a in datos["alertas"]}
    assert "Norte" in sedes
    assert "Sur" in sedes
    assert sedes["Norte"]["nivel_alerta"] in ("ATENCION", "CRITICO")
    assert sedes["Sur"]["nivel_alerta"] == "NORMAL"


def test_consultar_lote_inexistente_retorna_404(client: TestClient):
    """Verifica que solicitar un lote desconocido retorne 404 Not Found."""
    resp = client.get("/api/v1/batches/inexistente_999")
    assert resp.status_code == 404
    assert "No se encontró ningún lote" in resp.json()["detail"]


def test_archivo_extension_invalida_retorna_400(client: TestClient):
    """Verifica que extensiones no soportadas sean rechazadas con 400 Bad Request."""
    archivos = {"file": ("informe.pdf", b"%PDF-1.4...", "application/pdf")}
    resp = client.post("/api/v1/predict/file", files=archivos)
    assert resp.status_code == 400
    assert "Formato no soportado" in resp.json()["detail"]


def test_archivo_vacio_retorna_400(client: TestClient):
    """Verifica que subir un archivo con 0 bytes devuelva 400 Bad Request."""
    archivos = {"file": ("vacio.csv", b"", "text/csv")}
    resp = client.post("/api/v1/predict/file", files=archivos)
    assert resp.status_code == 400
    assert "está vacío" in resp.json()["detail"]


def test_archivo_mayor_500_filas_retorna_202_accepted(client: TestClient):
    """Verifica que archivos mayores a 500 filas arranquen en segundo plano con 202 Accepted."""
    filas = ["reseña,sucursal"]
    for i in range(505):
        filas.append(f"La comida estuvo muy rica reseña número {i},Sede Principal")
    csv_grande = "\n".join(filas)

    archivos = {"file": ("lote_grande.csv", csv_grande.encode("utf-8"), "text/csv")}
    resp = client.post("/api/v1/predict/file", files=archivos)

    # Debe retornar 202 ACCEPTED y estado PROCESSING
    assert resp.status_code == 202
    datos = resp.json()
    assert datos["estado"] == "PROCESSING"
    assert datos["total_filas"] == 505
    assert "batch_id" in datos
