"""Router para la gestión y consulta de lotes de análisis (Batches) y alertas operativas.

Endpoints:
- GET /api/v1/batches/{id}: Consulta el estado, avance y resultados de un lote procesado.
- GET /api/v1/batches/{id}/download: Descarga el archivo Excel (.xlsx) con los resultados incorporados.
- GET /api/v1/batches/{id}/alerts: Consulta el diagnóstico de alertas por sucursal y palabras críticas.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Path
from fastapi.responses import Response

from api.schemas import LoteEstadoResponse, AlertasLoteResponse
from api.service import ServicioPrediccion
from api.dependencies import obtener_servicio_prediccion

router = APIRouter(prefix="/batches", tags=["Gestión de Lotes y Alertas por Sede"])


@router.get(
    "/{batch_id}",
    response_model=LoteEstadoResponse,
    status_code=status.HTTP_200_OK,
    summary="Consultar el estado y resultados de un lote",
    description=(
        "Permite consultar el estado (PROCESSING, COMPLETED, FAILED), progreso, "
        "resumen estadístico y listado de opiniones procesadas para un lote específico."
    ),
)
def obtener_estado_lote(
    batch_id: str = Path(..., description="Identificador único del lote (ej. 8 caracteres alfanuméricos)."),
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> LoteEstadoResponse:
    """Retorna la información consolidada de un lote previamente cargado."""
    lote = servicio.obtener_lote(batch_id)
    if not lote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró ningún lote con el identificador '{batch_id}'.",
        )
    return lote


@router.get(
    "/{batch_id}/download",
    status_code=status.HTTP_200_OK,
    summary="Descargar archivo Excel procesado con predicciones incorporadas",
    description=(
        "Descarga un archivo Excel (.xlsx) que conserva todas las columnas originales "
        "del archivo cargado más las 4 nuevas columnas de diagnóstico: "
        "'sentimiento', 'certeza' (decimal), 'probabilidad_positiva' y 'probabilidad_negativa'."
    ),
)
def descargar_archivo_lote(
    batch_id: str = Path(..., description="Identificador único del lote."),
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> Response:
    """Genera la descarga en formato Excel (.xlsx) del lote solicitado."""
    archivo_info = servicio.obtener_archivo_descarga(batch_id)
    if not archivo_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe archivo descargable disponible para el lote '{batch_id}'.",
        )

    excel_bytes, nombre_archivo = archivo_info
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{nombre_archivo}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.get(
    "/{batch_id}/alerts",
    response_model=AlertasLoteResponse,
    status_code=status.HTTP_200_OK,
    summary="Consultar diagnóstico de alertas operativas por sucursal",
    description=(
        "Evalúa la tasa de insatisfacción por cada sede detectada en el archivo y clasifica "
        "el nivel de alerta (NORMAL <15%, ATENCION 15-25%, CRITICO >25%), "
        "extrayendo los términos negativos más frecuentes de las quejas."
    ),
)
def obtener_alertas_sucursales(
    batch_id: str = Path(..., description="Identificador único del lote."),
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> AlertasLoteResponse:
    """Retorna el reporte de alertas por sucursal para la toma de decisiones gerenciales."""
    alertas = servicio.obtener_alertas_lote(batch_id)
    if not alertas:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontraron datos de alertas para el lote '{batch_id}'.",
        )
    return alertas

