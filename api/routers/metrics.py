"""Router para consulta y comparación de métricas de los modelos (CU5 y CU6)."""

from fastapi import APIRouter, Depends
from typing import List, Dict, Any

from api.schemas import MetricasResponse, ModeloDetalleDTO
from api.service import ServicioPrediccion
from api.dependencies import obtener_servicio_prediccion

router = APIRouter(prefix="/metrics", tags=["Métricas y Evaluación"])


@router.get(
    "",
    response_model=MetricasResponse,
    summary="Ver métricas del modelo",
    description="Retorna las métricas del modelo en producción (Accuracy, F1-macro, Precisión, Recall y Matriz de confusión).",
)
def obtener_metricas(
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> MetricasResponse:
    """Entrega el reporte de desempeño del modelo actual."""
    return servicio.obtener_metricas()


@router.get(
    "/compare",
    response_model=List[ModeloDetalleDTO],
    summary="Comparar modelos (Naive Bayes vs Regresión Logística)",
    description="Entrega la comparativa directa entre los modelos evaluados para alimentar las tablas del frontend.",
)
def comparar_modelos(
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> List[ModeloDetalleDTO]:
    """Entrega la lista de modelos con sus respectivas métricas para comparación lado a lado."""
    metricas_resp = servicio.obtener_metricas()
    return metricas_resp.modelos

