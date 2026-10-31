"""Router para verificación del estado del servicio (Healthcheck - CU7)."""

from fastapi import APIRouter, Depends
from api.schemas import EstadoResponse
from api.service import ServicioPrediccion
from api.dependencies import obtener_servicio_prediccion

router = APIRouter(tags=["Salud y Monitoreo"])


@router.get(
    "/health",
    response_model=EstadoResponse,
    summary="Consultar estado del servicio",
    description="Permite verificar que la API está operativa y si el modelo entrenado está cargado en memoria.",
)
def consultar_salud(
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> EstadoResponse:
    """Retorna el estado de disponibilidad y detalles de los artefactos del sistema."""
    return servicio.estado()

