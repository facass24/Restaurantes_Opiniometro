"""Router para análisis de sentimiento de reseñas individuales y por lote CSV.

Implementa los casos de uso:
- CU1: Analizar sentimiento de una reseña.
- CU2: Analizar reseñas desde un archivo CSV.
- CU3: Ver preprocesamiento paso a paso.
- CU4: Validar texto ingresado.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import Response

from api.schemas import ReseñaRequest, PrediccionResponse, PrediccionLoteResponse
from api.service import ServicioPrediccion
from api.dependencies import obtener_servicio_prediccion

router = APIRouter(prefix="/predict", tags=["Predicciones de Sentimiento"])


@router.post(
    "",
    response_model=PrediccionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analizar sentimiento de una reseña",
    description=(
        "Clasifica una reseña de restaurante en español como Positiva o Negativa, "
        "calculando su probabilidad y exponiendo la traza del preprocesamiento NLP paso a paso."
    ),
)
def predecir_sentimiento(
    request: ReseñaRequest,
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> PrediccionResponse:
    """Ejecuta la inferencia sobre el texto validado."""
    return servicio.predecir(request.texto, modelo_solicitado=request.modelo)


@router.post(
    "/csv",
    response_model=PrediccionLoteResponse,
    status_code=status.HTTP_200_OK,
    summary="Analizar reseñas desde un archivo CSV",
    description=(
        "Sube un archivo .csv con reseñas de restaurantes. El sistema detecta "
        "automáticamente la columna de texto y clasifica todas las opiniones en lote."
    ),
)
async def predecir_csv(
    file: UploadFile = File(..., description="Archivo CSV con opiniones."),
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> PrediccionLoteResponse:
    """Procesa un archivo CSV y entrega las predicciones con estadísticas consolidadas."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo proporcionado debe tener extensión .csv",
        )

    contenido = await file.read()
    if not contenido or len(contenido.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo CSV está vacío.",
        )

    try:
        lote_response, _ = servicio.procesar_archivo_csv(contenido)
        return lote_response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error al procesar el archivo CSV: {str(e)}",
        )


@router.post(
    "/csv/export",
    summary="Exportar archivo CSV procesado con predicciones añadidas",
    description="Sube un archivo CSV y lo descarga con las columnas de sentimiento y probabilidad agregadas.",
)
async def exportar_csv_con_predicciones(
    file: UploadFile = File(..., description="Archivo CSV con opiniones."),
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> Response:
    """Retorna un archivo CSV descargable con los resultados anexados."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo debe tener formato .csv",
        )

    contenido = await file.read()
    try:
        _, csv_bytes = servicio.procesar_archivo_csv(contenido)
        nombre_descarga = f"predicciones_{file.filename}"
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={nombre_descarga}"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No se pudo generar el archivo procesado: {str(e)}",
        )

