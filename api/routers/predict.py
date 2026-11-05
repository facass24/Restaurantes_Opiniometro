"""Router para análisis de sentimiento de reseñas individuales y por lote CSV.

Implementa los casos de uso:
- CU1: Analizar sentimiento de una reseña.
- CU2: Analizar reseñas desde un archivo CSV.
- CU3: Ver preprocesamiento paso a paso.
- CU4: Validar texto ingresado.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.responses import Response
import uuid
from datetime import datetime

from src.config import BATCHES_DIR
from api.schemas import (
    ReseñaRequest,
    PrediccionResponse,
    PrediccionLoteResponse,
    LoteEstadoResponse,
)
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


@router.post(
    "/file",
    response_model=LoteEstadoResponse,
    summary="Analizar reseñas desde archivo CSV o Excel (.xlsx)",
    description=(
        "Sube un archivo .csv o .xlsx con opiniones gastronómicas. "
        "Detecta automáticamente columnas de reseña, sucursal y fecha. "
        "Archivos de hasta 500 filas se procesan directamente (200 OK); "
        "lotes mayores se ejecutan en segundo plano (202 Accepted) asignando un batch_id."
    ),
)
async def predecir_archivo(
    background_tasks: BackgroundTasks,
    response: Response,
    file: UploadFile = File(..., description="Archivo .csv o .xlsx con opiniones."),
    servicio: ServicioPrediccion = Depends(obtener_servicio_prediccion),
) -> LoteEstadoResponse:
    """Procesa un archivo tabular (CSV o Excel) identificando sucursales y calculando alertas."""
    nombre_lower = file.filename.lower()
    if not (nombre_lower.endswith(".csv") or nombre_lower.endswith(".xlsx") or nombre_lower.endswith(".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato no soportado. Proporcione un archivo con extensión .csv o .xlsx",
        )

    contenido = await file.read()
    if not contenido or len(contenido.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo proporcionado está vacío.",
        )

    # Validar lectura previa y conteo de filas
    try:
        df_preview = servicio._leer_dataframe_archivo(contenido, file.filename)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error al analizar la estructura del archivo: {str(e)}",
        )

    total_filas = len(df_preview)
    batch_id = str(uuid.uuid4())[:8]

    if total_filas <= 500:
        # Procesamiento síncrono inmediato (hasta 500 filas)
        try:
            lote_resp, _, _ = servicio.procesar_archivo_tabular(
                contenido, file.filename, batch_id=batch_id
            )
            response.status_code = status.HTTP_200_OK
            return lote_resp
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error durante el procesamiento del archivo: {str(e)}",
            )
    else:
        # Procesamiento asíncrono en segundo plano para lotes grandes (> 500 filas)
        fecha_creacion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        estado_inicial = LoteEstadoResponse(
            batch_id=batch_id,
            nombre_archivo=file.filename,
            estado="PROCESSING",
            progreso_porcentaje=0.0,
            total_filas=total_filas,
            filas_procesadas=0,
            fecha_creacion=fecha_creacion,
            resumen=None,
            resultados=None,
            archivo_descargable=False,
            error=None,
        )

        # Registrar estado inicial en disco
        try:
            ruta_json = BATCHES_DIR / f"{batch_id}.json"
            import json
            with open(ruta_json, "w", encoding="utf-8") as f:
                json.dump(estado_inicial.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        background_tasks.add_task(
            servicio.ejecutar_lote_en_segundo_plano,
            batch_id,
            contenido,
            file.filename,
        )
        response.status_code = status.HTTP_202_ACCEPTED
        return estado_inicial


