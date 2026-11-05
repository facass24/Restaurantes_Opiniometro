"""Esquemas de datos Pydantic (DTOs) para la API FastAPI.

Define los contratos de entrada y salida, asegurando validación estricta
según el caso de uso CU4 (Validar texto ingresado) y el Diagrama de Clases.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import re


class ReseñaRequest(BaseModel):
    """Modelo de solicitud para clasificar una reseña individual."""

    texto: str = Field(
        ...,
        description="Texto de la reseña del restaurante en español.",
        json_schema_extra={"example": "La comida estuvo excelente, el mole delicioso pero el servicio un poco lento."},
    )
    modelo: Optional[Literal["auto", "logistic_regression", "naive_bayes"]] = Field(
        default="auto",
        description="Modelo opcional para realizar la inferencia.",
    )

    @field_validator("texto")
    @classmethod
    def validar_texto(cls, valor: str) -> str:
        """Valida que el texto no esté vacío, no sea excesivamente largo y tenga contenido semántico."""
        if not valor or not valor.strip():
            raise ValueError("La reseña no puede estar vacía ni contener solo espacios en blanco.")

        limpio = valor.strip()
        if len(limpio) < 3:
            raise ValueError("La reseña es demasiado corta (mínimo 3 caracteres requeridos).")

        if len(limpio) > 5000:
            raise ValueError("La reseña excede la longitud máxima permitida de 5000 caracteres.")

        # Verificar que contenga al menos algunas letras (evitar cadenas con solo números o símbolos)
        if not re.search(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]", limpio):
            raise ValueError("La reseña debe contener caracteres alfabéticos válidos.")

        return limpio


class ResultadoPreprocesamientoDTO(BaseModel):
    """Representación serializable del preprocesamiento paso a paso (CU3)."""

    original: str = Field(..., description="Texto original ingresado por el usuario.")
    normalizado: str = Field(..., description="Texto en minúsculas y caracteres normalizados.")
    tokens: List[str] = Field(..., description="Lista de palabras tokenizadas.")
    sin_stopwords: List[str] = Field(
        ..., description="Tokens tras filtrar stop words (conservando negaciones)."
    )
    reducidos: List[str] = Field(
        ..., description="Tokens reducidos mediante stemming o lematización."
    )


class PalabraClaveDTO(BaseModel):
    """Explicabilidad: palabra influyente en la decisión del modelo."""

    palabra: str
    peso: float
    polaridad: Literal["positivo", "negativo"]


class PrediccionResponse(BaseModel):
    """Respuesta estructurada para la predicción de una reseña (CU1 y CU3)."""

    clase: Literal["positivo", "negativo"] = Field(
        ..., description="Sentimiento asignado a la reseña."
    )
    probabilidad: float = Field(
        ..., description="Probabilidad de pertenecer a la clase predicha (0.0 a 1.0)."
    )
    confianza_porcentaje: float = Field(
        ..., description="Porcentaje de confianza redondeado (0 a 100%)."
    )
    modelo_usado: str = Field(..., description="Nombre del modelo que generó la predicción.")
    tiempo_ms: float = Field(..., description="Tiempo de inferencia en milisegundos.")
    preprocesamiento: ResultadoPreprocesamientoDTO = Field(
        ..., description="Desglose didáctico del texto paso a paso."
    )
    palabras_clave: Optional[List[PalabraClaveDTO]] = Field(
        default=None, description="Palabras con mayor peso explicativo."
    )


class ItemPrediccionLote(BaseModel):
    """Resultado individual de una fila procesada en lote."""

    indice: int
    texto_resumen: str
    clase: Literal["positivo", "negativo"]
    probabilidad: float


class ResumenLoteDTO(BaseModel):
    """Estadísticas consolidadas del lote procesado (CU2)."""

    total_procesadas: int
    positivas: int
    negativas: int
    porcentaje_positivas: float
    porcentaje_negativas: float


class PrediccionLoteResponse(BaseModel):
    """Respuesta consolidada para predicciones por archivo CSV."""

    resumen: ResumenLoteDTO
    resultados: List[ItemPrediccionLote]


class MetricasDTO(BaseModel):
    """Métricas de evaluación de un modelo de machine learning."""

    accuracy: float
    f1_macro: float
    precision: float
    recall: float
    matriz_confusion: List[List[int]]
    precision_negativo: Optional[float] = None
    recall_negativo: Optional[float] = None
    f1_negativo: Optional[float] = None


class ModeloDetalleDTO(BaseModel):
    """Detalle de configuración y rendimiento de un modelo evaluado."""

    nombre: str
    identificador: str
    descripcion: str
    es_activo: bool
    metricas: MetricasDTO


class MetricasResponse(BaseModel):
    """Respuesta para los endpoints de métricas y comparativa (CU5 y CU6)."""

    modelo_produccion: str
    modelos: List[ModeloDetalleDTO]


class EstadoResponse(BaseModel):
    """Respuesta del estado del servicio para healthcheck y monitoreo (CU7)."""

    estado: Literal["ok", "degraded", "error"]
    modelo_cargado: bool
    tipo_modelo: str
    version: str
    detalles: dict


# ============================================================================
# DTOs para Procesamiento Avanzado de Archivos (CSV / Excel) y Alertas por Sede
# ============================================================================

class ReseñaItemProcesadaDTO(BaseModel):
    """Detalle de una reseña analizada dentro de un archivo procesado."""

    indice: int = Field(..., description="Número secuencial de fila en el archivo.")
    reseña_original: str = Field(..., description="Texto original de la reseña.")
    sentimiento: Literal["Positivo", "Negativo"] = Field(
        ..., description="Sentimiento asignado a la opinión."
    )
    certeza: float = Field(
        ..., description="Porcentaje numérico de certeza calibrada (ej. 94.5)."
    )
    probabilidad_positiva: float = Field(
        ..., description="Probabilidad matemática de clase positiva (0.0 a 1.0)."
    )
    probabilidad_negativa: float = Field(
        ..., description="Probabilidad matemática de clase negativa (0.0 a 1.0)."
    )
    sucursal: Optional[str] = Field(
        default=None, description="Sucursal o sede donde se originó la opinión."
    )
    fecha: Optional[str] = Field(
        default=None, description="Fecha o marca temporal registrada."
    )


class ResumenLoteAvanzadoDTO(BaseModel):
    """Estadísticas consolidadas del lote analizado."""

    total_procesadas: int = Field(..., description="Total de opiniones procesadas.")
    positivas: int = Field(..., description="Cantidad de opiniones positivas.")
    negativas: int = Field(..., description="Cantidad de opiniones negativas.")
    porcentaje_satisfaccion: float = Field(
        ..., description="Porcentaje de opiniones positivas sobre el total procesado."
    )
    porcentaje_insatisfaccion: float = Field(
        ..., description="Porcentaje de opiniones negativas sobre el total procesado."
    )
    tiene_sucursales: bool = Field(
        default=False, description="Indica si el archivo contenía columna de sucursal."
    )
    tiene_fechas: bool = Field(
        default=False, description="Indica si el archivo contenía columna de fecha."
    )


class AlertaSucursalDTO(BaseModel):
    """Diagnóstico y nivel de alerta operativa para una sucursal específica."""

    sucursal: str = Field(..., description="Nombre de la sucursal o sede.")
    total_reseñas: int = Field(..., description="Total de reseñas de esta sucursal.")
    positivas: int = Field(..., description="Reseñas positivas recibidas.")
    negativas: int = Field(..., description="Quejas o reseñas negativas recibidas.")
    porcentaje_insatisfaccion: float = Field(
        ..., description="Porcentaje de quejas sobre el total de la sede."
    )
    nivel_alerta: Literal["NORMAL", "ATENCION", "CRITICO"] = Field(
        ..., description="Nivel de alerta (NORMAL <15%, ATENCION 15-25%, CRITICO >25%)."
    )
    mensaje: str = Field(..., description="Resumen cualitativo de la situación operativa.")
    palabras_criticas: List[str] = Field(
        default_factory=list,
        description="Términos negativos más frecuentes detectados en la sucursal.",
    )


class AlertasLoteResponse(BaseModel):
    """Resumen consolidado de alertas por sucursal de un lote procesado."""

    batch_id: str = Field(..., description="Identificador único del lote.")
    fecha_analisis: str = Field(..., description="Marca temporal del análisis.")
    total_sucursales: int = Field(..., description="Cantidad total de sucursales evaluadas.")
    sucursales_en_alerta: int = Field(
        ..., description="Cantidad de sucursales en estado ATENCIÓN o CRÍTICO."
    )
    alertas: List[AlertaSucursalDTO] = Field(
        ..., description="Detalle de alertas por cada sucursal."
    )


class LoteEstadoResponse(BaseModel):
    """Estado, progreso y resultados de un lote de análisis."""

    batch_id: str = Field(..., description="Identificador único del lote.")
    nombre_archivo: str = Field(..., description="Nombre del archivo original cargado.")
    estado: Literal["PROCESSING", "COMPLETED", "FAILED"] = Field(
        ..., description="Estado de procesamiento del lote."
    )
    progreso_porcentaje: float = Field(
        default=0.0, description="Porcentaje de avance (0.0 a 100.0)."
    )
    total_filas: int = Field(default=0, description="Total de filas a procesar.")
    filas_procesadas: int = Field(default=0, description="Filas analizadas hasta el momento.")
    fecha_creacion: str = Field(..., description="Fecha de creación del lote.")
    resumen: Optional[ResumenLoteAvanzadoDTO] = Field(
        default=None, description="Resumen estadístico (disponible al completar)."
    )
    resultados: Optional[List[ReseñaItemProcesadaDTO]] = Field(
        default=None, description="Listado de reseñas clasificadas."
    )
    archivo_descargable: bool = Field(
        default=False, description="Indica si el archivo Excel procesado está listo para descarga."
    )
    error: Optional[str] = Field(
        default=None, description="Mensaje de error en caso de fallo."
    )


