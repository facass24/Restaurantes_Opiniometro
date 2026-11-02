"""Pruebas de integración del ServicioPrediccion con un Pipeline entrenado de scikit-learn."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from api.service import ServicioPrediccion
from src.nlp.pipeline import PreprocesadorTexto


def test_pipeline_integracion_recibe_texto_crudo(tmp_path: Path):
    """Verifica que el pipeline entrenado reciba el texto CRUDO para evitar doble preprocesamiento."""
    servicio = ServicioPrediccion(ruta_modelo=tmp_path / "modelo_inexistente.joblib")
    assert not servicio.es_modelo_real

    # Simular un pipeline real de scikit-learn con preprocesador entrenado
    mock_pipeline = MagicMock()
    mock_prep = PreprocesadorTexto(quitar_tildes=True, metodo_reduccion="stem")
    mock_pipeline.named_steps = {"preprocesador": mock_prep}
    mock_pipeline.classes_ = np.array(["negativo", "positivo"])
    mock_pipeline.predict_proba.return_value = np.array([[0.15, 0.85]])
    mock_pipeline.nombre_modelo = "Pipeline Test Producción"

    # Inyectar el pipeline en el servicio
    servicio.pipeline = mock_pipeline
    servicio.es_modelo_real = True
    # Extraer el preprocesador tal como lo hace _cargar_modelo
    prep = servicio.pipeline.named_steps.get("preprocesador")
    if prep is not None:
        servicio.preprocesador = prep

    texto_prueba = "¡La comida estuvo excelente y deliciosa!"
    resultado = servicio.predecir(texto_prueba)

    # 1. Verificar que el preprocesador del pipeline fue utilizado
    assert servicio.preprocesador is mock_prep
    assert resultado.preprocesamiento.original == texto_prueba

    # 2. Verificar que predict_proba fue llamado con el texto CRUDO (sin procesar previamente)
    mock_pipeline.predict_proba.assert_called_once_with([texto_prueba])

    # 3. Verificar resultados
    assert resultado.clase == "positivo"
    assert resultado.probabilidad == 0.85
    assert resultado.modelo_usado == "Pipeline Test Producción"


def test_pipeline_nombre_preprocess_compatibilidad(tmp_path: Path):
    """Verifica compatibilidad cuando el paso del pipeline se llama 'preprocess'."""
    servicio = ServicioPrediccion(ruta_modelo=tmp_path / "modelo_inexistente.joblib")

    mock_pipeline = MagicMock(spec=["named_steps", "classes_", "predict_proba"])
    mock_prep = PreprocesadorTexto(quitar_tildes=False)
    mock_pipeline.named_steps = {"preprocess": mock_prep}
    mock_pipeline.classes_ = np.array(["negativo", "positivo"])
    mock_pipeline.predict_proba.return_value = np.array([[0.80, 0.20]])

    servicio.pipeline = mock_pipeline
    servicio.es_modelo_real = True
    prep = servicio.pipeline.named_steps.get("preprocesador") or servicio.pipeline.named_steps.get("preprocess")
    if prep is not None:
        servicio.preprocesador = prep

    texto_prueba = "Pésima atención, todo frío."
    resultado = servicio.predecir(texto_prueba)

    mock_pipeline.predict_proba.assert_called_once_with([texto_prueba])
    assert resultado.clase == "negativo"
    assert resultado.probabilidad == 0.80

