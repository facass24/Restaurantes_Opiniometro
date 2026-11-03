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


def test_palabras_clave_explicabilidad_modelo_real(tmp_path: Path):
    """Verifica el cálculo de palabras clave con coeficientes x TF-IDF para explicabilidad (Paso 5)."""
    servicio = ServicioPrediccion(ruta_modelo=tmp_path / "modelo_inexistente.joblib")

    # Mock de componentes para regresión logística
    mock_prep = PreprocesadorTexto()
    mock_tfidf = MagicMock()
    # Simular matriz dispersa (1 fila, 2 columnas)
    import scipy.sparse as sp
    matriz_sparse = sp.csr_matrix([[0.6, 0.4]])
    mock_tfidf.transform.return_value = matriz_sparse
    mock_tfidf.get_feature_names_out.return_value = np.array(["excelente", "comida"])

    mock_model = MagicMock()
    mock_model.coef_ = np.array([[2.5, 1.2]])
    mock_model.classes_ = np.array(["negativo", "positivo"])
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

    mock_pipeline = MagicMock(spec=["named_steps", "classes_", "predict_proba", "nombre_modelo"])
    mock_pipeline.named_steps = {
        "preprocess": mock_prep,
        "tfidf": mock_tfidf,
        "model": mock_model,
    }
    mock_pipeline.classes_ = mock_model.classes_
    mock_pipeline.predict_proba = mock_model.predict_proba
    mock_pipeline.nombre_modelo = "Regresión Logística Test"

    servicio.pipeline = mock_pipeline
    servicio.es_modelo_real = True
    servicio.preprocesador = mock_prep

    resultado = servicio.predecir("Excelente comida")

    # Verificar que el paso 5 de palabras clave no está vacío
    assert resultado.palabras_clave is not None
    assert len(resultado.palabras_clave) > 0
    primera_palabra = resultado.palabras_clave[0]
    assert primera_palabra.palabra in ["excelente", "comida"]
    assert primera_palabra.polaridad == "positivo"
    assert primera_palabra.peso > 0


def test_estado_health_degraded_sin_modelo(tmp_path: Path):
    """Verifica que el servicio reporte 'degraded' si no cargó el modelo real."""
    servicio = ServicioPrediccion(ruta_modelo=tmp_path / "modelo_inexistente.joblib")
    estado = servicio.estado()
    assert estado.estado == "degraded"
    assert estado.modelo_cargado is False

    # Al cargar un modelo real ficticio, debe pasar a 'ok'
    servicio.es_modelo_real = True
    estado_ok = servicio.estado()
    assert estado_ok.estado == "ok"
    assert estado_ok.modelo_cargado is True

