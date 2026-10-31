# Opiniómetro de Restaurantes

Clasificación de sentimiento (positivo / negativo) de reseñas de restaurantes
en español, con Naive Bayes y Regresión Logística.

Proyecto integrador de Tópicos y Programación Avanzada.

## Requisitos

- Python 3.12+

## Estructura del Proyecto

- `api/`: Backend desarrollado en FastAPI con documentación Swagger automática en `/docs`.
- `src/nlp/`: Pipeline modular de NLP (normalización, tokenización, stop words con preservación de negaciones, reducción).
- `src/models/`: Scripts de entrenamiento, validación cruzada y comparación de modelos.
- `frontend/`: Interfaz de usuario en React + Vite.
- `data/`: Datos crudos y procesados (dataset Rest-Mex).
- `notebooks/`: Análisis exploratorio de datos (EDA) y experimentación.
- `artifacts/`: Modelos serializados (`.joblib`) y métricas consolidadas (`metrics.json`).
- `tests/`: Batería de pruebas automatizadas con pytest.
- `docs/`: Informe técnico y diagramas de arquitectura.
- `params.yaml`: Parámetros del experimento.

## Cómo levantar el Backend y probar la API

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar las pruebas automatizadas (pytest)
```bash
pytest -v
```

### 3. Iniciar el servidor de desarrollo FastAPI
```bash
uvicorn api.main:app --reload
```
La documentación interactiva OpenAPI (Swagger) estará disponible en:
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
