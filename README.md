# Opiniómetro de Restaurantes

Clasificación de sentimiento (positivo / negativo) de reseñas de restaurantes en español,
con Naive Bayes y Regresión Logística sobre TF-IDF. Proyecto integrador de Tópicos y
Programación Avanzada.

**Resultado:** Regresión Logística con F1 macro 0,93 en el conjunto de prueba (la línea base,
que siempre dice "positivo", saca 0,48). Detalle en `docs/resultados_experimentos.md`.

## Requisitos

- Python [completar versión probada, por ejemplo 3.12]
- Conexión a internet la primera vez (descarga del dataset, de NLTK y de spaCy)

## Instalación desde cero

```bash
git clone https://github.com/facass24/Restaurantes_Opiniometro.git
cd Restaurantes_Opiniometro

python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m spacy download es_core_news_sm
```

## Entrenar el modelo (un solo comando)

```bash
python -m src.models.train
```

Descarga y prepara el dataset Rest-Mex si falta, entrena el modelo de `params.yaml`, guarda el
pipeline completo en `artifacts/model.joblib`, evalúa en el conjunto de prueba y exporta
`artifacts/metrics.json`, que lee la API. Tarda unos minutos la primera vez (preprocesa unas
80 mil reseñas); las siguientes veces usa una caché.

Opciones:

```bash
python -m src.models.train --sin-evaluar          # solo entrena y guarda el modelo
python -m src.models.train --repetir-evaluacion   # vuelve a usar el conjunto de prueba (con motivo)
```

El conjunto de prueba se usa una sola vez: si ya se evaluó con la misma configuración, el
comando reutiliza los resultados guardados.

## Levantar la API

```bash
uvicorn api.main:app --reload
```

Documentación interactiva: http://127.0.0.1:8000/docs

| Endpoint | Para qué sirve |
|---|---|
| `POST /api/v1/predict` | Clase, probabilidad y preprocesamiento paso a paso de una reseña |
| `POST /api/v1/predict/csv` | Predicción de un archivo CSV con varias reseñas |
| `POST /api/v1/predict/csv/export` | Descarga del CSV con la clase y la probabilidad agregadas |
| `GET /api/v1/metrics` | Métricas del modelo en producción |
| `GET /api/v1/metrics/compare` | Comparación con Naive Bayes y la línea base |
| `GET /api/v1/health` | Estado del servicio y del modelo |

Si `artifacts/model.joblib` no existe, la API usa un modelo heurístico de respaldo y
`/health` lo indica (`tipo_modelo`). Para usar el modelo real, ejecuta antes el entrenamiento.

## Frontend

[Completar con las instrucciones del frontend: carpeta `frontend/`, `npm install`, `npm run dev`.]

## Pruebas

```bash
pytest -q
```

## Estructura

- `api/`: backend FastAPI (rutas, esquemas Pydantic y servicio de predicción).
- `src/nlp/`: pipeline de NLP (normalización, tokenización, stop words que conservan negaciones, stemming y lematización).
- `src/data/`: preparación del dataset.
- `src/models/`: entrenamiento, validación cruzada, evaluación y análisis de errores.
- `frontend/`: interfaz web.
- `data/`: datos (el CSV crudo no se sube; ver `data/README.md`).
- `notebooks/`: análisis exploratorio y normalización.
- `artifacts/`: modelo y métricas generados por el entrenamiento (no se suben a git).
- `docs/`: informes y resultados de los experimentos.
- `tests/`: pruebas automatizadas.
- `params.yaml`: parámetros de todo el experimento (semilla, TF-IDF, modelo final).

## Reproducibilidad

El preprocesamiento de texto viaja dentro del pipeline guardado (`preprocess → tfidf → model`):
el entrenamiento y la API usan exactamente el mismo código. La semilla es 42 y las versiones de
las librerías están fijadas en `requirements.txt`.

## Autoría

Ver `CONTRIBUCIONES.md`. Uso de asistentes de IA declarado en `docs/informe_modelado.md`.