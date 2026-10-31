"""Configuración global del proyecto Opiniómetro de Restaurantes.

Define las rutas base, parámetros del sistema y configuración de la API.
"""

from pathlib import Path
import yaml

# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "restaurantes.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PARAMS_PATH = BASE_DIR / "params.yaml"

# Configuración de la API FastAPI
API_TITLE = "Opiniómetro de Restaurantes API"
API_DESCRIPTION = (
    "API para el análisis de sentimiento (positivo / negativo) de reseñas de restaurantes "
    "en español, integrando modelos de Machine Learning (Naive Bayes y Regresión Logística) "
    "y pipeline de NLP explicable paso a paso."
)
API_VERSION = "1.0.0"
API_V1_PREFIX = "/api/v1"

# Orígenes permitidos para CORS (Frontend React/Vite y entornos locales)
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]

def cargar_parametros() -> dict:
    """Carga los parámetros definidos en params.yaml con valores de respaldo."""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {
        "seed": 42,
        "tfidf": {
            "max_features": 5000,
            "min_df": 2,
            "max_df": 0.90,
            "sublinear_tf": True,
        },
    }

PARAMS = cargar_parametros()

