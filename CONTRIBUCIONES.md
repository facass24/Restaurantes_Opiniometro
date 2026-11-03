# Contribuciones

|        Integrante           |      Rol    |   Que hizo  |
|-----------------------------|-------------|-------------|
|   Sanchez Cuellar Melissa   | Datos y NLP |   Script de descarga y documentación del dataset Rest-Mex (`data/descargar_datos.py`, `data/README.md`). Análisis exploratorio (`notebooks/01_eda.ipynb`). Preparación del dataset limpio con `preparar_datos()` (`src/data/preparar_datos.py`). Normalización completa a partir de la versión inicial de Kelly (`src/nlp/normalizer.py`): conservar la ñ, reducir letras repetidas, quitar URLs, emojis y puntuación, y tildes por defecto. Tokenización con casos del dominio a partir de la versión inicial de Kelly (`src/nlp/tokenizer.py`): "10/10" y `$precio`. Ejemplos antes/después y justificación (`notebooks/02_normalizacion_tokenizacion.ipynb`). Pruebas del normalizador y el tokenizador (`tests/test_normalizer.py`, `tests/test_tokenizer.py`). |
| Fabiola Marioma Quispe Lero |   Modelado  | (pendiente) |
| Duran Marcelo Kelly Allison |   Backend   | Arquitectura de la API en FastAPI, contratos Pydantic (CU1 a CU7), procesamiento por lotes CSV, suite de pruebas con pytest, y FiltroStopWords preservando negaciones ('no', 'nunca', 'pero'). |
| Wendy | Frontend | (pendiente) |
