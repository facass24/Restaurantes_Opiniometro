# Dataset: Rest-Mex 2025

## Origen
- **Fuente original:** Rest-Mex 2025 – *Researching on Evaluating Sentiment and Textual Instances Selection for Mexican Magical Towns*, tarea del foro IberLEF 2025, organizada por CIMAT (México).
  Sitio oficial: https://sites.google.com/cimat.mx/rest-mex-2025
- **Copia utilizada:** https://github.com/lahernandezguevara/scala-spark-reviews (archivo `Rest-Mex_2025_train.csv`).
- **Contenido:** opiniones de turistas en español (México) sobre hoteles, restaurantes y atractivos de Pueblos Mágicos.

## Licencia y uso
- Uso exclusivamente académico y de investigación, según lo indicado en el repositorio de la copia.
- Se cita a los organizadores de Rest-Mex como fuente original.

## Fecha de obtención
- 31 de octubre de 2026.

## Cómo descargarlo
Desde la raíz del proyecto, con el entorno virtual activo:

    python data/descargar_datos.py

El archivo se guarda en `data/raw/Rest-Mex_2025_train.csv`. No se sube al repositorio por su tamaño (aprox. 89 MB).

## Columnas
| Columna | Descripción |
|---|---|
| Title | Título de la opinión |
| Review | Texto de la reseña |
| Polarity | Calificación de 1 a 5 |
| Town | Pueblo Mágico |
| Region | Estado de México |
| Type | Tipo de lugar: Hotel, Restaurant o Attractive |

## Tamaño
- Total: 208.051 reseñas.
- Restaurantes (`Type = Restaurant`): 86.720 reseñas.