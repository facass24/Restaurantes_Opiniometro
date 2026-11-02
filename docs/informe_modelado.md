# Informe técnico: modelado

Rol de modelado. Este documento resume las decisiones y los resultados del modelado; el detalle
de cada experimento, con sus tablas y comandos, está en `docs/resultados_experimentos.md`.

## 1. Problema y datos

**Tarea:** clasificar el sentimiento (positivo o negativo) de reseñas de restaurantes en español.

**Dataset:** Rest-Mex 2025 (IberLEF 2025), reseñas de TripAdvisor sobre destinos turísticos de
México publicadas entre 2002 y 2024. El conjunto de entrenamiento tiene 208.051 reseñas; este
proyecto usa solo las de tipo restaurante.

- **Origen:** corpus de la evaluación Rest-Mex 2025 (Álvarez-Carmona et al., *Overview of Rest-Mex
  at IberLEF 2025*, Procesamiento del Lenguaje Natural, 75, pp. 499-511). La copia usada se
  descargó de `github.com/lahernandezguevara/scala-spark-reviews` (archivo `Rest-Mex_2025_train.csv`).
- **Licencia:** la copia usada no especifica licencia. El corpus original se distribuye con
  registro y términos de uso académico de la evaluación; en este proyecto se usa solo con fines
  académicos.
- **Fecha de obtención:** 31 de octubre de 2026 (según `data/README.md`).

**Etiquetado y limpieza:** 1 y 2 estrellas = negativo; 4 y 5 = positivo; se descartan las de 3
estrellas por ambiguas. Se quitan textos vacíos y duplicados. El texto es el título más la reseña.

| | Reseñas |
|---|---|
| Restaurantes en el corpus | 86.720 |
| Tras descartar 3 estrellas, vacíos y duplicados | 80.713 |
| Entrenamiento (80 %) / prueba (20 %) | 64.570 / 16.143 |
| Negativas / positivas (total) | 4.846 (6,0 %) / 75.867 (94,0 %) |

La división es estratificada y con semilla fija (42). El conjunto de prueba no se usó para
ningún ajuste: solo para la evaluación final.

## 2. Modelos y por qué

Se compararon dos modelos vistos en clase, más una línea base:

- **Línea base:** predice siempre "positivo". Sirve para demostrar que los modelos aportan valor.
- **Naive Bayes multinomial:** rápido y adecuado para frecuencias de palabras; asume que las
  palabras son independientes entre sí.
- **Regresión logística:** aprende un peso por término, que permite interpretar qué palabras
  empujan hacia cada clase, y entrega probabilidades calibradas.

Ambos usan el pipeline de NLP del proyecto (normalización, tokenización, stop words que
conservan la negación, stemming) y TF-IDF con vocabulario de 5.000 términos, frecuencia mínima
de 2 reseñas, máxima de 90 % y frecuencia sublineal (`params.yaml`).

## 3. Metodología

- **Métrica:** F1 macro. El 94 % de las reseñas es positivo, así que la accuracy engaña: la
  línea base saca 0,94 sin detectar ninguna reseña negativa.
- **Selección:** validación cruzada estratificada de 5 pliegues sobre el entrenamiento, con
  búsqueda de hiperparámetros (`GridSearchCV`). El vocabulario TF-IDF se aprende dentro de cada
  pliegue.
- **Desbalance:** se probaron prior uniforme, `class_weight="balanced"` y submuestreo de la
  clase positiva (solo en entrenamiento). Regresión logística rindió mejor sin tratar; Naive
  Bayes necesitó el submuestreo.
- **Stemming contra lematización:** empate técnico (diferencias de ±0,004, dentro de la variación
  entre pliegues, y en sentidos opuestos). Se eligió stemming porque es unas 9 veces más rápido.
- **Unigramas contra unigramas + bigramas:** los bigramas mejoran a los dos modelos (alrededor
  de 0,01 de F1 macro), porque capturan frases como "no me gustó".

## 4. Resultados

| Modelo | F1 macro (validación cruzada) | F1 macro (prueba) | accuracy (prueba) | precisión negativa | recall negativa |
|---|---|---|---|---|---|
| Línea base (siempre positivo) | 0,4845 | 0,4845 | 0,9400 | 0,0000 | 0,0000 |
| Naive Bayes + submuestreo (bigramas) | 0,916 | 0,9074 | 0,9785 | 0,8025 | 0,8514 |
| **Regresión logística (bigramas, C = 10)** | **0,926** | **0,9282** | **0,9846** | **0,9187** | **0,8163** |

**Modelo en producción:** regresión logística con stemming, unigramas + bigramas y C = 10. Gana
en F1 macro, tiene mucha mejor precisión sobre las reseñas negativas, y sus probabilidades
reflejan la proporción real de las clases porque entrena con todos los datos. Naive Bayes
detecta algo más de negativas (mayor recall) a costa de marcar muchas positivas como negativas.

Matriz de confusión en la prueba (filas = clase real, columnas = predicha; orden negativo,
positivo): `[[791, 178], [70, 15104]]`.

## 5. Análisis de errores

El modelo comete 248 errores en 16.143 reseñas (1,54 %): 178 negativas que pasa por positivas
y 70 positivas marcadas como negativas.

- Los errores se concentran en las reseñas de borde: 2 estrellas (77 % de acierto) y 1 estrella
  (86 %), contra 99,9 % en 5 estrellas.
- En una lectura manual de 20 errores al azar, 10 eran reseñas mixtas (elogios y críticas),
  6 giros narrativos (comparan con el pasado o empiezan mal y terminan bien), 2 quejas por hechos
  sin palabras de sentimiento, 1 posible error de etiqueta y 1 caso de negación con una palabra
  fuera de la lista.
- Los errores son de poca confianza: la mediana de su probabilidad es 0,76 y solo el 25 % supera
  0,90.
- La presencia de "pero" o de negación marca la clase difícil más que causar el fallo: la
  negación aparece sobre todo en reseñas negativas.

**Limitación principal:** un modelo de bolsa de palabras no capta el contraste dentro de una
reseña ("buena comida, pero..."). Mejoras posibles: análisis por aspectos (comida, servicio,
precio) y ajustar el umbral de decisión si se prioriza detectar reseñas negativas.

## 6. Reproducibilidad

Con un solo comando, desde los datos crudos hasta el modelo guardado:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m src.models.train
```

El comando descarga y prepara los datos si faltan, entrena el modelo de `params.yaml`, guarda el
pipeline completo en `artifacts/model.joblib`, evalúa en la prueba una sola vez y exporta
`artifacts/metrics.json`, que lee la API. La semilla es fija (42) y las versiones de las
librerías están en `requirements.txt` (Python [completar versión]).

El preprocesamiento viaja dentro del pipeline guardado (`preprocess → tfidf → model`): el
entrenamiento y la API usan exactamente el mismo código, y la API nunca lo reimplementa.

## 7. Uso de asistentes de IA

Se usó un asistente de IA (Claude, de Anthropic) como apoyo en el código de entrenamiento,
evaluación y pruebas del modelado, y en la revisión del análisis de resultados. Todo el código
fue revisado, ejecutado y probado por la integrante, que puede explicarlo y modificarlo.