# Resultados de los experimentos de modelado

Todos los resultados son de **validación cruzada estratificada de 5 pliegues sobre el conjunto
de entrenamiento** (semilla 42). El conjunto de prueba no se usó en ningún paso.
Métrica principal: **F1 macro** (el dataset tiene 94 % de reseñas positivas y 6 % negativas).
Las columnas `precision_neg` y `recall_neg` se refieren a la clase "negativo".

Preprocesamiento base: stemming, TF-IDF con los parámetros de `params.yaml`
(`max_features`, `min_df`, `max_df`, `sublinear_tf`).

## 1. Línea base y modelos sin tratar (unigramas)

| Experimento | accuracy | F1 macro | precision_neg | recall_neg |
|---|---|---|---|---|
| Línea base (siempre "positivo") | 0,9400 | 0,4845 | – | – |
| Naive Bayes sin tratar | 0,9640 | 0,7805 | 0,9693 | 0,4137 |
| Regresión Logística sin tratar | 0,9786 | 0,8923 | 0,9319 | 0,6946 |

Comando: `python -m src.models.comparar_modelos`

## 2. Tratamiento del desbalance (unigramas)

| Experimento | accuracy | F1 macro | precision_neg | recall_neg |
|---|---|---|---|---|
| Naive Bayes, prior uniforme (`fit_prior=False`) | 0,9459 | 0,8220 | 0,5283 | 0,9288 |
| Naive Bayes + submuestreo (ratio 2) | 0,9757 | 0,8969 | 0,7737 | 0,8432 |
| Regresión Logística, `class_weight="balanced"` | 0,9659 | 0,8743 | 0,6510 | 0,9337 |
| Regresión Logística + submuestreo (ratio 2) | 0,9724 | 0,8912 | 0,7151 | 0,9007 |

El submuestreo se aplica solo a la parte de entrenamiento de cada pliegue; la validación
conserva la proporción real 94/6.

Comando: `python -m src.models.comparar_modelos`

## 3. Búsqueda de hiperparámetros (unigramas)

| Modelo | Mejor configuración | F1 macro | desviación |
|---|---|---|---|
| Regresión Logística | C = 10, sin `class_weight` | 0,9175 | 0,0034 |
| Naive Bayes + submuestreo | alpha = 2,0, ratio = 2 | 0,8985 | 0,0049 |

Con `class_weight="balanced"`, la Regresión Logística rindió menos que sin él para todos
los valores de C probados (entre 0,874 y 0,890).

Comando: `python -m src.models.buscar_hiperparametros`

## 4. Experimento propio: stemming contra lematización (unigramas)

| Modelo | Stemming | Lematización | Diferencia |
|---|---|---|---|
| Regresión Logística (C = 10) | 0,9175 | 0,9134 | −0,0041 |
| Naive Bayes + submuestreo | 0,8985 | 0,9027 | +0,0042 |

Las diferencias son del tamaño de la variación entre pliegues (0,003 a 0,006) y van en
direcciones opuestas: no hay una diferencia real. Se eligió **stemming** porque es
aproximadamente 9 veces más rápido (0,9 s contra 8,2 s por 1000 reseñas).

Comando: `python -m src.models.comparar_reduccion`

## 5. Unigramas contra unigramas + bigramas (stemming)

| Modelo | Unigramas | Unigramas + bigramas | Diferencia |
|---|---|---|---|
| Regresión Logística (C = 10) | 0,9175 | 0,9258 | +0,0083 |
| Naive Bayes + submuestreo | 0,8985 | 0,9108 | +0,0122 |

Los bigramas mejoran a los dos modelos, en la misma dirección.

Comando: `python -m src.models.comparar_ngramas`

## 6. Reajuste con bigramas

| Modelo | Mejor configuración | F1 macro | desviación |
|---|---|---|---|
| Regresión Logística | C = 20 (C = 10 queda en 0,9258) | 0,9262 | 0,0056 |
| Naive Bayes + submuestreo | alpha = 2,0, ratio = 3 (empata con alpha = 1,0, ratio = 4) | 0,9167 | 0,0029 |

Comando: `python -m src.models.buscar_hiperparametros --ngramas 1 2 --grid grid_bigramas`

## 7. Decisión

**Modelo elegido: Regresión Logística** con stemming, unigramas + bigramas y **C = 10**
(registrado en `params.yaml`, bloque `modelo_final`).

- F1 macro de validación cruzada ≈ 0,926, contra ≈ 0,917 de Naive Bayes + submuestreo.
- C = 10 y C = 20 empatan dentro del ruido (diferencia de 0,0004); se elige el más simple
  (mayor regularización).
- Mayor precisión sobre las reseñas negativas (0,91 contra 0,84).
- Sus probabilidades reflejan la proporción real de las clases, porque entrena con todos
  los datos; con submuestreo quedarían desplazadas hacia "negativo".
- Sus coeficientes permiten interpretar el aporte de cada palabra.

Limitaciones: la diferencia con Naive Bayes es moderada (~0,009, unas dos veces la
variación entre pliegues); el F1 macro de la validación cruzada es una estimación que se
verificará una sola vez con el conjunto de prueba en la Fase 4.

## 8. Evaluación final en el conjunto de prueba

Se evaluó una sola vez (16.143 reseñas, 969 negativas), con el modelo de la sección 7
(Regresión Logística, stemming, unigramas + bigramas, C = 10).

  | Clase | precisión | recall | F1 |
  |---|---|---|---|
  | negativo | 0,9187 | 0,8163 | 0,8645 |
  | positivo | 0,9884 | 0,9954 | 0,9919 |

Accuracy 0,9846 y F1 macro 0,9282, consistente con el 0,926 de la validación cruzada.
Matriz de confusión: 791 negativas y 15.104 positivas bien clasificadas; 178 negativas
pasaron por positivas y 70 positivas por negativas.

Comando: `python -m src.models.evaluate`

## 9. Comparación de modelos en el conjunto de prueba

Los tres modelos se evaluaron una sola vez sobre las mismas 16.143 reseñas de prueba
(969 negativas, 15.174 positivas). Estos números **no se usaron para elegir el modelo**:
la decisión de la sección 7 se tomó con validación cruzada, antes de mirar la prueba.

| Modelo | accuracy | F1 macro | precision_neg | recall_neg |
|---|---|---|---|---|
| Línea base (siempre positivo) | 0,9400 | 0,4845 | 0,0000 | 0,0000 |
| Naive Bayes + submuestreo | 0,9785 | 0,9074 | 0,8025 | 0,8514 |
| Regresión Logística (modelo final) | 0,9846 | 0,9282 | 0,9187 | 0,8163 |

- **Contra la línea base:** con accuracy 0,94 no detecta ninguna reseña negativa (F1 macro 0,4845).
  Los dos modelos aportan valor real.
- **Regresión Logística contra Naive Bayes:** la Regresión Logística gana en F1 macro por
  0,021 y tiene mucha mejor precisión sobre las negativas (0,92 contra 0,80): cuando
  marca una reseña como negativa, casi siempre acierta.
- **Qué cede:** Naive Bayes tiene algo más de recall sobre las negativas (0,85 contra 0,82,
  unas 34 reseñas de 969). Es decir, deja pasar menos negativas, a costa de marcar muchas
  positivas como negativas.
- **Consistencia con la validación cruzada:** la dirección es la misma (0,926 contra 0,916
  en validación cruzada); la brecha es algo mayor en la prueba (0,021 contra 0,010).

Comando: `python -m src.models.comparar_prueba`

## 10. Análisis de errores (conjunto de prueba)

Se analizaron los 248 errores del modelo final (1,54 % de las 16.143 reseñas de prueba):
178 negativas que pasaron por positivas y 70 positivas marcadas como negativas. Este
análisis describe los fallos; no se usó para cambiar el modelo.

| Estrellas originales | reseñas | accuracy | errores |
|---|---|---|---|
| 1 | 501 | 0,856 | 72 |
| 2 | 468 | 0,774 | 106 |
| 4 | 3.403 | 0,984 | 53 |
| 5 | 11.771 | 0,999 | 17 |

- El 72 % de los errores son negativas perdidas. Las de 2 estrellas son las más difíciles
  (77 % de acierto): son reseñas de borde, que la regla 1-2 = negativo etiqueta como
  negativas aunque en la muestra leída suelen mezclar elogios y críticas.
- Los errores son de poca confianza: la mediana de la probabilidad es 0,76 y el 40 % está
  por debajo de 0,70. Solo el 25 % son errores seguros (probabilidad de 0,90 o más).

### Negación y "pero"

| Fenómeno | % de la prueba | accuracy con | accuracy sin | % de los errores |
|---|---|---|---|---|
| contiene "pero" | 20,7 | 0,966 | 0,990 | 46,0 |
| contiene negación | 31,2 | 0,966 | 0,993 | 69,8 |
| texto largo (más de 100 palabras) | 7,6 | 0,978 | 0,985 | 10,9 |

A primera vista la negación concentra la mayoría de los errores, pero la lectura es
engañosa: la negación aparece sobre todo en reseñas negativas (14,6 % de las reseñas con
negación son negativas, contra 2,1 % de las que no la tienen), y las negativas son la
clase difícil. Dentro de la clase negativa, las reseñas con negación se clasifican mejor
(accuracy 0,851) que las que no la tienen (0,755); con "pero" no hay diferencia (0,820
contra 0,833). Estas marcas identifican la clase difícil más que explicar el fallo.

### Lectura manual de 20 errores al azar (semilla 42)

13 eran reseñas mixtas (elogios y críticas en la misma reseña, con o sin "pero"), 4 giros
narrativos (compara con el pasado o empieza mal y termina bien), 2 quejas por hechos sin
palabras de sentimiento y 1 posible error de etiqueta.

### Posibles errores de etiqueta

Algunas etiquetas parecen inconsistentes con el texto (por ejemplo, "Muy buen restaurante...
Recomendable" con 1 estrella, o "Malísimo... pésimo servicio" con 5). Parte de los errores
son del dato y no del modelo, así que el desempeño real podría ser algo mejor.

### Limitaciones

Un modelo de bolsa de palabras no capta bien el contraste dentro de una reseña ("buena
comida, pero..."), que es el patrón de error más frecuente. Mejoras posibles: análisis
por aspectos (comida, servicio, precio) y ajustar el umbral de decisión si se prioriza
detectar negativas.

Comando: `python -m src.models.analizar_errores`