"""Entrenamiento del modelo final.

Entrena, con TODO el conjunto de entrenamiento, la configuracion elegida en
la Fase 3 (bloque `modelo_final` de params.yaml) y devuelve el Pipeline completo.
Fase 5 extiende este archivo para que todo corra con un solo comando.

El conjunto de prueba NO se toca aqui: se usa una sola vez en evaluate.py (4.2).

Uso (desde la raiz del proyecto):
    python -m src.models.train
"""

import time

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import PARAMS
from src.models.data import cargar_datos, dividir
from src.models.experiments import crear_modelo, crear_vectorizador, textos_preprocesados
from src.nlp.pipeline import PreprocesadorTexto

NOMBRES_MODELO = {
    "logistic_regression": "Regresión Logística (TF-IDF)",
    "naive_bayes": "Naive Bayes multinomial (TF-IDF)",
}


def entrenar_modelo_final(train: pd.DataFrame) -> Pipeline:
    """Entrena la configuración de `modelo_final` y devuelve el Pipeline completo.

    Pasos:
      1. Preprocesar los textos de entrenamiento una sola vez (usa la caché de artifacts/cache).
      2. Ajustar TF-IDF + clasificador sobre esos textos.
      3. Ensamblar el Pipeline final: preprocess -> tfidf -> model. Recibe texto crudo,
         así el preprocesamiento viaja junto al modelo y la API no lo reimplementa.
    """
    config = PARAMS["modelo_final"]
    reduccion = config["reduccion"]
    ngram_range = tuple(config["ngram_range"])
    opciones = config.get("opciones", {})

    # 1) El preprocesador no aprende nada de los datos, así que precalcularlo es
    #    equivalente a ejecutarlo dentro del Pipeline, pero no se repite en cada ejecución.
    textos = textos_preprocesados(train["texto"], reduccion)

    # 2) El vocabulario TF-IDF y los pesos del modelo se aprenden SOLO con train.
    nucleo = Pipeline([
        ("tfidf", crear_vectorizador(ngram_range)),
        ("model", crear_modelo(config["modelo"], **opciones)),
    ])
    nucleo.fit(textos, train["sentimiento"])

    # 3) Pipeline de producción con los pasos ya ajustados (no se vuelve a entrenar).
    final = Pipeline([
        ("preprocess", PreprocesadorTexto(metodo_reduccion=reduccion)),
        ("tfidf", nucleo.named_steps["tfidf"]),
        ("model", nucleo.named_steps["model"]),
    ])
    final.nombre_modelo = NOMBRES_MODELO[config["modelo"]]  # lo muestra la API

    verificar_consistencia(final, nucleo, train, textos)
    return final


def verificar_consistencia(final: Pipeline, nucleo: Pipeline, train: pd.DataFrame,
                           textos: pd.Series, n: int = 100) -> None:
    """Comprueba que el Pipeline con texto crudo da las mismas probabilidades que el
    entrenamiento sobre texto preprocesado (mismo preprocesamiento en train y producción)."""
    muestra = train["texto"].head(n)
    desde_crudo = final.predict_proba(muestra.tolist())
    desde_preprocesado = nucleo.predict_proba(textos.loc[muestra.index])
    if not (abs(desde_crudo - desde_preprocesado) < 1e-9).all():
        raise AssertionError("El Pipeline guardado no reproduce lo entrenado")


def main() -> None:
    df = cargar_datos()
    train, _ = dividir(df)   # la prueba se descarta: no se usa hasta evaluate.py
    print(f"Entrenamiento: {len(train):,} reseñas | {train['sentimiento'].value_counts().to_dict()}")
    print(f"Configuración: {dict(PARAMS['modelo_final'])}")

    inicio = time.perf_counter()
    final = entrenar_modelo_final(train)
    print(f"Entrenado en {time.perf_counter() - inicio:.1f} s | "
          f"vocabulario: {len(final.named_steps['tfidf'].vocabulary_):,} términos | "
          f"clases: {list(final.classes_)}")

    print("\nPredicciones de ejemplo (texto crudo -> clase, probabilidad):")
    ejemplos = [
        "La comida estaba riquísima y el servicio excelente",
        "No me gustó nada, pésimo servicio y comida fría",
        "La comida riquísima pero el servicio muy lento",
    ]
    for texto, probas in zip(ejemplos, final.predict_proba(ejemplos)):
        i = probas.argmax()
        print(f"  {final.classes_[i]:<9} {probas[i]:.2%}  <- {texto}")


if __name__ == "__main__":
    main()