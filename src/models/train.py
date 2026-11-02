"""Entrenamiento reproducible: de los datos crudos a los modelos guardados, con un solo comando.

    python -m src.models.train

Pasos (cada uno se salta solo si ya está hecho):
  1. Datos: descarga Rest-Mex y genera data/raw/restaurantes.csv si faltan.
  2. Entrena la configuración de `modelo_final` (params.yaml) con TODO el entrenamiento.
  3. Guarda el Pipeline completo (preprocesamiento + TF-IDF + modelo) en artifacts/model.joblib.
  4. Evalúa en la prueba, compara con la línea base y Naive Bayes, y exporta artifacts/metrics.json.

El paso 4 respeta la regla de usar la prueba una sola vez: si ya hay resultados y la
configuración no cambió, los reutiliza; si cambió, se niega, salvo con --repetir-evaluacion.
"""

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ARTIFACTS_DIR, BASE_DIR, MODEL_PATH, PARAMS
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


def comprobar_recursos() -> None:
    """Falla pronto y con instrucciones si falta el corpus de stop words de NLTK.

    Sin él, el filtro usa en silencio una lista de respaldo y los resultados dejan de ser
    reproducibles entre computadoras.
    """
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        raise SystemExit("Falta el corpus de stop words de NLTK. Instálalo con:\n"
                         "  python -m nltk.downloader stopwords")


def asegurar_datos() -> None:
    """Deja listo data/raw/restaurantes.csv (descarga y prepara solo lo que falte)."""
    from data.descargar_datos import descargar
    from src.data.preparar_datos import RUTA_CRUDA, preparar_datos

    if (BASE_DIR / PARAMS["data"]["raw_path"]).exists():
        return
    if not RUTA_CRUDA.exists():
        descargar()
    preparar_datos()


def guardar_modelo(final: Pipeline) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Pipeline guardado en {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


def evaluar_y_exportar(final: Pipeline, train: pd.DataFrame, test: pd.DataFrame, repetir: bool) -> None:
    """Paso 4: métricas en la prueba (una sola vez) y metrics.json para la API."""
    from src.models import exportar_metricas
    from src.models.comparar_prueba import RUTA_COMPARATIVA, comparar_y_guardar
    from src.models.evaluate import RUTA_RESULTADO, evaluar_y_guardar

    if RUTA_RESULTADO.exists() and not repetir:
        guardado = json.loads(RUTA_RESULTADO.read_text(encoding="utf-8"))["modelo_final"]
        if guardado != dict(PARAMS["modelo_final"]):
            raise SystemExit(
                "La configuración de modelo_final cambió desde la última evaluación en la prueba.\n"
                "Repetirla es una decisión tuya: usa --repetir-evaluacion y deja el motivo en el commit.")
        print("El conjunto de prueba ya se evaluó con esta configuración: se reutilizan los resultados.")
    else:
        evaluar_y_guardar(final, train, test)
        comparar_y_guardar(train, test)
    if not RUTA_COMPARATIVA.exists():
        comparar_y_guardar(train, test)
    exportar_metricas.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo final y genera los artefactos.")
    parser.add_argument("--repetir-evaluacion", action="store_true",
                        help="vuelve a usar el conjunto de prueba aunque ya se haya evaluado")
    parser.add_argument("--sin-evaluar", action="store_true",
                        help="solo entrena y guarda model.joblib (no toca la prueba)")
    args = parser.parse_args()

    comprobar_recursos()
    asegurar_datos()
    train, test = dividir(cargar_datos())
    print(f"Entrenamiento: {len(train):,} reseñas | {train['sentimiento'].value_counts().to_dict()}")
    print(f"Configuración: {dict(PARAMS['modelo_final'])}")

    inicio = time.perf_counter()
    final = entrenar_modelo_final(train)
    print(f"Entrenado en {time.perf_counter() - inicio:.1f} s | "
          f"vocabulario: {len(final.named_steps['tfidf'].vocabulary_):,} términos | "
          f"clases: {list(final.classes_)}")
    guardar_modelo(final)

    print("\nPredicciones de ejemplo (texto crudo -> clase, probabilidad):")
    ejemplos = [
        "La comida estaba riquísima y el servicio excelente",
        "No me gustó nada, pésimo servicio y comida fría",
        "La comida riquísima pero el servicio muy lento",
    ]
    for texto, probas in zip(ejemplos, final.predict_proba(ejemplos)):
        i = probas.argmax()
        print(f"  {final.classes_[i]:<9} {probas[i]:.2%}  <- {texto}")

    if not args.sin_evaluar:
        print()
        evaluar_y_exportar(final, train, test, args.repetir_evaluacion)


if __name__ == "__main__":
    main()