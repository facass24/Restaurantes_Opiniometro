"""Tabla comparativa en el conjunto de prueba (Fase 4.3).

Reporta, sobre la prueba, la línea base y el candidato que perdió en la Fase 3
(Naive Bayes + submuestreo), junto con el modelo final que ya evaluó evaluate.py.

IMPORTANTE: esto NO sirve para elegir modelo. La elección se hizo en la Fase 3 con
validación cruzada, antes de mirar la prueba. Aquí solo se reportan los números.
Cada modelo se evalúa una sola vez; el resultado queda en artifacts/comparativa_prueba.json.

Uso (desde la raíz del proyecto, después de `python -m src.models.evaluate`):
    python -m src.models.comparar_prueba
    python -m src.models.comparar_prueba --repetir
"""

import argparse
import json

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.config import ARTIFACTS_DIR, PARAMS
from src.models.data import cargar_datos, dividir
from src.models.evaluate import RUTA_RESULTADO, calcular_metricas
from src.models.experiments import (crear_modelo, crear_vectorizador, submuestrear,
                                    textos_preprocesados)

RUTA_COMPARATIVA = ARTIFACTS_DIR / "comparativa_prueba.json"


def metricas_linea_base(train, test) -> dict:
    """Predice siempre la clase más frecuente (positivo)."""
    modelo = DummyClassifier(strategy="most_frequent").fit(train["texto"], train["sentimiento"])
    return calcular_metricas(test["sentimiento"], modelo.predict(test["texto"]))


def metricas_naive_bayes(train, test) -> dict:
    """Naive Bayes + submuestreo, con la configuración de `modelo_alternativo` (params.yaml).

    El submuestreo se aplica SOLO al entrenamiento; la prueba conserva la proporción real 94/6.
    """
    cfg = PARAMS["modelo_alternativo"]
    X_train = textos_preprocesados(train["texto"], cfg["reduccion"])
    X_test = textos_preprocesados(test["texto"], cfg["reduccion"])
    X_sub, y_sub = submuestrear(X_train, train["sentimiento"], cfg["undersample_ratio"], PARAMS["seed"])
    modelo = Pipeline([
        ("tfidf", crear_vectorizador(tuple(cfg["ngram_range"]))),
        ("model", crear_modelo(cfg["modelo"], **cfg["opciones"])),
    ]).fit(X_sub, y_sub)
    return calcular_metricas(test["sentimiento"], modelo.predict(X_test))


def tabla(resultados: dict) -> str:
    """Tabla en Markdown lista para pegar en docs/resultados_experimentos.md."""
    filas = ["| Modelo | accuracy | F1 macro | precision_neg | recall_neg |", "|---|---|---|---|---|"]
    for nombre, m in resultados.items():
        neg = m["por_clase"]["negativo"]
        filas.append(f"| {nombre} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | "
                     f"{neg['precision']:.4f} | {neg['recall']:.4f} |")
    return "\n".join(filas)


def comparar_y_guardar(train, test) -> dict:
    """Evalúa línea base y Naive Bayes en la prueba, junto al modelo final ya evaluado."""
    final = json.loads(RUTA_RESULTADO.read_text(encoding="utf-8"))["metricas"]
    resultados = {
        "Línea base (siempre positivo)": metricas_linea_base(train, test),
        "Naive Bayes + submuestreo": metricas_naive_bayes(train, test),
        "Regresión Logística (modelo final)": final,
    }
    print(tabla(resultados))

    RUTA_COMPARATIVA.write_text(json.dumps({
        "modelo_alternativo": dict(PARAMS["modelo_alternativo"]),
        "n_prueba": len(test),
        "resultados": resultados,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {RUTA_COMPARATIVA}")
    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparativa de modelos en el conjunto de prueba.")
    parser.add_argument("--repetir", action="store_true", help="vuelve a evaluar en la prueba")
    args = parser.parse_args()

    if RUTA_COMPARATIVA.exists() and not args.repetir:
        print("La comparativa YA se calculó (regla: la prueba se usa una sola vez). Guardada:\n")
        print(tabla(json.loads(RUTA_COMPARATIVA.read_text(encoding="utf-8"))["resultados"]))
        return
    if not RUTA_RESULTADO.exists():
        raise SystemExit("Primero evalúa el modelo final: python -m src.models.evaluate")

    train, test = dividir(cargar_datos())
    comparar_y_guardar(train, test)


if __name__ == "__main__":
    main()