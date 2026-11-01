"""Evaluación del modelo final en el conjunto de prueba (Fase 4.2).

El conjunto de prueba se usa UNA sola vez, con el modelo ya elegido en la Fase 3.
Para que esa regla se cumpla de verdad, el resultado se guarda en
artifacts/resultado_prueba.json y una segunda ejecución se niega a repetirlo, salvo
que se pida explícitamente (porque cambió el modelo o el preprocesamiento).

Uso (desde la raíz del proyecto):
    python -m src.models.evaluate              # evalúa y guarda
    python -m src.models.evaluate --repetir    # vuelve a evaluar (lo decides tú, con motivo)
"""

import argparse
import json

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.config import ARTIFACTS_DIR, PARAMS
from src.models.data import cargar_datos, dividir
from src.models.train import entrenar_modelo_final

ETIQUETAS = ["negativo", "positivo"]   # orden fijo: filas/columnas de la matriz de confusión
RUTA_RESULTADO = ARTIFACTS_DIR / "resultado_prueba.json"


def calcular_metricas(y_real, y_pred) -> dict:
    """Accuracy, precisión/recall/F1 por clase, promedios macro y matriz de confusión.

    Matriz: filas = clase real, columnas = clase predicha, en el orden de ETIQUETAS.
        [[negativas bien, negativas -> positivo],
         [positivas -> negativo, positivas bien]]
    """
    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_real, y_pred, labels=ETIQUETAS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_real, y_pred, average="macro",
                                         labels=ETIQUETAS, zero_division=0)), 4),
        "precision_macro": round(float(precision.mean()), 4),
        "recall_macro": round(float(recall.mean()), 4),
        "por_clase": {
            clase: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                    "f1": round(float(f), 4), "soporte": int(s)}
            for clase, p, r, f, s in zip(ETIQUETAS, precision, recall, f1, soporte)
        },
        "matriz_confusion": confusion_matrix(y_real, y_pred, labels=ETIQUETAS).tolist(),
    }


def evaluar_en_prueba(final, test: pd.DataFrame) -> dict:
    """Predice con el Pipeline de producción (texto CRUDO) y calcula las métricas."""
    y_pred = final.predict(test["texto"].tolist())
    return calcular_metricas(test["sentimiento"], y_pred)


def mostrar(metricas: dict) -> None:
    pc = metricas["por_clase"]
    print(f"\n{'':<10}{'precisión':>10}{'recall':>9}{'F1':>8}{'soporte':>9}")
    for clase in ETIQUETAS:
        m = pc[clase]
        print(f"{clase:<10}{m['precision']:>10.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}{m['soporte']:>9,}")
    print(f"\naccuracy = {metricas['accuracy']:.4f} | F1 macro = {metricas['f1_macro']:.4f} "
          f"| precisión macro = {metricas['precision_macro']:.4f} | recall macro = {metricas['recall_macro']:.4f}")
    (nn, np_), (pn, pp) = metricas["matriz_confusion"]
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(f"{'':<18}{'pred. negativo':>16}{'pred. positivo':>16}")
    print(f"{'real negativo':<18}{nn:>16,}{np_:>16,}")
    print(f"{'real positivo':<18}{pn:>16,}{pp:>16,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el modelo final en el conjunto de prueba.")
    parser.add_argument("--repetir", action="store_true",
                        help="vuelve a usar el conjunto de prueba aunque ya se haya evaluado")
    args = parser.parse_args()

    if RUTA_RESULTADO.exists() and not args.repetir:
        print("El conjunto de prueba YA se evaluó (regla: se usa una sola vez). Resultado guardado:")
        guardado = json.loads(RUTA_RESULTADO.read_text(encoding="utf-8"))
        print(f"  modelo: {guardado['modelo_final']}")
        mostrar(guardado["metricas"])
        print("\nSi cambiaste el modelo o el preprocesamiento, usa --repetir.")
        return

    df = cargar_datos()
    train, test = dividir(df)
    print(f"Entrenamiento: {len(train):,} | prueba: {len(test):,} | {test['sentimiento'].value_counts().to_dict()}")
    final = entrenar_modelo_final(train)

    metricas = evaluar_en_prueba(final, test)
    mostrar(metricas)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADO.write_text(json.dumps({
        "modelo_final": dict(PARAMS["modelo_final"]),
        "n_entrenamiento": len(train),
        "n_prueba": len(test),
        "metricas": metricas,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {RUTA_RESULTADO}")


if __name__ == "__main__":
    main()