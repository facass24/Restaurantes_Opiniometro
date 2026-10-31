"""Búsqueda de hiperparámetros con validación cruzada (solo sobre train).

Uso (desde la raíz del repo):  python -m src.models.buscar_hiperparametros
"""

import time

from src.models.data import cargar_datos, dividir
from src.models.experiments import buscar_naive_bayes, buscar_regresion_logistica, textos_preprocesados


def main() -> None:
    df = cargar_datos()
    train, _ = dividir(df)          # el test no se usa en esta fase
    X = textos_preprocesados(train["texto"], "stem")
    y = train["sentimiento"]

    t0 = time.perf_counter()
    lr = buscar_regresion_logistica(X, y)
    print(f"\nRegresión Logística (GridSearchCV, F1 macro) [{time.perf_counter() - t0:.0f} s]\n")
    print(lr.round(4).to_string(index=False))

    t0 = time.perf_counter()
    nb = buscar_naive_bayes(X, y)
    print(f"\nNaive Bayes con submuestreo (F1 macro) [{time.perf_counter() - t0:.0f} s]\n")
    print(nb.round(4).to_string(index=False))


if __name__ == "__main__":
    main()