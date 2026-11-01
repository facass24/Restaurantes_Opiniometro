"""Busqueda de hiperparametros con validacion cruzada (solo sobre train).

Uso (desde la raiz del repo):
    python -m src.models.buscar_hiperparametros                                      # unigramas 
    python -m src.models.buscar_hiperparametros --ngramas 1 2 --grid grid_bigramas   # con bigramas 
"""

import argparse
import time

from src.models.data import cargar_datos, dividir
from src.models.experiments import buscar_naive_bayes, buscar_regresion_logistica, textos_preprocesados


def main() -> None:
    parser = argparse.ArgumentParser(description="Búsqueda de hiperparámetros (solo train).")
    parser.add_argument("--ngramas", type=int, nargs=2, default=[1, 1], metavar=("MIN", "MAX"))
    parser.add_argument("--grid", default="grid", help="bloque de params.yaml con las rejillas")
    args = parser.parse_args()
    rango = tuple(args.ngramas)

    df = cargar_datos()
    train, _ = dividir(df)          # el test no se usa en esta fase
    X = textos_preprocesados(train["texto"], "stem")
    y = train["sentimiento"]

    t0 = time.perf_counter()
    lr = buscar_regresion_logistica(X, y, rango, args.grid)
    print(f"\nRegresion Logistica (GridSearchCV, F1 macro, n-gramas {rango}) [{time.perf_counter() - t0:.0f} s]\n")
    print(lr.round(4).to_string(index=False))

    t0 = time.perf_counter()
    nb = buscar_naive_bayes(X, y, rango, args.grid)
    print(f"\nNaive Bayes con submuestreo (F1 macro, n-gramas {rango}) [{time.perf_counter() - t0:.0f} s]\n")
    print(nb.round(4).to_string(index=False))


if __name__ == "__main__":
    main()