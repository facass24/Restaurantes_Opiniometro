"""Unigramas contra unigramas + bigramas (mismos modelos e hiperparámetros).

Uso (desde la raíz del repo):  python -m src.models.comparar_ngramas
"""

import pandas as pd

from src.models.data import cargar_datos, dividir
from src.models.experiments import evaluar_candidatos, textos_preprocesados

RANGOS = {"unigramas": (1, 1), "unigramas + bigramas": (1, 2)}


def main() -> None:
    df = cargar_datos()
    train, _ = dividir(df)          # el test no se usa en esta fase
    y = train["sentimiento"]
    X = textos_preprocesados(train["texto"], "stem")

    tablas = []
    for nombre, rango in RANGOS.items():
        tabla = evaluar_candidatos(X, y, ngram_range=rango)
        tabla.insert(0, "ngramas", nombre)
        tablas.append(tabla)

    todo = pd.concat(tablas, ignore_index=True)
    print("\nUnigramas contra unigramas + bigramas (validación cruzada sobre train, stemming)\n")
    print(todo.round(4).to_string(index=False))

    f1 = todo.pivot(index="modelo", columns="ngramas", values="f1_macro")
    f1["bigramas - unigramas"] = f1["unigramas + bigramas"] - f1["unigramas"]
    print("\nF1 macro por modelo y diferencia (bigramas - unigramas)\n")
    print(f1.round(4).to_string())


if __name__ == "__main__":
    main()