"""Compara la línea base, Naive Bayes y Regresión Logística con validación cruzada.

Uso (desde la raíz del repo):  python -m src.models.comparar_modelos
"""

import pandas as pd

from src.models.data import cargar_datos, dividir
from src.models.experiments import (
    evaluar_linea_base,
    evaluar_modelo,
    evaluar_modelo_submuestreo,
    textos_preprocesados,
)

# (etiqueta, modelo, opciones del modelo, ¿con submuestreo?)
EXPERIMENTOS = [
    ("NB sin tratar", "naive_bayes", {}, False),
    ("NB prior uniforme (fit_prior=False)", "naive_bayes", {"fit_prior": False}, False),
    ("NB + submuestreo", "naive_bayes", {}, True),
    ("LR sin tratar", "logistic_regression", {}, False),
    ("LR class_weight=balanced", "logistic_regression", {"class_weight": "balanced"}, False),
    ("LR + submuestreo", "logistic_regression", {}, True),
]


def main() -> None:
    df = cargar_datos()
    train, _ = dividir(df)          # el test no se usa en esta fase
    X = textos_preprocesados(train["texto"], "stem")
    y = train["sentimiento"]

    filas = [{"experimento": "línea base (siempre positivo)", **evaluar_linea_base(train)}]
    for etiqueta, modelo, opciones, submuestreo in EXPERIMENTOS:
        evaluar = evaluar_modelo_submuestreo if submuestreo else evaluar_modelo
        filas.append({"experimento": etiqueta, **evaluar(modelo, X, y, **opciones)})

    tabla = pd.DataFrame(filas)
    print("\nValidación cruzada sobre train (stemming, unigramas)\n")
    print(tabla.round(4).to_string(index=False))


if __name__ == "__main__":
    main()