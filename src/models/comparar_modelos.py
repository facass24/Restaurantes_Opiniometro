"""Compara la línea base, Naive Bayes y Regresión Logística con validación cruzada.

Uso (desde la raíz del repo):  python -m src.models.comparar_modelos
"""

import pandas as pd

from src.models.data import cargar_datos, dividir
from src.models.experiments import evaluar_linea_base, evaluar_modelo, textos_preprocesados


def main() -> None:
    df = cargar_datos()
    train, _ = dividir(df)          # el test no se usa en esta fase
    X = textos_preprocesados(train["texto"], "stem")
    y = train["sentimiento"]

    base = evaluar_linea_base(train)
    filas = [{"modelo": "línea base (siempre positivo)", **base}]
    for nombre in ("naive_bayes", "logistic_regression"):
        filas.append({"modelo": nombre, **evaluar_modelo(nombre, X, y)})

    tabla = pd.DataFrame(filas)
    print("\nValidación cruzada sobre train (stemming, unigramas, sin tratar el desbalance)\n")
    print(tabla.round(4).to_string(index=False))


if __name__ == "__main__":
    main()