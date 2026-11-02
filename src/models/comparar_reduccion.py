"""Experimento propio: stemming contra lematización (mismos modelos e hiperparámetros).

Uso (desde la raíz del repo):  python -m src.models.comparar_reduccion
La primera vez, la lematización del train tarda varios minutos; luego queda en caché.
"""

import pandas as pd

from src.models.data import cargar_datos, dividir
from src.models.experiments import evaluar_candidatos, textos_preprocesados
from src.nlp.reducer import _NLP_SPACY


def main() -> None:
    if _NLP_SPACY is None:
        raise SystemExit("spaCy no cargó: 'lema' haría stemming y la comparación no tendría sentido.")

    df = cargar_datos()
    train, _ = dividir(df)          # el test no se usa en esta fase
    y = train["sentimiento"]

    tablas = []
    for metodo in ("stem", "lema"):
        X = textos_preprocesados(train["texto"], metodo)
        tabla = evaluar_candidatos(X, y)
        tabla.insert(0, "reduccion", metodo)
        tablas.append(tabla)

    todo = pd.concat(tablas, ignore_index=True)
    print("\nStemming contra lematización (validación cruzada sobre train, unigramas)\n")
    print(todo.round(4).to_string(index=False))

    f1 = todo.pivot(index="modelo", columns="reduccion", values="f1_macro")
    f1["lema - stem"] = f1["lema"] - f1["stem"]
    print("\nF1 macro por modelo y diferencia (lema - stem)\n")
    print(f1.round(4).to_string())


if __name__ == "__main__":
    main()