"""Análisis de errores del modelo final en el conjunto de prueba (Fase 4.4).

Responde: ¿en qué reseñas falla el modelo y por qué?
  - negativas perdidas: reseñas negativas que el modelo marca como positivas
  - falsas alarmas:     reseñas positivas que el modelo marca como negativas

Se agrupan por fenómenos lingüísticos (reseñas mixtas con "pero", negaciones, texto largo)
y por estrellas originales (2 y 4 son los casos de borde). Si el CSV crudo de Rest-Mex
está en data/raw/, se recuperan las estrellas; si no, ese bloque se omite.

Uso (desde la raíz del proyecto):
    python -m src.models.analizar_errores
"""

import json
import re

import pandas as pd

from src.config import ARTIFACTS_DIR
from src.data.preparar_datos import RUTA_CRUDA, construir_texto
from src.models.data import cargar_datos, dividir
from src.models.train import entrenar_modelo_final

NEGACIONES = {"no", "nunca", "jamás", "jamas", "tampoco", "nada"}
TIPOS = {("negativo", "positivo"): "negativa perdida", ("positivo", "negativo"): "falsa alarma"}


def cargar_estrellas() -> pd.Series | None:
    """Estrellas originales (1, 2, 4, 5) indexadas por texto. None si no está el CSV crudo."""
    if not RUTA_CRUDA.exists():
        return None
    crudo = pd.read_csv(RUTA_CRUDA)
    crudo = crudo[(crudo["Type"] == "Restaurant") & (crudo["Polarity"] != 3)].copy()
    crudo["texto"] = [construir_texto(t, r) for t, r in zip(crudo["Title"], crudo["Review"])]
    crudo = crudo[crudo["texto"] != ""].drop_duplicates(subset="texto")   # igual que preparar_datos
    return crudo.set_index("texto")["Polarity"].astype(int)


def tabla_predicciones(final, test: pd.DataFrame, estrellas: pd.Series | None) -> pd.DataFrame:
    """Cada reseña de prueba con su predicción, confianza y marcas lingüísticas."""
    proba = final.predict_proba(test["texto"].tolist())
    clases = list(final.classes_)
    palabras = test["texto"].map(lambda t: set(re.findall(r"\w+", t.lower())))
    tabla = pd.DataFrame({
        "texto": test["texto"].to_numpy(),
        "real": test["sentimiento"].to_numpy(),
        "predicho": [clases[i] for i in proba.argmax(axis=1)],
        "probabilidad": proba.max(axis=1).round(4),
        "tiene_pero": palabras.map(lambda w: "pero" in w).to_numpy(),
        "tiene_negacion": palabras.map(lambda w: bool(NEGACIONES & w)).to_numpy(),
        "texto_largo": (test["texto"].str.split().str.len() > 100).to_numpy(),
    })
    tabla["estrellas"] = test["texto"].map(estrellas).to_numpy() if estrellas is not None else pd.NA
    tabla["correcto"] = tabla["real"] == tabla["predicho"]
    tabla["tipo_error"] = [TIPOS.get((r, p), "") for r, p in zip(tabla["real"], tabla["predicho"])]
    return tabla


def por_fenomeno(tabla: pd.DataFrame) -> pd.DataFrame:
    """Accuracy con y sin cada fenómeno, y qué parte de los errores lo tiene."""
    errores = tabla[~tabla["correcto"]]
    filas = []
    for nombre, col in [('contiene "pero" (reseña mixta)', "tiene_pero"),
                        ("contiene negación", "tiene_negacion"),
                        ("texto largo (>100 palabras)", "texto_largo")]:
        con = tabla[tabla[col]]
        sin = tabla[~tabla[col]]
        filas.append({
            "fenómeno": nombre,
            "% de la prueba": 100 * len(con) / len(tabla),
            "accuracy con": con["correcto"].mean(),
            "accuracy sin": sin["correcto"].mean(),
            "% de los errores": 100 * errores[col].mean(),
        })
    return pd.DataFrame(filas)


def por_estrellas(tabla: pd.DataFrame) -> pd.DataFrame:
    grupos = tabla.dropna(subset=["estrellas"]).groupby("estrellas")["correcto"]
    return pd.DataFrame({"reseñas": grupos.size(), "accuracy": grupos.mean(),
                         "errores": grupos.apply(lambda s: int((~s).sum()))}).reset_index()


def ejemplos(tabla: pd.DataFrame, tipo: str, n: int = 5, ancho: int = 260) -> list[dict]:
    """Los errores más seguros del modelo (los más instructivos) de un tipo."""
    sel = tabla[tabla["tipo_error"] == tipo].sort_values("probabilidad", ascending=False).head(n)
    return [{"texto": t[:ancho] + ("..." if len(t) > ancho else ""), "estrellas": None if pd.isna(e) else int(e),
             "probabilidad": float(p)} for t, e, p in zip(sel["texto"], sel["estrellas"], sel["probabilidad"])]


def main() -> None:
    train, test = dividir(cargar_datos())
    final = entrenar_modelo_final(train)
    estrellas = cargar_estrellas()
    tabla = tabla_predicciones(final, test, estrellas)

    errores = tabla[~tabla["correcto"]]
    conteo = errores["tipo_error"].value_counts()
    print(f"\nErrores: {len(errores)} de {len(tabla):,} ({100 * len(errores) / len(tabla):.2f} %)")
    for tipo, n in conteo.items():
        print(f"  {tipo}: {n}")

    fen = por_fenomeno(tabla)
    print("\nPor fenómeno lingüístico:")
    print(fen.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    est = por_estrellas(tabla)
    if len(est):
        print("\nPor estrellas originales (2 y 4 son los casos de borde):")
        print(est.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    resumen = {"errores": int(len(errores)), "por_tipo": conteo.to_dict(),
               "por_fenomeno": fen.round(4).to_dict(orient="records"),
               "por_estrellas": est.round(4).to_dict(orient="records"),
               "ejemplos": {t: ejemplos(tabla, t) for t in TIPOS.values()}}
    for tipo in TIPOS.values():
        print(f"\nErrores más seguros — {tipo}:")
        for e in resumen["ejemplos"][tipo]:
            print(f"  [{e['estrellas']}★, {e['probabilidad']:.1%}] {e['texto']}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    errores.drop(columns="correcto").to_csv(ARTIFACTS_DIR / "errores.csv", index=False, encoding="utf-8")
    (ARTIFACTS_DIR / "analisis_errores.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nGuardado: {ARTIFACTS_DIR / 'errores.csv'} y analisis_errores.json")


if __name__ == "__main__":
    main()