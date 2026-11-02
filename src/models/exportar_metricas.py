"""Exporta artifacts/metrics.json en el formato que lee la API.

No vuelve a evaluar nada: toma los resultados ya calculados sobre el conjunto de prueba
(resultado_prueba.json del modelo final y comparativa_prueba.json del alternativo).

Uso (desde la raiz del proyecto):
    python -m src.models.exportar_metricas
"""

import json

from src.config import METRICS_PATH
from src.models.comparar_prueba import RUTA_COMPARATIVA
from src.models.evaluate import RUTA_RESULTADO
from src.models.train import NOMBRES_MODELO


def a_metricas_api(m: dict) -> dict:
    """Pasa las metricas de evaluate.py al formato MetricasDTO de la API (promedios macro).

    La matriz de confusion queda con filas = clase real y columnas = clase predicha,
    en el orden [negativo, positivo].
    """
    return {
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "precision": m["precision_macro"],
        "recall": m["recall_macro"],
        "matriz_confusion": m["matriz_confusion"],
    }


def describir(cfg: dict, n_prueba: int, submuestreo: int | None = None) -> str:
    ngramas = "unigramas + bigramas" if list(cfg["ngram_range"]) == [1, 2] else "unigramas"
    reduccion = "stemming" if cfg["reduccion"] == "stem" else "lematización"
    opciones = ", ".join(f"{k}={v}" for k, v in cfg["opciones"].items())
    texto = f"TF-IDF con {ngramas} y {reduccion}; {opciones}"
    if submuestreo:
        texto += f"; submuestreo (positivas = {submuestreo} x negativas)"
    n = f"{n_prueba:,}".replace(",", ".")
    return f"{texto}. Metricas sobre {n} reseñas de prueba."


def main() -> None:
    if not RUTA_RESULTADO.exists() or not RUTA_COMPARATIVA.exists():
        raise SystemExit("Faltan los resultados de prueba. Ejecuta primero:\n"
                         "  python -m src.models.evaluate\n  python -m src.models.comparar_prueba")

    final = json.loads(RUTA_RESULTADO.read_text(encoding="utf-8"))
    comparativa = json.loads(RUTA_COMPARATIVA.read_text(encoding="utf-8"))
    cfg_final, cfg_alt = final["modelo_final"], comparativa["modelo_alternativo"]
    n_prueba = final["n_prueba"]
    nombre_final = NOMBRES_MODELO[cfg_final["modelo"]]

    modelos = [
        {
            "nombre": nombre_final,
            "identificador": cfg_final["modelo"],
            "descripcion": describir(cfg_final, n_prueba),
            "es_activo": True,
            "metricas": a_metricas_api(final["metricas"]),
        },
        {
            "nombre": NOMBRES_MODELO[cfg_alt["modelo"]],
            "identificador": cfg_alt["modelo"],
            "descripcion": describir(cfg_alt, n_prueba, cfg_alt.get("undersample_ratio")),
            "es_activo": False,
            "metricas": a_metricas_api(comparativa["resultados"]["Naive Bayes + submuestreo"]),
        },
    ]

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"modelo_produccion": nombre_final, "modelos": modelos},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado en {METRICS_PATH}")
    for m in modelos:
        met = m["metricas"]
        activo = " (en producción)" if m["es_activo"] else ""
        print(f"  {m['nombre']}{activo}: F1 macro {met['f1_macro']:.4f} | accuracy {met['accuracy']:.4f}")


if __name__ == "__main__":
    main()