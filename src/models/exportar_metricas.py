"""Exporta artifacts/metrics.json en el formato que lee la API.

No vuelve a evaluar nada: toma los resultados ya calculados sobre el conjunto de prueba
(resultado_prueba.json del modelo final y comparativa_prueba.json del alternativo).

Uso (desde la raiz del proyecto):
    python -m src.models.exportar_metricas
"""

import json
import unicodedata

from src.config import METRICS_PATH
from src.models.comparar_prueba import RUTA_COMPARATIVA
from src.models.evaluate import RUTA_RESULTADO
from src.models.train import NOMBRES_MODELO


def a_metricas_api(m: dict) -> dict:
    """Pasa las métricas de evaluate.py al formato MetricasDTO de la API.

    precision y recall son promedios macro. La clase minoritaria (negativo) va aparte, porque
    es la que importa en este problema. La matriz de confusión queda con filas = clase real y
    columnas = clase predicha, en el orden [negativo, positivo].
    """
    neg = m["por_clase"]["negativo"]
    return {
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "precision": m["precision_macro"],
        "recall": m["recall_macro"],
        "matriz_confusion": m["matriz_confusion"],
        "precision_negativo": neg["precision"],
        "recall_negativo": neg["recall"],
        "f1_negativo": neg["f1"],
    }


def buscar_resultado(resultados: dict, clave: str) -> dict:
    """Busca un modelo en comparativa_prueba.json por una parte de su nombre (sin tildes ni mayúsculas)."""
    for nombre, metricas in resultados.items():
        sin_tildes = "".join(c for c in unicodedata.normalize("NFD", nombre.lower())
                             if unicodedata.category(c) != "Mn")
        if clave in sin_tildes:
            return metricas
    raise KeyError(f"No hay un modelo con '{clave}' en comparativa_prueba.json: {list(resultados)}")


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

    resultados = comparativa["resultados"]
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
            "metricas": a_metricas_api(buscar_resultado(resultados, "naive")),
        },
        {
            "nombre": "Línea base (siempre positivo)",
            "identificador": "baseline",
            "descripcion": "Predice siempre la clase más frecuente (positivo). Es la referencia "
                           f"mínima que los modelos deben superar. Métricas sobre {n_prueba:,} reseñas de prueba."
                           .replace(",", "."),
            "es_activo": False,
            "metricas": a_metricas_api(buscar_resultado(resultados, "base")),
        },
    ]

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"modelo_produccion": nombre_final, "modelos": modelos},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado en {METRICS_PATH}")
    for m in modelos:
        met = m["metricas"]
        activo = " (en producción)" if m["es_activo"] else ""
        print(f"  {m['nombre']}{activo}: F1 macro {met['f1_macro']:.4f} | accuracy {met['accuracy']:.4f} "
              f"| recall neg. {met['recall_negativo']:.4f}")


if __name__ == "__main__":
    main()