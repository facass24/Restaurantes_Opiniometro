"""Pruebas del modelado con un dataset sintético pequeño (rápidas, sin descargar Rest-Mex).

Cubren lo que se defiende en la Fase 5: división estratificada y reproducible, un solo
preprocesamiento en entrenamiento y producción, pipeline que se guarda y se recarga,
métricas con la matriz de confusión bien orientada y formato compatible con la API.
"""

import random

import joblib
import pandas as pd
import pytest

from api.schemas import MetricasResponse
from src.models import experiments
from src.models.comparar_prueba import metricas_linea_base
from src.models.data import dividir
from src.models.evaluate import calcular_metricas
from src.models.exportar_metricas import a_metricas_api, describir
from src.models.train import NOMBRES_MODELO, entrenar_modelo_final

POSITIVAS = ["la comida estuvo deliciosa", "excelente servicio y muy amables", "todo riquísimo, lo recomiendo",
             "me encantó el lugar, ambiente agradable", "platillos frescos y atención rápida"]
NEGATIVAS = ["la comida estaba fría y mala", "pésimo servicio, muy lentos", "no me gustó, todo caro",
             "horrible experiencia, nunca volvería", "platillos crudos y atención grosera"]
EXTRAS = ["fuimos el sábado", "pedimos tacos", "estaba lleno", "con mi familia", "en la noche", "con amigos"]


def _dataset(n_pos: int = 160, n_neg: int = 40) -> pd.DataFrame:
    """Reseñas sintéticas desbalanceadas (80 / 20), todas distintas entre sí."""
    azar = random.Random(0)
    filas = []
    for i in range(n_pos + n_neg):
        positivo = i < n_pos
        base = azar.choice(POSITIVAS if positivo else NEGATIVAS)
        filas.append({"texto": f"{base} {azar.choice(EXTRAS)} {i}",
                      "sentimiento": "positivo" if positivo else "negativo"})
    return pd.DataFrame(filas)


@pytest.fixture()
def cache_temporal(tmp_path, monkeypatch):
    """Las pruebas no deben escribir en artifacts/cache del proyecto."""
    monkeypatch.setattr(experiments, "CARPETA_CACHE", tmp_path / "cache")
    return tmp_path


@pytest.fixture()
def modelo_entrenado(cache_temporal):
    train, test = dividir(_dataset())
    return entrenar_modelo_final(train), train, test


def test_division_reproducible_y_sin_mezclar():
    df = _dataset()
    train1, test1 = dividir(df)
    train2, test2 = dividir(df)
    assert test1["texto"].tolist() == test2["texto"].tolist()      # misma semilla, misma división
    assert not set(train1["texto"]) & set(test1["texto"])          # nada de la prueba en entrenamiento
    assert len(train1) + len(test1) == len(df)


def test_division_estratificada_conserva_las_proporciones():
    train, test = dividir(_dataset())
    prop = lambda d: (d["sentimiento"] == "negativo").mean()
    assert abs(prop(train) - prop(test)) < 0.02


def test_el_pipeline_final_recibe_texto_crudo(modelo_entrenado):
    final, _, _ = modelo_entrenado
    assert list(final.named_steps) == ["preprocess", "tfidf", "model"]
    assert list(final.classes_) == ["negativo", "positivo"]
    assert final.predict(["No me gustó, pésimo servicio y comida fría"])[0] == "negativo"
    assert final.predict(["Excelente atención, todo riquísimo"])[0] == "positivo"
    assert final.nombre_modelo == NOMBRES_MODELO["logistic_regression"]


def test_pipeline_guardado_y_recargado_predice_igual(modelo_entrenado, tmp_path):
    final, _, test = modelo_entrenado
    ruta = tmp_path / "model.joblib"
    joblib.dump(final, ruta)
    recargado = joblib.load(ruta)
    textos = test["texto"].tolist()
    assert (recargado.predict_proba(textos) == final.predict_proba(textos)).all()
    assert recargado.nombre_modelo == final.nombre_modelo


def test_la_cache_de_preprocesamiento_no_cambia_el_resultado(cache_temporal):
    textos = _dataset()["texto"].head(30)
    primera = experiments.textos_preprocesados(textos, "stem")     # calcula y guarda
    segunda = experiments.textos_preprocesados(textos, "stem")     # lee de la caché
    assert primera.tolist() == segunda.tolist()
    assert len(list((cache_temporal / "cache").glob("*.joblib"))) == 1


def test_la_huella_del_preprocesamiento_es_estable():
    assert experiments.huella_nlp() == experiments.huella_nlp()
    assert len(experiments.huella_nlp()) == 8


def test_matriz_de_confusion_orientada_filas_real_columnas_predicho():
    real = ["negativo", "negativo", "positivo", "positivo"]
    pred = ["negativo", "positivo", "positivo", "positivo"]
    m = calcular_metricas(real, pred)
    assert m["matriz_confusion"] == [[1, 1], [0, 2]]               # [negativo, positivo]
    assert m["por_clase"]["negativo"]["recall"] == 0.5
    assert m["por_clase"]["positivo"]["precision"] == pytest.approx(0.6667, abs=1e-4)


def test_linea_base_tiene_accuracy_alta_pero_f1_macro_bajo(cache_temporal):
    train, test = dividir(_dataset())
    m = metricas_linea_base(train, test)
    assert m["accuracy"] == pytest.approx(0.8, abs=0.05)           # acierta la mayoría
    assert m["f1_macro"] < 0.5                                     # pero no detecta ninguna negativa
    assert m["por_clase"]["negativo"]["recall"] == 0.0


def test_metricas_exportadas_cumplen_el_esquema_de_la_api():
    real = ["negativo", "positivo", "positivo", "negativo"]
    m = calcular_metricas(real, ["negativo", "positivo", "positivo", "positivo"])
    cfg = {"reduccion": "stem", "ngram_range": [1, 2], "opciones": {"C": 10}}
    datos = {
        "modelo_produccion": "Regresión Logística (TF-IDF)",
        "modelos": [{"nombre": "Regresión Logística (TF-IDF)", "identificador": "logistic_regression",
                     "descripcion": describir(cfg, 4), "es_activo": True, "metricas": a_metricas_api(m)}],
    }
    respuesta = MetricasResponse(**datos)
    metricas = respuesta.modelos[0].metricas
    assert metricas.f1_macro == m["f1_macro"]
    # La clase minoritaria (negativo) debe llegar a la API, no quedar en null
    assert metricas.recall_negativo == m["por_clase"]["negativo"]["recall"]
    assert metricas.precision_negativo is not None and metricas.f1_negativo is not None