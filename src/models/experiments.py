"""Linea base, validacion cruzada, modelos y pipeline."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import BASE_DIR, PARAMS
from src.nlp.pipeline import PreprocesadorTexto

import hashlib
import joblib

def crear_cv() -> StratifiedKFold:
    """Validacion cruzada estratificada, con semilla y pliegues desde params.yaml."""
    return StratifiedKFold(
        n_splits=PARAMS["experiments"]["cv_folds"],
        shuffle=True,
        random_state=PARAMS["seed"],
    )


def evaluar_linea_base(train: pd.DataFrame) -> dict:
    """Linea base: predecir siempre la clase mas frecuente (solo sobre train)."""
    modelo = DummyClassifier(strategy="most_frequent")
    resultado = cross_validate(
        modelo,
        train["texto"],
        train["sentimiento"],
        cv=crear_cv(),
        scoring={
            "accuracy": "accuracy",
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        },
    )
    return {
        "accuracy": float(resultado["test_accuracy"].mean()),
        "f1_macro": float(resultado["test_f1_macro"].mean()),
    }


def crear_modelo(nombre: str):
    """Devuelve el clasificador final del pipeline."""
    if nombre == "naive_bayes":
        return MultinomialNB()
    if nombre == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=PARAMS["seed"])
    raise ValueError(f"Modelo desconocido: {nombre}")


def crear_vectorizador(ngram_range=(1, 1)) -> TfidfVectorizer:
    """TF-IDF con los parametros de params.yaml (vocabulario, min_df, max_df, sublinear)."""
    return TfidfVectorizer(
        token_pattern=r"\S+",   # el preprocesador ya separo los tokens por espacios
        lowercase=False,         # ya esta en minusculas: no repetir trabajo
        ngram_range=tuple(ngram_range),
        **PARAMS["tfidf"],
    )


def construir_pipeline(modelo: str, metodo_reduccion: str = "stem", ngram_range=(1, 1)) -> Pipeline:
    """Pipeline completo: texto crudo entra, etiqueta y probabilidad salen."""
    return Pipeline([
        ("preprocess", PreprocesadorTexto(metodo_reduccion=metodo_reduccion)),
        ("tfidf", crear_vectorizador(ngram_range)),
        ("model", crear_modelo(modelo)),
    ])


CARPETA_CACHE = BASE_DIR / "artifacts" / "cache"

def textos_preprocesados(textos: pd.Series, metodo: str) -> pd.Series:
    """Preprocesa una sola vez y guarda el resultado en artifacts/cache/."""
    huella = hashlib.md5("\n".join(textos).encode("utf-8")).hexdigest()[:10]
    ruta = CARPETA_CACHE / f"{metodo}_{huella}.joblib"
    if ruta.exists():
        return pd.Series(joblib.load(ruta), index=textos.index)
    salida = PreprocesadorTexto(metodo_reduccion=metodo).transform(textos)
    CARPETA_CACHE.mkdir(parents=True, exist_ok=True)
    joblib.dump(salida, ruta)
    return pd.Series(salida, index=textos.index)

def evaluar_modelo(nombre: str, X: pd.Series, y: pd.Series, ngram_range=(1, 1)) -> dict:
    """Validación cruzada de TF-IDF + modelo sobre textos ya preprocesados (solo train)."""
    pipeline = Pipeline([
        ("tfidf", crear_vectorizador(ngram_range)),
        ("model", crear_modelo(nombre)),
    ])
    resultado = cross_validate(
        pipeline,
        X,
        y,
        cv=crear_cv(),
        scoring={
            "f1_macro": "f1_macro",
            "accuracy": "accuracy",
            "precision_negativo": make_scorer(precision_score, pos_label="negativo", zero_division=0),
            "recall_negativo": make_scorer(recall_score, pos_label="negativo", zero_division=0),
        },
    )
    return {
        "f1_macro": float(resultado["test_f1_macro"].mean()),
        "f1_macro_std": float(resultado["test_f1_macro"].std()),
        "accuracy": float(resultado["test_accuracy"].mean()),
        "precision_negativo": float(resultado["test_precision_negativo"].mean()),
        "recall_negativo": float(resultado["test_recall_negativo"].mean()),
    }