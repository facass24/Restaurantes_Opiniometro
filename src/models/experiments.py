"""Linea base, validacion cruzada y experimentos de modelado."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import PARAMS


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