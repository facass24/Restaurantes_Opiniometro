"""Pruebas del tokenizador (src/nlp/tokenizer.py)."""

from src.nlp.tokenizer import Tokenizador

tokenizador = Tokenizador()


def test_separa_palabras():
    assert tokenizador.tokenizar("no me gusto muy lento") == ["no", "me", "gusto", "muy", "lento"]


def test_calificacion_es_un_solo_token():
    assert tokenizador.tokenizar("todo perfecto 10/10") == ["todo", "perfecto", "10/10"]


def test_precio_se_agrupa():
    assert tokenizador.tokenizar("pagamos $150 algo caro") == ["pagamos", "$precio", "algo", "caro"]


def test_varios_precios():
    assert tokenizador.tokenizar("entre $80 y $99.50") == ["entre", "$precio", "y", "$precio"]


def test_conserva_enie():
    assert tokenizador.tokenizar("el año pasado") == ["el", "año", "pasado"]


def test_texto_vacio():
    assert tokenizador.tokenizar("") == []