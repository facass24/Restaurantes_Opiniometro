"""Pruebas del normalizador de texto (src/nlp/normalizer.py)."""

from src.nlp.normalizer import Normalizador

normalizador = Normalizador(quitar_tildes=True)


def test_minusculas():
    assert normalizador.normalizar("EXCELENTE Comida") == "excelente comida"


def test_conserva_la_enie():
    assert normalizador.normalizar("El año pasado comí piña") == "el año pasado comi piña"


def test_quita_tildes():
    assert normalizador.normalizar("atención rápida") == "atencion rapida"


def test_reduce_letras_repetidas():
    assert normalizador.normalizar("RIQUÍSIMOOOO, muuuuy bueeeno") == "riquisimo muy bueno"


def test_conserva_letras_dobles_validas():
    assert normalizador.normalizar("La llama y el carro") == "la llama y el carro"


def test_no_altera_numeros():
    assert normalizador.normalizar("Pagamos 1000") == "pagamos 1000"


def test_quita_urls():
    assert normalizador.normalizar("Vean https://tripadvisor.com y www.sitio.com") == "vean y"


def test_quita_emojis():
    assert normalizador.normalizar("Delicioso 😋👍 volveremos") == "delicioso volveremos"


def test_quita_puntuacion():
    assert normalizador.normalizar("¡¡Excelente!! ¿Volvería? Sí.") == "excelente volveria si"


def test_conserva_calificacion_y_precio():
    assert normalizador.normalizar("Todo 10/10, pagamos $150.") == "todo 10/10 pagamos $150"


def test_texto_vacio():
    assert normalizador.normalizar("") == ""
    assert normalizador.normalizar(None) == ""