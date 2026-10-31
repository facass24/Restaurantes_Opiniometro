"""Módulo de tokenización para reseñas en español."""

import re
from typing import List


class Tokenizador:
    """Divide un texto normalizado en unidades léxicas (tokens)."""

    def __init__(self, preservar_emojis: bool = True):
        self.preservar_emojis = preservar_emojis
        # Expresión regular que captura palabras, contracciones y opcionalmente emojis
        self._regex_palabras = re.compile(r"\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9_]+\b")

    def tokenizar(self, texto: str) -> List[str]:
        """Extrae tokens alfanuméricos ignorando signos de puntuación sueltos."""
        if not texto:
            return []
        return self._regex_palabras.findall(texto)

