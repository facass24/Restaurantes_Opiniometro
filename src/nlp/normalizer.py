"""Módulo de normalización de texto para reseñas en español."""

import re
# Vocales acentuadas -> vocales simples (la ñ no se incluye a propósito)
TABLA_TILDES = str.maketrans("áéíóúü", "aeiouu")

# URLs: no aportan sentimiento (solo 22 reseñas del dataset las tienen)
PATRON_URL = re.compile(r"https?://\S+|www\.\S+")

# Emojis: el dataset no contiene ninguno, así que el modelo no los conoce
PATRON_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F1E6-\U0001F1FF\uFE0F]")

class Normalizador:
    """Normaliza texto eliminando ruido, tildes y caracteres innecesarios."""

    def __init__(self, quitar_tildes: bool = False, quitar_numeros: bool = False):
        """Inicializa el normalizador.

        Args:
            quitar_tildes: Si True, transforma vocales acentuadas a vocales simples.
            quitar_numeros: Si True, elimina dígitos del texto.
        """
        self.quitar_tildes = quitar_tildes
        self.quitar_numeros = quitar_numeros

    def normalizar(self, texto: str) -> str:
        """Aplica la secuencia de normalización sobre el texto crudo."""
        if not texto or not isinstance(texto, str):
            return ""

        # 1. Convertir a minúsculas
        texto_limpio = texto.lower().strip()
        # 1.1 Quitar URLs y emojis (antes de reducir repeticiones, para que
        # 'www' no se convierta en 'w' y la URL siga siendo reconocible)
        texto_limpio = PATRON_URL.sub(" ", texto_limpio)
        texto_limpio = PATRON_EMOJI.sub(" ", texto_limpio)
        # 2. Reducir letras repetidas 3 o más veces a una sola
        # ('riquísimoooo' -> 'riquísimo'). Las dobles se conservan porque son
        # válidas en español ('llama', 'carro', 'leer'). Solo se aplica a letras
        # para no alterar números ('1000' no debe quedar como '10').
        texto_limpio = re.sub(r"([a-zñáéíóúü])\1{2,}", r"\1", texto_limpio)

        # 3. Quitar tildes si fue configurado (conservando la ñ)
        # Se reemplazan solo las vocales acentuadas. No se usa unicodedata
        # porque convierte "ñ" en "n" y cambia palabras ("año" -> "ano").
        if self.quitar_tildes:
            texto_limpio = texto_limpio.translate(TABLA_TILDES)

        # 4. Quitar números si fue configurado
        if self.quitar_numeros:
            texto_limpio = re.sub(r"\d+", "", texto_limpio)

        # 5. Normalizar espacios en blanco consecutivos
        texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()

        return texto_limpio

