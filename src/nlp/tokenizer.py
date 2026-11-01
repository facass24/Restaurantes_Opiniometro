"""Módulo de tokenización para reseñas de restaurantes en español."""

import re
from typing import List

# Orden de búsqueda: primero los casos del dominio, luego palabras normales.
#   \d+/\d+            -> calificaciones como "10/10" (un solo token)
#   \$\d+(?:[.,]\d+)?  -> precios como "$150" o "$99.50"
#   \w+                -> palabras y números (incluye ñ y vocales con tilde)
PATRON_TOKEN = re.compile(r"\d+/\d+|\$\d+(?:[.,]\d+)?|\w+")

# Todos los precios se agrupan en un solo token: el monto exacto no aporta
# sentimiento, pero mencionar el precio sí (suele ir con "caro" o "barato").
TOKEN_PRECIO = "$precio"


class Tokenizador:
    """Divide un texto normalizado en tokens, tratando casos del dominio."""

    def tokenizar(self, texto: str) -> List[str]:
        """Separa el texto en tokens, conservando calificaciones y agrupando precios."""
        if not texto:
            return []
        tokens = PATRON_TOKEN.findall(texto)
        return [TOKEN_PRECIO if token.startswith("$") else token for token in tokens]