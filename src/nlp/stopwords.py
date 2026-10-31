"""Módulo de filtrado de Stop Words para reseñas en español.

Contribución técnica de Integrante C (Backend y Stop Words).
Lógica lingüística:
Las listas estándar de stop words (como NLTK o spaCy) eliminan indiscriminadamente
adverbios de negación ('no', 'nunca', 'jamás', 'tampoco') y conjunciones adversativas ('pero').
En análisis de sentimientos, eliminar estas palabras altera catastróficamente la polaridad:
    - "No me gustó la comida" -> "gustó comida" (clasificado erróneamente como positivo).
Por lo tanto, este filtro conserva explícitamente las palabras clave de polaridad y negación.
"""

from typing import Iterable, Set
import nltk

# Descarga preventiva del corpus de stopwords si no estuviera disponible
try:
    from nltk.corpus import stopwords
    _NLTK_STOPWORDS_ES = set(stopwords.words("spanish"))
except Exception:
    # Conjunto de respaldo en caso de entorno sin conexión inicial a NLTK data
    _NLTK_STOPWORDS_ES = {
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
        "con", "una", "su", "al", "lo", "como", "mas", "más", "sus", "le", "ya", "o", "este",
        "es", "porque", "esta", "son", "entre", "esta", "cuando", "muy", "sin", "sobre",
        "tambien", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
        "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos",
        "e", "esto", "mi", "antes", "algunos", "que", "unos", "yo", "otro", "otras", "otra", "el",
        "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella",
        "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tu", "te", "ti", "tu", "tus",
        "ellas", "nosotras", "vosostros", "vosostras", "os", "mio", "mia", "mios", "mias", "tuyo",
        "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros",
        "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estas",
        "esta", "estamos", "estais", "estan", "este", "estes", "estemos", "esteis", "esten", "estare",
        "estaras", "estara", "estaremos", "estareis", "estaran", "estaria", "estarias", "estariamos",
        "estariais", "estarian", "estaba", "estabas", "estabamos", "estabais", "estaban", "estuve",
        "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera", "estuvieras",
        "estuvieramos", "estuvierais", "estuvieran", "estuviese", "estuvieses", "estuviesemos",
        "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas", "estad",
        "he", "has", "ha", "hemos", "habeis", "han", "haya", "hayas", "hayamos", "hayais", "hayan",
        "habre", "habras", "habra", "habremos", "habreis", "habran", "habria", "habrias", "habriamos",
        "habriais", "habrian", "habia", "habias", "habiamos", "habiais", "habian", "hube", "hubiste",
        "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubieramos", "hubierais",
        "hubieran", "hubiese", "hubieses", "hubiesemos", "hubieseis", "hubiesen", "habiendo", "habido",
        "habida", "habidos", "habidas", "soy", "eres", "es", "somos", "sois", "son", "sea", "seas",
        "seamos", "seais", "sean", "sere", "seras", "sera", "seremos", "sereis", "seran", "seria",
        "serias", "seriamos", "seriais", "serian", "era", "eras", "eramos", "erais", "eran", "fui",
        "fuiste", "fue", "fuimos", "fuisteis", "fueron", "fuera", "fueras", "fueramos", "fuerais",
        "fueran", "fuese", "fueses", "fuesemos", "fueseis", "fuesen", "siendo", "sido", "tengo",
        "tienes", "tiene", "tenemos", "teneis", "tienen", "tenga", "tengas", "tengamos", "tengais",
        "tengan", "tendre", "tendras", "tendra", "tendremos", "tendreis", "tendran", "tendria",
        "tendrias", "tendriamos", "tendriais", "tendrian", "tenia", "tenias", "teniamos", "teniais",
        "tenian", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera", "tuvieras",
        "tuvieramos", "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuviesemos", "tuvieseis",
        "tuviesen", "teniendo", "tenido", "tenida", "tenidos", "tenidas", "tened",
    }


class FiltroStopWords:
    """Filtra palabras vacías (stop words) en español preservando negaciones y conjunciones clave."""

    # Palabras que NUNCA deben removerse porque modifican la polaridad del sentimiento
    PALABRAS_CONSERVAR: Set[str] = {
        "no",
        "nunca",
        "jamas",
        "jamás",
        "tampoco",
        "nada",
        "sin",
        "pero",
        "ni",
        "menos",
        "poco",
    }

    # Palabras opcionales muy frecuentes en reseñas que no aportan polaridad
    PALABRAS_DOMINIO: Set[str] = {
        "restaurante",
        "lugar",
        "comida",
    }

    def __init__(
        self,
        conservar: Iterable[str] | None = None,
        incluir_dominio: bool = False,
    ):
        """Inicializa el filtro de stop words.

        Args:
            conservar: Conjunto de palabras a mantener fuera del listado de stop words.
                       Si es None, usa PALABRAS_CONSERVAR por defecto.
            incluir_dominio: Si True, agrega palabras hiperfrecuentes del dominio gastronómico
                             como stop words adicionales.
        """
        self.conservar: Set[str] = (
            set(conservar) if conservar is not None else set(self.PALABRAS_CONSERVAR)
        )
        base = set(_NLTK_STOPWORDS_ES)

        # Quitar de la lista de descarte aquellas que debemos conservar
        self.stop_words: Set[str] = base - self.conservar

        if incluir_dominio:
            self.stop_words.update(self.PALABRAS_DOMINIO)

    def filtrar(self, tokens: list[str]) -> list[str]:
        """Filtra una lista de tokens removiendo las stop words definidas."""
        return [token for token in tokens if token.lower() not in self.stop_words]

