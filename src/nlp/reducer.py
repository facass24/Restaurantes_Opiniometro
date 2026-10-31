"""Módulo de reducción morfológica (Stemming y Lematización) en español."""

from typing import List
from nltk.stem.snowball import SnowballStemmer

try:
    import spacy
    try:
        _NLP_SPACY = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except Exception:
        _NLP_SPACY = None
except ImportError:
    _NLP_SPACY = None


class Reductor:
    """Reduce tokens a su raíz morfológica mediante lematización o stemming."""

    def __init__(self, metodo: str = "stem"):
        """Inicializa el reductor.

        Args:
            metodo: 'stem' para SnowballStemmer o 'lema' para spaCy es_core_news_sm.
        """
        if metodo not in ("stem", "lema"):
            raise ValueError("El método de reducción debe ser 'stem' o 'lema'.")

        self.metodo = metodo
        self._stemmer = SnowballStemmer("spanish")
        self._spacy = _NLP_SPACY

    def reducir(self, tokens: List[str]) -> List[str]:
        """Aplica lematización o stemming a la lista de tokens."""
        if not tokens:
            return []

        if self.metodo == "stem" or self._spacy is None:
            return [self._stemmer.stem(token) for token in tokens]

        # Lematización con spaCy
        doc = self._spacy(" ".join(tokens))
        return [token.lemma_ for token in doc if not token.is_space]

