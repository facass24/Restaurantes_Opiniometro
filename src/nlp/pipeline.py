"""Pipeline unificado de NLP para scikit-learn y análisis didáctico paso a paso."""

from dataclasses import dataclass
from typing import List, Iterable
from sklearn.base import BaseEstimator, TransformerMixin

from src.nlp.normalizer import Normalizador
from src.nlp.tokenizer import Tokenizador
from src.nlp.stopwords import FiltroStopWords
from src.nlp.reducer import Reductor


@dataclass
class ResultadoPreprocesamiento:
    """Estructura que encapsula la traza completa de transformación del texto."""

    original: str
    normalizado: str
    tokens: List[str]
    sin_stopwords: List[str]
    reducidos: List[str]

    def texto_final(self) -> str:
        """Retorna la cadena final reconstruida a partir de los tokens reducidos."""
        return " ".join(self.reducidos)


class PreprocesadorTexto(BaseEstimator, TransformerMixin):
    """Transformador compatible con scikit-learn que ejecuta el preprocesamiento NLP completo."""

    def __init__(
        self,
        quitar_tildes: bool = True,
        quitar_numeros: bool = False,
        metodo_reduccion: str = "stem",
        conservar_stopwords: Iterable[str] | None = None,
        incluir_dominio_stopwords: bool = False,
    ):
        self.quitar_tildes = quitar_tildes
        self.quitar_numeros = quitar_numeros
        self.metodo_reduccion = metodo_reduccion
        self.conservar_stopwords = conservar_stopwords
        self.incluir_dominio_stopwords = incluir_dominio_stopwords

        # Instanciar los subcomponentes modulares
        self.normalizador = Normalizador(
            quitar_tildes=self.quitar_tildes,
            quitar_numeros=self.quitar_numeros,
        )
        self.tokenizador = Tokenizador()
        self.filtro_stopwords = FiltroStopWords(
            conservar=self.conservar_stopwords,
            incluir_dominio=self.incluir_dominio_stopwords,
        )
        self.reductor = Reductor(metodo=self.metodo_reduccion)

    def fit(self, X, y=None):
        """Ajuste de scikit-learn (sin estado que aprender)."""
        return self

    def paso_a_paso(self, texto: str) -> ResultadoPreprocesamiento:
        """Ejecuta el preprocesamiento registrando cada etapa para explicabilidad."""
        texto_str = str(texto) if texto is not None else ""
        normalizado = self.normalizador.normalizar(texto_str)
        tokens = self.tokenizador.tokenizar(normalizado)
        sin_stopwords = self.filtro_stopwords.filtrar(tokens)
        reducidos = self.reductor.reducir(sin_stopwords)

        return ResultadoPreprocesamiento(
            original=texto_str,
            normalizado=normalizado,
            tokens=tokens,
            sin_stopwords=sin_stopwords,
            reducidos=reducidos,
        )

    def transform(self, X: Iterable[str]) -> List[str]:
        """Transforma una colección de textos en textos limpios listos para TF-IDF."""
        return [self.paso_a_paso(texto).texto_final() for texto in X]

