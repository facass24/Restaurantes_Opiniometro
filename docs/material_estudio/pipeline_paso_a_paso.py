import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import unicodedata

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline

SEMILLA = 42

# ---------- datos sinteticos (10 % negativas), como en 1.4 y 1.5 ----------
rng = random.Random(SEMILLA)
POS = ["rica", "excelente", "delicioso", "buena", "genial", "recomiendo", "rapido", "amable"]
NEG = ["mala", "pesima", "fria", "lento", "sucio", "caro", "horrible", "tarde"]
NEU = ["comida", "servicio", "restaurante", "lugar", "atencion", "mesa", "plato", "precio"]


def generar(etiqueta):
    propias, ajenas = (POS, NEG) if etiqueta == 1 else (NEG, POS)
    palabras = [rng.choice(NEU) for _ in range(3)]
    for _ in range(2):
        palabras.append(rng.choice(propias) if rng.random() < 0.85 else rng.choice(ajenas))
    rng.shuffle(palabras)
    return " ".join(palabras)


etiquetas = [1] * 900 + [0] * 100
textos = np.array([generar(e) for e in etiquetas])
y = np.array(etiquetas)
X_tr, X_te, y_tr, y_te = train_test_split(textos, y, test_size=0.20, stratify=y, random_state=SEMILLA)

print("=== PASO 1: hacerlo a mano contra el Pipeline (deben ser identicos) ===")
vec = TfidfVectorizer()
Xtr_m = vec.fit_transform(X_tr)          # fit_transform SOLO con el train
clf = LogisticRegression(C=10, max_iter=1000).fit(Xtr_m, y_tr)
pred_manual = clf.predict(vec.transform(X_te))   # solo transform con el test

pipe = Pipeline([("vectorizador", TfidfVectorizer()),
                 ("clasificador", LogisticRegression(C=10, max_iter=1000))])
pipe.fit(X_tr, y_tr)
pred_pipe = pipe.predict(X_te)
print("Predicciones identicas:", bool(np.array_equal(pred_manual, pred_pipe)))
print("Pasos del pipeline:", [nombre for nombre, _ in pipe.steps])
print("Pipeline.fit hace: vectorizador.fit_transform -> clasificador.fit")
print("Pipeline.predict hace: vectorizador.transform -> clasificador.predict")

print("\n=== PASO 2: que se filtra si ajustas el TF-IDF con todo ===")
entrena = ["comida rica", "servicio lento"]
prueba = ["comida deliciosa", "servicio pesimo"]
v_mal = TfidfVectorizer().fit(entrena + prueba)
v_bien = TfidfVectorizer().fit(entrena)
print("Vocabulario (train + test, MAL):", list(v_mal.get_feature_names_out()))
print("Vocabulario (solo train, BIEN)  :", list(v_bien.get_feature_names_out()))
j1 = list(v_mal.get_feature_names_out()).index("comida")
j2 = list(v_bien.get_feature_names_out()).index("comida")
print(f"IDF de 'comida': MAL = {v_mal.idf_[j1]:.3f} | BIEN = {v_bien.idf_[j2]:.3f}")
print("Palabras del test que el modelo 'conoceria' sin haberlas visto:",
      sorted(set(v_mal.get_feature_names_out()) - set(v_bien.get_feature_names_out())))

print("\n=== PASO 3: la fuga del TF-IDF es sutil (no usa las etiquetas) ===")
vec_all = TfidfVectorizer().fit(textos)   # MAL: ve tambien el test
m_mal = LogisticRegression(C=10, max_iter=1000).fit(vec_all.transform(X_tr), y_tr)
f_mal = f1_score(y_te, m_mal.predict(vec_all.transform(X_te)), average="macro")
f_bien = f1_score(y_te, pred_pipe, average="macro")
print(f"F1 macro con TF-IDF ajustado con todo: {f_mal:.3f} | solo con train: {f_bien:.3f}")
print("La diferencia es pequena porque el TF-IDF no mira y (no es supervisado).")

print("\n=== PASO 4: la fuga GRAVE: usar las etiquetas fuera de la validacion ===")
print("100 filas y 2000 columnas de PURO RUIDO, etiquetas al azar: no hay nada que aprender.")
cv = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
fuera, dentro = [], []
for semilla in range(10):
    r = np.random.RandomState(semilla)
    X_ruido = r.randn(100, 2000)
    y_ruido = r.randint(0, 2, 100)
    X_sel = SelectKBest(f_classif, k=20).fit_transform(X_ruido, y_ruido)     # MAL: elige con TODAS las etiquetas
    fuera.append(cross_val_score(LogisticRegression(), X_sel, y_ruido, cv=cv).mean())
    pipe_ok = Pipeline([("seleccion", SelectKBest(f_classif, k=20)), ("clasificador", LogisticRegression())])
    dentro.append(cross_val_score(pipe_ok, X_ruido, y_ruido, cv=cv).mean())   # BIEN: elige solo con cada train
print(f"Seleccion FUERA de la validacion (promedio de 10 repeticiones): accuracy = {np.mean(fuera):.2f}  <- ilusion")
print(f"Seleccion DENTRO del pipeline    (promedio de 10 repeticiones): accuracy = {np.mean(dentro):.2f}  <- azar, lo honesto")

print("\n=== PASO 5: un preprocesador propio con la interfaz de scikit-learn ===")


class PreprocesadorTexto(BaseEstimator, TransformerMixin):
    """Version provisional: minusculas, sin tildes, sin URLs/numeros/puntuacion.
    La de A (src/nlp) debe respetar esta misma interfaz: fit(X, y=None) y transform(X)."""

    STOPWORDS = {"el", "la", "los", "las", "un", "una", "de", "del", "y", "en", "que", "es",
                 "me", "se", "lo", "al", "con", "por", "para", "muy", "mi", "su",
                 "no", "nunca", "pero", "nada", "sin"}
    NEGACIONES = {"no", "nunca", "pero", "nada", "sin"}

    def __init__(self, quitar_tildes=True, quitar_stopwords=True, conservar_negaciones=True):
        # Regla de scikit-learn: __init__ solo guarda los parametros, sin logica
        self.quitar_tildes = quitar_tildes
        self.quitar_stopwords = quitar_stopwords
        self.conservar_negaciones = conservar_negaciones

    def fit(self, X, y=None):
        return self                       # no aprende nada, pero debe existir

    def _limpiar(self, texto):
        t = texto.lower()
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)
        if self.quitar_tildes:
            t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
        t = re.sub(r"[^a-zñ\s]", " ", t)  # quita numeros, puntuacion y emojis
        tokens = t.split()
        if self.quitar_stopwords:
            quitar = self.STOPWORDS - self.NEGACIONES if self.conservar_negaciones else self.STOPWORDS
            tokens = [w for w in tokens if w not in quitar]
        return " ".join(tokens)

    def transform(self, X):
        return [self._limpiar(t) for t in X]


ejemplo = "¡NO me gustó la comida! Pésima atención :( www.sitio.com"
print("Antes  :", ejemplo)
print("Despues (conserva negaciones):", PreprocesadorTexto().transform([ejemplo])[0])
print("Despues (quita 'no')         :", PreprocesadorTexto(conservar_negaciones=False).transform([ejemplo])[0])

print("\nParametros que GridSearch podria ajustar:")
pipe_es = Pipeline([("preprocesador", PreprocesadorTexto()),
                    ("vectorizador", TfidfVectorizer()),
                    ("clasificador", LogisticRegression(max_iter=1000))])
print([k for k in pipe_es.get_params() if k.count("__") == 1 and k.startswith(("prepro", "vector", "clasif"))][:9])

pos = ["La comida estuvo buena", "Muy buena atención", "Buena comida y buen sabor", "Todo estuvo bueno",
       "Servicio bueno y rápido", "Buena la comida", "Excelente lugar, lo recomiendo", "Me gustó mucho",
       "Rica comida y amable servicio", "Todo muy rico"]
neg = ["Comida mala", "Servicio pésimo", "Estuvo malo", "Pésima atención", "Mala comida y fría",
       "No estuvo bueno el servicio", "No es bueno", "No estuvo buena la comida", "No me gustó nada",
       "Nunca volvería, no es rico"]
datos = pos + neg
y_es = np.array([1] * len(pos) + [0] * len(neg))
pares = ["Me gustó mucho", "No me gustó mucho", "No es buena"]
for conservar in [True, False]:
    p = Pipeline([("preprocesador", PreprocesadorTexto(conservar_negaciones=conservar)),
                  ("vectorizador", TfidfVectorizer()),
                  ("clasificador", LogisticRegression(C=10, max_iter=1000))]).fit(datos, y_es)
    for t in pares:
        print(f"conservar_negaciones={conservar!s:5s} | P(positivo | '{t}') = {p.predict_proba([t])[0, 1]:.3f}")

print("\n=== PASO 6: guardar y cargar (lo que hace train.py y lo que hace la API) ===")
final = Pipeline([("preprocesador", PreprocesadorTexto()),
                  ("vectorizador", TfidfVectorizer()),
                  ("clasificador", LogisticRegression(C=10, max_iter=1000))]).fit(datos, y_es)
with tempfile.TemporaryDirectory() as carpeta:
    ruta = os.path.join(carpeta, "model.joblib")
    joblib.dump({"pipeline": final, "sklearn": sklearn.__version__}, ruta)
    print("archivo guardado, tamano:", os.path.getsize(ruta), "bytes")
    cargado = joblib.load(ruta)
    pipe_api = cargado["pipeline"]
    crudo = "¡Qué rica estuvo la comida!!! 10/10"
    print("Texto CRUDO que llega a la API:", crudo)
    print("P(positivo) con el pipeline original:", round(float(final.predict_proba([crudo])[0, 1]), 4))
    print("P(positivo) con el pipeline cargado :", round(float(pipe_api.predict_proba([crudo])[0, 1]), 4))
    print("version de scikit-learn con que se entreno:", cargado["sklearn"])
    print("preprocesamiento (para el frontend):", pipe_api.named_steps["preprocesador"].transform([crudo]))

print("\n=== PASO 7: trampa clasica de joblib: la clase definida en __main__ ===")
codigo_clase = textwrap.dedent('''
    from sklearn.base import BaseEstimator, TransformerMixin
    class Mayusculas(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return [t.upper() for t in X]
''')
with tempfile.TemporaryDirectory() as carpeta:
    with open(os.path.join(carpeta, "mi_modulo.py"), "w") as f:
        f.write(codigo_clase)
    # (a) la clase vive en el script que entrena (__main__)
    with open(os.path.join(carpeta, "entrena_main.py"), "w") as f:
        f.write(codigo_clase + "\nimport joblib\njoblib.dump(Mayusculas(), 'a.joblib')\n")
    # (b) la clase vive en un modulo importable
    with open(os.path.join(carpeta, "entrena_modulo.py"), "w") as f:
        f.write("import joblib\nfrom mi_modulo import Mayusculas\njoblib.dump(Mayusculas(), 'b.joblib')\n")
    # la "API" es otro proceso que carga el archivo
    with open(os.path.join(carpeta, "api.py"), "w") as f:
        f.write("import sys, joblib\nprint(type(joblib.load(sys.argv[1])).__name__)\n")
    for script in ["entrena_main.py", "entrena_modulo.py"]:
        subprocess.run([sys.executable, script], cwd=carpeta, check=True)
    for archivo, etiqueta in [("a.joblib", "(a) clase en __main__ "), ("b.joblib", "(b) clase en un modulo")]:
        res = subprocess.run([sys.executable, "api.py", archivo], cwd=carpeta, capture_output=True, text=True)
        if res.returncode == 0:
            print(etiqueta, "-> la API carga el modelo:", res.stdout.strip())
        else:
            print(etiqueta, "-> la API FALLA:", res.stderr.strip().splitlines()[-1])