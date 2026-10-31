import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

np.set_printoptions(precision=4, suppress=True)


def limpio(clases_, probas):
    """Convierte a tipos simples de Python para imprimir sin ruido."""
    return {str(c): round(float(p), 3) for c, p in zip(clases_, probas)}


resenas = ["comida rica", "servicio excelente", "comida excelente",
           "comida fria", "servicio lento"]
clases = np.array(["positivo", "positivo", "positivo", "negativo", "negativo"])

print("=== PASO 1: conteos por clase ===")
cv = CountVectorizer()
X = cv.fit_transform(resenas).toarray()
vocab = list(cv.get_feature_names_out())
V = len(vocab)
print("Vocabulario:", vocab, "| V =", V)
conteo = {}
for c in ["positivo", "negativo"]:
    conteo[c] = X[clases == c].sum(axis=0)
    print(c, conteo[c], "| total palabras:", conteo[c].sum())

print("\n=== PASO 2: priors ===")
prior = {c: (clases == c).mean() for c in ["positivo", "negativo"]}
print({k: float(v) for k, v in prior.items()})

print("\n=== PASO 3: P(palabra|clase) con alpha=1 ===")
alpha = 1.0
likelihood = {}
for c in ["positivo", "negativo"]:
    likelihood[c] = (conteo[c] + alpha) / (conteo[c].sum() + alpha * V)
    print(c, {w: round(float(p), 3) for w, p in zip(vocab, likelihood[c])})

def clasificar_a_mano(texto):
    x = cv.transform([texto]).toarray()[0]
    puntaje = {c: prior[c] * np.prod(likelihood[c] ** x) for c in prior}
    total = sum(puntaje.values())
    return {c: round(float(p / total), 3) for c, p in puntaje.items()}, puntaje

print("\n=== PASO 4: clasificar a mano ===")
for t in ["comida excelente", "servicio lento"]:
    prob, punt = clasificar_a_mano(t)
    print(t, "| puntajes sin normalizar:", {k: round(float(v), 4) for k, v in punt.items()}, "| prob:", prob)

print("\n=== PASO 5: lo mismo con MultinomialNB ===")
nb = MultinomialNB(alpha=1.0).fit(X, clases)
for t in ["comida excelente", "servicio lento"]:
    p = nb.predict_proba(cv.transform([t]))[0]
    print(t, limpio(nb.classes_, p))
print("prior de sklearn:", limpio(nb.classes_, np.exp(nb.class_log_prior_)))
print("log P(w|clase) existe en nb.feature_log_prob_, forma:", nb.feature_log_prob_.shape)

print("\n=== PASO 6: por que logaritmos ===")
print("Multiplicar 100 probabilidades de 0.01 ->", 0.01 ** 100, "(todavia representable)")
print("Multiplicar 200 probabilidades de 0.01 ->", 0.01 ** 200, "(se vuelve 0: underflow)")
print("Sumando logs 200 veces ->", round(200 * np.log(0.01), 1), "(sin problema)")

print("\n=== PASO 7: efecto de alpha ===")
for a in [0.0001, 1.0, 10.0]:
    m = MultinomialNB(alpha=a).fit(X, clases)
    p = m.predict_proba(cv.transform(["comida excelente"]))[0]
    print("alpha =", a, "->", limpio(m.classes_, p))

print("\n=== PASO 8: fit_prior (desbalance) ===")
desb = ["rica comida", "excelente servicio", "comida excelente", "rico todo",
        "excelente lugar", "rica atencion", "excelente sabor", "comida fria"]
y_d = np.array(["positivo"] * 7 + ["negativo"])
cv2 = CountVectorizer()
X2 = cv2.fit_transform(desb)
for fp in [True, False]:
    m = MultinomialNB(fit_prior=fp).fit(X2, y_d)
    print("fit_prior =", fp, "| prior:", limpio(m.classes_, np.exp(m.class_log_prior_)))

print("\n=== PASO 9: palabras mas indicativas ===")
dif = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]   # positivo - negativo
orden = np.argsort(dif)[::-1]
for i in orden:
    print(f"{vocab[i]:10s} log P(w|pos) - log P(w|neg) = {dif[i]: .3f}")

print("\n=== PASO 10: debilidad con la negacion ===")
ent = ["la comida estuvo buena", "muy buena atencion", "buena comida y buen sabor",
       "todo estuvo bueno", "servicio bueno y rapido", "buena la comida",
       "comida mala", "servicio pesimo", "estuvo malo", "pesima atencion",
       "mala comida y fria"]
y = np.array(["positivo"] * 6 + ["negativo"] * 5)
prueba = "no estuvo bueno"
v = CountVectorizer()
m = MultinomialNB().fit(v.fit_transform(ent), y)
print("A) unigramas, sin ejemplos con 'no' ->", limpio(m.classes_, m.predict_proba(v.transform([prueba]))[0]))

# Si el entrenamiento incluye negaciones, los bigramas "no estuvo" / "no es" pueden aprenderse
ent2 = ent + ["no estuvo bueno el servicio", "no es bueno", "no estuvo buena la comida", "no es buena"]
y2 = np.array(list(y) + ["negativo"] * 4)
for rango in [(1, 1), (1, 2)]:
    v = CountVectorizer(ngram_range=rango)
    m = MultinomialNB().fit(v.fit_transform(ent2), y2)
    print("B) con ejemplos de negacion, ngram", rango, "->", limpio(m.classes_, m.predict_proba(v.transform([prueba]))[0]))

print("\n=== PASO 11: con TF-IDF tambien funciona ===")
tf = TfidfVectorizer()
m = MultinomialNB(alpha=1.0).fit(tf.fit_transform(resenas), clases)
p = m.predict_proba(tf.transform(["comida excelente"]))[0]
print("comida excelente (TF-IDF) ->", limpio(m.classes_, p))