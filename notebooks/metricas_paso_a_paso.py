import random
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# En todo el script: 1 = positivo, 0 = negativo. La clase "rara" es la 0.

print("=== PASO 1: el modelo tramposo (siempre dice positivo) ===")
real = np.array([1] * 90 + [0] * 10)
tramposo = np.ones(100, dtype=int)
print("accuracy :", accuracy_score(real, tramposo))
print("F1 macro :", round(f1_score(real, tramposo, average="macro", zero_division=0), 3))
print("recall de la clase negativa:", recall_score(real, tramposo, pos_label=0, zero_division=0))
print(confusion_matrix(real, tramposo, labels=[0, 1]))

print("\n=== PASO 2: matriz de confusion de un modelo real, a mano ===")
# filas = realidad, columnas = prediccion
# real neg: 6 acertadas, 4 confundidas | real pos: 5 confundidas, 85 acertadas
y_real = np.array([0] * 10 + [1] * 90)
y_pred = np.array([0] * 6 + [1] * 4 + [0] * 5 + [1] * 85)
print(confusion_matrix(y_real, y_pred, labels=[0, 1]))
TP, FN, FP, TN = 6, 4, 5, 85          # tomando "negativo" como la clase de interes
prec_neg = TP / (TP + FP)
rec_neg = TP / (TP + FN)
f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg)
print(f"Clase negativa: precision = 6/11 = {prec_neg:.3f} | recall = 6/10 = {rec_neg:.3f} | F1 = {f1_neg:.3f}")
prec_pos = 85 / (85 + 4)
rec_pos = 85 / (85 + 5)
f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos)
print(f"Clase positiva: precision = 85/89 = {prec_pos:.3f} | recall = 85/90 = {rec_pos:.3f} | F1 = {f1_pos:.3f}")
print("F1 macro a mano :", round((f1_neg + f1_pos) / 2, 3))
print("F1 macro sklearn:", round(f1_score(y_real, y_pred, average="macro"), 3))
print("accuracy        :", accuracy_score(y_real, y_pred))

print("\n=== PASO 3: macro vs weighted vs micro ===")
for prom in ["macro", "weighted", "micro"]:
    print(f"F1 {prom:8s}:", round(f1_score(y_real, y_pred, average=prom), 3))
print("(weighted y micro quedan cerca de la accuracy: el 90 % positivo tapa a la clase rara)")

print("\n=== PASO 4: por que media armonica ===")
p, r = 1.0, 0.1
print(f"precision = {p}, recall = {r} -> promedio simple = {(p + r) / 2:.2f} | F1 (armonica) = {2 * p * r / (p + r):.2f}")

print("\n=== PASO 5: el umbral cambia precision y recall ===")
# P(positivo) que dio un modelo a 20 resenas; las 4 primeras son realmente negativas
y_u = np.array([0, 0, 0, 0] + [1] * 16)
p_pos = np.array([0.10, 0.35, 0.45, 0.60,
                  0.30, 0.55, 0.62, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.97, 0.99])
for umbral in [0.3, 0.5, 0.7]:
    pred = (p_pos >= umbral).astype(int)
    pr = precision_score(y_u, pred, pos_label=0, zero_division=0)
    rc = recall_score(y_u, pred, pos_label=0, zero_division=0)
    print(f"umbral {umbral} | negativas detectadas: {int(((pred == 0) & (y_u == 0)).sum())}/4 | precision neg = {pr:.2f} | recall neg = {rc:.2f}")

print("\n=== PASO 6: experimento con datos desbalanceados (10 % negativas) ===")
rng = random.Random(42)
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
textos = [generar(e) for e in etiquetas]
y_all = np.array(etiquetas)

print("proporcion de negativas:", round(float((y_all == 0).mean()), 3), "| positivas:", round(float((y_all == 1).mean()), 3))
Xtr_t, Xte_t, ytr, yte = train_test_split(textos, y_all, test_size=0.20, stratify=y_all, random_state=42)
print("con stratify    -> negativas en train:", int((ytr == 0).sum()), "| en test:", int((yte == 0).sum()))
_, _, ytr2, yte2 = train_test_split(textos, y_all, test_size=0.20, random_state=3)
print("sin stratify(3) -> negativas en train:", int((ytr2 == 0).sum()), "| en test:", int((yte2 == 0).sum()))

tfidf = TfidfVectorizer()
Xtr = tfidf.fit_transform(Xtr_t)
Xte = tfidf.transform(Xte_t)

modelos = {
    "Linea base (siempre positivo)": DummyClassifier(strategy="most_frequent"),
    "NB (fit_prior=True)": MultinomialNB(),
    "NB (fit_prior=False)": MultinomialNB(fit_prior=False),
    "LR (sin class_weight)": LogisticRegression(),
    "LR (balanced)": LogisticRegression(class_weight="balanced"),
}
print(f"\n{'modelo':32s} {'accuracy':>9s} {'F1 macro':>9s} {'recall neg':>11s} {'precision neg':>14s}")
resultados = {}
for nombre, m in modelos.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    resultados[nombre] = pred
    print(f"{nombre:32s} {accuracy_score(yte, pred):9.3f} {f1_score(yte, pred, average='macro'):9.3f} "
          f"{recall_score(yte, pred, pos_label=0, zero_division=0):11.3f} {precision_score(yte, pred, pos_label=0, zero_division=0):14.3f}")

print("\n=== PASO 7: reporte completo del mejor modelo ===")
pred = resultados["LR (balanced)"]
print(confusion_matrix(yte, pred, labels=[0, 1]))
print(classification_report(yte, pred, target_names=["negativo", "positivo"], digits=3))

print("=== PASO 8: error clasico, rebalancear ANTES de dividir ===")
idx_neg = [k for k, e in enumerate(etiquetas) if e == 0]

# MAL: duplicar negativas en todo el conjunto y luego dividir
textos_mal = textos + [textos[k] for k in idx_neg] * 8
y_mal = np.array(etiquetas + [0] * (len(idx_neg) * 8))
a, b_, c, d = train_test_split(textos_mal, y_mal, test_size=0.20, stratify=y_mal, random_state=42)
t_mal = TfidfVectorizer()
m_mal = LogisticRegression(C=100).fit(t_mal.fit_transform(a), c)
f1_mal = f1_score(d, m_mal.predict(t_mal.transform(b_)), average="macro")

# BIEN: dividir primero, duplicar solo dentro del entrenamiento, evaluar en el test original
tr_txt = list(Xtr_t)
tr_y = list(ytr)
extra = [t for t, e in zip(Xtr_t, ytr) if e == 0] * 8
tr_txt += extra
tr_y += [0] * len(extra)
t_bien = TfidfVectorizer()
m_bien = LogisticRegression(C=100).fit(t_bien.fit_transform(tr_txt), tr_y)
f1_bien = f1_score(yte, m_bien.predict(t_bien.transform(Xte_t)), average="macro")
print("MAL  (duplicar antes de dividir): F1 macro =", round(f1_mal, 3), "<- inflado, hay copias en train y test")
print("BIEN (duplicar solo en train)   : F1 macro =", round(f1_bien, 3), "<- honesto, test original intacto")