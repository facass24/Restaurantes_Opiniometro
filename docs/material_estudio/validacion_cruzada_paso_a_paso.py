import random
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SEMILLA = 42

# ---------- datos sinteticos (10 % negativas), igual que en la subtarea 1.4 ----------
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

# El test se aparta UNA vez y no se toca hasta el PASO 9
X_tr, X_te, y_tr, y_te = train_test_split(textos, y, test_size=0.20, stratify=y, random_state=SEMILLA)
print("train:", len(y_tr), "| test (guardado):", len(y_te))


def pipe(clasificador, **tfidf_kw):
    return Pipeline([("tfidf", TfidfVectorizer(**tfidf_kw)), ("clf", clasificador)])


print("\n=== PASO 1: una sola division es una moneda al aire ===")
puntajes = []
for semilla in range(10):
    a, b, c, d = train_test_split(textos, y, test_size=0.20, stratify=y, random_state=semilla)
    m = pipe(LogisticRegression(class_weight="balanced")).fit(a, c)
    puntajes.append(f1_score(d, m.predict(b), average="macro"))
print("F1 macro con 10 divisiones distintas:", np.round(puntajes, 3))
print(f"minimo = {min(puntajes):.3f} | maximo = {max(puntajes):.3f} | media = {np.mean(puntajes):.3f} | desv = {np.std(puntajes):.3f}")

print("\n=== PASO 2: como se reparten las particiones (20 resenas: 15 pos, 5 neg, ordenadas) ===")
y_demo = np.array([1] * 15 + [0] * 5)
X_demo = np.arange(20).reshape(-1, 1)
print("KFold SIN mezclar ni estratificar -> negativas por particion:")
print("  ", [int((y_demo[test] == 0).sum()) for _, test in KFold(5).split(X_demo)])
print("StratifiedKFold -> negativas por particion:")
print("  ", [int((y_demo[test] == 0).sum()) for _, test in StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(X_demo, y_demo)])

print("\n=== PASO 3: validacion cruzada de 5 particiones (solo con el train) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
candidatos = {
    "Linea base": pipe(DummyClassifier(strategy="most_frequent")),
    "NB (fit_prior=False)": pipe(MultinomialNB(fit_prior=False)),
    "LR (balanced)": pipe(LogisticRegression(class_weight="balanced")),
}
medias = {}
for nombre, p in candidatos.items():
    s = cross_val_score(p, X_tr, y_tr, cv=cv, scoring="f1_macro")
    medias[nombre] = s
    print(f"{nombre:22s} folds = {np.round(s, 3)} | media = {s.mean():.3f} | desv = {s.std():.3f}")

print("\n=== PASO 4: el pipeline reajusta el TF-IDF en cada particion (sin fuga) ===")
res = cross_validate(pipe(LogisticRegression(class_weight="balanced")), X_tr, y_tr, cv=cv,
                     scoring="f1_macro", return_estimator=True)
for i, est in enumerate(res["estimator"], 1):
    v = est.named_steps["tfidf"]
    j = list(v.get_feature_names_out()).index("comida")
    print(f"particion {i}: IDF de 'comida' aprendido solo con su train = {v.idf_[j]:.4f}")

print("\n=== PASO 5: GridSearchCV para la regresion logistica ===")
rejilla_lr = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1, 10],
    "clf__class_weight": [None, "balanced"],
}
gs_lr = GridSearchCV(pipe(LogisticRegression(max_iter=1000)), rejilla_lr, cv=cv, scoring="f1_macro")
gs_lr.fit(X_tr, y_tr)
combos = len(gs_lr.cv_results_["params"])
print(f"combinaciones = {combos} | entrenamientos = {combos} x 5 particiones = {combos * 5}")
orden = np.argsort(gs_lr.cv_results_["rank_test_score"])
print(f"{'rank':>4s} {'media':>7s} {'desv':>6s}  parametros")
for i in orden:
    r = gs_lr.cv_results_
    pr = {k.split('__')[1]: v for k, v in r["params"][i].items()}
    print(f"{r['rank_test_score'][i]:4d} {r['mean_test_score'][i]:7.3f} {r['std_test_score'][i]:6.3f}  {pr}")
print("MEJOR LR:", {k.split('__')[1]: v for k, v in gs_lr.best_params_.items()}, "| F1 macro CV =", round(gs_lr.best_score_, 3))

print("\n=== PASO 6: el mejor C quedo en el borde de la rejilla -> ampliarla ===")
rejilla_ext = {"clf__C": [1, 10, 100, 1000]}
gs_ext = GridSearchCV(pipe(LogisticRegression(max_iter=1000)), rejilla_ext, cv=cv, scoring="f1_macro").fit(X_tr, y_tr)
for i in np.argsort(gs_ext.cv_results_["rank_test_score"]):
    r = gs_ext.cv_results_
    print(f"C = {r['params'][i]['clf__C']:5d} | media = {r['mean_test_score'][i]:.3f} | desv = {r['std_test_score'][i]:.3f}")
print("Cuantas negativas predice el modelo en el train (hay 80 reales) segun C:")
for C_ in [0.1, 1, 10, 100]:
    m = pipe(LogisticRegression(C=C_, max_iter=1000)).fit(X_tr, y_tr)
    print(f"  C = {C_:5} -> predice {int((m.predict(X_tr) == 0).sum())} negativas")

print("\n=== PASO 7: GridSearchCV para Naive Bayes ===")
rejilla_nb = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.1, 0.5, 1.0],
    "clf__fit_prior": [True, False],
}
gs_nb = GridSearchCV(pipe(MultinomialNB()), rejilla_nb, cv=cv, scoring="f1_macro")
gs_nb.fit(X_tr, y_tr)
orden = np.argsort(gs_nb.cv_results_["rank_test_score"])
for i in orden[:4]:
    r = gs_nb.cv_results_
    pr = {k.split('__')[1]: v for k, v in r["params"][i].items()}
    print(f"{r['rank_test_score'][i]:4d} {r['mean_test_score'][i]:7.3f} {r['std_test_score'][i]:6.3f}  {pr}")
print("MEJOR NB:", {k.split('__')[1]: v for k, v in gs_nb.best_params_.items()}, "| F1 macro CV =", round(gs_nb.best_score_, 3))

print("\n=== PASO 8: comparar con la desviacion en mente ===")
i_lr = gs_lr.best_index_
i_nb = gs_nb.best_index_
m_lr, s_lr = gs_lr.cv_results_["mean_test_score"][i_lr], gs_lr.cv_results_["std_test_score"][i_lr]
m_nb, s_nb = gs_nb.cv_results_["mean_test_score"][i_nb], gs_nb.cv_results_["std_test_score"][i_nb]
print(f"LR: {m_lr:.3f} +- {s_lr:.3f}")
print(f"NB: {m_nb:.3f} +- {s_nb:.3f}")
print(f"diferencia = {abs(m_lr - m_nb):.3f} | desviacion mayor = {max(s_lr, s_nb):.3f}")
print("-> la diferencia supera la desviacion" if abs(m_lr - m_nb) > max(s_lr, s_nb)
      else "-> la diferencia es MENOR que la desviacion: empate tecnico, se elige el mas simple/explicable")

print("\n=== PASO 9: el test se usa UNA sola vez, con el modelo ya elegido ===")
for nombre, gs in [("LR", gs_lr), ("NB", gs_nb)]:
    pred = gs.predict(X_te)       # refit=True: ya viene reentrenado con todo el train
    print(f"{nombre}: F1 macro en test = {f1_score(y_te, pred, average='macro'):.3f}")