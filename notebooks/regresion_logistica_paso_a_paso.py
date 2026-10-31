import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight

np.set_printoptions(precision=3, suppress=True)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


print("=== PASO 1: la sigmoide convierte z en probabilidad ===")
for z in [-6, -4, -2, 0, 2, 4, 6]:
    print(f"z = {z:2d}  ->  P(positivo) = {sigmoide(z):.3f}")

print("\n=== PASO 2: odds y log-odds ===")
p = 0.8
print(f"P = {p} -> odds = {p/(1-p):.2f} -> ln(odds) = {np.log(p/(1-p)):.3f}")
print(f"z = 1.386 -> sigmoide = {sigmoide(1.386):.3f} (la sigmoide deshace el logaritmo de los odds)")

print("\n=== PASO 3: calculo a mano con pesos inventados ===")
pesos = {"excelente": 2.0, "rica": 1.5, "lento": -1.8, "fria": -1.6}
b = 0.1
for texto in ["comida excelente", "servicio lento"]:
    z = b + sum(pesos.get(w, 0.0) for w in texto.split())
    print(f"{texto}: z = {z:.1f} -> P(positivo) = {sigmoide(z):.3f}")

print("\n=== PASO 4: entrenar de verdad y reproducir predict_proba ===")
resenas = ["comida rica", "servicio excelente", "comida excelente",
           "comida fria", "servicio lento"]
y = np.array([1, 1, 1, 0, 0])          # 1 = positivo, 0 = negativo
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(resenas)
vocab = list(tfidf.get_feature_names_out())
lr = LogisticRegression(C=1.0).fit(X, y)
print("clases:", lr.classes_, "(coef_ describe la clase 1 = positivo)")
print("pesos :", {w: round(float(p), 3) for w, p in zip(vocab, lr.coef_[0])})
print("sesgo b:", round(float(lr.intercept_[0]), 3))
for texto in ["comida excelente", "servicio lento"]:
    x = tfidf.transform([texto])
    z = float(x.toarray()[0] @ lr.coef_[0] + lr.intercept_[0])
    print(f"{texto}: z = {z:.3f} | sigmoide(z) = {sigmoide(z):.3f} | predict_proba = {lr.predict_proba(x)[0, 1]:.3f}")

print("\n=== PASO 5: la funcion de perdida (log-loss) ===")
proba = lr.predict_proba(X)[:, 1]
perdida_i = -(y * np.log(proba) + (1 - y) * np.log(1 - proba))
print("P(positivo) por resena :", np.round(proba, 3))
print("perdida por resena     :", np.round(perdida_i, 3))
print("promedio a mano        :", round(float(perdida_i.mean()), 4))
print("log_loss de sklearn    :", round(float(log_loss(y, proba)), 4))
print("Castigo cuando la resena real es positiva y el modelo dice P(positivo) = q:")
for q in [0.4, 0.1, 0.01]:
    print(f"  q = {q:.2f} -> perdida = {-np.log(q):.2f}")

print("\n=== PASO 6: gradiente descendente desde cero ===")
Xd = X.toarray()
n = len(y)
C = 1.0
w = np.zeros(Xd.shape[1])
bias = 0.0
lr_tasa = 1.0
for it in range(0, 5001):
    p_ = sigmoide(Xd @ w + bias)
    perd = -np.mean(y * np.log(p_) + (1 - y) * np.log(1 - p_)) + (w @ w) / (2 * C * n)
    if it in (0, 10, 100, 1000, 5000):
        print(f"iteracion {it:4d} | perdida = {perd:.4f}")
    grad_w = Xd.T @ (p_ - y) / n + w / (C * n)
    grad_b = np.mean(p_ - y)
    w -= lr_tasa * grad_w
    bias -= lr_tasa * grad_b
print("pesos mios    :", np.round(w, 3))
print("pesos sklearn :", np.round(lr.coef_[0], 3))
print("sesgo mio / sklearn:", round(float(bias), 3), "/", round(float(lr.intercept_[0]), 3))

print("\n=== PASO 7: el parametro C (regularizacion) ===")
for C_ in [0.01, 1.0, 100.0]:
    m = LogisticRegression(C=C_).fit(X, y)
    pe = m.predict_proba(tfidf.transform(["comida excelente"]))[0, 1]
    print(f"C = {C_:6.2f} | peso de 'excelente' = {m.coef_[0][vocab.index('excelente')]: .3f} | P(pos | comida excelente) = {pe:.3f}")

print("\n=== PASO 8: class_weight='balanced' ===")
y_d = np.array([1] * 7 + [0])
pesos_clase = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_d)
print("formula: n_muestras / (n_clases * conteo_de_la_clase) = 8 / (2 * conteo)")
print("peso clase 0 (negativo):", round(float(pesos_clase[0]), 3), "| peso clase 1 (positivo):", round(float(pesos_clase[1]), 3))
desb = ["rica comida", "excelente servicio", "comida excelente", "rico todo",
        "excelente lugar", "rica atencion", "excelente sabor", "comida fria"]
cv2 = TfidfVectorizer()
X2 = cv2.fit_transform(desb)
for cw in [None, "balanced"]:
    m = LogisticRegression(class_weight=cw).fit(X2, y_d)
    pe = m.predict_proba(cv2.transform(["comida fria"]))[0, 1]
    print(f"class_weight = {str(cw):9s} | P(positivo | 'comida fria') = {pe:.3f}")

print("\n=== PASO 9: pesos interpretables y odds ===")
orden = np.argsort(lr.coef_[0])[::-1]
for i in orden:
    print(f"{vocab[i]:10s} peso = {lr.coef_[0][i]: .3f} | multiplica los odds de positivo por {np.exp(lr.coef_[0][i]):.2f}")

print("\n=== PASO 10: la negacion con regresion logistica ===")
ent = ["la comida estuvo buena", "muy buena atencion", "buena comida y buen sabor",
       "todo estuvo bueno", "servicio bueno y rapido", "buena la comida",
       "comida mala", "servicio pesimo", "estuvo malo", "pesima atencion",
       "mala comida y fria"]
yy = np.array([1] * 6 + [0] * 5)
prueba = "no estuvo bueno"
v = TfidfVectorizer()
m = LogisticRegression().fit(v.fit_transform(ent), yy)
print("A) unigramas, sin ejemplos con 'no'     -> P(pos) =", round(float(m.predict_proba(v.transform([prueba]))[0, 1]), 3))
ent2 = ent + ["no estuvo bueno el servicio", "no es bueno", "no estuvo buena la comida", "no es buena"]
yy2 = np.array(list(yy) + [0] * 4)
for rango in [(1, 1), (1, 2)]:
    v = TfidfVectorizer(ngram_range=rango)
    m = LogisticRegression().fit(v.fit_transform(ent2), yy2)
    pe = m.predict_proba(v.transform([prueba]))[0, 1]
    print(f"B) con ejemplos de negacion, ngram {rango} -> P(pos) = {pe:.3f}")
v = TfidfVectorizer(ngram_range=(1, 1))
m = LogisticRegression().fit(v.fit_transform(ent2), yy2)
print("peso de 'no' (unigramas):", round(float(m.coef_[0][list(v.get_feature_names_out()).index('no')]), 3))