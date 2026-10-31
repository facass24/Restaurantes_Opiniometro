import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

np.set_printoptions(precision=3, suppress=True)

docs = ["la comida muy rica", "la comida muy mala", "servicio muy lento"]
N = len(docs)

print("=== PASO 1: bolsa de palabras (conteos) ===")
cv = CountVectorizer()
conteos = cv.fit_transform(docs).toarray()
vocab = list(cv.get_feature_names_out())
print("Vocabulario:", vocab)
print(conteos)

print("\n=== PASO 2: df e IDF a mano ===")
df = (conteos > 0).sum(axis=0)
idf_clasico = np.log(N / df)
idf_sklearn = np.log((1 + N) / (1 + df)) + 1
print("palabra  :", vocab)
print("df       :", df)
print("IDF clasico ln(N/df)           :", idf_clasico)
print("IDF sklearn ln((1+N)/(1+df))+1 :", idf_sklearn)

print("\n=== PASO 3: TF x IDF y normalizacion L2 (D1 a mano) ===")
tf_d1 = conteos[0]
crudo = tf_d1 * idf_sklearn
norma = np.sqrt((crudo ** 2).sum())
print("TF x IDF de D1       :", crudo)
print("norma (longitud)     :", round(norma, 4))
print("D1 normalizado       :", crudo / norma)

print("\n=== PASO 4: lo mismo con TfidfVectorizer ===")
tfidf = TfidfVectorizer()
M = tfidf.fit_transform(docs)
print("idf_ de sklearn:", tfidf.idf_)
print(M.toarray())
print("Coincide con mi calculo a mano:", np.allclose(M.toarray()[0], crudo / norma))
print("Longitud de cada vector:", np.linalg.norm(M.toarray(), axis=1))

print("\n=== PASO 5: fit vs transform (palabras nuevas se ignoran) ===")
nuevo = tfidf.transform(["la comida es pesima"])
print("Vector de 'la comida es pesima':", nuevo.toarray()[0])
print("Solo 'la' y 'comida' existen en el vocabulario; 'es' y 'pesima' se ignoran.")

print("\n=== PASO 6: sublinear_tf ===")
tf_crudo = np.array([1, 2, 5, 10])
print("tf        :", tf_crudo)
print("1 + ln(tf):", 1 + np.log(tf_crudo))

print("\n=== PASO 7: min_df y max_df ===")
v = TfidfVectorizer(min_df=2, max_df=0.90).fit(docs)
print("min_df=2, max_df=0.90 ->", list(v.get_feature_names_out()))
print("(muy esta en 3/3 = 100 % > 90 %, se descarta; las que estan en 1 doc tampoco)")

print("\n=== PASO 8: n-gramas ===")
v = TfidfVectorizer(ngram_range=(1, 2)).fit(["no me gusto la comida"])
print(list(v.get_feature_names_out()))