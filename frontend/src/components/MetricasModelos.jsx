import { useEffect, useState } from "react";
import api from "../services/api.js";

const valido = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0;
const porcentaje = (v) => (valido(v) ? `${(v * 100).toFixed(2)}%` : "—");
const entero = (v) => (typeof v === "number" ? v.toLocaleString("en-US") : "—");

function matrizValida(m) {
  return (
    Array.isArray(m) &&
    m.length === 2 &&
    m.every(
      (f) =>
        Array.isArray(f) &&
        f.length === 2 &&
        f.every((v) => Number.isInteger(v) && v >= 0)
    ) &&
    m.flat().some((v) => v > 0)
  );
}

export default function MetricasModelos() {
  const [final, setFinal] = useState(null);
  const [modelos, setModelos] = useState([]);
  const [cargando, setCargando] = useState(true);
  const [errorConexion, setErrorConexion] = useState(null);

  const cargarMetricas = async () => {
    setCargando(true);
    setErrorConexion(null);

    try {
      const [resFinal, resCompare] = await Promise.allSettled([
        api.get("/api/v1/metrics"),
        api.get("/api/v1/metrics/compare"),
      ]);

      let modeloActivo = null;

      if (resFinal.status === "fulfilled") {
        const data = resFinal.value?.data;
        const lista = Array.isArray(data?.modelos) ? data.modelos : [];
        modeloActivo =
          lista.find((m) => m?.es_activo) ??
          lista.find(
            (m) =>
              m?.identificador === data?.modelo_produccion ||
              m?.nombre === data?.modelo_produccion
          ) ??
          null;
        setFinal(modeloActivo);
      } else {
        throw new Error(
          "No se pudo consultar /api/v1/metrics. Verifica que el backend esté activo en el puerto 8000."
        );
      }

      if (resCompare.status === "fulfilled") {
        const listaComp = Array.isArray(resCompare.value?.data)
          ? resCompare.value.data.filter((m) => m && typeof m === "object")
          : [];
        setModelos(listaComp);
      } else if (modeloActivo) {
        setModelos([modeloActivo]);
      }
    } catch (err) {
      setErrorConexion(
        err.message || "Error al conectar con la API de métricas del backend."
      );
      setFinal(null);
      setModelos([]);
    } finally {
      setCargando(false);
    }
  };

  useEffect(() => {
    cargarMetricas();
  }, []);

  const metricas = final?.metricas ?? {};
  const matriz = matrizValida(metricas.matriz_confusion)
    ? metricas.matriz_confusion
    : null;
  const total = matriz ? matriz.flat().reduce((a, b) => a + b, 0) : null;

  // Extraer valores de la matriz: fila 0 = Negativo [TN, FP], fila 1 = Positivo [FN, TP]
  const tn = matriz ? matriz[0][0] : null;
  const fp = matriz ? matriz[0][1] : null;
  const fn = matriz ? matriz[1][0] : null;

  // Precision Negativa = TN / (TN + FN)
  const precisionNegativo =
    metricas.precision_negativo ??
    metricas.precision_negativa ??
    (tn !== null && fn !== null && tn + fn > 0 ? tn / (tn + fn) : null);

  // Recall Negativo = TN / (TN + FP)
  const recallNegativo =
    metricas.recall_negativo ??
    metricas.recall_negativa ??
    (tn !== null && fp !== null && tn + fp > 0 ? tn / (tn + fp) : null);

  const tarjetas = [
    ["F1-Score Macro", porcentaje(metricas.f1_macro)],
    ["Precision Macro", porcentaje(metricas.precision)],
    ["Recall Macro", porcentaje(metricas.recall)],
    ["Precisión negativa", porcentaje(precisionNegativo)],
    ["Recall negativo", porcentaje(recallNegativo)],
    ["Muestras de prueba", total !== null ? entero(total) : "—"],
  ];

  const etiquetas = [
    ["TN · Verdadero negativo", "FP · Falso positivo"],
    ["FN · Falso negativo", "TP · Verdadero positivo"],
  ];

  return (
    <section className="space-y-5" aria-busy={cargando}>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <div>
          <h2 className="text-lg font-semibold text-stone-900">Métricas de modelos</h2>
          <p className="mt-1 text-sm text-stone-500">
            {final?.nombre ? `Modelo final: ${final.nombre}` : "Datos oficiales del backend"}
          </p>
        </div>
        {errorConexion && (
          <button
            onClick={cargarMetricas}
            className="self-start sm:self-auto rounded-lg bg-stone-900 px-3 py-1.5 text-xs font-semibold text-white hover:bg-stone-800 transition"
          >
            Reintentar conexión
          </button>
        )}
      </div>

      {cargando && (
        <div className="rounded-2xl border border-stone-200 bg-white p-6 text-center shadow-sm">
          <p className="text-sm text-stone-500">Cargando métricas oficiales desde el backend…</p>
        </div>
      )}

      {errorConexion && (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-5 text-rose-900 shadow-sm">
          <h3 className="font-semibold text-sm">Error de conexión con el backend</h3>
          <p className="mt-1 text-xs text-rose-700">{errorConexion}</p>
          <p className="mt-2 text-xs text-stone-600">
            No se muestran datos de referencia para evitar inconsistencias con el modelo en evaluación.
          </p>
        </div>
      )}

      {!cargando && !errorConexion && !final && (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-amber-900 shadow-sm">
          <h3 className="font-semibold text-sm">Sin métricas disponibles</h3>
          <p className="mt-1 text-xs text-amber-800">
            El backend respondió pero no se encontró la información del modelo evaluado en `metrics.json`.
          </p>
        </div>
      )}

      {!cargando && !errorConexion && final && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {tarjetas.map(([titulo, valor]) => (
              <article key={titulo} className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
                <h3 className="text-sm text-stone-500">{titulo}</h3>
                <p className="mt-2 text-2xl font-semibold text-stone-900">{valor}</p>
              </article>
            ))}
          </div>

          <article className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-stone-900">Matriz de confusión</h3>
            <p className="mt-1 text-sm text-stone-500">
              Eje Y (filas): clase real. Eje X (columnas): clase predicha. Orden: Negativo, Positivo.
            </p>

            {matriz ? (
              <div className="mt-5 overflow-x-auto">
                <div role="table" aria-label="Matriz de confusión" className="min-w-[480px] space-y-2">
                  <div role="row" className="grid grid-cols-[8rem_1fr_1fr] gap-2 text-center text-sm font-semibold text-stone-600">
                    <span role="columnheader">Real ↓ / Predicha →</span>
                    <span role="columnheader">Predicho Negativo</span>
                    <span role="columnheader">Predicho Positivo</span>
                  </div>
                  {matriz.map((fila, i) => (
                    <div role="row" key={i} className="grid grid-cols-[8rem_1fr_1fr] gap-2">
                      <span role="rowheader" className="self-center text-sm font-semibold text-stone-700">
                        Real {i === 0 ? "Negativo" : "Positivo"}
                      </span>
                      {fila.map((valor, j) => (
                        <div
                          role="cell"
                          key={j}
                          className={`rounded-xl border p-4 text-center ${
                            i === j
                              ? "border-emerald-200 bg-emerald-50 text-emerald-900"
                              : "border-amber-200 bg-amber-50 text-amber-900"
                          }`}
                        >
                          <p className="text-xs font-semibold">{etiquetas[i][j]}</p>
                          <p className="my-2 text-2xl font-bold">{entero(valor)}</p>
                          <p className="text-xs">
                            Real {i === 0 ? "Negativo" : "Positivo"} · Predicho {j === 0 ? "Negativo" : "Positivo"}
                          </p>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="mt-4 p-4 rounded-xl bg-stone-50 border border-stone-200 text-center text-xs text-stone-500">
                Matriz de confusión no provista en el archivo de métricas.
              </div>
            )}
            <p className="mt-4 text-xs text-stone-500">Verde: predicción correcta (TN y TP). Ámbar: error (FP y FN).</p>
          </article>

          {modelos.length > 0 && (
            <div className="overflow-x-auto rounded-2xl border border-stone-200 bg-white shadow-sm">
              <table className="min-w-full text-left text-sm">
                <caption className="px-4 py-3 text-left font-semibold text-stone-900">
                  Comparativa de modelos
                </caption>
                <thead className="bg-stone-50 text-xs text-stone-500">
                  <tr>
                    {["Modelo", "Accuracy", "F1 Macro", "Precision Macro", "Recall Macro"].map((t) => (
                      <th scope="col" key={t} className="px-4 py-3 font-medium">
                        {t}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {modelos.map((m, i) => (
                    <tr key={m.identificador || i} className="border-t border-stone-100">
                      <th scope="row" className="px-4 py-3 font-medium text-stone-800">
                        {m.nombre || "Modelo sin nombre"}
                        {m.es_activo && (
                          <span className="ml-2 rounded-full bg-orange-100 px-2 py-1 text-xs text-orange-800">
                            En producción
                          </span>
                        )}
                      </th>
                      {["accuracy", "f1_macro", "precision", "recall"].map((k) => (
                        <td key={k} className="px-4 py-3">
                          {porcentaje(m.metricas?.[k])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}