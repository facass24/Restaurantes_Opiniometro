import { useEffect, useState } from "react";
import api from "../services/api.js";

const MATRIZ = [[791, 178], [70, 15104]];
const REFERENCIA = {
  f1_macro: 0.9282,
  precision: (791 / 861 + 15104 / 15282) / 2,
  recall: (791 / 969 + 15104 / 15174) / 2,
  accuracy: 15895 / 16143,
};
const valido = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0;
const porcentaje = (v) => valido(v) ? `${(v * 100).toFixed(2)}%` : "—";
const entero = (v) => v.toLocaleString("en-US");
function matrizValida(m) {
  return Array.isArray(m) && m.length === 2 && m.every((f) =>
    Array.isArray(f) && f.length === 2 && f.every((v) => Number.isInteger(v) && v >= 0))
    && m.flat().some((v) => v > 0);
}

export default function MetricasModelos() {
  const [final, setFinal] = useState(null);
  const [modelos, setModelos] = useState([]);
  const [cargando, setCargando] = useState(true);
  const [avisos, setAvisos] = useState([]);

  useEffect(() => {
    let activo = true;
    async function cargarFinal() {
      try {
        const { data } = await api.get("/api/v1/metrics");
        const lista = Array.isArray(data?.modelos) ? data.modelos : [];
        const modelo = lista.find((m) => m?.es_activo) ?? lista.find((m) =>
          m?.identificador === data?.modelo_produccion || m?.nombre === data?.modelo_produccion);
        if (activo) {
          setFinal(modelo ?? null);
          if (!modelo) setAvisos((a) => [...a, "Sin datos del modelo final: se muestran valores de referencia."]);
        }
      } catch {
        if (activo) setAvisos((a) => [...a, "No se pudo consultar el modelo final: se muestran valores de referencia."]);
      }
    }
    async function cargarComparacion() {
      try {
        const { data } = await api.get("/api/v1/metrics/compare");
        const lista = Array.isArray(data) ? data.filter((m) => m && typeof m === "object") : [];
        if (activo) {
          setModelos(lista);
          if (!lista.length) setAvisos((a) => [...a, "Comparación vacía: se muestra el modelo final como referencia."]);
        }
      } catch {
        if (activo) setAvisos((a) => [...a, "Comparación no disponible: se muestra el modelo final como referencia."]);
      }
    }
    Promise.all([cargarFinal(), cargarComparacion()]).finally(() => {
      if (activo) setCargando(false);
    });
    return () => { activo = false; };
  }, []);

  const recibidas = final?.metricas ?? {};
  const metricas = Object.fromEntries(Object.entries(REFERENCIA).map(([k, v]) =>
    [k, valido(recibidas[k]) ? recibidas[k] : v]));
  const matriz = matrizValida(recibidas.matriz_confusion) ? recibidas.matriz_confusion : MATRIZ;
  const total = matriz.flat().reduce((a, b) => a + b, 0);
  const filas = modelos.length ? modelos : [{ nombre: final?.nombre || "Modelo final (referencia)", metricas }];
  const tarjetas = [["F1-Score Macro", porcentaje(metricas.f1_macro)],
    ["Precision Macro", porcentaje(metricas.precision)], ["Recall Macro", porcentaje(metricas.recall)],
    ["Muestras Totales", entero(total)]];
  const etiquetas = [["TN · Verdadero negativo", "FP · Falso positivo"],
    ["FN · Falso negativo", "TP · Verdadero positivo"]];

  return (
    <section className="space-y-5" aria-busy={cargando}>
      <div>
        <h2 className="text-lg font-semibold text-stone-900">Métricas de modelos</h2>
        <p className="mt-1 text-sm text-stone-500">Modelo final: {final?.nombre || "Valores de referencia"}</p>
        {cargando && <p role="status" className="mt-2 text-sm text-stone-500">Actualizando métricas…</p>}
        {avisos.map((a) => <p role="status" key={a} className="mt-2 text-sm text-amber-800">{a}</p>)}
      </div>
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
        <p className="mt-1 text-sm text-stone-500">Eje Y (filas): clase real. Eje X (columnas): clase predicha. Orden: Negativo, Positivo.</p>
        <div className="mt-5 overflow-x-auto">
          <div role="table" aria-label="Matriz de confusión" className="min-w-[480px] space-y-2">
            <div role="row" className="grid grid-cols-[8rem_1fr_1fr] gap-2 text-center text-sm font-semibold text-stone-600">
              <span role="columnheader">Real ↓ / Predicha →</span>
              <span role="columnheader">Predicho Negativo</span>
              <span role="columnheader">Predicho Positivo</span>
            </div>
            {matriz.map((fila, i) => (
              <div role="row" key={i} className="grid grid-cols-[8rem_1fr_1fr] gap-2">
                <span role="rowheader" className="self-center text-sm font-semibold text-stone-700">Real {i === 0 ? "Negativo" : "Positivo"}</span>
                {fila.map((valor, j) => (
                  <div role="cell" key={j} className={`rounded-xl border p-4 text-center ${i === j
                    ? "border-emerald-200 bg-emerald-50 text-emerald-900"
                    : "border-amber-200 bg-amber-50 text-amber-900"}`}>
                    <p className="text-xs font-semibold">{etiquetas[i][j]}</p>
                    <p className="my-2 text-2xl font-bold">{entero(valor)}</p>
                    <p className="text-xs">Real {i === 0 ? "Negativo" : "Positivo"} · Predicho {j === 0 ? "Negativo" : "Positivo"}</p>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
        <p className="mt-4 text-xs text-stone-500">Verde: predicción correcta (TN y TP). Ámbar: error (FP y FN).</p>
      </article>
      <div className="overflow-x-auto rounded-2xl border border-stone-200 bg-white shadow-sm">
        <table className="min-w-full text-left text-sm">
          <caption className="px-4 py-3 text-left font-semibold text-stone-900">Comparativa de modelos</caption>
          <thead className="bg-stone-50 text-xs text-stone-500">
            <tr>{["Modelo", "Accuracy", "F1 Macro", "Precision Macro", "Recall Macro"].map((t) =>
              <th scope="col" key={t} className="px-4 py-3 font-medium">{t}</th>)}</tr>
          </thead>
          <tbody>{filas.map((m, i) => (
            <tr key={m.identificador || i} className="border-t border-stone-100">
              <th scope="row" className="px-4 py-3 font-medium text-stone-800">
                {m.nombre || "Modelo sin nombre"}
                {m.es_activo && <span className="ml-2 rounded-full bg-orange-100 px-2 py-1 text-xs text-orange-800">En producción</span>}
              </th>
              {["accuracy", "f1_macro", "precision", "recall"].map((k) =>
                <td key={k} className="px-4 py-3">{porcentaje(m.metricas?.[k])}</td>)}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  );
}
