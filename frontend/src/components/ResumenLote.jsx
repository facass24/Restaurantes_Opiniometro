import { useState } from "react";
import { descargarExcelLote } from "../services/api.js";

export default function ResumenLote({ lote }) {
  const [descargando, setDescargando] = useState(false);
  const [errorDescarga, setErrorDescarga] = useState(null);

  if (!lote || !lote.resumen) return null;

  const { resumen, batch_id, nombre_archivo, archivo_descargable } = lote;
  const {
    total_procesadas,
    positivas,
    negativas,
    porcentaje_satisfaccion,
    porcentaje_insatisfaccion,
    tiene_sucursales,
  } = resumen;

  const handleDescargar = async () => {
    setDescargando(true);
    setErrorDescarga(null);
    try {
      const nombreSugerido = nombre_archivo.replace(/\.[^/.]+$/, "") + "_con_predicciones.xlsx";
      await descargarExcelLote(batch_id, nombreSugerido);
    } catch (err) {
      setErrorDescarga("No se pudo descargar el archivo: " + err.message);
    } finally {
      setDescargando(false);
    }
  };

  // Determinar nivel de alerta general
  let colorAlerta = "bg-emerald-50 text-emerald-800 border-emerald-200";
  let textoAlerta = "Normal (<15% quejas)";
  if (porcentaje_insatisfaccion > 25) {
    colorAlerta = "bg-rose-50 text-rose-800 border-rose-200";
    textoAlerta = "Crítico (>25% quejas)";
  } else if (porcentaje_insatisfaccion >= 15) {
    colorAlerta = "bg-amber-50 text-amber-800 border-amber-200";
    textoAlerta = "Atención (15-25% quejas)";
  }

  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6 space-y-5">
      {/* Encabezado y botón de exportación */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between border-b border-stone-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold tracking-wider text-orange-700 uppercase">
              Resultados del Lote
            </span>
            <span className="text-xs text-stone-400">• ID: {batch_id}</span>
          </div>
          <h2 className="text-xl font-semibold text-stone-900 mt-0.5">
            {nombre_archivo}
          </h2>
          <p className="text-xs text-stone-500 mt-0.5">
            Procesado con inferencia masiva scikit-learn
          </p>
        </div>

        {archivo_descargable && (
          <div className="flex flex-col items-start sm:items-end gap-1.5">
            <button
              type="button"
              onClick={handleDescargar}
              disabled={descargando}
              className="inline-flex items-center gap-2 rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-emerald-800 disabled:opacity-50"
            >
              {descargando ? (
                <>
                  <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Descargando...</span>
                </>
              ) : (
                <>
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <span>Exportar Excel (.xlsx)</span>
                </>
              )}
            </button>
            <span className="text-[11px] text-stone-400">
              Incluye las 4 columnas de diagnóstico calculadas
            </span>
          </div>
        )}
      </div>

      {errorDescarga && (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
          {errorDescarga}
        </div>
      )}

      {/* Tarjetas métricas */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* Total procesadas */}
        <div className="rounded-xl border border-stone-200 bg-stone-50/70 p-4">
          <p className="text-xs font-medium text-stone-500">Total Analizadas</p>
          <p className="mt-1 text-2xl font-bold text-stone-900">{total_procesadas}</p>
          <p className="mt-0.5 text-[11px] text-stone-400">
            {tiene_sucursales ? "Con sedes detectadas" : "Opiniones tabulares"}
          </p>
        </div>

        {/* Positivas */}
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-4">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium text-emerald-800">Positivas</p>
            <span className="text-xs font-bold text-emerald-700">
              {porcentaje_satisfaccion.toFixed(1)}%
            </span>
          </div>
          <p className="mt-1 text-2xl font-bold text-emerald-900">{positivas}</p>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-emerald-100">
            <div
              className="h-full bg-emerald-600 rounded-full"
              style={{ width: `${porcentaje_satisfaccion}%` }}
            />
          </div>
        </div>

        {/* Negativas */}
        <div className="rounded-xl border border-rose-200 bg-rose-50/40 p-4">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium text-rose-800">Negativas (Quejas)</p>
            <span className="text-xs font-bold text-rose-700">
              {porcentaje_insatisfaccion.toFixed(1)}%
            </span>
          </div>
          <p className="mt-1 text-2xl font-bold text-rose-900">{negativas}</p>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-rose-100">
            <div
              className="h-full bg-rose-600 rounded-full"
              style={{ width: `${porcentaje_insatisfaccion}%` }}
            />
          </div>
        </div>

        {/* Estado Alerta General */}
        <div className={`rounded-xl border p-4 ${colorAlerta}`}>
          <p className="text-xs font-medium">Diagnóstico General</p>
          <p className="mt-1 text-base font-bold uppercase tracking-wide">
            {porcentaje_insatisfaccion > 25
              ? "Crítico"
              : porcentaje_insatisfaccion >= 15
              ? "Atención"
              : "Saludable"}
          </p>
          <p className="mt-0.5 text-[11px] opacity-80">{textoAlerta}</p>
        </div>
      </div>
    </section>
  );
}

