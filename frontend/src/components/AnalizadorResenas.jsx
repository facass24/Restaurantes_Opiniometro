import { useState } from "react";
import { predecirSentimiento } from "../services/api.js";
import { normalizarResultado } from "../services/resultadoAnalisis.js";
import ResultadoSentimiento from "./ResultadoSentimiento.jsx";
import NlpPasoAPaso from "./NlpPasoAPaso.jsx";

export default function AnalizadorResenas() {
  const [texto, setTexto] = useState("");
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState(null);
  const [resultado, setResultado] = useState(null);

  const handleAnalizar = async (e) => {
    e?.preventDefault();
    if (!texto.trim()) return;

    setCargando(true);
    setError(null);

    try {
      const inicio = performance.now();
      const data = await predecirSentimiento(texto.trim());
      setResultado(normalizarResultado(data, texto.trim(), performance.now() - inicio));
    } catch (err) {
      setError(err.message);
      setResultado(null);
    } finally {
      setCargando(false);
    }
  };

  const handleLimpiar = () => {
    setTexto("");
    setResultado(null);
    setError(null);
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6">
        <h2 className="text-lg font-semibold text-stone-900">Analizar Reseña</h2>
        <p className="mt-1 text-sm text-stone-500">
          Ingresa una reseña sobre un restaurante para clasificar su sentimiento mediante PLN y Machine Learning.
        </p>

        <form onSubmit={handleAnalizar} className="mt-4 space-y-4">
          <div>
            <textarea
              rows={4}
              value={texto}
              onChange={(e) => setTexto(e.target.value)}
              placeholder="Ejemplo: La atención fue excelente y la comida llegó rápido, volvería sin duda..."
              className="w-full rounded-xl border border-stone-300 p-3 text-sm text-stone-800 placeholder-stone-400 focus:border-stone-500 focus:outline-none focus:ring-1 focus:ring-stone-500"
              disabled={cargando}
            />
          </div>

          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
              <span className="font-semibold">Error:</span> {error}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={cargando || !texto.trim()}
              className="rounded-xl bg-stone-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {cargando ? "Analizando..." : "Analizar Sentimiento"}
            </button>

            {(texto || resultado) && (
              <button
                type="button"
                onClick={handleLimpiar}
                disabled={cargando}
                className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50"
              >
                Limpiar
              </button>
            )}
          </div>
        </form>
      </section>

      {resultado && (
        <div className="space-y-6">
          <ResultadoSentimiento resultado={resultado} />

          <NlpPasoAPaso
            preprocesamiento={resultado.preprocesamiento}
            palabrasClave={resultado.palabras_clave}
          />
        </div>
      )}
    </div>
  );
}