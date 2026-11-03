import { useState } from "react";
import { predecirSentimiento } from "../services/api.js";
import { normalizarResultado, crearEjemplo } from "../services/resultadoAnalisis.js";

const PLACEHOLDER =
  "Escribe una reseña (ej: La comida estuvo deliciosa pero la atención tardó mucho)...";

export default function AnalizadorResenas({ onResultado }) {
  const [texto, setTexto] = useState("");
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState("");

  async function analizar(evento) {
    evento.preventDefault();
    if (cargando) return;
    setError("");

    if (!texto.trim()) {
      setError("Escribe una reseña antes de analizar.");
      return;
    }

    setCargando(true);
    onResultado(null);
    try {
      const inicio = performance.now();
      const data = await predecirSentimiento(texto);
      onResultado(normalizarResultado(data, texto, performance.now() - inicio));
    } catch (err) {
      setError(err.message);
    } finally {
      setCargando(false);
    }
  }

  return (
    <section className="space-y-6">
      <form
        onSubmit={analizar}
        className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6"
      >
        <label htmlFor="resena" className="block text-sm font-medium text-stone-800">
          Reseña del restaurante
        </label>
        <textarea
          id="resena"
          rows={6}
          value={texto}
          onChange={(e) => setTexto(e.target.value)}
          placeholder={PLACEHOLDER}
          className="mt-2 w-full resize-y rounded-xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm text-stone-800 outline-none transition placeholder:text-stone-400 focus:border-orange-500 focus:bg-white focus:ring-2 focus:ring-orange-200"
        />
        <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-stone-400">{texto.trim().length} caracteres</p>
          <button
            type="submit"
            disabled={cargando}
            className="inline-flex items-center justify-center rounded-full bg-orange-700 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-orange-800 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {cargando ? (
              <>
                <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                Analizando...
              </>
            ) : (
              "Analizar Sentimiento"
            )}
          </button>
        </div>
      </form>

      {error && (
        <div
          role="alert"
          className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800"
        >
          {error}
        </div>
      )}

      <button type="button" disabled={cargando}
        onClick={() => { setError(""); onResultado(crearEjemplo()); }}
        className="rounded-full border border-stone-300 bg-white px-5 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50">
        Probar con un ejemplo simulado
      </button>
    </section>
  );
}
