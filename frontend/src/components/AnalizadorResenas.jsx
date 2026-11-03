import { useState } from "react";
import api from "../services/api.js";
import { normalizarResultado } from "../services/resultadoAnalisis.js";
import ResultadoSentimiento from "./ResultadoSentimiento.jsx";
import NlpPasoAPaso from "./NlpPasoAPaso.jsx";

export default function AnalizadorResenas() {
  const [texto, setTexto] = useState("");
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState(null);
  const [resultado, setResultado] = useState(null);

  const manejarEnvio = async (e) => {
    e.preventDefault();
    if (!texto.trim()) return;

    setCargando(true);
    setError(null);

    try {
      const { data } = await api.post("/api/v1/predict", { texto: texto.trim() });
      setResultado(normalizarResultado(data));
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          "Error al conectar con el backend para analizar la reseña."
      );
      setResultado(null);
    } finally {
      setCargando(false);
    }
  };

  return (
    <section className="space-y-6">
      <form onSubmit={manejarEnvio} className="space-y-4">
        <div>
          <label htmlFor="resena" className="block text-sm font-medium text-stone-700">
            Escribe o pega una reseña
          </label>
          <textarea
            id="resena"
            rows="4"
            value={texto}
            onChange={(e) => setTexto(e.target.value)}
            placeholder="Ejemplo: La comida estuvo deliciosa y el servicio fue muy atento..."
            className="mt-1 block w-full rounded-xl border border-stone-300 p-3 shadow-sm focus:border-stone-500 focus:outline-none focus:ring-1 focus:ring-stone-500"
            disabled={cargando}
          />
        </div>

        {error && (
          <div className="rounded-lg bg-rose-50 p-3 text-sm text-rose-700 border border-rose-200">
            {error}
          </div>
        )}

        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={cargando || !texto.trim()}
            className="rounded-xl bg-stone-900 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-stone-800 disabled:opacity-50 transition"
          >
            {cargando ? "Analizando…" : "Analizar Sentimiento"}
          </button>
        </div>
      </form>

      {resultado && (
        <div className="space-y-6">
          <ResultadoSentimiento resultado={resultado} />
          <NlpPasoAPaso
            preprocesamiento={resultado.preprocesamiento}
            palabrasClave={resultado.palabrasClave}
          />
        </div>
      )}
    </section>
  );
}