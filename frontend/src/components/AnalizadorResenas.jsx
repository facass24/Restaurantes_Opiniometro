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

  const manejarEnvio = async (e) => {
    e.preventDefault();
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
    }
    } catch (err) {
      let mensajeError = "Ocurrió un error al analizar la reseña.";
    
      const detail = err.response?.data?.detail;
      if (typeof detail === "string") {
        mensajeError = detail;
      } else if (Array.isArray(detail)) {
        // Cuando FastAPI devuelve lista de errores de validación (422)
        mensajeError = detail.map((e) => e.msg || JSON.stringify(e)).join(", ");
      } else if (err.message) {
        mensajeError = err.message;
      }
    
      setError(mensajeError);
    }
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