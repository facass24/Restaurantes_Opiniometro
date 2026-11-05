import { useState } from "react";
import { predecirSentimiento } from "../services/api.js";
import { normalizarResultado } from "../services/resultadoAnalisis.js";
import ResultadoSentimiento from "./ResultadoSentimiento.jsx";
import NlpPasoAPaso from "./NlpPasoAPaso.jsx";

const EJEMPLOS = [
  {
    tipo: "positiva",
    etiqueta: "Excelente experiencia",
    texto: "La atención fue maravillosa, la pasta al dente perfecta y el postre sublime. Volveremos sin dudarlo.",
  },
  {
    tipo: "negativa",
    etiqueta: "Queja por servicio",
    texto: "Pésima experiencia. Tardaron más de 45 minutos en traer los platos y la carne estaba completamente fría y dura.",
  },
  {
    tipo: "desafio",
    etiqueta: "Negación / Frase corta",
    texto: "No estuvo nada rico el almuerzo.",
  },
];

export default function AnalizadorResenas() {
  const [texto, setTexto] = useState("");
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState(null);
  const [resultado, setResultado] = useState(null);

  const longitudMinima = 3;
  const esValido = texto.trim().length >= longitudMinima;

  const handleAnalizar = async (e) => {
    e?.preventDefault();
    if (!esValido || cargando) return;

    setCargando(true);
    setError(null);

    try {
      const inicio = performance.now();
      const data = await predecirSentimiento(texto.trim());
      console.log("DATOS CRUDOS DE LA API:", data);
      const resNormalizado = normalizarResultado(data, texto.trim(), performance.now() - inicio);
      console.log("RESULTADO NORMALIZADO:", resNormalizado);
      setResultado(resNormalizado);
    } catch (err) {
      setError(err.message || "Ocurrió un error inesperado al conectar con el servidor.");
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
      <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm transition-all sm:p-7">
        <div className="flex flex-col gap-1 border-b border-stone-100 pb-4">
          <h2 className="text-xl font-bold tracking-tight text-stone-900">
            Analizador de Reseñas en Tiempo Real
          </h2>
          <p className="text-sm text-stone-500">
            Escribe o selecciona una reseña gastronómica para clasificar su polaridad mediante PLN y modelos de Machine Learning.
          </p>
        </div>

        {/* Ejemplos rápidos para la defensa / pruebas */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-stone-400">
            Probar ejemplo:
          </span>
          {EJEMPLOS.map((ej, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setTexto(ej.texto);
                setError(null);
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 bg-stone-50 px-3 py-1 text-xs font-medium text-stone-700 transition hover:border-stone-300 hover:bg-stone-100 active:scale-95"
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  ej.tipo === "positiva"
                    ? "bg-emerald-500"
                    : ej.tipo === "negativa"
                    ? "bg-rose-500"
                    : "bg-amber-500"
                }`}
              />
              {ej.etiqueta}
            </button>
          ))}
        </div>

        <form onSubmit={handleAnalizar} className="mt-4 space-y-4">
          <div className="relative">
            <textarea
              rows={4}
              value={texto}
              onChange={(e) => {
                setTexto(e.target.value);
                if (error) setError(null);
              }}
              placeholder="Escribe aquí tu opinión sobre el restaurante (ej. 'La comida estuvo exquisita pero la atención fue lenta')..."
              className="w-full resize-y rounded-xl border border-stone-300 p-4 text-sm leading-relaxed text-stone-800 placeholder-stone-400 shadow-inner transition focus:border-stone-900 focus:outline-none focus:ring-1 focus:ring-stone-900 disabled:bg-stone-50"
              disabled={cargando}
            />
            <div className="mt-1 flex items-center justify-between px-1 text-xs text-stone-400">
              <span>Mínimo 3 caracteres para clasificar</span>
              <span className={texto.length > 0 && !esValido ? "font-medium text-rose-500" : ""}>
                {texto.length} caracteres
              </span>
            </div>
          </div>

          {error && (
            <div className="flex items-start gap-3 rounded-xl border border-rose-200 bg-rose-50/80 p-4 text-sm text-rose-800 shadow-sm animate-in fade-in">
              <svg className="mt-0.5 h-5 w-5 flex-shrink-0 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <div>
                <p className="font-semibold">Validación de entrada:</p>
                <p className="mt-0.5 text-xs text-rose-700">{error}</p>
              </div>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button
              type="submit"
              disabled={cargando || !esValido}
              className="inline-flex items-center gap-2 rounded-xl bg-stone-900 px-6 py-2.5 text-sm font-semibold text-white shadow transition hover:bg-stone-800 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {cargando ? (
                <>
                  <svg className="h-4 w-4 animate-spin text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                  </svg>
                  <span>Analizando texto...</span>
                </>
              ) : (
                <>
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Analizar Sentimiento</span>
                </>
              )}
            </button>

            {(texto || resultado) && (
              <button
                type="button"
                onClick={handleLimpiar}
                disabled={cargando}
                className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 transition hover:bg-stone-50 hover:text-stone-900 active:scale-95 disabled:opacity-50"
              >
                Limpiar campos
              </button>
            )}
          </div>
        </form>
      </section>

      {resultado && (
        <div className="space-y-6 transition-all duration-300">
          <ResultadoSentimiento resultado={resultado} />
          <NlpPasoAPaso
  preprocesamiento={{
    ...resultado.preprocesamiento,
    clase: resultado.clase,
    sentimiento: resultado.sentimiento,
  }}
  palabrasClave={resultado.palabras_clave}
/>
        </div>
      )}
    </div>
  );
}