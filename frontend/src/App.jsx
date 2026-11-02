import { useState } from "react";
import AnalizadorResenas from "./components/AnalizadorResenas.jsx";
import MetricasModelos from "./components/MetricasModelos.jsx";
import ResultadoSentimiento from "./components/ResultadoSentimiento.jsx";
import NlpPasoAPaso from "./components/NlpPasoAPaso.jsx";

const PESTANAS = [
  { id: "analizar", etiqueta: "Analizar reseña" },
  { id: "metricas", etiqueta: "Métricas de modelos" },
];

export default function App() {
  const [pestaña, setPestaña] = useState("analizar");
  const [resultadoAnalisis, setResultadoAnalisis] = useState(null);

  return (
    <div className="min-h-screen">
      <header className="border-b border-stone-200/80 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 px-4 py-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold tracking-[0.18em] text-orange-700 uppercase">
              Opiniómetro
            </p>
            <h1 className="mt-1 text-2xl font-semibold text-stone-900">
              Análisis de sentimiento en reseñas
            </h1>
            <p className="mt-1 text-sm text-stone-500">
              Clasificación positivo / negativo con FastAPI, Naive Bayes y Regresión Logística.
            </p>
          </div>
          <nav className="flex rounded-full border border-stone-200 bg-stone-50 p-1">
            {PESTANAS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setPestaña(item.id)}
                className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                  pestaña === item.id
                    ? "bg-orange-700 text-white shadow-sm"
                    : "text-stone-600 hover:text-stone-900"
                }`}
              >
                {item.etiqueta}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8">
        {pestaña === "analizar" ? (
          <div className="space-y-6">
            <AnalizadorResenas onResultado={setResultadoAnalisis} />
            {resultadoAnalisis && (
              <>
                {resultadoAnalisis.simulado && (
                  <p role="status" className="rounded-xl bg-amber-50 p-4 text-sm text-amber-900">
                    Modo de prueba: ejemplo fijo simulado; no es un análisis de tu reseña.
                  </p>
                )}
                <ResultadoSentimiento resultado={resultadoAnalisis} />
                <NlpPasoAPaso preprocesamiento={resultadoAnalisis.preprocesamiento}
                  palabrasClave={resultadoAnalisis.palabras_clave} />
              </>
            )}
          </div>
        ) : <MetricasModelos />}
      </main>
    </div>
  );
}
