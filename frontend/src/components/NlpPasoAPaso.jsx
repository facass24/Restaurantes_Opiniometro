import { useState } from "react";

function chips(valor) {
  if (Array.isArray(valor)) return valor;
  if (!valor) return [];
  return String(valor).split(/\s+/).filter(Boolean);
}

export default function NlpPasoAPaso({ preprocesamiento, palabrasClave = [] }) {
  const [abierto, setAbierto] = useState("original");

  if (!preprocesamiento) return null;

  const pasos = [
    {
      id: "original",
      titulo: "1. Original",
      descripcion: "Texto tal como lo escribió el usuario.",
      contenido: preprocesamiento.original,
      tipo: "texto",
    },
    {
      id: "normalizado",
      titulo: "2. Limpieza / Normalización",
      descripcion: "Minúsculas, tildes y caracteres normalizados.",
      contenido: preprocesamiento.normalizado,
      tipo: "texto",
    },
    {
      id: "tokens",
      titulo: "3. Tokenización",
      descripcion: "El texto se parte en unidades (tokens).",
      contenido: preprocesamiento.tokens,
      tipo: "chips",
    },
    {
      id: "reducidos",
      titulo: "4. Lematización / Stemming",
      descripcion: "Raíces o lemas. También se filtran stopwords conservando negaciones.",
      contenido: preprocesamiento.reducidos?.length
        ? preprocesamiento.reducidos
        : preprocesamiento.sin_stopwords,
      extra: preprocesamiento.sin_stopwords,
      tipo: "chips",
    },
    {
      id: "tfidf",
      titulo: "5. Representación TF-IDF",
      descripcion:
        "El modelo no expone el vector completo; se muestran las palabras con mayor peso en la decisión.",
      contenido: palabrasClave,
      tipo: "pesos",
    },
  ];

  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6">
      <h2 className="text-lg font-semibold text-stone-900">NLP paso a paso</h2>
      <p className="mt-1 text-sm text-stone-500">
        Recorrido didáctico del pipeline: original → limpieza → tokens → reducción → TF-IDF.
      </p>

      <div className="mt-4 space-y-2">
        {pasos.map((paso) => {
          const estaAbierto = abierto === paso.id;
          const maxPeso = paso.tipo === "pesos"
  ? Math.max(...(paso.contenido || []).map((i) => Math.abs(i.peso)), 0.0001)
  : 1;
          return (
            <div key={paso.id} className="overflow-hidden rounded-xl border border-stone-200">
              <button
                type="button"
                aria-expanded={estaAbierto}
                onClick={() => setAbierto(estaAbierto ? null : paso.id)}
                className="flex w-full items-center justify-between bg-stone-50 px-4 py-3 text-left"
              >
                <span>
                  <span className="block text-sm font-semibold text-stone-800">{paso.titulo}</span>
                  <span className="block text-xs text-stone-500">{paso.descripcion}</span>
                </span>
                <span className="text-stone-400">{estaAbierto ? "−" : "+"}</span>
              </button>

              {estaAbierto && (
                <div className="border-t border-stone-200 px-4 py-3">
                  {paso.tipo === "texto" && (
                    <p className="text-sm leading-relaxed text-stone-700">{paso.contenido || "—"}</p>
                  )}

                  {paso.tipo === "chips" && (
                    <div className="flex flex-wrap gap-1.5">
                      {chips(paso.contenido).length === 0 && (
                        <span className="text-sm text-stone-400">Sin tokens</span>
                      )}
                      {chips(paso.contenido).map((token, idx) => (
                        <span key={`${token}-${idx}`} className="token-chip">
                          {token}
                        </span>
                      ))}
                    </div>
                  )}

                  {paso.id === "reducidos" && paso.extra?.length > 0 && (
                    <p className="mt-3 text-xs text-stone-500">
                      Sin stopwords: {paso.extra.join(" · ")}
                    </p>
                  )}

                  {paso.tipo === "pesos" && (
                    <div className="space-y-2">
                      {(!paso.contenido || paso.contenido.length === 0) && (
                        <p className="text-sm text-stone-400">
                          No hay pesos TF-IDF disponibles para esta predicción.
                        </p>
                      )}
                      {(paso.contenido || []).map((item) => {
                        const positivo = item.polaridad ? item.polaridad === "positivo" : item.peso >= 0;
                        const ancho = (Math.abs(item.peso) / maxPeso) * 100;
                        return (
                          <div key={`${item.palabra}-${item.peso}`} className="grid grid-cols-[7rem_1fr_4rem] items-center gap-2">
                            <span className="truncate font-mono text-xs text-stone-700">
                              {item.palabra}
                            </span>
                            <div className="h-2 rounded-full bg-stone-100">
                              <div
                                className={`h-full rounded-full ${positivo ? "bg-emerald-500" : "bg-rose-500"}`}
                                style={{ width: `${ancho}%` }}
                              />
                            </div>
                            <span className="text-right text-xs text-stone-500">
                              {item.peso.toFixed(2)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
