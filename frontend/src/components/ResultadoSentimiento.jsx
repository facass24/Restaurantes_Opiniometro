export default function ResultadoSentimiento({ resultado }) {
  if (!resultado) return null;

  // 1. Extraer probabilidades de forma segura
  const probPositivo = Number(
    resultado.probabilidades?.positivo ?? 
    resultado.probabilidades?.Positivo ?? 
    0
  );
  const probNegativo = Number(
    resultado.probabilidades?.negativo ?? 
    resultado.probabilidades?.Negativo ?? 
    0
  );

  // 2. Determinar si es positivo evaluando el texto O las probabilidades
  const rawSentimiento = (
    resultado.sentimiento || 
    resultado.prediccion || 
    resultado.polaridad || 
    resultado.clase || 
    (probPositivo >= probNegativo ? "Positivo" : "Negativo")
  ).toString().toLowerCase();

  const esPositivo = rawSentimiento.includes("pos") || rawSentimiento === "1" || probPositivo >= 0.5;

  // 3. Determinar el texto a mostrar en el título
  const textoSentimiento = esPositivo ? "Positivo" : "Negativo";

  // 4. Si la confianza viene en 0 o undefined, usar la probabilidad dominante
  const confianza = Number(
    resultado.confianza || 
    (esPositivo ? probPositivo : probNegativo) || 
    0
  );

  const latencia = resultado.latenciaMs ?? resultado.latencia_ms ?? 0;

  return (
    <section
      className={`relative overflow-hidden rounded-2xl border p-6 shadow-sm transition-all sm:p-7 ${
        esPositivo
          ? "border-emerald-200 bg-emerald-50/40 text-emerald-950"
          : "border-rose-200 bg-rose-50/40 text-rose-950"
      }`}
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3.5">
          <div
            className={`flex h-12 w-12 items-center justify-center rounded-2xl shadow-sm ${
              esPositivo ? "bg-emerald-500 text-white" : "bg-rose-500 text-white"
            }`}
          >
            {esPositivo ? (
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ) : (
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-2xl font-black uppercase tracking-tight">
                {textoSentimiento}
              </h3>
              <span
                className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                  esPositivo ? "bg-emerald-100 text-emerald-800" : "bg-rose-100 text-rose-800"
                }`}
              >
                {(confianza * 100).toFixed(1)}% certeza
              </span>
            </div>
            <p className="text-xs text-stone-500">
              Modelo en inferencia: <span className="font-medium text-stone-700">{resultado.modelo || "Regresión Logística"}</span>
            </p>
          </div>
        </div>

        {/* Badge de Latencia */}
        <div className="flex items-center gap-1.5 self-start rounded-xl border border-stone-200 bg-white px-3 py-1.5 text-xs font-semibold text-stone-700 shadow-sm sm:self-center">
          <svg className="h-4 w-4 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
          </svg>
          <span>{Number(latencia).toFixed(0)} ms</span>
          <span className="text-stone-400 font-normal">latencia</span>
        </div>
      </div>

      {/* Barras de distribución de probabilidades */}
      <div className="mt-6 border-t border-stone-200/60 pt-4">
        <div className="flex justify-between text-xs font-semibold text-stone-600">
          <span>Distribución de polaridad</span>
          <span>Positivo {(probPositivo * 100).toFixed(1)}% · Negativo {(probNegativo * 100).toFixed(1)}%</span>
        </div>
        <div className="mt-2 flex h-2.5 w-full overflow-hidden rounded-full bg-stone-200/70 p-0.5">
          <div
            className="rounded-l-full bg-emerald-500 transition-all duration-500"
            style={{ width: `${Math.max(probPositivo * 100, 3)}%` }}
            title={`Positivo: ${(probPositivo * 100).toFixed(1)}%`}
          />
          <div
            className="rounded-r-full bg-rose-500 transition-all duration-500"
            style={{ width: `${Math.max(probNegativo * 100, 3)}%` }}
            title={`Negativo: ${(probNegativo * 100).toFixed(1)}%`}
          />
        </div>
      </div>
    </section>
  );
}