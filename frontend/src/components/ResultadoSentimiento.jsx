function MedidorConfianza({ porcentaje, positivo }) {
  const radio = 42;
  const circunferencia = 2 * Math.PI * radio;
  const progreso = circunferencia * (1 - Math.min(Math.max(porcentaje, 0), 100) / 100);
  const color = positivo ? "#059669" : "#e11d48";

  return (
    <div className="flex flex-col items-center">
      <div className="relative h-28 w-28">
        <svg viewBox="0 0 108 108" className="h-28 w-28 -rotate-90">
          <circle cx="54" cy="54" r={radio} fill="none" stroke="#e7e5e4" strokeWidth="10" />
          <circle
            cx="54"
            cy="54"
            r={radio}
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circunferencia}
            strokeDashoffset={progreso}
          />
        </svg>
        <p className="absolute inset-0 flex items-center justify-center text-xl font-semibold text-stone-900">
          {porcentaje.toFixed(0)}%
        </p>
      </div>
      <p className="mt-1 text-xs font-medium tracking-wide text-stone-500 uppercase">Confianza</p>
    </div>
  );
}

export default function ResultadoSentimiento({ resultado }) {
  if (!resultado) return null;
  const esPositivo = resultado.clase === "positivo";
  const probabilidadPct = Math.round(resultado.probabilidad * 1000) / 10;

  return (
    <article className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold tracking-wide text-stone-500 uppercase">Resultado</p>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <span
              className={`rounded-full px-4 py-1.5 text-sm font-semibold ${
                esPositivo
                  ? "bg-emerald-100 text-emerald-800"
                  : "bg-rose-100 text-rose-800"
              }`}
            >
              {esPositivo ? "Positivo" : "Negativo"}
            </span>
            <span className="text-sm text-stone-500">
              Modelo: {resultado.modelo_usado} · {resultado.tiempo_ms?.toFixed?.(1) ?? resultado.tiempo_ms} ms
            </span>
          </div>
        </div>
        <MedidorConfianza
          porcentaje={resultado.confianza_porcentaje ?? probabilidadPct}
          positivo={esPositivo}
        />
      </div>

      <div className="mt-2">
        <div className="mb-4 grid grid-cols-2 gap-3 text-sm">
          {Object.entries(resultado.probabilidades ?? {}).map(([clase, probabilidad]) => (
            <div key={clase} className="rounded-xl bg-stone-50 p-3">
              <span className="capitalize text-stone-600">{clase}</span>
              <p className="font-semibold text-stone-900">{(probabilidad * 100).toFixed(1)}%</p>
            </div>
          ))}
        </div>
        <div className="mb-1 flex items-center justify-between text-sm">
          <span className="text-stone-600">Probabilidad de la clase predicha</span>
          <span className="font-medium text-stone-900">{probabilidadPct}%</span>
        </div>
        <div className="h-3 overflow-hidden rounded-full bg-stone-100">
          <div
            className={`h-full rounded-full transition-all ${esPositivo ? "bg-emerald-500" : "bg-rose-500"}`}
            style={{ width: `${Math.min(probabilidadPct, 100)}%` }}
          />
        </div>
      </div>
    </article>
  );
}
