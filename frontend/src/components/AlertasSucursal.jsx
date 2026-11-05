import { useState } from "react";

export default function AlertasSucursal({ alertasData }) {
  const [filtroAlerta, setFiltroAlerta] = useState("TODOS");

  if (!alertasData || !alertasData.alertas || alertasData.alertas.length === 0) {
    return null;
  }

  const { total_sucursales, sucursales_en_alerta, alertas } = alertasData;

  const alertasFiltradas = alertas.filter((a) => {
    if (filtroAlerta === "EN_ALERTA") {
      return a.nivel_alerta === "CRITICO" || a.nivel_alerta === "ATENCION";
    }
    if (filtroAlerta === "CRITICO") return a.nivel_alerta === "CRITICO";
    if (filtroAlerta === "ATENCION") return a.nivel_alerta === "ATENCION";
    if (filtroAlerta === "NORMAL") return a.nivel_alerta === "NORMAL";
    return true;
  });

  const getEstilosNivel = (nivel) => {
    switch (nivel) {
      case "CRITICO":
        return {
          badge: "bg-rose-100 text-rose-800 border-rose-300",
          cardBorder: "border-rose-300 bg-rose-50/20",
          barra: "bg-rose-600",
          icono: (
            <svg className="h-4 w-4 text-rose-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          ),
        };
      case "ATENCION":
        return {
          badge: "bg-amber-100 text-amber-800 border-amber-300",
          cardBorder: "border-amber-300 bg-amber-50/20",
          barra: "bg-amber-600",
          icono: (
            <svg className="h-4 w-4 text-amber-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
        };
      default:
        return {
          badge: "bg-emerald-100 text-emerald-800 border-emerald-300",
          cardBorder: "border-stone-200 bg-white",
          barra: "bg-emerald-600",
          icono: (
            <svg className="h-4 w-4 text-emerald-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ),
        };
    }
  };

  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6 space-y-5">
      {/* Encabezado */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-stone-100 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold tracking-wider text-orange-700 uppercase">
              Monitoreo por Sede
            </span>
            <span className="inline-flex items-center rounded-full bg-stone-100 px-2.5 py-0.5 text-xs font-medium text-stone-600">
              {total_sucursales} {total_sucursales === 1 ? "sucursal" : "sucursales"}
            </span>
          </div>
          <h3 className="text-lg font-semibold text-stone-900 mt-0.5">
            Diagnóstico de Alertas Operativas
          </h3>
          <p className="text-xs text-stone-500">
            Regla de decisión: Normal (&lt;15%), Atención (15-25%), Crítico (&gt;25% de quejas).
          </p>
        </div>

        {/* Indicador de sedes en alerta */}
        <div className="flex items-center gap-2">
          {sucursales_en_alerta > 0 ? (
            <div className="flex items-center gap-2 rounded-xl bg-rose-50 border border-rose-200 px-3 py-1.5 text-xs font-medium text-rose-800">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-600"></span>
              </span>
              <span>{sucursales_en_alerta} sede(s) requieren atención</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 rounded-xl bg-emerald-50 border border-emerald-200 px-3 py-1.5 text-xs font-medium text-emerald-800">
              <svg className="h-4 w-4 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span>Todas las sedes operando en nivel normal</span>
            </div>
          )}
        </div>
      </div>

      {/* Filtros rápidos */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-stone-500 font-medium mr-1">Filtrar:</span>
        <button
          type="button"
          onClick={() => setFiltroAlerta("TODOS")}
          className={`rounded-lg px-3 py-1 text-xs font-medium transition ${
            filtroAlerta === "TODOS"
              ? "bg-stone-900 text-white"
              : "bg-stone-100 text-stone-600 hover:bg-stone-200"
          }`}
        >
          Todas ({alertas.length})
        </button>
        {sucursales_en_alerta > 0 && (
          <button
            type="button"
            onClick={() => setFiltroAlerta("EN_ALERTA")}
            className={`rounded-lg px-3 py-1 text-xs font-medium transition ${
              filtroAlerta === "EN_ALERTA"
                ? "bg-rose-700 text-white"
                : "bg-rose-50 text-rose-700 hover:bg-rose-100"
            }`}
          >
            En Alerta ({sucursales_en_alerta})
          </button>
        )}
        <button
          type="button"
          onClick={() => setFiltroAlerta("NORMAL")}
          className={`rounded-lg px-3 py-1 text-xs font-medium transition ${
            filtroAlerta === "NORMAL"
              ? "bg-emerald-700 text-white"
              : "bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
          }`}
        >
          Normales ({alertas.filter((a) => a.nivel_alerta === "NORMAL").length})
        </button>
      </div>

      {/* Grid de tarjetas por sucursal */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {alertasFiltradas.map((alerta) => {
          const estilos = getEstilosNivel(alerta.nivel_alerta);
          return (
            <div
              key={alerta.sucursal}
              className={`rounded-xl border p-4.5 transition-all shadow-xs flex flex-col justify-between ${estilos.cardBorder}`}
            >
              <div>
                {/* Cabecera de la sucursal */}
                <div className="flex items-start justify-between gap-2">
                  <div className="truncate">
                    <p className="font-semibold text-stone-900 truncate text-base">
                      {alerta.sucursal}
                    </p>
                    <p className="text-xs text-stone-500">
                      {alerta.total_reseñas} {alerta.total_reseñas === 1 ? "opinión" : "opiniones"} registradas
                    </p>
                  </div>
                  <span
                    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${estilos.badge}`}
                  >
                    {estilos.icono}
                    {alerta.nivel_alerta}
                  </span>
                </div>

                {/* Tasa de quejas */}
                <div className="mt-4 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-stone-600">Tasa de Insatisfacción:</span>
                    <span className="font-bold text-stone-900">
                      {alerta.porcentaje_insatisfaccion.toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-stone-200">
                    <div
                      className={`h-full ${estilos.barra} rounded-full`}
                      style={{ width: `${Math.min(alerta.porcentaje_insatisfaccion, 100)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[11px] text-stone-400">
                    <span>{alerta.positivas} positivas</span>
                    <span>{alerta.negativas} quejas</span>
                  </div>
                </div>

                {/* Palabras críticas */}
                {alerta.palabras_criticas && alerta.palabras_criticas.length > 0 && (
                  <div className="mt-3.5">
                    <p className="text-[11px] font-semibold tracking-wide text-stone-500 uppercase">
                      Términos recurrentes en quejas:
                    </p>
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {alerta.palabras_criticas.map((palabra, idx) => (
                        <span
                          key={idx}
                          className="rounded-md bg-stone-100 px-2 py-0.5 text-[11px] font-medium text-stone-700 border border-stone-200"
                        >
                          "{palabra}"
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Mensaje descriptivo */}
              <div className="mt-4 pt-3 border-t border-stone-100 text-xs text-stone-600 italic">
                {alerta.mensaje}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

