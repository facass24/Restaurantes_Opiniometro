import { useState, useMemo } from "react";

export default function TablaResultados({ resultados = [] }) {
  const [busqueda, setBusqueda] = useState("");
  const [filtroSentimiento, setFiltroSentimiento] = useState("TODOS");
  const [filtroSucursal, setFiltroSucursal] = useState("TODAS");
  const [paginaActual, setPaginaActual] = useState(1);
  const [filasPorPagina, setFilasPorPagina] = useState(10);
  const [expandidas, setExpandidas] = useState({});

  // Extraer lista única de sucursales si existen
  const sucursalesUnicas = useMemo(() => {
    const sedes = new Set();
    resultados.forEach((item) => {
      if (item.sucursal) sedes.add(item.sucursal);
    });
    return Array.from(sedes).sort();
  }, [resultados]);

  const tieneSucursal = sucursalesUnicas.length > 0;
  const tieneFecha = useMemo(() => resultados.some((r) => Boolean(r.fecha)), [resultados]);

  // Filtrado reactivo
  const resultadosFiltrados = useMemo(() => {
    return resultados.filter((item) => {
      // Filtro de sentimiento
      if (filtroSentimiento === "POSITIVO" && item.sentimiento !== "Positivo") return false;
      if (filtroSentimiento === "NEGATIVO" && item.sentimiento !== "Negativo") return false;

      // Filtro de sucursal
      if (filtroSucursal !== "TODAS" && item.sucursal !== filtroSucursal) return false;

      // Búsqueda de texto
      if (busqueda.trim()) {
        const query = busqueda.toLowerCase();
        const textoMatch = item.reseña_original?.toLowerCase().includes(query);
        const sucursalMatch = item.sucursal?.toLowerCase().includes(query);
        if (!textoMatch && !sucursalMatch) return false;
      }

      return true;
    });
  }, [resultados, filtroSentimiento, filtroSucursal, busqueda]);

  // Paginación
  const totalPaginas = Math.ceil(resultadosFiltrados.length / filasPorPagina) || 1;
  const indiceInicio = (paginaActual - 1) * filasPorPagina;
  const itemsPagina = resultadosFiltrados.slice(indiceInicio, indiceInicio + filasPorPagina);

  const toggleExpandir = (indice) => {
    setExpandidas((prev) => ({ ...prev, [indice]: !prev[indice] }));
  };

  const cambiarPagina = (nueva) => {
    if (nueva >= 1 && nueva <= totalPaginas) {
      setPaginaActual(nueva);
    }
  };

  if (!resultados || resultados.length === 0) return null;

  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6 space-y-4">
      {/* Encabezado y controles */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-lg font-semibold text-stone-900">
            Detalle de Reseñas Clasificadas
          </h3>
          <p className="text-xs text-stone-500">
            Mostrando {resultadosFiltrados.length} de {resultados.length} opiniones totales
          </p>
        </div>

        {/* Buscador */}
        <div className="relative w-full sm:w-72">
          <input
            type="text"
            value={busqueda}
            onChange={(e) => {
              setBusqueda(e.target.value);
              setPaginaActual(1);
            }}
            placeholder="Buscar por palabra o sede..."
            className="w-full rounded-xl border border-stone-300 py-2 pl-9 pr-3 text-xs text-stone-800 placeholder-stone-400 focus:border-stone-500 focus:outline-none focus:ring-1 focus:ring-stone-500"
          />
          <svg
            className="absolute left-3 top-2.5 h-4 w-4 text-stone-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* Barra de Filtros */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-y border-stone-100 py-3">
        {/* Filtros de Sentimiento */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => {
              setFiltroSentimiento("TODOS");
              setPaginaActual(1);
            }}
            className={`rounded-lg px-3 py-1 text-xs font-medium transition ${
              filtroSentimiento === "TODOS"
                ? "bg-stone-900 text-white"
                : "bg-stone-100 text-stone-600 hover:bg-stone-200"
            }`}
          >
            Todas ({resultados.length})
          </button>
          <button
            type="button"
            onClick={() => {
              setFiltroSentimiento("POSITIVO");
              setPaginaActual(1);
            }}
            className={`rounded-lg px-3 py-1 text-xs font-medium transition ${
              filtroSentimiento === "POSITIVO"
                ? "bg-emerald-700 text-white"
                : "bg-emerald-50 text-emerald-800 hover:bg-emerald-100"
            }`}
          >
            Positivas ({resultados.filter((r) => r.sentimiento === "Positivo").length})
          </button>
          <button
            type="button"
            onClick={() => {
              setFiltroSentimiento("NEGATIVO");
              setPaginaActual(1);
            }}
            className={`rounded-lg px-3 py-1 text-xs font-medium transition ${
              filtroSentimiento === "NEGATIVO"
                ? "bg-rose-700 text-white"
                : "bg-rose-50 text-rose-800 hover:bg-rose-100"
            }`}
          >
            Negativas ({resultados.filter((r) => r.sentimiento === "Negativo").length})
          </button>
        </div>

        {/* Filtro por Sucursal (si aplica) */}
        {tieneSucursal && (
          <div className="flex items-center gap-2">
            <label className="text-xs text-stone-500 font-medium">Sede:</label>
            <select
              value={filtroSucursal}
              onChange={(e) => {
                setFiltroSucursal(e.target.value);
                setPaginaActual(1);
              }}
              className="rounded-lg border border-stone-300 bg-white py-1 px-2.5 text-xs text-stone-700 focus:outline-none focus:ring-1 focus:ring-stone-500"
            >
              <option value="TODAS">Todas las sedes</option>
              {sucursalesUnicas.map((sede) => (
                <option key={sede} value={sede}>
                  {sede}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Tabla */}
      <div className="overflow-x-auto rounded-xl border border-stone-200">
        <table className="min-w-full divide-y divide-stone-200 text-left text-xs">
          <thead className="bg-stone-50 text-stone-600 font-medium">
            <tr>
              <th scope="col" className="w-12 px-3 py-3 text-center">#</th>
              <th scope="col" className="px-4 py-3">Opinión del Cliente</th>
              <th scope="col" className="w-32 px-3 py-3 text-center">Sentimiento</th>
              <th scope="col" className="w-28 px-3 py-3 text-center">Certeza</th>
              {tieneSucursal && <th scope="col" className="w-32 px-3 py-3">Sucursal</th>}
              {tieneFecha && <th scope="col" className="w-28 px-3 py-3">Fecha</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-100 bg-white">
            {itemsPagina.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-8 text-center text-stone-500">
                  No se encontraron opiniones que coincidan con los filtros seleccionados.
                </td>
              </tr>
            ) : (
              itemsPagina.map((item) => {
                const esPositivo = item.sentimiento === "Positivo";
                const expandida = expandidas[item.indice];
                const esLargo = item.reseña_original?.length > 120;

                return (
                  <tr key={item.indice} className="hover:bg-stone-50/60 transition">
                    <td className="px-3 py-3 text-center font-mono text-stone-400">
                      {item.indice}
                    </td>
                    <td className="px-4 py-3 text-stone-800">
                      <p className="leading-relaxed">
                        {esLargo && !expandida
                          ? `${item.reseña_original.slice(0, 120)}...`
                          : item.reseña_original}
                      </p>
                      {esLargo && (
                        <button
                          type="button"
                          onClick={() => toggleExpandir(item.indice)}
                          className="mt-1 text-[11px] font-medium text-orange-700 hover:underline"
                        >
                          {expandida ? "Ver menos" : "Ver reseña completa"}
                        </button>
                      )}
                    </td>
                    <td className="px-3 py-3 text-center whitespace-nowrap">
                      <span
                        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${
                          esPositivo
                            ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                            : "bg-rose-50 text-rose-800 border border-rose-200"
                        }`}
                      >
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${
                            esPositivo ? "bg-emerald-600" : "bg-rose-600"
                          }`}
                        />
                        {item.sentimiento}
                      </span>
                    </td>
                    <td className="px-3 py-3 text-center whitespace-nowrap">
                      <div className="flex flex-col items-center gap-1">
                        <span className="font-semibold text-stone-800">
                          {typeof item.certeza === "number" ? `${item.certeza.toFixed(1)}%` : item.certeza}
                        </span>
                        <div className="h-1 w-16 overflow-hidden rounded-full bg-stone-200">
                          <div
                            className={`h-full ${esPositivo ? "bg-emerald-600" : "bg-rose-600"}`}
                            style={{ width: `${item.certeza || 0}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    {tieneSucursal && (
                      <td className="px-3 py-3 text-stone-600 whitespace-nowrap">
                        {item.sucursal ? (
                          <span className="rounded-md bg-stone-100 px-2 py-0.5 text-stone-700">
                            {item.sucursal}
                          </span>
                        ) : (
                          <span className="text-stone-400 italic">No especificada</span>
                        )}
                      </td>
                    )}
                    {tieneFecha && (
                      <td className="px-3 py-3 text-stone-500 whitespace-nowrap text-[11px]">
                        {item.fecha || "—"}
                      </td>
                    )}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Controles de paginación */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between pt-2">
        <div className="flex items-center gap-2 text-xs text-stone-500">
          <span>Filas por página:</span>
          <select
            value={filasPorPagina}
            onChange={(e) => {
              setFilasPorPagina(Number(e.target.value));
              setPaginaActual(1);
            }}
            className="rounded-lg border border-stone-300 bg-white py-1 px-2 text-xs text-stone-700 focus:outline-none focus:ring-1 focus:ring-stone-500"
          >
            <option value={10}>10</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
          </select>
          <span>
            Página {paginaActual} de {totalPaginas}
          </span>
        </div>

        <div className="flex items-center gap-1.5 self-end sm:self-auto">
          <button
            type="button"
            onClick={() => cambiarPagina(paginaActual - 1)}
            disabled={paginaActual <= 1}
            className="rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-xs font-medium text-stone-700 hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Anterior
          </button>
          <button
            type="button"
            onClick={() => cambiarPagina(paginaActual + 1)}
            disabled={paginaActual >= totalPaginas}
            className="rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-xs font-medium text-stone-700 hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Siguiente
          </button>
        </div>
      </div>
    </section>
  );
}

