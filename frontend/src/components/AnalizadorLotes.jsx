import { useState, useEffect, useRef } from "react";
import CargaArchivo from "./CargaArchivo.jsx";
import ResumenLote from "./ResumenLote.jsx";
import AlertasSucursal from "./AlertasSucursal.jsx";
import TablaResultados from "./TablaResultados.jsx";
import {
  subirArchivoLote,
  obtenerEstadoLote,
  obtenerAlertasLote,
} from "../services/api.js";

export default function AnalizadorLotes() {
  const [cargando, setCargando] = useState(false);
  const [progreso, setProgreso] = useState(null);
  const [error, setError] = useState(null);
  const [lote, setLote] = useState(null);
  const [alertas, setAlertas] = useState(null);

  const pollingRef = useRef(null);

  // Limpiar polling al desmontar
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const iniciarPolling = (batchId) => {
    if (pollingRef.current) clearInterval(pollingRef.current);

    pollingRef.current = setInterval(async () => {
      try {
        const estadoActual = await obtenerEstadoLote(batchId);
        setProgreso(estadoActual.progreso_porcentaje);

        if (estadoActual.estado === "COMPLETED") {
          clearInterval(pollingRef.current);
          setLote(estadoActual);
          setCargando(false);
          setProgreso(null);

          // Si tiene sucursales, cargar alertas
          if (estadoActual.resumen?.tiene_sucursales) {
            cargarAlertas(batchId);
          }
        } else if (estadoActual.estado === "FAILED") {
          clearInterval(pollingRef.current);
          setError(estadoActual.error || "Ocurrió un error al procesar el lote.");
          setCargando(false);
          setProgreso(null);
        }
      } catch (err) {
        clearInterval(pollingRef.current);
        setError("Error al consultar el avance del lote: " + err.message);
        setCargando(false);
        setProgreso(null);
      }
    }, 1500);
  };

  const cargarAlertas = async (batchId) => {
    try {
      const dataAlertas = await obtenerAlertasLote(batchId);
      setAlertas(dataAlertas);
    } catch (err) {
      // Las alertas son complementarias, no bloquean la visualización del lote
      console.warn("No se pudieron cargar alertas de sede:", err.message);
    }
  };

  const handleProcesar = async (archivo) => {
    setCargando(true);
    setProgreso(0);
    setError(null);
    setLote(null);
    setAlertas(null);

    try {
      const data = await subirArchivoLote(archivo, (porcentaje) => {
        setProgreso(porcentaje);
      });

      if (data.estado === "COMPLETED") {
        setLote(data);
        setCargando(false);
        setProgreso(null);

        if (data.resumen?.tiene_sucursales) {
          cargarAlertas(data.batch_id);
        }
      } else if (data.estado === "PROCESSING") {
        // Iniciar sondeo en segundo plano para lotes de > 500 filas
        iniciarPolling(data.batch_id);
      } else if (data.estado === "FAILED") {
        setError(data.error || "El procesamiento del lote ha fallado.");
        setCargando(false);
        setProgreso(null);
      }
    } catch (err) {
      setError(err.message);
      setCargando(false);
      setProgreso(null);
    }
  };

  const handleReiniciar = () => {
    setLote(null);
    setAlertas(null);
    setError(null);
    setProgreso(null);
    setCargando(false);
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      {/* Selector y Carga de Archivo */}
      <CargaArchivo
        onProcesar={handleProcesar}
        cargando={cargando}
        progreso={progreso}
      />

      {/* Banner de Error General */}
      {error && (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 flex items-start gap-3">
          <svg className="h-5 w-5 shrink-0 text-rose-600 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <p className="font-semibold">No se pudo procesar el archivo</p>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Resultados una vez completado */}
      {lote && (
        <div className="space-y-6">
          {/* Botón superior para procesar otro archivo */}
          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleReiniciar}
              className="inline-flex items-center gap-1.5 rounded-xl border border-stone-300 bg-white px-3.5 py-1.5 text-xs font-medium text-stone-700 hover:bg-stone-50"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span>Subir otro archivo</span>
            </button>
          </div>

          {/* 1. Resumen Estadístico con Descarga Excel */}
          <ResumenLote lote={lote} />

          {/* 2. Alertas por Sucursal (si existen sedes en el archivo) */}
          {alertas && <AlertasSucursal alertasData={alertas} />}

          {/* 3. Tabla Interactiva de Opiniones con Certeza y Paginación */}
          <TablaResultados resultados={lote.resultados || []} />
        </div>
      )}
    </div>
  );
}

