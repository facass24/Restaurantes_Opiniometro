import { useState, useRef } from "react";

export default function CargaArchivo({ onProcesar, cargando, progreso }) {
  const [archivo, setArchivo] = useState(null);
  const [arrastrando, setArrastrando] = useState(false);
  const [errorLocal, setErrorLocal] = useState(null);
  const inputRef = useRef(null);

  const formatosPermitidos = [".xlsx", ".xls", ".csv"];

  const validarArchivo = (file) => {
    if (!file) return false;
    const nombre = file.name.toLowerCase();
    const esValido = formatosPermitidos.some((ext) => nombre.endsWith(ext));
    if (!esValido) {
      setErrorLocal("Formato no compatible. Por favor sube un archivo Excel (.xlsx, .xls) o CSV (.csv).");
      return false;
    }
    if (file.size === 0) {
      setErrorLocal("El archivo seleccionado está vacío.");
      return false;
    }
    setErrorLocal(null);
    return true;
  };

  const handleSeleccionar = (e) => {
    const file = e.target.files?.[0];
    if (file && validarArchivo(file)) {
      setArchivo(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setArrastrando(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setArrastrando(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setArrastrando(false);
    const file = e.dataTransfer.files?.[0];
    if (file && validarArchivo(file)) {
      setArchivo(file);
    }
  };

  const handleLimpiar = () => {
    setArchivo(null);
    setErrorLocal(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!archivo || cargando) return;
    onProcesar(archivo);
  };

  const formatearTamano = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-stone-900">
            Importar Reseñas desde Excel o CSV
          </h2>
          <p className="text-sm text-stone-500">
            Sube un archivo con opiniones de clientes para clasificar el sentimiento en lote y detectar alertas operativas.
          </p>
        </div>
        <span className="inline-flex self-start rounded-full bg-stone-100 px-3 py-1 text-xs font-medium text-stone-600 sm:self-auto">
          .xlsx, .xls, .csv
        </span>
      </div>

      <form onSubmit={handleSubmit} className="mt-5 space-y-4">
        <input
          ref={inputRef}
          type="file"
          accept=".xlsx,.xls,.csv"
          onChange={handleSeleccionar}
          className="hidden"
          disabled={cargando}
        />

        {/* Zona Drag & Drop */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !cargando && inputRef.current?.click()}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition ${
            arrastrando
              ? "border-orange-500 bg-orange-50/50"
              : archivo
              ? "border-stone-300 bg-stone-50/50 hover:bg-stone-50"
              : "border-stone-300 bg-white hover:border-stone-400 hover:bg-stone-50/40"
          } ${cargando ? "cursor-not-allowed opacity-60" : ""}`}
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-orange-100 text-orange-700">
            <svg
              className="h-6 w-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.8}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
          </div>

          <p className="mt-3 text-sm font-medium text-stone-800">
            {arrastrando
              ? "Suelta el archivo aquí..."
              : "Haz clic para seleccionar o arrastra tu archivo aquí"}
          </p>
          <p className="mt-1 text-xs text-stone-500">
            Detecta automáticamente columnas de <strong>reseña</strong>, <strong>sucursal</strong> y <strong>fecha</strong>.
          </p>
        </div>

        {/* Ficha de archivo seleccionado */}
        {archivo && (
          <div className="flex items-center justify-between rounded-xl border border-stone-200 bg-stone-50 p-3.5">
            <div className="flex items-center gap-3 overflow-hidden">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-100 font-bold text-xs text-emerald-800 uppercase">
                {archivo.name.split(".").pop()}
              </span>
              <div className="truncate">
                <p className="truncate text-sm font-medium text-stone-900">
                  {archivo.name}
                </p>
                <p className="text-xs text-stone-500">
                  {formatearTamano(archivo.size)}
                </p>
              </div>
            </div>

            {!cargando && (
              <button
                type="button"
                onClick={handleLimpiar}
                className="shrink-0 rounded-lg p-1.5 text-stone-400 hover:bg-stone-200 hover:text-stone-700"
                title="Quitar archivo"
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        )}

        {/* Barra de progreso / Estado de carga */}
        {cargando && (
          <div className="space-y-2 rounded-xl bg-orange-50/70 p-4 border border-orange-200">
            <div className="flex items-center justify-between text-xs font-medium text-orange-900">
              <span className="flex items-center gap-2">
                <svg className="h-4 w-4 animate-spin text-orange-700" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Procesando inferencia vectorizada de reseñas...
              </span>
              <span>{progreso !== null ? `${progreso}%` : "Analizando..."}</span>
            </div>
            {progreso !== null && (
              <div className="h-2 w-full overflow-hidden rounded-full bg-orange-200">
                <div
                  className="h-full bg-orange-700 transition-all duration-300"
                  style={{ width: `${progreso}%` }}
                />
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {errorLocal && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
            <span className="font-semibold">Atención:</span> {errorLocal}
          </div>
        )}

        {/* Acciones */}
        <div className="flex flex-wrap items-center gap-3 pt-1">
          <button
            type="submit"
            disabled={!archivo || cargando}
            className="flex items-center gap-2 rounded-xl bg-stone-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {cargando ? (
              <>
                <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span>Procesando...</span>
              </>
            ) : (
              <>
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Analizar Archivo</span>
              </>
            )}
          </button>

          {archivo && !cargando && (
            <button
              type="button"
              onClick={handleLimpiar}
              className="rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm font-medium text-stone-700 hover:bg-stone-50"
            >
              Cambiar archivo
            </button>
          )}
        </div>
      </form>
    </section>
  );
}

