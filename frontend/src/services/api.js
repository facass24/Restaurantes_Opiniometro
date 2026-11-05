import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 60000,
  headers: {
    "Content-Type": "application/json",
  },
});

function extraerMensajeError(error) {
  const detalle = error.response?.data?.detail;

  if (typeof detalle === "string") {
    return detalle;
  }

  if (Array.isArray(detalle)) {
    return detalle
      .map((item) => item.msg || item.message || JSON.stringify(item))
      .join(" ");
  }

  if (error.code === "ECONNABORTED") {
    return "La solicitud tardó demasiado. Intenta de nuevo.";
  }

  if (!error.response) {
    return "No se pudo conectar con la API en http://127.0.0.1:8000. Verifica que FastAPI esté en ejecución.";
  }

  return "Ocurrió un error al procesar la solicitud.";
}

export async function predecirSentimiento(opinion) {
  try {
    const { data } = await api.post("/api/v1/predict", {
      texto: opinion,
      modelo: "auto",
    });
    return data;
  } catch (error) {
    throw new Error(extraerMensajeError(error));
  }
}

export async function obtenerMetricas() {
  try {
    const { data } = await api.get("/api/v1/metrics");
    return data;
  } catch (error) {
    throw new Error(extraerMensajeError(error));
  }
}

/**
 * Sube un archivo .xlsx, .xls o .csv para análisis de sentimiento en lote.
 * Retorna el estado y resultados del lote (LoteEstadoResponse).
 */
export async function subirArchivoLote(archivo, onProgreso = null) {
  try {
    const formData = new FormData();
    formData.append("file", archivo);

    const { data } = await api.post("/api/v1/predict/file", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
      onUploadProgress: (progreso) => {
        if (onProgreso && progreso.total) {
          const porcentaje = Math.round((progreso.loaded * 100) / progreso.total);
          onProgreso(porcentaje);
        }
      },
    });
    return data;
  } catch (error) {
    throw new Error(extraerMensajeError(error));
  }
}

/**
 * Consulta el estado de procesamiento de un lote mediante su batch_id.
 */
export async function obtenerEstadoLote(batchId) {
  try {
    const { data } = await api.get(`/api/v1/batches/${batchId}`);
    return data;
  } catch (error) {
    throw new Error(extraerMensajeError(error));
  }
}

/**
 * Obtiene el diagnóstico de alertas por sucursal de un lote procesado.
 */
export async function obtenerAlertasLote(batchId) {
  try {
    const { data } = await api.get(`/api/v1/batches/${batchId}/alerts`);
    return data;
  } catch (error) {
    throw new Error(extraerMensajeError(error));
  }
}

/**
 * Descarga el archivo Excel con las predicciones incorporadas.
 */
export async function descargarExcelLote(batchId, nombreSugerido = "predicciones_lote.xlsx") {
  try {
    const response = await api.get(`/api/v1/batches/${batchId}/download`, {
      responseType: "blob",
    });

    const blob = new Blob([response.data], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", nombreSugerido);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    throw new Error(extraerMensajeError(error));
  }
}

export default api;
