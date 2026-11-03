import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 30000,
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

  return "Ocurrió un error al procesar la reseña.";
}

export async function predecirSentimiento(opinion) {
  try {
    const { data } = await api.post("/api/v1/predict", { 
      texto: opinion,
      modelo: "auto"
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

export default api;
