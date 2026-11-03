const esNumero = (v) => typeof v === "number" && Number.isFinite(v);
const fraccion = (v) => esNumero(v) && v >= 0 && v <= 1;
const lista = (v) => Array.isArray(v) ? v.filter((item) => typeof item === "string") : [];

export function normalizarResultado(data, texto, tiempoMedido) {
  const etiqueta = String(data?.clase ?? data?.sentimiento ?? data?.sentiment ?? "").toLowerCase();
  const clase = ({ positivo: "positivo", positive: "positivo", negativo: "negativo", negative: "negativo" })[etiqueta];
  if (!clase) throw new Error("El servidor devolvió un resultado sin un sentimiento válido.");

  const probabilidades = data.probabilidades ?? data.probabilities ?? {};
  let positivo = probabilidades.positivo ?? probabilidades.positive;
  let negativo = probabilidades.negativo ?? probabilidades.negative;
  const confianza = data.confianza_porcentaje ?? data.confianza ?? data.confidence;
  const confianzaPct = esNumero(confianza) && confianza >= 0 && confianza <= 100
    ? (data.confianza_porcentaje != null || confianza > 1 ? confianza : confianza * 100) : null;
  const probabilidad = data.probabilidad ?? (clase === "positivo" ? positivo : negativo)
    ?? (confianzaPct == null ? null : confianzaPct / 100);
  if (!fraccion(probabilidad)) throw new Error("El servidor devolvió una probabilidad inválida.");
  if (!fraccion(positivo)) positivo = clase === "positivo" ? probabilidad : 1 - probabilidad;
  if (!fraccion(negativo)) negativo = clase === "negativo" ? probabilidad : 1 - probabilidad;

  const pasos = data.preprocesamiento ?? data.preprocessing ?? {};
  const pesos = data.palabras_clave ?? pasos.pesos ?? data.word_weights ?? [];
  return {
    clase, probabilidad, confianza_porcentaje: confianzaPct ?? probabilidad * 100,
    probabilidades: { positivo, negativo },
    modelo_usado: data.modelo_usado ?? data.model ?? "Modelo del servidor",
    tiempo_ms: esNumero(data.tiempo_ms) ? data.tiempo_ms
      : esNumero(data.response_time_ms) ? data.response_time_ms : tiempoMedido,
    preprocesamiento: {
      original: pasos.original ?? texto,
      normalizado: pasos.normalizado ?? pasos.texto_normalizado ?? pasos.normalized_text ?? "",
      tokens: lista(pasos.tokens),
      reducidos: lista(pasos.reducidos ?? pasos.lemas ?? pasos.lemmas),
      sin_stopwords: lista(pasos.sin_stopwords),
    },
    palabras_clave: Array.isArray(pesos) ? pesos.filter((p) => p && typeof p.palabra === "string" && esNumero(p.peso)) : [],
  };
}

export function crearEjemplo() {
  return {
    simulado: true,
    clase: "positivo", probabilidad: 0.947, confianza_porcentaje: 94.7,
    probabilidades: { positivo: 0.947, negativo: 0.053 },
    modelo_usado: "Regresión Logística (simulación)", tiempo_ms: 18.6,
    preprocesamiento: {
      original: "La comida estaba deliciosa y el personal fue amable, aunque el servicio tardó.",
      normalizado: "la comida estaba deliciosa y el personal fue amable aunque el servicio tardo",
      tokens: ["la", "comida", "estaba", "deliciosa", "y", "el", "personal", "fue", "amable", "aunque", "el", "servicio", "tardo"],
      sin_stopwords: ["comida", "deliciosa", "personal", "amable", "servicio", "tardo"],
      reducidos: ["comida", "delicioso", "personal", "amable", "servicio", "tardar"],
    },
    palabras_clave: [
      { palabra: "delicioso", peso: 1.82, polaridad: "positivo" },
      { palabra: "amable", peso: 1.34, polaridad: "positivo" },
      { palabra: "tardar", peso: -0.63, polaridad: "negativo" },
    ],
  };
}
