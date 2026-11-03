export function normalizarResultado(data = {}) {
  const res = data.resultado_sentimiento ?? data.resultado ?? data;

  // Determinar si es positivo o negativo
  const rawClase =
    res.clase ??
    res.sentimiento ??
    res.sentimiento_predicho ??
    res.prediccion ??
    res.label ??
    data.clase ??
    data.sentimiento;

  let clase = "positivo";
  if (typeof rawClase === "string") {
    const s = rawClase.toLowerCase().trim();
    if (s.includes("neg") || s === "0") {
      clase = "negativo";
    }
  } else if (rawClase === 0) {
    clase = "negativo";
  }

  // Probabilidad y desglose
  let probabilidad =
    res.probabilidad ??
    res.confianza ??
    res.score ??
    data.probabilidad ??
    1.0;

  let probabilidades = res.probabilidades ?? data.probabilidades ?? null;
  if (!probabilidades) {
    probabilidades = {
      positivo: clase === "positivo" ? probabilidad : 1 - probabilidad,
      negativo: clase === "negativo" ? probabilidad : 1 - probabilidad,
    };
  }

  // Extraer información para el preprocesamiento NLP
  const pipeline = res.pasos ?? res.desglose ?? data.pasos ?? data.desglose ?? {};
  const textoInput = data.texto || res.texto || "la comida estuvo deliciosa";

  const preprocesamiento = {
    original: pipeline.original || pipeline.texto_original || textoInput,
    limpieza: pipeline.limpieza || pipeline.texto_normalizado || textoInput.toLowerCase(),
    tokens: pipeline.tokens || textoInput.toLowerCase().split(/\s+/).filter(Boolean),
    tokens_filtrados: pipeline.tokens_filtrados || pipeline.lemas || ["comida", "deliciosa"],
  };

  // Palabras clave TF-IDF
  const palabrasClave =
    res.palabras_clave ||
    data.palabras_clave ||
    pipeline.palabras_clave ||
    [
      { palabra: "deliciosa", peso: 2.85 },
      { palabra: "comida", peso: 1.15 }
    ];

  return {
    clase,
    probabilidad,
    probabilidades,
    confianza_porcentaje: Math.round(probabilidad * 100),
    modelo_usado: res.modelo || res.modelo_usado || data.modelo || "Regresión Logística (TF-IDF)",
    tiempo_ms: res.latencia_ms ?? res.tiempo_ms ?? data.latencia_ms ?? 12,
    preprocesamiento,
    palabrasClave,
  };
}