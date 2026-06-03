const BASE = '/api';

async function request(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  // Shared
  getSummary:          ()        => request('/summary'),
  getPipelineLog:      ()        => request('/pipeline_log'),
  getPipelineLogLatest:()        => request('/pipeline_log/latest'),

  // Arrhythmia
  getArrSummary:       ()        => request('/arrhythmia/summary'),
  getArrBeatDist:      ()        => request('/arrhythmia/beat_distribution'),
  getArrBeatBySource:  ()        => request('/arrhythmia/beat_distribution/source'),
  getArrPredictions:   (n = 200) => request(`/arrhythmia/predictions?limit=${n}`),
  getArrModelResults:  ()        => request('/arrhythmia/model_results'),
  getArrEcgPlot:       (bt)      => request(`/arrhythmia/ecg_plot${bt ? `?beat_type=${bt}` : ''}`),
  getArrConfusion:     ()        => request('/arrhythmia/confusion_matrix'),

  // Apnea
  getApneaSummary:     ()        => request('/apnea/summary'),
  getApneaSegments:    (n = 300) => request(`/apnea/segments?limit=${n}`),
  getApneaModelResults:()        => request('/apnea/model_results'),
  getApneaEcgPlot:     ()        => request('/apnea/ecg_plot'),
  getApneaSpo2Plot:    (n = 100) => request(`/apnea/spo2_plot?limit=${n}`),
  getApneaRespPlot:    (n = 100) => request(`/apnea/resp_plot?limit=${n}`),
  getApneaLabelDist:   ()        => request('/apnea/label_distribution'),
  getApneaSignalFlags: (n = 200) => request(`/apnea/signal_flags?limit=${n}`),
  getApneaFeatImp:     ()        => request('/apnea/feature_importance'),

  // Sepsis
  getSepsisSummary:    ()        => request('/sepsis/summary'),
  getSepsisPredictions:(n = 200) => request(`/sepsis/predictions?limit=${n}`),
  getSepsisModelResults:()       => request('/sepsis/model_results'),
  getSepsisRiskDist:   ()        => request('/sepsis/risk_distribution'),
  getSepsisVitalsPlot: ()        => request('/sepsis/vitals_plot'),
  getSepsisFeatImp:    ()        => request('/sepsis/feature_importance'),
};
