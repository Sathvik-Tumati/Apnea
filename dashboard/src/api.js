const BASE = '/api';

async function request(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  // Shared
  getSummary: () => request('/summary'),
  getPipelineLog: () => request('/pipeline_log'),
  getPipelineLogLatest: () => request('/pipeline_log/latest'),

  // Apnea
  getApneaSummary: () => request('/apnea/summary'),
  getApneaRecords: () => request('/apnea/records'),
  getApneaSegments: (n = 300, rec) => request(`/apnea/segments?limit=${n}${rec ? `&record=${encodeURIComponent(rec)}` : ''}`),
  getApneaModelResults: () => request('/apnea/model_results'),
  getApneaEcgPlot: (rec) => request(`/apnea/ecg_plot${rec ? `?record=${encodeURIComponent(rec)}` : ''}`),
  getApneaSpo2Plot: (n = 100, rec) => request(`/apnea/spo2_plot?limit=${n}${rec ? `&record=${encodeURIComponent(rec)}` : ''}`),
  getApneaRespPlot: (n = 100, rec) => request(`/apnea/resp_plot?limit=${n}${rec ? `&record=${encodeURIComponent(rec)}` : ''}`),
  getApneaLabelDist: (rec) => request(`/apnea/label_distribution${rec ? `?record=${encodeURIComponent(rec)}` : ''}`),
  getApneaSignalFlags: (n = 200, rec) => request(`/apnea/signal_flags?limit=${n}${rec ? `&record=${encodeURIComponent(rec)}` : ''}`),
  getApneaFeatImp: () => request('/apnea/feature_importance'),
};
