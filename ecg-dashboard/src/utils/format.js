const IST = { timeZone: 'Asia/Kolkata', hour12: false };

export const toISTShort = (r) => {
  if (!r && r !== 0) return '—';
  try {
    const d = new Date(r);
    if (isNaN(d)) return String(r);
    return d.toLocaleString('en-IN', { ...IST, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' IST';
  } catch {
    return String(r);
  }
};

export const toISTFull = (r) => {
  if (!r && r !== 0) return '—';
  try {
    const d = new Date(r);
    if (isNaN(d)) return String(r);
    return d.toLocaleString('en-IN', {
      ...IST,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }) + ' IST';
  } catch {
    return String(r);
  }
};

const SEV = { 'Normal': '#22c55e', 'Mild OSA': '#eab308', 'Moderate OSA': '#f97316', 'Severe OSA': '#ef4444' };

export const sevC = (s) => SEV[s] || '#64748b';
export const sevBg = (s) => sevC(s) + '18';
export const ahiSev = (a) => a < 5 ? 'Normal' : a < 15 ? 'Mild OSA' : a < 30 ? 'Moderate OSA' : 'Severe OSA';
