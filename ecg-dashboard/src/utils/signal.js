export function detectRPeaks(ecg, fs = 125) {
  const N = ecg.length;
  if (N < 30) return [];
  
  let s = 0, s2 = 0;
  for (let i = 0; i < N; i++) {
    s += ecg[i];
    s2 += ecg[i] * ecg[i];
  }
  
  const mean = s / N;
  const std = Math.sqrt(Math.max(0, s2 / N - mean * mean));
  
  // Adaptive threshold
  const thr = mean + 0.55 * std;
  const minD = Math.round(0.27 * fs);
  
  const peaks = [];
  for (let i = 2; i < N - 2; i++) {
    if (ecg[i] >= thr && ecg[i] >= ecg[i - 1] && ecg[i] >= ecg[i + 1] && ecg[i] >= ecg[i - 2] && ecg[i] >= ecg[i + 2]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minD) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

export function computeEDR(ecg, peaks, fs = 125) {
  if (peaks.length < 2) return { t: [], v: [] };
  
  const totalS = ecg.length / fs;
  const pt = peaks.map(p => p / fs);
  const pv = peaks.map(p => {
    const lo = Math.max(0, p - 15);
    const hi = Math.min(ecg.length - 1, p + 15);
    let mx = -Infinity, mn = Infinity;
    for (let i = lo; i <= hi; i++) {
      if (ecg[i] > mx) mx = ecg[i];
      if (ecg[i] < mn) mn = ecg[i];
    }
    return mx - mn;
  });
  
  const respFs = 4;
  const n = Math.round(totalS * respFs);
  const t = [], v = [];
  
  for (let i = 0; i < n; i++) {
    const ti = i / respFs;
    t.push(ti);
    
    if (ti <= pt[0]) {
      v.push(pv[0]);
      continue;
    }
    if (ti >= pt[pt.length - 1]) {
      v.push(pv[pv.length - 1]);
      continue;
    }
    
    let j = 0;
    while (j < pt.length - 1 && pt[j + 1] < ti) j++;
    
    // Linear interpolation
    v.push(pv[j] * (1 - (ti - pt[j]) / (pt[j + 1] - pt[j])) + pv[j + 1] * ((ti - pt[j]) / (pt[j + 1] - pt[j])));
  }
  
  return { t, v };
}
