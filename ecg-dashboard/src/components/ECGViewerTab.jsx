import React, { useMemo, useCallback, useRef, useEffect } from 'react';
import { Empty } from './Shared';
import { detectRPeaks, computeEDR } from '../utils/signal';
import { toISTFull } from '../utils/format';

function ECGCanvas({ ecg, peaks }) {
  const ref = useRef(null);
  const [size, setSize] = React.useState({ w: 0, h: 0 });

  useEffect(() => {
    const ob = new ResizeObserver(entries => {
      for (let e of entries) setSize({ w: e.contentRect.width, h: e.contentRect.height });
    });
    if (ref.current) ob.observe(ref.current.parentElement);
    return () => ob.disconnect();
  }, []);
  
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !ecg?.length || size.w === 0 || size.h === 0) return;
    
    const dpr = window.devicePixelRatio || 1;
    const W = size.w;
    const H = size.h;
    
    cv.width = W * dpr;
    cv.height = H * dpr;
    
    const ctx = cv.getContext('2d');
    ctx.scale(dpr, dpr);
    
    const P = { t: 18, r: 16, b: 28, l: 50 };
    const pW = W - P.l - P.r;
    const pH = H - P.t - P.b;
    const N = ecg.length;
    
    ctx.fillStyle = '#080f1e';
    ctx.fillRect(0, 0, W, H);
    
    // minor grid
    ctx.strokeStyle = '#0c1828';
    ctx.lineWidth = 0.4;
    for (let i = 0; i <= 30; i++) {
      const x = P.l + (i / 30) * pW;
      ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.t + pH); ctx.stroke();
    }
    for (let i = 0; i <= 8; i++) {
      const y = P.t + (i / 8) * pH;
      ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(P.l + pW, y); ctx.stroke();
    }
    
    // 5s marks
    ctx.strokeStyle = '#131f38';
    ctx.lineWidth = 0.8;
    for (let i = 0; i <= 6; i++) {
      const x = P.l + (i / 6) * pW;
      ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.t + pH); ctx.stroke();
    }
    
    ctx.fillStyle = '#1e3a5f';
    ctx.font = '9px Inter';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 6; i++) {
      ctx.fillText((i * 5) + 's', P.l + (i / 6) * pW, H - 5);
    }
    ctx.fillText('Time (seconds)', W / 2, H - 0.5);
    
    // amplitude range
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) {
      if (ecg[i] < mn) mn = ecg[i];
      if (ecg[i] > mx) mx = ecg[i];
    }
    const pad = (mx - mn) * 0.1 || 0.5;
    const lo = mn - pad;
    const hi = mx + pad;
    const rng = hi - lo;
    
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const v = lo + (i / 4) * rng;
      const y = P.t + pH - (i / 4) * pH;
      ctx.fillText(v.toFixed(2), P.l - 5, y + 3);
    }
    
    // ECG trace
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 1.3;
    ctx.shadowColor = '#00e67640';
    ctx.shadowBlur = 4;
    
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = P.l + (i / (N - 1)) * pW;
      const y = P.t + pH - ((ecg[i] - lo) / rng) * pH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // R-peaks
    if (peaks?.length) {
      ctx.fillStyle = '#ff4444';
      peaks.forEach(p => {
        const x = P.l + (p / (N - 1)) * pW;
        const y = P.t + pH - ((ecg[p] - lo) / rng) * pH;
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [ecg, peaks, size.w, size.h]);
  
  return <canvas ref={ref} style={{ width: '100%', height: '100%' }} />;
}

function EDRCanvas({ edr }) {
  const ref = useRef(null);
  const [size, setSize] = React.useState({ w: 0, h: 0 });

  useEffect(() => {
    const ob = new ResizeObserver(entries => {
      for (let e of entries) setSize({ w: e.contentRect.width, h: e.contentRect.height });
    });
    if (ref.current) ob.observe(ref.current.parentElement);
    return () => ob.disconnect();
  }, []);
  
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !edr?.v?.length || size.w === 0 || size.h === 0) return;
    
    const dpr = window.devicePixelRatio || 1;
    const W = size.w;
    const H = size.h;
    
    cv.width = W * dpr;
    cv.height = H * dpr;
    
    const ctx = cv.getContext('2d');
    ctx.scale(dpr, dpr);
    
    ctx.fillStyle = '#080f1e';
    ctx.fillRect(0, 0, W, H);
    
    const P = { t: 8, r: 16, b: 22, l: 50 };
    const pW = W - P.l - P.r;
    const pH = H - P.t - P.b;
    
    const { t, v } = edr;
    const N = v.length;
    
    let mn = Infinity, mx = -Infinity;
    v.forEach(x => {
      if (x < mn) mn = x;
      if (x > mx) mx = x;
    });
    const rng = mx - mn || 1;
    
    ctx.strokeStyle = '#7c3aed';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#7c3aed40';
    ctx.shadowBlur = 4;
    
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = P.l + (t[i] / 30) * pW;
      const y = P.t + pH - ((v[i] - mn) / rng) * pH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    ctx.fillStyle = '#1e3a5f';
    ctx.font = '9px Inter';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 30; i += 5) {
      const x = P.l + (i / 30) * pW;
      ctx.fillText(`${i}s`, x, P.t + pH + 6);
    }
  }, [edr, size]);
  
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block', borderRadius: 8 }} />
    </div>
  );
}

export function SpO2Canvas({ spo2 }) {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  
  useEffect(() => {
    const ob = new ResizeObserver(e => {
      if (e[0]?.contentRect) {
        setSize({ w: e[0].contentRect.width, h: e[0].contentRect.height });
      }
    });
    if (ref.current) ob.observe(ref.current.parentElement);
    return () => ob.disconnect();
  }, []);
  
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !spo2 || size.w === 0 || size.h === 0) return;
    
    const dpr = window.devicePixelRatio || 1;
    const W = size.w;
    const H = size.h;
    
    cv.width = W * dpr;
    cv.height = H * dpr;
    
    const ctx = cv.getContext('2d');
    ctx.scale(dpr, dpr);
    
    ctx.fillStyle = '#080f1e';
    ctx.fillRect(0, 0, W, H);
    
    const P = { t: 8, r: 16, b: 22, l: 50 };
    const pW = W - P.l - P.r;
    const pH = H - P.t - P.b;
    
    const N = spo2.length;
    let valid = false;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < N; i++) {
      if (spo2[i] > 0) {
        valid = true;
        if (spo2[i] < mn) mn = spo2[i];
        if (spo2[i] > mx) mx = spo2[i];
      }
    }
    
    if (!valid) {
      ctx.fillStyle = '#1e3a5f';
      ctx.font = '11px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('No SpO2 Data for this segment', W / 2, H / 2);
      return;
    }
    
    mn = Math.max(0, mn - 2);
    mx = Math.min(100, mx + 2);
    const rng = mx - mn || 1;
    
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#38bdf840';
    ctx.shadowBlur = 4;
    
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const v = spo2[i];
      const x = P.l + (i / Math.max(1, N - 1)) * pW;
      const y = v > 0 ? P.t + pH - ((v - mn) / rng) * pH : P.t + pH;
      if (v > 0) {
        if (i === 0 || spo2[i-1] === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    ctx.fillStyle = '#1e3a5f';
    ctx.font = '9px Inter';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${mx.toFixed(1)}%`, P.l - 6, P.t);
    ctx.fillText(`${mn.toFixed(1)}%`, P.l - 6, P.t + pH);
    
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 30; i += 5) {
      const x = P.l + (i / 30) * pW;
      ctx.fillText(`${i}s`, x, P.t + pH + 6);
    }
  }, [spo2, size]);
  
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block', borderRadius: 8 }} />
    </div>
  );
}

function ProbGauge({ prob, threshold = 0.55 }) {
  const p = typeof prob === 'number' && !isNaN(prob) ? Math.min(1, Math.max(0, prob)) : 0;
  const col = p >= threshold ? '#ef4444' : '#22c55e';
  const range = 240, start = 210;
  const arc = pct => {
    const ang = (start - pct * range) * (Math.PI / 180);
    return { x: 65 + 52 * Math.cos(ang), y: 70 - 52 * Math.sin(ang) };
  };
  
  const a0 = arc(0);
  const ap = arc(p);
  const at = arc(threshold);
  const lg = p * range > 180 ? 1 : 0;
  
  return (
    <div style={{ textAlign: 'center' }}>
      <svg width={130} height={92} viewBox="0 0 130 92">
        <path d={`M ${a0.x} ${a0.y} A 52 52 0 1 1 ${arc(1).x} ${arc(1).y}`} fill="none" stroke="#1a2d4d" strokeWidth={9} strokeLinecap="round" />
        {p > 0.002 && (
          <path d={`M ${a0.x} ${a0.y} A 52 52 0 ${lg} 1 ${ap.x} ${ap.y}`} fill="none" stroke={col} strokeWidth={9} strokeLinecap="round" />
        )}
        <circle cx={at.x} cy={at.y} r={5} fill="#eab308" />
        <text x={65} y={62} textAnchor="middle" fill={col} fontSize={20} fontWeight={800} fontFamily="JetBrains Mono">{p.toFixed(3)}</text>
        <text x={65} y={75} textAnchor="middle" fill="#334155" fontSize={8}>APNEA PROB</text>
        <text x={65} y={85} textAnchor="middle" fill="#1e3a5f" fontSize={7}>● {threshold} threshold</text>
      </svg>
    </div>
  );
}

export function ECGViewerTab({ merged, ecgMap, spo2Map, selSeg, onSel }) {
  const [filter, setFilter] = React.useState('all');
  
  const filtered = useMemo(() => {
    if (!merged) return [];
    return merged.filter(r => {
      const isA = r.apnea_pred === 1;
      const isF = r.hr_gate_pass === 0 || (r.quality_flag && r.quality_flag !== 'OK');
      if (filter === 'apnea') return isA;
      if (filter === 'flagged') return isF;
      if (filter === 'normal') return !isA && !isF;
      return true;
    });
  }, [merged, filter]);

  const cur = filtered?.find(r => r.segment_idx === selSeg) ?? filtered?.[0] ?? merged?.[0];
  const idx = filtered?.findIndex(r => r.segment_idx === cur?.segment_idx);
  const curIdx = idx >= 0 ? idx : 0;
  const total = filtered?.length ?? 0;
  
  const ecgRaw = useMemo(() => ecgMap && cur ? ecgMap[String(cur.segment_idx)] : null, [ecgMap, cur?.segment_idx]);
  const ecg = useMemo(() => ecgRaw ? Array.from(ecgRaw) : null, [ecgRaw]);
  const spo2Raw = useMemo(() => spo2Map && cur ? spo2Map[String(cur.segment_idx)] : null, [spo2Map, cur?.segment_idx]);
  const spo2 = useMemo(() => spo2Raw ? Array.from(spo2Raw) : null, [spo2Raw]);
  const peaks = useMemo(() => ecg ? detectRPeaks(ecg) : [], [ecg]);
  const edr = useMemo(() => (ecg && peaks.length >= 2) ? computeEDR(ecg, peaks) : { t: [], v: [] }, [ecg, peaks]);
  
  const nav = useCallback(d => {
    if (!filtered?.length) return;
    const ni = Math.max(0, Math.min(total - 1, curIdx + d));
    onSel(filtered[ni].segment_idx);
  }, [curIdx, total, filtered, onSel]);
  
  if (!merged?.length) return <Empty msg="Upload infer_results_ADM.csv to use the ECG viewer" />;
  
  const isA = cur?.apnea_pred === 1;
  const isF = cur?.hr_gate_pass === 0 || (cur?.quality_flag && cur.quality_flag !== 'OK');
  const bc = isA ? '#ef4444' : isF ? '#eab308' : '#22c55e';
  
  const FEATS = [
    ['ECG HR', cur?.ecg_hr_bpm, 'bpm'], ['Ref HR', cur?.ref_hr_bpm, 'bpm'], ['HR Diff', cur?.hr_diff_bpm, 'bpm'],
    ['RR Mean', cur?.rr_mean, 'ms'], ['RR Std', cur?.rr_std, 'ms'], ['RMSSD', cur?.rmssd, 'ms'],
    ['pNN50', cur?.pnn50, '%'], ['LF/HF', cur?.lf_hf_ratio, ''], ['Resp Rate', cur?.resp_rate_bpm, 'bpm'],
    ['Resp Var', cur?.resp_rate_variability, ''], ['Flatline', cur?.flatline_duration_s, 's'],
    ['Resp Amp μ', cur?.resp_amplitude_mean, ''], ['Resp Amp σ', cur?.resp_amplitude_std, ''],
    ['SpO2 Mean', cur?.has_spo2 ? cur?.spo2_mean : null, '%'], ['SpO2 Min', cur?.has_spo2 ? cur?.spo2_min : null, '%'],
    ['Has SpO2', cur?.has_spo2 ? 'Yes' : 'No', ''], ['Signal Quality', cur?.signal_quality, ''],
    ['Quality Flag', cur?.quality_flag, ''], ['Rhythm', cur?.dominant_rhythm, ''],
    ['Ectopy', cur?.ectopy_present ? 'Yes' : 'No', ''], ['HR Gate', cur?.hr_gate_pass === 1 ? 'Pass' : 'Fail', ''],
    ['fs', cur?.fs_source_hz, 'Hz'],
  ];
  
  return (
    <div className="fi" style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto', padding: '10px 14px' }}>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexShrink: 0 }}>
          <select value={filter} onChange={e => setFilter(e.target.value)} style={{ background: '#0c1526', border: '1px solid #1a2d4d', color: '#e2e8f0', padding: '4px 8px', borderRadius: 6, fontSize: 11, outline: 'none', fontWeight: 600 }}>
            <option value="all">All Segments</option>
            <option value="apnea">Apnea Only</option>
            <option value="flagged">Flagged Only</option>
            <option value="normal">Normal Only</option>
          </select>
          <button onClick={() => nav(-1)} disabled={curIdx <= 0} style={{ background: '#0c1526', border: '1px solid #1a2d4d', color: '#475569', borderRadius: 6, padding: '5px 12px', fontSize: 14 }}>◀</button>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center' }}>
            <input type="range" min={0} max={Math.max(0, total - 1)} value={curIdx} onChange={e => {
              if (filtered[+e.target.value]) onSel(filtered[+e.target.value].segment_idx);
            }} style={{ width: '100%' }} />
          </div>
          <button onClick={() => nav(1)} disabled={curIdx >= total - 1} style={{ background: '#0c1526', border: '1px solid #1a2d4d', color: '#475569', borderRadius: 6, padding: '5px 12px', fontSize: 14 }}>▶</button>
          <span style={{ color: '#475569', fontSize: 11, minWidth: 36, textAlign: 'right', fontFamily: 'JetBrains Mono' }}>{curIdx + 1}/{total || 1}</span>
        </div>
        
        <div style={{ background: '#0c1526', border: `2px solid ${bc}`, borderRadius: 8, padding: '8px 14px', marginBottom: 8, display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', flexShrink: 0 }}>
          <span style={{ color: '#1e3a5f', fontSize: 10 }}>SEG</span>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 22, fontWeight: 800, color: '#e2e8f0' }}>#{cur?.segment_idx ?? '—'}</span>
          <span style={{ color: '#334155', fontSize: 11 }}>{toISTFull(cur?.timestamp)}</span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            {isA ? <span style={{ background: '#ef444420', color: '#ef4444', border: '1px solid #ef444430', borderRadius: 5, padding: '2px 10px', fontSize: 11, fontWeight: 700 }}>⚠ APNEA</span> :
              <span style={{ background: '#22c55e12', color: '#22c55e', border: '1px solid #22c55e30', borderRadius: 5, padding: '2px 10px', fontSize: 11, fontWeight: 600 }}>✓ NORMAL</span>}
            {isF && <span style={{ background: '#eab30820', color: '#eab308', border: '1px solid #eab30830', borderRadius: 5, padding: '2px 10px', fontSize: 11 }}>⚑ FLAGGED</span>}
          </div>
        </div>
        
        <div style={{ background: '#080f1e', border: '1px solid #111f33', borderRadius: 8, padding: 8, marginBottom: 6, position: 'relative', height: 220, flexShrink: 0 }}>
          <div style={{ position: 'absolute', top: 6, left: 10, color: '#1e3a5f', fontSize: 9, zIndex: 1, pointerEvents: 'none' }}>
            {`ECG WAVEFORM · 125 Hz · 30 s · ${peaks.length} R-peaks `}
            <span style={{ color: '#ef4444' }}>● R-peaks</span>
          </div>
          {ecg ? <ECGCanvas ecg={ecg} peaks={peaks} /> : <Empty msg="Upload ADM_segments.csv to view ECG waveform" />}
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, flexShrink: 0, marginBottom: 6 }}>
          <div style={{ background: '#080f1e', border: '1px solid #111f33', borderRadius: 8, padding: 8, position: 'relative', height: 88, flexShrink: 0 }}>
            <div style={{ position: 'absolute', top: 6, left: 10, color: '#1e3a5f', fontSize: 9, zIndex: 1, pointerEvents: 'none' }}>RESPIRATORY PROXY (EDR · ~4 Hz)</div>
            {edr.v.length > 0 ? <EDRCanvas edr={edr} /> :
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#1e3a5f', fontSize: 11 }}>
                {ecg ? 'Too few R-peaks for EDR' : 'Load segments.csv'}
              </div>}
          </div>
          
          <div style={{ background: '#080f1e', border: '1px solid #111f33', borderRadius: 8, padding: 8, position: 'relative', height: 88, flexShrink: 0 }}>
            <div style={{ position: 'absolute', top: 6, left: 10, color: '#1e3a5f', fontSize: 9, zIndex: 1, pointerEvents: 'none' }}>SPO2 WAVEFORM (1 Hz)</div>
            {spo2 ? <SpO2Canvas spo2={spo2} /> :
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#1e3a5f', fontSize: 11 }}>
                {cur?.has_spo2 ? 'SpO2 not in CSV (update pipeline)' : 'No SpO2 for segment'}
              </div>}
          </div>
        </div>
      </div>
      
      <div style={{ width: 212, background: '#080f1e', borderLeft: '1px solid #111f33', overflowY: 'auto', padding: '12px 10px', flexShrink: 0 }}>
        <ProbGauge prob={cur?.apnea_prob} />
        {cur?.bilstm_prob != null && (
          <div style={{ textAlign: 'center', marginTop: 2, marginBottom: 8 }}>
            <div style={{ color: '#1e3a5f', fontSize: 9, marginBottom: 2 }}>BILSTM PROB</div>
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: 16, fontWeight: 700, color: cur.bilstm_pred === 1 ? '#a78bfa' : '#334155' }}>
              {cur.bilstm_prob.toFixed(3)}
            </div>
          </div>
        )}
        
        <div style={{ borderTop: '1px solid #111f33', paddingTop: 8 }}>
          {FEATS.map(([l, v, u]) => (
            <div key={l} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', padding: '3px 2px', borderBottom: '1px solid #0a111e' }}>
              <span style={{ color: '#1e3a5f', fontSize: 9 }}>{l}</span>
              <span style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: v == null ? '#0f1c2e' : '#64748b' }}>
                {v == null ? 'N/A' : typeof v === 'number' ? v.toFixed(2) : v}
                {u && v != null && <span style={{ color: '#1e3a5f', fontSize: 8 }}> {u}</span>}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
