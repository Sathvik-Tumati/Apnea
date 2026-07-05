import React, { useState, useMemo } from 'react';
import { Empty } from './Shared';
import { toISTShort } from '../utils/format';

export function SegTable({ merged, hasBL, onSel, onTab }) {
  const [pg, setPg] = useState(0);
  const [sf, setSf] = useState('all');
  
  const fil = useMemo(() => {
    if (!merged) return [];
    if (sf === 'all') return merged;
    if (sf === 'apneaX') return merged.filter(r => r.apnea_pred === 1);
    if (sf === 'apneaB') return merged.filter(r => r.bilstm_pred === 1);
    if (sf === 'disagree') return merged.filter(r => r.bilstm_pred != null && r.apnea_pred !== r.bilstm_pred);
    if (sf === 'flag') return merged.filter(r => r.hr_gate_pass === 0);
    return merged;
  }, [merged, sf]);
  
  if (!merged?.length) return <Empty msg="Upload infer_results_ADM.csv to view segment table" />;
  
  const psz = 100;
  const pgs = Math.ceil(fil.length / psz);
  const cur = fil.slice(pg * psz, (pg + 1) * psz);
  
  const TH = ({ children }) => (
    <th style={{ padding: '6px 8px', color: '#334155', fontSize: 9, fontWeight: 700, textTransform: 'uppercase', textAlign: 'left', borderBottom: '1px solid #1a2d4d', position: 'sticky', top: 0, background: '#0a111e', whiteSpace: 'nowrap' }}>
      {children}
    </th>
  );
  
  const TD = ({ children, mono, col }) => (
    <td style={{ padding: '5px 8px', color: col || '#1e3a5f', fontSize: 11, fontFamily: mono ? 'JetBrains Mono' : 'inherit', borderBottom: '1px solid #0a111e', whiteSpace: 'nowrap' }}>
      {children}
    </td>
  );

  // Colour-code ΔRR: green ≤50ms, amber ≤100ms, red >100ms
  const rrDiffCol = (d) => {
    if (d == null || isNaN(d)) return '#64748b';
    const abs = Math.abs(d);
    if (abs <= 50)  return '#22c55e';
    if (abs <= 100) return '#eab308';
    return '#ef4444';
  };
  
  return (
    <div className="fi" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', padding: '10px 14px', borderBottom: '1px solid #1a2d4d', gap: 14 }}>
        <select
          value={sf}
          onChange={e => { setSf(e.target.value); setPg(0); }}
          style={{ background: '#0c1526', color: '#e2e8f0', border: '1px solid #1a2d4d', borderRadius: 6, padding: '4px 10px', fontSize: 12, outline: 'none' }}
        >
          <option value="all">All {merged.length} segments</option>
          <option value="apneaX">XGBoost Apnea ({merged.filter(r => r.apnea_pred === 1).length})</option>
          {hasBL && <option value="apneaB">BiLSTM Apnea ({merged.filter(r => r.bilstm_pred === 1).length})</option>}
          {hasBL && <option value="disagree">Disagreements ({merged.filter(r => r.bilstm_pred != null && r.apnea_pred !== r.bilstm_pred).length})</option>}
          <option value="flag">Flagged/Excluded ({merged.filter(r => r.hr_gate_pass === 0).length})</option>
        </select>
        
        <div style={{ color: '#1e3a5f', fontSize: 11, marginLeft: 'auto' }}>
          Page {pg + 1} of {Math.max(1, pgs)} ({fil.length} matches)
        </div>
        
        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={() => setPg(Math.max(0, pg - 1))} disabled={pg === 0} style={{ background: '#0c1526', border: '1px solid #1a2d4d', color: '#475569', borderRadius: 4, padding: '2px 8px', fontSize: 12 }}>◀</button>
          <button onClick={() => setPg(Math.min(pgs - 1, pg + 1))} disabled={pg >= pgs - 1} style={{ background: '#0c1526', border: '1px solid #1a2d4d', color: '#475569', borderRadius: 4, padding: '2px 8px', fontSize: 12 }}>▶</button>
        </div>
      </div>
      
      <div style={{ flex: 1, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <TH>Seg</TH><TH>Time (IST)</TH><TH>XGB Prob</TH>
              {hasBL && <TH>BiLSTM Prob</TH>}
              <TH>ECG HR</TH><TH>Ref HR</TH>
              <TH>RR (ms)</TH><TH>Ref RR (ms)</TH><TH>ΔRR (ms)</TH>
              <TH>SpO2 µ</TH><TH>LF/HF</TH><TH>Resp Rate</TH><TH>HR Gate</TH>
            </tr>
          </thead>
          <tbody>
            {cur.map(r => (
              <tr
                key={r.segment_idx}
                onClick={() => { onSel(r.segment_idx); onTab(1); }}
                style={{ cursor: 'pointer' }}
                onMouseEnter={e => e.currentTarget.style.background = '#0f1c2e'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <TD mono col="#60a5fa">#{r.segment_idx}</TD>
                <TD col="#334155" fontSize={10}>{toISTShort(r.timestamp)}</TD>
                <TD mono col={r.apnea_pred === 1 ? '#ef4444' : '#22c55e'}>{r.apnea_prob?.toFixed(3)}</TD>
                {hasBL && <TD mono col={r.bilstm_pred === 1 ? '#a78bfa' : r.bilstm_pred === 0 ? '#22c55e' : '#334155'}>{r.bilstm_prob?.toFixed(3) ?? '—'}</TD>}
                <TD mono col="#64748b">{r.ecg_hr_bpm?.toFixed(1)}</TD>
                <TD mono col="#64748b">{r.ref_hr_bpm?.toFixed(1)}</TD>
                {/* RR columns */}
                <TD mono col="#94a3b8">{r.rr_mean != null ? Number(r.rr_mean).toFixed(0) : '—'}</TD>
                {(() => {
                  const refRR = (r.ref_hr_bpm != null && r.ref_hr_bpm > 0)
                    ? 60000.0 / r.ref_hr_bpm
                    : null;
                  const ecgRR = r.rr_mean != null ? Number(r.rr_mean) : null;
                  const diff  = (ecgRR != null && refRR != null) ? ecgRR - refRR : null;
                  return (
                    <>
                      <TD mono col="#94a3b8">{refRR != null ? refRR.toFixed(0) : '—'}</TD>
                      <TD mono col={rrDiffCol(diff)}>
                        {diff != null ? (diff >= 0 ? '+' : '') + diff.toFixed(0) : '—'}
                      </TD>
                    </>
                  );
                })()}
                <TD mono col="#64748b">{r.has_spo2 ? r.spo2_mean?.toFixed(1) : '—'}</TD>
                <TD mono col="#64748b">{r.lf_hf_ratio?.toFixed(2)}</TD>
                <TD mono col="#64748b">{r.resp_rate_bpm?.toFixed(1)}</TD>
                <TD col={r.hr_gate_pass === 1 ? '#22c55e' : '#ef4444'}>{r.hr_gate_pass === 1 ? 'PASS' : 'FAIL'}</TD>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
