import React, { useMemo } from 'react';
import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, ReferenceLine, Line, Legend } from 'recharts';
import { Card, Lbl, Empty, St, CS } from './Shared';
import { toISTShort } from '../utils/format';

export function ModelsTab({ merged, hasBL, onSel, onTab }) {
  if (!merged?.length) return <Empty msg="Upload infer_results_ADM.csv to compare models" />;
  
  const cd = merged.map(r => ({ idx: r.segment_idx, xp: r.apnea_prob, bp: r.bilstm_prob }));
  
  const stats = useMemo(() => {
    let agree = 0, xonly = 0, bonly = 0, both0 = 0, both1 = 0;
    const disagree = [];
    
    merged.forEach(r => {
      const x = r.apnea_pred === 1, b = r.bilstm_pred === 1;
      if (r.bilstm_pred == null) {
        agree++;
        return;
      }
      
      if (x && b) both1++;
      else if (!x && !b) both0++;
      else if (x) {
        xonly++;
        disagree.push(r);
      } else {
        bonly++;
        disagree.push(r);
      }
      
      if (x === b) agree++;
    });
    
    return { agree, xonly, bonly, both0, both1, disagree, total: merged.length };
  }, [merged]);
  
  const ap = stats.total ? ((stats.agree / stats.total) * 100).toFixed(1) : 0;
  
  const cellC = r => {
    const x = r.apnea_pred === 1, b = r.bilstm_pred === 1;
    if (r.hr_gate_pass === 0) return '#0f1c2e';
    if (r.bilstm_pred == null) return x ? '#ef4444' : '#22c55e15';
    if (x && b) return '#ef4444';
    if (!x && !b) return '#22c55e15';
    return x ? '#3b82f6' : '#f97316';
  };
  
  return (
    <div className="fi" style={{ height: '100%', overflowY: 'auto', padding: '12px 18px' }}>
      {!hasBL && (
        <div style={{ background: '#eab30810', border: '1px solid #eab30830', borderRadius: 6, padding: '6px 12px', fontSize: 11, color: '#eab308', marginBottom: 10 }}>
          ⚠ BiLSTM file not loaded — upload bilstm_infer_results.csv for full comparison
        </div>
      )}
      
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <St label="Agreement" value={ap + '%'} color="#22c55e" />
        <St label="Both Apnea" value={stats.both1} color="#ef4444" />
        <St label="XGB Only" value={stats.xonly} color="#3b82f6" />
        <St label="BiLSTM Only" value={stats.bonly} color="#f97316" />
        <St label="Both Normal" value={stats.both0} color="#334155" />
      </div>
      
      <Card style={{ marginBottom: 12 }}>
        <Lbl>Model Probabilities per Segment</Lbl>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={cd} margin={{ top: 5, right: 18, left: 40, bottom: 22 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0e1d30" />
            <XAxis dataKey="idx" stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 9 }} label={{ value: 'Segment Index', position: 'insideBottom', fill: '#1e3a5f', fontSize: 9, dy: 8 }} />
            <YAxis domain={[0, 1]} stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 9 }} label={{ value: 'Apnea Prob', angle: -90, position: 'insideLeft', fill: '#1e3a5f', fontSize: 9, dx: 8 }} />
            <Tooltip {...CS} formatter={(v, n) => [v?.toFixed(3), n]} />
            <ReferenceLine y={0.55} stroke="#eab30860" strokeDasharray="4 2" label={{ value: '0.55', fill: '#eab308', fontSize: 8 }} />
            <Line type="monotone" dataKey="xp" name="XGBoost" stroke="#60a5fa" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />
            {hasBL && <Line type="monotone" dataKey="bp" name="BiLSTM" stroke="#a78bfa" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />}
            <Legend wrapperStyle={{ fontSize: 10, color: '#334155' }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      
      <Card style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 16, marginBottom: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <Lbl>Per-Segment Agreement Grid (click → ECG viewer)</Lbl>
          <div style={{ display: 'flex', gap: 8, fontSize: 9, flexWrap: 'wrap' }}>
            {[
              ['Both Apnea', '#ef4444'], ['Both Normal', '#22c55e30'], ['XGB Only', '#3b82f6'],
              ['BiLSTM Only', '#f97316'], ['Flagged', '#0f1c2e']
            ].map(([l, c]) => (
              <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 3, color: '#1e3a5f' }}>
                <span style={{ width: 9, height: 9, background: c, borderRadius: 2, border: '1px solid #1a2d4d20' }} />
                {l}
              </span>
            ))}
          </div>
        </div>
        
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, maxHeight: 180, overflowY: 'auto' }}>
          {merged.map(r => (
            <div
              key={r.segment_idx}
              title={`Seg ${r.segment_idx}: XGB=${r.apnea_pred} BiLSTM=${r.bilstm_pred ?? '?'} · ${toISTShort(r.timestamp)}`}
              onClick={() => { onSel(r.segment_idx); onTab(1); }}
              style={{ width: 11, height: 11, borderRadius: 2, background: cellC(r), cursor: 'pointer', border: '1px solid #00000020', transition: 'transform 0.1s' }}
              onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.9)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
            />
          ))}
        </div>
      </Card>
      
      <Card>
        <Lbl>Disagreements — {stats.disagree.length} segments (click row → ECG viewer)</Lbl>
        <div style={{ overflowX: 'auto', maxHeight: 280, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 11 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#0a111e' }}>
              <tr>
                {['Seg', 'Time (IST)', 'XGB Prob', 'BiLSTM Prob', 'ECG HR', 'Quality Flag'].map(hd => (
                  <th key={hd} style={{ padding: '5px 8px', color: '#1e3a5f', fontWeight: 600, textAlign: 'left', borderBottom: '1px solid #1a2d4d', whiteSpace: 'nowrap' }}>
                    {hd}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {stats.disagree.map(r => (
                <tr
                  key={r.segment_idx}
                  onClick={() => { onSel(r.segment_idx); onTab(1); }}
                  style={{ cursor: 'pointer', borderBottom: '1px solid #0a111e' }}
                  onMouseEnter={e => e.currentTarget.style.background = '#0f1c2e'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '4px 8px', fontFamily: 'JetBrains Mono', color: '#60a5fa' }}>#{r.segment_idx}</td>
                  <td style={{ padding: '4px 8px', color: '#1e3a5f', fontSize: 10, whiteSpace: 'nowrap' }}>{toISTShort(r.timestamp)}</td>
                  <td style={{ padding: '4px 8px', fontFamily: 'JetBrains Mono', color: r.apnea_pred === 1 ? '#ef4444' : '#22c55e' }}>{r.apnea_prob?.toFixed(3)}</td>
                  <td style={{ padding: '4px 8px', fontFamily: 'JetBrains Mono', color: r.bilstm_pred === 1 ? '#a78bfa' : '#334155' }}>{r.bilstm_prob?.toFixed(3) ?? '—'}</td>
                  <td style={{ padding: '4px 8px', fontFamily: 'JetBrains Mono', color: '#64748b' }}>{r.ecg_hr_bpm?.toFixed(1)}</td>
                  <td style={{ padding: '4px 8px', color: r.quality_flag !== 'OK' ? '#eab308' : '#1e3a5f', fontSize: 10 }}>{r.quality_flag}</td>
                </tr>
              ))}
              {stats.disagree.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', color: '#1e3a5f', padding: 24, fontSize: 12 }}>✓ Models fully agree</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
