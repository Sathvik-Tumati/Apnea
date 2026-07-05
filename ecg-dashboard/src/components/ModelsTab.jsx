import React, { useMemo } from 'react';
import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, ReferenceLine, Line, Legend } from 'recharts';
import { Card, Lbl, Empty, St, CS } from './Shared';
import { toISTShort } from '../utils/format';

export function ModelsTab({ merged, onSel, onTab }) {
  if (!merged?.length) return <Empty msg="Upload infer_results_ADM.csv to compare models" />;
  
  const cd = merged.map(r => ({ idx: r.segment_idx, xp: r.apnea_prob }));
  
  const stats = useMemo(() => {
    let apnea = 0, normal = 0;
    
    merged.forEach(r => {
      const x = r.apnea_pred === 1;
      if (x) apnea++;
      else normal++;
    });
    
    return { apnea, normal, total: merged.length };
  }, [merged]);
  
  const cellC = r => {
    const x = r.apnea_pred === 1;
    if (r.hr_gate_pass === 0) return '#0f1c2e';
    return x ? '#ef4444' : '#22c55e15';
  };
  
  return (
    <div className="fi" style={{ height: '100%', overflowY: 'auto', padding: '12px 18px' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <St label="Total" value={stats.total} color="#22c55e" />
        <St label="Apnea" value={stats.apnea} color="#ef4444" />
        <St label="Normal" value={stats.normal} color="#334155" />
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
            <Legend wrapperStyle={{ fontSize: 10, color: '#334155' }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      
      <Card style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 16, marginBottom: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <Lbl>Per-Segment Grid (click → ECG viewer)</Lbl>
          <div style={{ display: 'flex', gap: 8, fontSize: 9, flexWrap: 'wrap' }}>
            {[
              ['Apnea', '#ef4444'], ['Normal', '#22c55e30'], ['Flagged', '#0f1c2e']
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
              title={`Seg ${r.segment_idx}: XGB=${r.apnea_pred} · ${toISTShort(r.timestamp)}`}
              onClick={() => { onSel(r.segment_idx); onTab(1); }}
              style={{ width: 11, height: 11, borderRadius: 2, background: cellC(r), cursor: 'pointer', border: '1px solid #00000020', transition: 'transform 0.1s' }}
              onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.9)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}
            />
          ))}
        </div>
      </Card>
      

    </div>
  );
}
