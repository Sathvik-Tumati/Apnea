import React, { useMemo } from 'react';
import { ResponsiveContainer, ComposedChart, CartesianGrid, XAxis, YAxis, Tooltip, ReferenceArea, ReferenceLine, Line } from 'recharts';
import { Card, Lbl, Empty, St, SevBadge, CS } from './Shared';
import { toISTShort, ahiSev } from '../utils/format';

export function OverviewTab({ merged, summary, sleepMap, hasBL, onSel, onTab }) {
  if (!merged?.length) return <Empty msg="Upload infer_results_ADM.csv to see the overview" />;
  
  const admId = merged[0]?.admission_id || summary?.admission_id || '—';
  
  const blAhi = useMemo(() => {
    if (!hasBL) return summary?.ahi_bilstm ?? null;
    const nA = merged.filter(r => r.bilstm_pred === 1).length;
    const dur = summary?.duration_min;
    if (!dur) return null;
    return nA / Math.max(dur / 60, 1e-6);
  }, [merged, hasBL, summary]);
  
  const cd = useMemo(() => merged.map(r => ({
    idx: r.segment_idx,
    ecg: r.ecg_hr_bpm,
    ref: r.ref_hr_bpm,
    xp: r.apnea_prob,
    bp: r.bilstm_prob,
    xa: r.apnea_pred === 1 ? (r.ecg_hr_bpm || 60) : null,
    ba: r.bilstm_pred === 1 ? (r.ecg_hr_bpm || 60) : null,
    fl: r.hr_gate_pass === 0 ? (r.ecg_hr_bpm || 60) : null,
    ts: r.timestamp
  })), [merged]);
  
  const slA = useMemo(() => {
    const a = [];
    let st = null;
    cd.forEach((d, i) => {
      const sl = sleepMap?.get(d.idx)?.is_sleep;
      if (sl === 1 && st == null) st = d.idx;
      if (sl !== 1 && st != null) {
        a.push([st, cd[i - 1]?.idx ?? d.idx]);
        st = null;
      }
    });
    if (st != null) a.push([st, cd[cd.length - 1]?.idx]);
    return a;
  }, [cd, sleepMap]);
  
  const evDot = (col) => (props) => {
    if (props.value == null) return null;
    return <circle key={props.index} cx={props.cx} cy={props.cy} r={5} fill={col} stroke="#000" strokeWidth={0.5} />;
  };
  
  return (
    <div className="fi" style={{ height: '100%', overflowY: 'auto', padding: '14px 18px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 14 }}>
        <div style={{ fontSize: 18, fontWeight: 800, color: '#e2e8f0', fontFamily: 'JetBrains Mono' }}>{admId}</div>
        {summary?.duration_min && <div style={{ color: '#334155', fontSize: 12 }}>{Math.round(summary.duration_min)} min recording</div>}
      </div>
      
      <Card style={{ marginBottom: 12 }}>
        <Lbl>AHI Results</Lbl>
        <div style={{ display: 'flex', gap: 32, alignItems: 'center', flexWrap: 'wrap' }}>
          <div>
            <div style={{ color: '#475569', fontSize: 10, marginBottom: 4 }}>XGBoost AHI</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
              <span style={{ fontSize: 40, fontWeight: 900, color: '#60a5fa', fontFamily: 'JetBrains Mono', lineHeight: 1 }}>
                {summary?.ahi_proxy != null ? (+summary.ahi_proxy).toFixed(1) : '—'}
              </span>
              <span style={{ color: '#1e3a5f', fontSize: 12 }}>/hr</span>
            </div>
            {summary?.severity && <div style={{ marginTop: 4 }}><SevBadge s={summary.severity} /></div>}
          </div>
          <div style={{ width: 1, height: 60, background: '#1a2d4d' }} />
          <div>
            <div style={{ color: '#475569', fontSize: 10, marginBottom: 4 }}>BiLSTM AHI</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
              <span style={{ fontSize: 40, fontWeight: 900, color: '#a78bfa', fontFamily: 'JetBrains Mono', lineHeight: 1 }}>
                {blAhi != null ? blAhi.toFixed(1) : '—'}
              </span>
              <span style={{ color: '#1e3a5f', fontSize: 12 }}>/hr</span>
            </div>
            {blAhi != null && <div style={{ marginTop: 4 }}><SevBadge s={ahiSev(blAhi)} /></div>}
          </div>
          {!hasBL && (
            <div style={{ color: '#1e3a5f', fontSize: 11, marginLeft: 'auto', padding: '6px 12px', background: '#0d1d30', borderRadius: 6, border: '1px solid #1a2d4d' }}>
              BiLSTM file not loaded
            </div>
          )}
        </div>
      </Card>
      
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <St label="Total Segs" value={summary?.total_segments} />
        <St label="Scored" value={summary?.scored_segments} />
        <St label="Flagged" value={summary?.flagged_segments} color="#f97316" />
        <St label="Apnea Segs" value={summary?.n_apnea} color="#ef4444" />
        <St label="Apnea %" value={summary?.apnea_pct} unit="%" color={summary?.apnea_pct > 30 ? '#ef4444' : summary?.apnea_pct > 15 ? '#f97316' : '#e2e8f0'} />
        <St label="Duration" value={summary?.duration_min ? Math.round(summary.duration_min) : null} unit="min" />
        <St label="Mean HR Δ" value={summary?.mean_hr_diff_bpm} unit="bpm" mono />
      </div>
      
      <Card>
        <Lbl>Full Recording Timeline (IST) — blue shading=sleep | ● XGB apnea | ● BiLSTM apnea | ● HR flagged</Lbl>
        <div style={{ display: 'flex', gap: 14, marginBottom: 6, fontSize: 10, flexWrap: 'wrap' }}>
          {[
            ['ECG HR', '#60a5fa'], ['Ref HR', '#34d399'], ['XGB Prob', '#f97316'],
            ['BiLSTM Prob', '#a78bfa'], ['XGB Apnea', '#ef4444'], ['BiLSTM Apnea', '#fb923c'], ['HR Flagged', '#eab308']
          ].map(([l, c]) => (
            <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#334155' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: c }} />
              {l}
            </span>
          ))}
        </div>
        
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart 
            data={cd} 
            margin={{ top: 6, right: 60, left: 8, bottom: 20 }} 
            onClick={e => {
              if (e?.activePayload?.[0]) {
                onSel(e.activePayload[0].payload.idx);
                onTab(1);
              }
            }}
            style={{ cursor: 'pointer' }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#0e1d30" />
            {slA.map(([x1, x2], i) => <ReferenceArea key={i} x1={x1} x2={x2} fill="rgba(59,130,246,0.07)" strokeOpacity={0} />)}
            
            <XAxis dataKey="idx" stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 9 }} label={{ value: 'Segment Index', position: 'insideBottom', fill: '#1e3a5f', fontSize: 9, dy: 8 }} />
            <YAxis yAxisId="hr" stroke="#1a2d4d" domain={['auto', 'auto']} tick={{ fill: '#1e3a5f', fontSize: 9 }} label={{ value: 'HR (bpm)', angle: -90, position: 'insideLeft', fill: '#1e3a5f', fontSize: 9 }} />
            <YAxis yAxisId="p" orientation="right" domain={[0, 1]} stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 9 }} label={{ value: 'Prob', angle: 90, position: 'insideRight', fill: '#1e3a5f', fontSize: 9 }} />
            
            <Tooltip
              {...CS}
              labelFormatter={v => { const d = cd.find(r => r.idx === v); return d ? `Seg ${d.idx} · ${toISTShort(d.ts)}` : `Seg ${v}`; }}
              formatter={(val, name) => [typeof val === 'number' ? val.toFixed(3) : val, name]}
            />
            
            <Line yAxisId="hr" dataKey="ecg" name="ECG HR (bpm)" stroke="#60a5fa" strokeWidth={1.4} dot={false} connectNulls isAnimationActive={false} />
            <Line yAxisId="hr" dataKey="ref" name="Ref HR (bpm)" stroke="#34d399" strokeWidth={1} dot={false} connectNulls strokeDasharray="4 2" isAnimationActive={false} />
            <Line yAxisId="p" dataKey="xp" name="XGB Prob" stroke="#f97316" strokeWidth={1} dot={false} connectNulls isAnimationActive={false} />
            {hasBL && <Line yAxisId="p" dataKey="bp" name="BiLSTM Prob" stroke="#a78bfa" strokeWidth={1} dot={false} connectNulls isAnimationActive={false} />}
            
            <Line yAxisId="hr" dataKey="xa" name="XGB Apnea" stroke="none" strokeWidth={0} dot={evDot('#ef4444')} activeDot={false} isAnimationActive={false} connectNulls={false} />
            <Line yAxisId="hr" dataKey="ba" name="BiLSTM Apnea" stroke="none" strokeWidth={0} dot={evDot('#fb923c')} activeDot={false} isAnimationActive={false} connectNulls={false} />
            <Line yAxisId="hr" dataKey="fl" name="HR Flagged" stroke="none" strokeWidth={0} dot={evDot('#eab308')} activeDot={false} isAnimationActive={false} connectNulls={false} />
            
            <ReferenceLine yAxisId="p" y={0.55} stroke="#ef444440" strokeDasharray="4 2" label={{ value: '0.55', fill: '#ef4444', fontSize: 8 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
