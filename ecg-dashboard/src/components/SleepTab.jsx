import React from 'react';
import { ResponsiveContainer, ComposedChart, CartesianGrid, XAxis, YAxis, Tooltip, ReferenceArea, ReferenceLine, Line, AreaChart, Area } from 'recharts';
import { Card, Lbl, Empty, CS } from './Shared';
import { toISTShort } from '../utils/format';

export function SleepTab({ sleepWindows, onSel, onTab }) {
  if (!sleepWindows?.length) return <Empty msg="Upload sleep_windows.csv to inspect the sleep filter" />;
  
  const cd = sleepWindows.map(r => ({
    idx: r.segment_idx,
    hr: r.mean_hr,
    sdnn: r.sdnn,
    in_w: (r.in_sleep_hours === true || r.in_sleep_hours === 1 || r.in_sleep_hours === 'True' || r.in_sleep_hours === 'true') ? 1 : 0,
    score: r.sleep_score,
    is_s: r.is_sleep,
    ts: r.timestamp_utc || r.timestamp_ist
  }));
  
  const wins = [];
  let ws = null;
  sleepWindows.forEach((r, i) => {
    if (r.is_sleep === 1 && ws == null) ws = { idx: r.segment_idx, ts: r.timestamp_utc };
    if (r.is_sleep !== 1 && ws != null) {
      const prev = sleepWindows[i - 1] || r;
      wins.push({ s: ws.idx, e: prev.segment_idx, ts: ws.ts, te: prev.timestamp_utc, dur: (prev.segment_idx - ws.idx + 1) * 0.5 });
      ws = null;
    }
  });
  
  if (ws != null) {
    const last = sleepWindows[sleepWindows.length - 1];
    wins.push({ s: ws.idx, e: last.segment_idx, ts: ws.ts, te: last.timestamp_utc, dur: (last.segment_idx - ws.idx + 1) * 0.5 });
  }
  
  const sAs = wins.map(w => [w.s, w.e]);
  const excl = cd.filter(d => d.in_w === 1 && d.is_s === 0).map(d => d.idx);
  
  const RA = ({ areas }) => areas.map(([x1, x2], i) => <ReferenceArea key={i} x1={x1} x2={x2} fill="rgba(59,130,246,0.08)" strokeOpacity={0} />);
  const AX = <XAxis dataKey="idx" stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 8 }} height={22} label={{ value: 'Segment Index', position: 'insideBottom', fill: '#1e3a5f', fontSize: 8, dy: 5 }} />;
  const ps = { marginBottom: 8, background: '#0c1526', border: '1px solid #1a2d4d', borderRadius: 8, padding: '8px 12px 4px' };
  
  const onClk = (e) => {
    if (e?.activePayload?.[0]) {
      onSel(e.activePayload[0].payload.idx);
      onTab(1);
    }
  };

  return (
    <div className="fi" style={{ height: '100%', overflowY: 'auto', padding: '12px 18px' }}>
      {excl.length > 0 && (
        <div style={{ background: '#eab30812', border: '1px solid #eab30830', borderRadius: 6, padding: '6px 12px', fontSize: 11, color: '#eab308', marginBottom: 10 }}>
          ⚑ {excl.length} segments inside the 9PM–9AM IST window but excluded by HR/SDNN gate
        </div>
      )}
      
      <div style={ps}>
        <div style={{ color: '#334155', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Heart Rate (bpm)</div>
        <ResponsiveContainer width="100%" height={130}>
          <ComposedChart data={cd} margin={{ top: 4, right: 18, left: 40, bottom: 18 }} onClick={onClk} style={{ cursor: 'pointer' }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0e1d30" />
            <RA areas={sAs} />
            {AX}
            <YAxis stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 8 }} label={{ value: 'HR (bpm)', angle: -90, position: 'insideLeft', fill: '#1e3a5f', fontSize: 8 }} />
            <Tooltip {...CS} formatter={v => [v?.toFixed(1), 'HR (bpm)']} labelFormatter={v => { const d = cd.find(r => r.idx === v) || {}; return `Seg ${d.idx || v} · ${toISTShort(d.ts)}`; }} />
            <Line type="monotone" dataKey="hr" stroke="#f87171" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div style={ps}>
        <div style={{ color: '#334155', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>SDNN (ms)</div>
        <ResponsiveContainer width="100%" height={130}>
          <ComposedChart data={cd} margin={{ top: 4, right: 18, left: 40, bottom: 18 }} onClick={onClk} style={{ cursor: "pointer" }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0e1d30" />
            <RA areas={sAs} />
            {AX}
            <YAxis stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 8 }} label={{ value: 'SDNN (ms)', angle: -90, position: 'insideLeft', fill: '#1e3a5f', fontSize: 8 }} />
            <Tooltip {...CS} formatter={v => [v?.toFixed(1), 'SDNN (ms)']} />
            <Line type="monotone" dataKey="sdnn" stroke="#fb923c" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <div style={ps}>
        <div style={{ color: '#334155', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>In Time Window (9PM–9AM IST)</div>
        <ResponsiveContainer width="100%" height={75}>
          <AreaChart data={cd} margin={{ top: 4, right: 18, left: 40, bottom: 18 }} onClick={onClk} style={{ cursor: 'pointer' }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0e1d30" />
            {AX}
            <YAxis domain={[0, 1]} ticks={[0, 1]} stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 8 }} />
            <Tooltip {...CS} formatter={v => [v ? 'In window' : 'Outside', '']} />
            <Area type="stepAfter" dataKey="in_w" fill="rgba(34,197,94,0.18)" stroke="#22c55e" strokeWidth={1} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      
      <div style={ps}>
        <div style={{ color: '#334155', fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Sleep Score (threshold = 0.50) · shaded = is_sleep=1</div>
        <ResponsiveContainer width="100%" height={130}>
          <ComposedChart data={cd} margin={{ top: 4, right: 18, left: 40, bottom: 18 }} onClick={onClk} style={{ cursor: "pointer" }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#0e1d30" />
            <RA areas={sAs} />
            {AX}
            <YAxis domain={[0, 1.1]} stroke="#1a2d4d" tick={{ fill: '#1e3a5f', fontSize: 8 }} label={{ value: 'Score', angle: -90, position: 'insideLeft', fill: '#1e3a5f', fontSize: 8 }} />
            <Tooltip {...CS} />
            <Area type="stepAfter" dataKey="is_s" fill="rgba(96,165,250,0.12)" stroke="none" isAnimationActive={false} />
            <Line type="monotone" dataKey="score" name="Sleep Score" stroke="#60a5fa" strokeWidth={1.5} dot={false} connectNulls isAnimationActive={false} />
            <ReferenceLine y={0.5} stroke="#eab308" strokeDasharray="4 2" label={{ value: '0.50', fill: '#eab308', fontSize: 8, position: 'insideTopLeft' }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      
      <Card>
        <Lbl>Sleep Windows Found: {wins.length}</Lbl>
        {wins.length === 0 && <div style={{ color: '#1e3a5f', fontSize: 12 }}>No sleep windows detected</div>}
        {wins.map((w, i) => (
          <div key={i} style={{ display: 'flex', gap: 16, padding: '6px 0', borderBottom: '1px solid #0d1624', alignItems: 'center', fontSize: 11, flexWrap: 'wrap' }}>
            <span style={{ color: '#60a5fa', fontFamily: 'JetBrains Mono', fontWeight: 700, fontSize: 12 }}>#{i + 1}</span>
            <span style={{ color: '#1e3a5f' }}>Segs {w.s}–{w.e}</span>
            <span style={{ color: '#334155' }}>{toISTShort(w.ts)}</span>
            <span style={{ color: '#0f1c2e' }}>→</span>
            <span style={{ color: '#334155' }}>{toISTShort(w.te)}</span>
            <span style={{ color: '#22c55e', marginLeft: 'auto', fontFamily: 'JetBrains Mono', fontWeight: 700 }}>{w.dur.toFixed(0)} min</span>
          </div>
        ))}
      </Card>
    </div>
  );
}
