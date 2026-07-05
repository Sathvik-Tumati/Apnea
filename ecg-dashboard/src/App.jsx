import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { UploadZone, FDEFS, parseCSV } from './components/UploadZone';
import { TabBar } from './components/Shared';
import { OverviewTab } from './components/OverviewTab';
import { ECGViewerTab } from './components/ECGViewerTab';
import { SleepTab } from './components/SleepTab';
import { ModelsTab } from './components/ModelsTab';
import { SegTable } from './components/SegTable';
import './index.css';

export default function App() {
  const [tab, setTab] = useState(0);
  const [selSeg, setSelSeg] = useState(0);
  const [raw, setRaw] = useState({ segments: null, ecgMap: null, inferXgb: null, sleepWindows: null, summary: null });
  
  const [st, setSt] = useState({});
  const [pg, setPg] = useState({});

  const onLoad = useCallback((key, result) => {
    setRaw(prev => {
      const next = { ...prev };
      if (key === 'segments') { next.segments = result.rows; next.ecgMap = result.ecgMap; }
      else if (key === 'inferXgb') next.inferXgb = result.rows;
      else if (key === 'sleep') next.sleepWindows = result.rows;
      else if (key === 'summary') next.summary = result.rows?.[0];
      return next;
    });
  }, []);

  const handle = useCallback(async (file, key) => {
    setSt(s => ({ ...s, [key]: 'loading' }));
    try {
      const r = await parseCSV(file, key, n => setPg(p => ({ ...p, [key]: n })));
      onLoad(key, r);
      setSt(s => ({ ...s, [key]: 'ok' }));
    } catch (e) {
      console.error(e);
      setSt(s => ({ ...s, [key]: 'err' }));
    }
  }, [onLoad]);

  // Auto-fetch data from backend on load
  useEffect(() => {
    FDEFS.forEach(async ({ key, file }) => {
      try {
        setSt(s => ({ ...s, [key]: 'loading' }));
        const res = await fetch(`/${file}`);
        if (!res.ok) throw new Error('Not found');
        const blob = await res.blob();
        blob.name = file;
        handle(blob, key);
      } catch (e) {
        setSt(s => ({ ...s, [key]: 'err' }));
      }
    });
  }, [handle]);

  const sleepMap = useMemo(() => {
    if (!raw.sleepWindows) return new Map();
    return new Map(raw.sleepWindows.map(r => [+r.segment_idx, r]));
  }, [raw.sleepWindows]);

  const merged = useMemo(() => {
    if (!raw.inferXgb) return null;
    return raw.inferXgb;
  }, [raw.inferXgb]);

  const admId = merged?.[0]?.admission_id || raw.summary?.admission_id || null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#060b16' }}>
      <div style={{ display: 'flex', alignItems: 'center', background: '#08111f', borderBottom: '1px solid #1a2d4d', padding: '10px 14px', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 14, height: 14, borderRadius: '50%', background: '#3b82f6', boxShadow: '0 0 10px #3b82f6' }} />
          <h1 style={{ fontSize: 13, fontWeight: 900, color: '#e2e8f0', letterSpacing: '0.08em', margin: 0, textTransform: 'uppercase' }}>
            ECG Apnea Verification Dashboard
          </h1>
        </div>
        <div style={{ marginLeft: 16, color: '#475569', fontSize: 10 }}>
          All times in IST (UTC+5:30) · Research prototype — not for clinical use
        </div>
        {admId && (
          <div style={{ marginLeft: 'auto', background: '#0f1e35', border: '1px solid #1a2d4d', borderRadius: 6, padding: '4px 14px', fontFamily: 'JetBrains Mono', fontSize: 12, color: '#60a5fa', fontWeight: 700 }}>
            {admId}
          </div>
        )}
      </div>

      <UploadZone st={st} pg={pg} handle={handle} />
      <TabBar active={tab} onSelect={setTab} />

      <div style={{ flex: 1, overflow: 'hidden' }}>
        {tab === 0 && <OverviewTab merged={merged} summary={raw.summary} sleepMap={sleepMap} onSel={setSelSeg} onTab={setTab} />}
        {tab === 1 && <ECGViewerTab merged={merged} ecgMap={raw.ecgMap} spo2Map={raw.spo2Map} selSeg={selSeg} onSel={setSelSeg} />}
        {tab === 2 && <SleepTab sleepWindows={raw.sleepWindows} onSel={setSelSeg} onTab={setTab} />}
        {tab === 3 && <ModelsTab merged={merged} onSel={setSelSeg} onTab={setTab} />}
        {tab === 4 && <SegTable merged={merged} onSel={setSelSeg} onTab={setTab} />}
      </div>
    </div>
  );
}
