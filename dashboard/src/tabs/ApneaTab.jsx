import { useState, useEffect, useCallback } from 'react';
import { api } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
  LineChart, Line,
  AreaChart, Area,
} from 'recharts';

const LABEL_COLORS = {
  normal: '#00e676',
  possible_hypopnea: '#ffab00',
  probable_apnea: '#ff8f00',
  definite_apnea: '#ff1744',
};

function SectionHeader({ title, subtitle, color }) {
  return (
    <div className="section-header">
      <div className="accent-bar" style={color ? { background: color } : undefined} />
      <div>
        <h2>{title}</h2>
        {subtitle && <div className="subtitle">{subtitle}</div>}
      </div>
    </div>
  );
}

export default function ApneaTab() {
  const [summary, setSummary] = useState(null);
  const [records, setRecords] = useState([]);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [labelDist, setLabelDist] = useState([]);
  const [segments, setSegments] = useState([]);
  const [modelResults, setModelResults] = useState(null);
  const [signalFlags, setSignalFlags] = useState([]);
  const [spo2Plot, setSpo2Plot] = useState([]);
  const [respPlot, setRespPlot] = useState([]);
  const [ecgPlot, setEcgPlot] = useState([]);
  const [featImp, setFeatImp] = useState([]);
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(false);

  // Initial load — record list + model-level data
  useEffect(() => {
    (async () => {
      try {
        const [s, recs, mr, fi] = await Promise.all([
          api.getApneaSummary(),
          api.getApneaRecords(),
          api.getApneaModelResults(),
          api.getApneaFeatImp(),
        ]);
        setSummary(s);
        setRecords(recs);
        setModelResults(mr?.[0] || null);
        setFeatImp(fi.slice(0, 12));
      } catch (err) {
        console.error('Apnea fetch error:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Reload per-patient chart data whenever selected record changes
  const loadChartData = useCallback(async (rec) => {
    setChartLoading(true);
    try {
      const [ld, seg, sf, sp, rp, ep] = await Promise.all([
        api.getApneaLabelDist(rec),
        api.getApneaSegments(100, rec),
        api.getApneaSignalFlags(100, rec),
        api.getApneaSpo2Plot(80, rec),
        api.getApneaRespPlot(80, rec),
        api.getApneaEcgPlot(rec),
      ]);
      setLabelDist(ld);
      setSegments(seg);
      setSignalFlags(sf);
      setSpo2Plot(sp);
      setRespPlot(rp);
      
      // ECG comes back as an object with `ecg` array. 
      // Downsample / limit to 1250 points (10s at 125Hz) to keep UI fast.
      if (ep && ep.ecg) {
        const sliced = ep.ecg.slice(0, 1250);
        setEcgPlot(sliced.map((val, idx) => ({ idx, val })));
      } else {
        setEcgPlot([]);
      }
    } catch (err) {
      console.error('Apnea chart reload error:', err);
    } finally {
      setChartLoading(false);
    }
  }, []);

  useEffect(() => {
    loadChartData(selectedRecord);
  }, [selectedRecord, loadChartData]);

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 120 }} />)}
      </div>
    );
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: '16px', alignItems: 'start' }}>

      {/* ── Left panel: Record selector ── */}
      <aside>
        <div className="glass-card" style={{ padding: '12px 0' }}>
          <div style={{
            padding: '6px 14px 10px',
            fontSize: 10,
            fontFamily: 'var(--font-sans)',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            color: 'var(--color-text-muted)',
            borderBottom: '1px solid var(--color-border)',
            marginBottom: 6,
          }}>
            Patient Records
          </div>
          {/* All records option */}
          <button
            onClick={() => setSelectedRecord(null)}
            style={{
              display: 'block',
              width: '100%',
              textAlign: 'left',
              padding: '8px 14px',
              background: selectedRecord === null ? 'var(--color-amber-glow)' : 'transparent',
              color: selectedRecord === null ? 'var(--color-amber)' : 'var(--color-text-secondary)',
              fontFamily: 'var(--font-sans)',
              fontSize: 12,
              fontWeight: 500,
              cursor: 'pointer',
              border: 'none',
              borderLeft: selectedRecord === null ? '3px solid var(--color-amber)' : '3px solid transparent',
              transition: 'all 0.15s ease',
            }}
          >
            All Patients
          </button>
          {records.map((r) => {
            const apneaPct = r.segment_count > 0 ? (r.apnea_count / r.segment_count * 100).toFixed(0) : 0;
            const isSel = selectedRecord === r.record;
            return (
              <button
                key={r.record}
                onClick={() => setSelectedRecord(r.record)}
                style={{
                  display: 'block',
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 14px',
                  background: isSel ? 'var(--color-amber-glow)' : 'transparent',
                  color: isSel ? 'var(--color-amber)' : 'var(--color-text-secondary)',
                  fontFamily: 'var(--font-sans)',
                  fontSize: 12,
                  fontWeight: 500,
                  cursor: 'pointer',
                  border: 'none',
                  borderLeft: isSel ? '3px solid var(--color-amber)' : '3px solid transparent',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{r.record}</div>
                <div style={{ fontSize: 10, color: 'var(--color-text-muted)', marginTop: 2 }}>
                  {r.segment_count} segs · {apneaPct}% apnea
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* ── Right panel: Charts ── */}
      <div className="space-y-6" style={{ opacity: chartLoading ? 0.55 : 1, transition: 'opacity 0.2s' }}>

        {/* Summary cards */}
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: 'Total Segments', value: summary?.total_segments, color: '' },
            { label: 'Apnea Segments', value: summary?.apnea_segments, color: 'red' },
            { label: 'Normal Segments', value: summary?.normal_segments, color: 'green' },
            { label: 'AUC-ROC', value: summary?.auc_roc != null ? summary.auc_roc.toFixed(3) : '—', color: 'purple' },
          ].map((c, i) => (
            <div key={c.label} className={`metric-card ${c.color} animate-fade-up`} style={{ animationDelay: `${i * 50}ms` }}>
              <div className="text-2xl font-bold" style={{ fontFamily: 'var(--font-mono)' }}>
                {typeof c.value === 'number' ? c.value.toLocaleString() : c.value}
              </div>
              <div className="mt-1 text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
                {c.label}
              </div>
            </div>
          ))}
        </div>

        {/* Label Distribution + Feature Importance */}
        <div className="grid grid-cols-2 gap-4">
          <section>
            <SectionHeader
              title="Label Distribution"
              subtitle={selectedRecord ? `Record: ${selectedRecord}` : '3-signal composite labels — all patients'}
              color="var(--color-amber)"
            />
            <div className="glass-card p-5" style={{ height: 280 }}>
              {labelDist.length > 0 ? (
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={labelDist}
                      dataKey="count"
                      nameKey="label_confidence"
                      cx="50%" cy="50%"
                      outerRadius={90} innerRadius={45}
                      paddingAngle={3}
                      label={({ label_confidence, percent }) =>
                        `${label_confidence} ${(percent * 100).toFixed(0)}%`
                      }
                      labelLine={false}
                    >
                      {labelDist.map((entry) => (
                        <Cell key={entry.label_confidence} fill={LABEL_COLORS[entry.label_confidence] || '#8b949e'} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full" style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
                  No data for this record
                </div>
              )}
            </div>
          </section>

          <section>
            <SectionHeader title="Feature Importance" subtitle="Top discriminative features (apnea vs normal)" color="var(--color-amber)" />
            <div className="glass-card p-5" style={{ height: 280 }}>
              <ResponsiveContainer>
                <BarChart data={featImp} layout="vertical" barSize={14}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#484f58', fontSize: 10 }} />
                  <YAxis dataKey="feature" type="category" width={140} tick={{ fill: '#8b949e', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }} />
                  <Bar dataKey="importance" fill="#ffab00" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>

        {/* SpO2 Trend */}
        <section>
          <SectionHeader title="SpO2 Trend" subtitle="Per-segment mean SpO2" color="var(--color-amber)" />
          <div className="glass-card p-5" style={{ height: 250 }}>
            <ResponsiveContainer>
              <AreaChart data={spo2Plot}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="segment_idx" tick={{ fill: '#484f58', fontSize: 10 }} label={{ value: 'Segment', position: 'insideBottom', offset: -5, fill: '#484f58' }} />
                <YAxis domain={['auto', 'auto']} tick={{ fill: '#484f58', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }} />
                <Area type="monotone" dataKey="spo2_mean" stroke="#00e5ff" fill="rgba(0,229,255,0.1)" strokeWidth={1.5} />
                <Area type="monotone" dataKey="spo2_min" stroke="#ff1744" fill="rgba(255,23,68,0.08)" strokeWidth={1} strokeDasharray="4 4" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Resp Rate */}
        <section>
          <SectionHeader title="Respiration Rate" subtitle="Breathing rate per segment (bpm)" color="var(--color-amber)" />
          <div className="glass-card p-5" style={{ height: 220 }}>
            <ResponsiveContainer>
              <LineChart data={respPlot}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="segment_idx" tick={{ fill: '#484f58', fontSize: 10 }} />
                <YAxis tick={{ fill: '#484f58', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }} />
                <Line type="monotone" dataKey="resp_rate_bpm" stroke="#00e676" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="flatline_duration_s" stroke="#ff1744" dot={false} strokeWidth={1} strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* ECG Waveform */}
        <section>
          <SectionHeader title="ECG Waveform (Lead II)" subtitle="10-second snippet from selected segment (125Hz)" color="var(--color-amber)" />
          <div className="glass-card p-5" style={{ height: 250 }}>
            {ecgPlot.length > 0 ? (
              <ResponsiveContainer>
                <LineChart data={ecgPlot}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.2} />
                  <XAxis dataKey="idx" hide />
                  <YAxis domain={['auto', 'auto']} tick={{ fill: '#484f58', fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
                    labelFormatter={() => ''}
                  />
                  <Line type="monotone" dataKey="val" stroke="#00e5ff" dot={false} strokeWidth={1.2} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full" style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
                No ECG data available
              </div>
            )}
          </div>
        </section>

        {/* Signal Flags Table */}
        <section>
          <SectionHeader title="Signal Flag Breakdown" subtitle="3-signal composite scoring per segment" color="var(--color-amber)" />
          <div className="glass-card overflow-auto" style={{ maxHeight: 360 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Record</th><th>Seg</th><th>Resp</th><th>SpO2</th><th>HRV</th><th>Σ Flags</th><th>Label</th>
                </tr>
              </thead>
              <tbody>
                {signalFlags.map((f, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: 10 }}>{f.record}</td>
                    <td>{f.segment_idx}</td>
                    <td><span className={`badge ${f.resp_flag ? 'red' : 'green'}`}>{f.resp_flag ? '⚡' : '○'}</span></td>
                    <td><span className={`badge ${f.spo2_flag ? 'red' : 'green'}`}>{f.spo2_flag ? '⚡' : '○'}</span></td>
                    <td><span className={`badge ${f.hrv_flag ? 'red' : 'green'}`}>{f.hrv_flag ? '⚡' : '○'}</span></td>
                    <td style={{ fontWeight: 700, color: f.signals_positive >= 2 ? 'var(--color-red)' : 'var(--color-green)' }}>
                      {f.signals_positive}/3
                    </td>
                    <td>
                      <span className="badge" style={{ background: `${LABEL_COLORS[f.label_confidence] || '#8b949e'}22`, color: LABEL_COLORS[f.label_confidence] || '#8b949e' }}>
                        {f.label_confidence}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Model Results */}
        {modelResults?.report && (
          <section>
            <SectionHeader title="LSTM Model Report" subtitle={`AUC-ROC: ${modelResults.auc_roc?.toFixed(4)}`} color="var(--color-amber)" />
            <div className="glass-card p-5 overflow-auto">
              <table className="data-table">
                <thead>
                  <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
                </thead>
                <tbody>
                  {['Normal', 'Apnea'].map((cls) => {
                    const r = modelResults.report[cls];
                    if (!r) return null;
                    return (
                      <tr key={cls}>
                        <td><span className={`badge ${cls === 'Apnea' ? 'red' : 'green'}`}>{cls}</span></td>
                        <td>{r.precision?.toFixed(3)}</td>
                        <td>{r.recall?.toFixed(3)}</td>
                        <td>{r['f1-score']?.toFixed(3)}</td>
                        <td>{r.support}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
