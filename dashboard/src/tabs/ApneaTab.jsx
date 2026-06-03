import { useState, useEffect } from 'react';
import { api } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line, ScatterChart, Scatter,
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
  const [labelDist, setLabelDist] = useState([]);
  const [segments, setSegments] = useState([]);
  const [modelResults, setModelResults] = useState(null);
  const [signalFlags, setSignalFlags] = useState([]);
  const [spo2Plot, setSpo2Plot] = useState([]);
  const [respPlot, setRespPlot] = useState([]);
  const [featImp, setFeatImp] = useState([]);
  const [ecgPlot, setEcgPlot] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [s, ld, seg, mr, sf, sp, rp, fi, ep] = await Promise.all([
          api.getApneaSummary(),
          api.getApneaLabelDist(),
          api.getApneaSegments(100),
          api.getApneaModelResults(),
          api.getApneaSignalFlags(100),
          api.getApneaSpo2Plot(80),
          api.getApneaRespPlot(80),
          api.getApneaFeatImp(),
          api.getApneaEcgPlot(),
        ]);
        setSummary(s);
        setLabelDist(ld);
        setSegments(seg);
        setModelResults(mr?.[0] || null);
        setSignalFlags(sf);
        setSpo2Plot(sp);
        setRespPlot(rp);
        setFeatImp(fi.slice(0, 12));
        setEcgPlot(ep);
      } catch (err) {
        console.error('Apnea fetch error:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => <div key={i} className="skeleton" style={{ height: 120 }} />)}
      </div>
    );
  }

  return (
    <div className="space-y-6">
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
          <SectionHeader title="Label Distribution" subtitle="AASM composite labels" color="var(--color-amber)" />
          <div className="glass-card p-5" style={{ height: 280 }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={labelDist}
                  dataKey="count"
                  nameKey="label_confidence"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  innerRadius={45}
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
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
                />
              </PieChart>
            </ResponsiveContainer>
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
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
                />
                <Bar dataKey="importance" fill="#ffab00" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* SpO2 Trend */}
      <section>
        <SectionHeader title="SpO2 Trend" subtitle="Per-segment mean SpO2 with label coloring" color="var(--color-amber)" />
        <div className="glass-card p-5" style={{ height: 250 }}>
          <ResponsiveContainer>
            <AreaChart data={spo2Plot}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="segment_idx" tick={{ fill: '#484f58', fontSize: 10 }} label={{ value: 'Segment', position: 'insideBottom', offset: -5, fill: '#484f58' }} />
              <YAxis domain={['auto', 'auto']} tick={{ fill: '#484f58', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
              />
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
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
              />
              <Line type="monotone" dataKey="resp_rate_bpm" stroke="#00e676" dot={false} strokeWidth={1.5} />
              <Line type="monotone" dataKey="flatline_duration_s" stroke="#ff1744" dot={false} strokeWidth={1} strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Signal Flags Table */}
      <section>
        <SectionHeader title="Signal Flag Breakdown" subtitle="AASM 4-signal scoring per segment" color="var(--color-amber)" />
        <div className="glass-card overflow-auto" style={{ maxHeight: 360 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Record</th>
                <th>Seg</th>
                <th>Resp</th>
                <th>SpO2</th>
                <th>HRV</th>
                <th>ABP</th>
                <th>Σ Flags</th>
                <th>Label</th>
              </tr>
            </thead>
            <tbody>
              {signalFlags.slice(0, 60).map((f, i) => (
                <tr key={i}>
                  <td style={{ fontSize: 10 }}>{f.record}</td>
                  <td>{f.segment_idx}</td>
                  <td><span className={`badge ${f.resp_flag ? 'red' : 'green'}`}>{f.resp_flag ? '⚡' : '○'}</span></td>
                  <td><span className={`badge ${f.spo2_flag ? 'red' : 'green'}`}>{f.spo2_flag ? '⚡' : '○'}</span></td>
                  <td><span className={`badge ${f.hrv_flag ? 'red' : 'green'}`}>{f.hrv_flag ? '⚡' : '○'}</span></td>
                  <td><span className={`badge ${f.abp_flag ? 'red' : 'green'}`}>{f.abp_flag ? '⚡' : '○'}</span></td>
                  <td style={{ fontWeight: 700, color: f.signals_positive >= 2 ? 'var(--color-red)' : 'var(--color-green)' }}>
                    {f.signals_positive}/4
                  </td>
                  <td>
                    <span className="badge" style={{
                      background: `${LABEL_COLORS[f.label_confidence] || '#8b949e'}22`,
                      color: LABEL_COLORS[f.label_confidence] || '#8b949e',
                    }}>
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
  );
}
