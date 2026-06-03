import { useState, useEffect } from 'react';
import { api } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line,
} from 'recharts';

const BEAT_COLORS = { N: '#00e5ff', VEB: '#ff1744', SVEB: '#ffab00', F: '#b388ff', Q: '#8b949e' };

function SectionHeader({ title, subtitle }) {
  return (
    <div className="section-header">
      <div className="accent-bar" />
      <div>
        <h2>{title}</h2>
        {subtitle && <div className="subtitle">{subtitle}</div>}
      </div>
    </div>
  );
}

export default function ArrhythmiaTab() {
  const [summary, setSummary] = useState(null);
  const [beatDist, setBeatDist] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [modelResults, setModelResults] = useState(null);
  const [confusion, setConfusion] = useState([]);
  const [ecgPlot, setEcgPlot] = useState(null);
  const [selectedBeat, setSelectedBeat] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [s, bd, p, mr, cm] = await Promise.all([
          api.getArrSummary(),
          api.getArrBeatDist(),
          api.getArrPredictions(500),
          api.getArrModelResults(),
          api.getArrConfusion(),
        ]);
        setSummary(s);
        setBeatDist(bd);
        setPredictions(p);
        setModelResults(mr?.[0] || null);
        setConfusion(cm);
      } catch (err) {
        console.error('Arrhythmia fetch error:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Load ECG plot when beat type selected
  useEffect(() => {
    if (selectedBeat) {
      api.getArrEcgPlot(selectedBeat).then(setEcgPlot).catch(console.error);
    }
  }, [selectedBeat]);

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton" style={{ height: 120 }} />
        ))}
      </div>
    );
  }

  const accuracy = modelResults?.accuracy;
  const report = modelResults?.report;
  const beatTypes = ['N', 'VEB', 'SVEB', 'F', 'Q'];

  // Build confusion matrix data
  const confMatrix = {};
  beatTypes.forEach((t) => { confMatrix[t] = {}; beatTypes.forEach((p) => { confMatrix[t][p] = 0; }); });
  confusion.forEach((r) => {
    if (confMatrix[r.true_label]) confMatrix[r.true_label][r.predicted_label] = r.count;
  });

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Raw Beats', value: summary?.raw_beats, color: '' },
          { label: 'Features', value: summary?.features, color: 'green' },
          { label: 'Predictions', value: summary?.predictions, color: 'amber' },
          { label: 'Accuracy', value: accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '—', color: 'purple' },
        ].map((c, i) => (
          <div key={c.label} className={`metric-card ${c.color} animate-fade-up`} style={{ animationDelay: `${i * 50}ms` }}>
            <div className="text-2xl font-bold" style={{ fontFamily: 'var(--font-mono)' }}>
              {typeof c.value === 'number' ? c.value.toLocaleString() : c.value || '—'}
            </div>
            <div className="mt-1 text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--color-text-secondary)' }}>
              {c.label}
            </div>
          </div>
        ))}
      </div>

      {/* Beat Distribution Chart */}
      <section>
        <SectionHeader title="Beat Distribution" subtitle="Counts by arrhythmia type across all databases" />
        <div className="glass-card p-5" style={{ height: 300 }}>
          <ResponsiveContainer>
            <BarChart data={beatDist} barSize={40}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="beat_type" tick={{ fill: '#8b949e', fontSize: 12 }} />
              <YAxis tick={{ fill: '#484f58', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
                labelStyle={{ color: '#e6edf3' }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {beatDist.map((entry) => (
                  <Cell key={entry.beat_type} fill={BEAT_COLORS[entry.beat_type] || '#8b949e'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Per-class report + Confusion Matrix */}
      <div className="grid grid-cols-2 gap-4">
        {/* Classification Report Table */}
        <section>
          <SectionHeader title="Classification Report" subtitle="Per-class precision, recall, F1" />
          <div className="glass-card overflow-auto" style={{ maxHeight: 320 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {report && beatTypes.map((bt) => {
                  const r = report[bt];
                  if (!r) return null;
                  return (
                    <tr key={bt}>
                      <td>
                        <span className="badge" style={{ background: `${BEAT_COLORS[bt]}22`, color: BEAT_COLORS[bt] }}>
                          {bt}
                        </span>
                      </td>
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

        {/* Confusion Matrix */}
        <section>
          <SectionHeader title="Confusion Matrix" subtitle="True vs Predicted labels" />
          <div className="glass-card p-4 overflow-auto">
            <table className="data-table" style={{ fontSize: 11 }}>
              <thead>
                <tr>
                  <th>True ↓ / Pred →</th>
                  {beatTypes.map((bt) => <th key={bt} style={{ color: BEAT_COLORS[bt], textAlign: 'center' }}>{bt}</th>)}
                </tr>
              </thead>
              <tbody>
                {beatTypes.map((trueLabel) => (
                  <tr key={trueLabel}>
                    <td style={{ color: BEAT_COLORS[trueLabel], fontWeight: 600 }}>{trueLabel}</td>
                    {beatTypes.map((predLabel) => {
                      const val = confMatrix[trueLabel]?.[predLabel] || 0;
                      const isDiag = trueLabel === predLabel;
                      return (
                        <td
                          key={predLabel}
                          style={{
                            textAlign: 'center',
                            fontWeight: isDiag ? 700 : 400,
                            color: isDiag ? 'var(--color-green)' : (val > 0 ? 'var(--color-red)' : 'var(--color-text-muted)'),
                            background: isDiag && val > 0 ? 'var(--color-green-glow)' : undefined,
                          }}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>

      {/* ECG Plot Viewer */}
      <section>
        <SectionHeader title="ECG Beat Viewer" subtitle="Select a beat type to view annotated waveform" />
        <div className="glass-card p-5">
          <div className="flex gap-2 mb-4">
            {beatTypes.map((bt) => (
              <button
                key={bt}
                className={`nav-tab ${selectedBeat === bt ? 'active' : ''}`}
                style={{ padding: '6px 14px', fontSize: 12 }}
                onClick={() => setSelectedBeat(bt)}
              >
                <span style={{ color: BEAT_COLORS[bt] }}>●</span>
                {bt}
              </button>
            ))}
          </div>
          {ecgPlot?.ecg ? (
            <div style={{ height: 200 }}>
              <ResponsiveContainer>
                <LineChart data={ecgPlot.ecg.map((v, i) => ({ i, v }))}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="i" tick={false} />
                  <YAxis domain={['auto', 'auto']} tick={{ fill: '#484f58', fontSize: 10 }} />
                  <Line type="monotone" dataKey="v" stroke="#00e5ff" dot={false} strokeWidth={1.5} />
                  <Tooltip
                    contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex items-center justify-center py-10" style={{ color: 'var(--color-text-muted)' }}>
              {selectedBeat ? 'Loading waveform…' : 'Click a beat type above to visualize'}
            </div>
          )}
        </div>
      </section>

      {/* Predictions Table */}
      <section>
        <SectionHeader title="Predictions" subtitle={`Showing ${predictions.length} predictions with confidence scores`} />
        <div className="glass-card overflow-auto" style={{ maxHeight: 400 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Beat Ref</th>
                <th>True Label</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>Match</th>
              </tr>
            </thead>
            <tbody>
              {predictions.slice(0, 100).map((p, i) => {
                const match = p.true_label === p.predicted_label;
                return (
                  <tr key={i}>
                    <td style={{ fontSize: 10 }}>{p.beat_ref}</td>
                    <td>
                      <span className="badge" style={{ background: `${BEAT_COLORS[p.true_label]}22`, color: BEAT_COLORS[p.true_label] }}>
                        {p.true_label}
                      </span>
                    </td>
                    <td>
                      <span className="badge" style={{ background: `${BEAT_COLORS[p.predicted_label]}22`, color: BEAT_COLORS[p.predicted_label] }}>
                        {p.predicted_label}
                      </span>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="conf-bar" style={{ width: 60 }}>
                          <div
                            className="fill"
                            style={{
                              width: `${(p.confidence * 100).toFixed(0)}%`,
                              background: match ? 'var(--color-green)' : 'var(--color-red)',
                            }}
                          />
                        </div>
                        <span style={{ fontSize: 11 }}>{(p.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${match ? 'green' : 'red'}`}>
                        {match ? '✓' : '✗'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
