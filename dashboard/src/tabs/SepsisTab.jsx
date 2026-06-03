import { useState, useEffect } from 'react';
import { api } from '../api';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend,
} from 'recharts';

function SectionHeader({ title, subtitle }) {
  return (
    <div className="section-header">
      <div className="accent-bar" style={{ background: 'var(--color-red)' }} />
      <div>
        <h2>{title}</h2>
        {subtitle && <div className="subtitle">{subtitle}</div>}
      </div>
    </div>
  );
}

export default function SepsisTab() {
  const [summary, setSummary] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [modelResults, setModelResults] = useState(null);
  const [riskDist, setRiskDist] = useState([]);
  const [vitals, setVitals] = useState(null);
  const [featImp, setFeatImp] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [s, p, mr, rd, vp, fi] = await Promise.all([
          api.getSepsisSummary(),
          api.getSepsisPredictions(300),
          api.getSepsisModelResults(),
          api.getSepsisRiskDist(),
          api.getSepsisVitalsPlot(),
          api.getSepsisFeatImp(),
        ]);
        setSummary(s);
        setPredictions(p);
        setModelResults(mr?.[0] || null);
        setRiskDist(rd);
        setVitals(vp);
        setFeatImp(fi.slice(0, 15));
      } catch (err) {
        console.error('Sepsis fetch error:', err);
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

  const accuracy = modelResults?.accuracy;
  const aucRoc = modelResults?.auc_roc;
  const report = modelResults?.report;

  // Build vitals chart data
  const vitalsData = vitals?.hr_series
    ? vitals.hr_series.map((_, i) => ({
        hour: i,
        HR: vitals.hr_series?.[i],
        SpO2: vitals.spo2_series?.[i],
        BP: vitals.bp_series?.[i],
        Temp: vitals.temp_series?.[i],
        RR: vitals.rr_series?.[i],
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-5 gap-3">
        {[
          { label: 'Total Patients', value: summary?.total_patients, color: '' },
          { label: 'Sepsis Cases', value: summary?.sepsis_patients, color: 'red' },
          { label: 'Prevalence', value: summary?.prevalence_pct != null ? `${summary.prevalence_pct}%` : '—', color: 'amber' },
          { label: 'Accuracy', value: accuracy != null ? `${(accuracy * 100).toFixed(1)}%` : '—', color: 'green' },
          { label: 'AUC-ROC', value: aucRoc != null ? aucRoc.toFixed(3) : '—', color: 'purple' },
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

      {/* Risk Distribution + Feature Importance */}
      <div className="grid grid-cols-2 gap-4">
        <section>
          <SectionHeader title="Risk Distribution" subtitle="Confidence score buckets by true label" />
          <div className="glass-card p-5" style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={riskDist} barSize={20}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="range" tick={{ fill: '#484f58', fontSize: 10 }} />
                <YAxis tick={{ fill: '#484f58', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
                />
                <Legend wrapperStyle={{ fontFamily: 'IBM Plex Sans', fontSize: 11 }} />
                <Bar dataKey="no_sepsis" stackId="a" fill="#00e676" name="No Sepsis" radius={[0, 0, 0, 0]} />
                <Bar dataKey="sepsis" stackId="a" fill="#ff1744" name="Sepsis" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section>
          <SectionHeader title="Feature Importance" subtitle="Top discriminative features" />
          <div className="glass-card p-5" style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={featImp} layout="vertical" barSize={12}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#484f58', fontSize: 10 }} />
                <YAxis dataKey="feature" type="category" width={160} tick={{ fill: '#8b949e', fontSize: 9 }} />
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
                />
                <Bar dataKey="importance" fill="#ff1744" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* Vitals Time-Series */}
      {vitalsData.length > 0 && (
        <section>
          <SectionHeader
            title="Patient Vitals"
            subtitle={`Subject ${vitals?.subject_id || '?'} — 24hr time-series`}
          />
          <div className="glass-card p-5" style={{ height: 280 }}>
            <ResponsiveContainer>
              <LineChart data={vitalsData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="hour" tick={{ fill: '#484f58', fontSize: 10 }} label={{ value: 'Hour', position: 'insideBottom', offset: -5, fill: '#484f58' }} />
                <YAxis tick={{ fill: '#484f58', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, fontFamily: 'IBM Plex Mono' }}
                />
                <Legend wrapperStyle={{ fontFamily: 'IBM Plex Sans', fontSize: 11 }} />
                <Line type="monotone" dataKey="HR" stroke="#ff1744" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="SpO2" stroke="#00e5ff" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="BP" stroke="#ffab00" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="RR" stroke="#00e676" dot={false} strokeWidth={1} strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* Classification Report */}
      {report && (
        <section>
          <SectionHeader title="Model Report" subtitle={`Accuracy: ${(accuracy * 100).toFixed(1)}% | AUC: ${aucRoc?.toFixed(4)}`} />
          <div className="glass-card p-5 overflow-auto">
            <table className="data-table">
              <thead>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
              </thead>
              <tbody>
                {['No Sepsis', 'Sepsis'].map((cls) => {
                  const r = report[cls];
                  if (!r) return null;
                  return (
                    <tr key={cls}>
                      <td><span className={`badge ${cls === 'Sepsis' ? 'red' : 'green'}`}>{cls}</span></td>
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

      {/* Predictions Table */}
      <section>
        <SectionHeader title="Predictions" subtitle={`${predictions.length} predictions sorted by confidence`} />
        <div className="glass-card overflow-auto" style={{ maxHeight: 400 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Subject ID</th>
                <th>True Label</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>Match</th>
              </tr>
            </thead>
            <tbody>
              {predictions.slice(0, 100).map((p, i) => {
                const trueLbl = String(p.true_label);
                const predLbl = String(p.predicted_label);
                const match = trueLbl === predLbl;
                const isSepsis = predLbl === '1';
                return (
                  <tr key={i}>
                    <td>{p.subject_id}</td>
                    <td>
                      <span className={`badge ${trueLbl === '1' ? 'red' : 'green'}`}>
                        {trueLbl === '1' ? 'Sepsis' : 'No Sepsis'}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${isSepsis ? 'red' : 'green'}`}>
                        {isSepsis ? 'Sepsis' : 'No Sepsis'}
                      </span>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="conf-bar" style={{ width: 60 }}>
                          <div
                            className="fill"
                            style={{
                              width: `${(p.confidence * 100).toFixed(0)}%`,
                              background: isSepsis ? 'var(--color-red)' : 'var(--color-green)',
                            }}
                          />
                        </div>
                        <span style={{ fontSize: 11 }}>{(p.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge ${match ? 'green' : 'red'}`}>{match ? '✓' : '✗'}</span>
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
