export default function OverviewTab({ summary, loading }) {
  const cards = [
    { label: 'Arrhythmia Beats',       value: summary?.arrhythmia_beats,       color: 'cyan',   icon: '♡' },
    { label: 'Arrhythmia Predictions', value: summary?.arrhythmia_predictions, color: 'green',  icon: '△' },
    { label: 'Apnea Segments',         value: summary?.apnea_segments,         color: 'amber',  icon: '◎' },
    { label: 'Sepsis Patients',        value: summary?.sepsis_patients,        color: 'red',    icon: '⚕' },
    { label: 'Sepsis Predictions',     value: summary?.sepsis_predictions,     color: 'purple', icon: '◇' },
    { label: 'Pipeline Stages',        value: summary?.pipeline_stages_run,    color: 'cyan',   icon: '▤' },
  ];

  return (
    <div className="space-y-5">
      {/* Stat cards grid */}
      <div className="grid grid-cols-3 gap-3">
        {cards.map((c, i) => (
          <div
            key={c.label}
            className={`metric-card ${c.color} animate-fade-up`}
            style={{ animationDelay: `${i * 60}ms` }}
          >
            {loading ? (
              <>
                <div className="skeleton" style={{ width: 60, height: 32, marginBottom: 8 }} />
                <div className="skeleton" style={{ width: 120, height: 12 }} />
              </>
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <div
                    className="text-3xl font-bold tracking-tight"
                    style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)' }}
                  >
                    {c.value != null ? c.value.toLocaleString() : '—'}
                  </div>
                  <span style={{ fontSize: '24px', opacity: 0.3 }}>{c.icon}</span>
                </div>
                <div
                  className="mt-1 text-xs font-medium uppercase tracking-wider"
                  style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-secondary)' }}
                >
                  {c.label}
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Module status cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          {
            title: 'Arrhythmia Module',
            desc: 'Beat-level ECG classification: N, VEB, SVEB, F, Q',
            color: 'var(--color-cyan)',
            dataSource: 'MIT-BIH / INCART / SCD-Holter CSVs',
            model: 'RandomForestClassifier',
          },
          {
            title: 'Apnea Module',
            desc: 'Sleep apnea detection with AASM multi-signal scoring',
            color: 'var(--color-amber)',
            dataSource: 'MIMIC-IV Waveform (PhysioNet)',
            model: 'Bidirectional LSTM',
          },
          {
            title: 'Sepsis Module',
            desc: 'ICU sepsis early warning from clinical vitals',
            color: 'var(--color-red)',
            dataSource: 'sepsis_icu_synthetic.csv',
            model: 'GradientBoostingClassifier',
          },
        ].map((mod, i) => (
          <div
            key={mod.title}
            className="glass-card p-5 animate-fade-up"
            style={{ animationDelay: `${(i + 6) * 60}ms` }}
          >
            <div className="flex items-center gap-2 mb-3">
              <div
                className="w-2 h-2 rounded-full"
                style={{ background: mod.color }}
              />
              <h3
                className="text-sm font-semibold"
                style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-primary)' }}
              >
                {mod.title}
              </h3>
            </div>
            <p
              className="text-xs mb-3"
              style={{ color: 'var(--color-text-secondary)', lineHeight: 1.5 }}
            >
              {mod.desc}
            </p>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-medium uppercase" style={{ color: 'var(--color-text-muted)', width: 50 }}>
                  Data
                </span>
                <span
                  className="text-[11px]"
                  style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)' }}
                >
                  {mod.dataSource}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-medium uppercase" style={{ color: 'var(--color-text-muted)', width: 50 }}>
                  Model
                </span>
                <span className="badge cyan">{mod.model}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
