import { useState, useEffect } from 'react';
import { api } from '../api';

const STATUS_COLORS = {
  started: 'var(--color-cyan)',
  done: 'var(--color-green)',
  failed: 'var(--color-red)',
  skipped: 'var(--color-amber)',
};

const STATUS_BG = {
  started: 'var(--color-cyan-glow)',
  done: 'var(--color-green-glow)',
  failed: 'var(--color-red-glow)',
  skipped: 'var(--color-amber-glow)',
};

export default function PipelineLogTab() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    (async () => {
      try {
        const data = await api.getPipelineLog();
        setLogs(Array.isArray(data) ? data : []);
      } catch (err) {
        console.error('Pipeline log error:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const filtered = filter === 'all'
    ? logs
    : logs.filter((l) => l.module === filter);

  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3, 4, 5].map((i) => <div key={i} className="skeleton" style={{ height: 48 }} />)}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Module filter buttons */}
      <div className="flex gap-2">
        {['all', 'arrhythmia', 'apnea', 'sepsis'].map((m) => (
          <button
            key={m}
            className={`nav-tab ${filter === m ? 'active' : ''}`}
            style={{ padding: '6px 14px', fontSize: 12 }}
            onClick={() => setFilter(m)}
          >
            {m === 'all' ? 'All Modules' : m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      {/* Timeline */}
      <div className="glass-card overflow-auto" style={{ maxHeight: 600 }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Module</th>
              <th>Stage</th>
              <th>Status</th>
              <th>Message</th>
              <th>Rows</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((log) => (
              <tr key={log.id}>
                <td style={{ color: 'var(--color-text-muted)', fontSize: 10 }}>{log.id}</td>
                <td>
                  <span
                    className="badge cyan"
                    style={{
                      background: log.module === 'arrhythmia' ? 'var(--color-cyan-glow)' :
                                  log.module === 'apnea' ? 'var(--color-amber-glow)' :
                                  'var(--color-red-glow)',
                      color: log.module === 'arrhythmia' ? 'var(--color-cyan)' :
                             log.module === 'apnea' ? 'var(--color-amber)' :
                             'var(--color-red)',
                    }}
                  >
                    {log.module}
                  </span>
                </td>
                <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 500 }}>{log.stage}</td>
                <td>
                  <span
                    className="badge"
                    style={{
                      background: STATUS_BG[log.status] || 'var(--color-cyan-glow)',
                      color: STATUS_COLORS[log.status] || 'var(--color-cyan)',
                    }}
                  >
                    {log.status === 'done' ? '✓' : log.status === 'failed' ? '✗' : '●'} {log.status}
                  </span>
                </td>
                <td
                  style={{
                    maxWidth: 200,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    fontSize: 11,
                    color: 'var(--color-text-muted)',
                  }}
                  title={log.message}
                >
                  {log.message || '—'}
                </td>
                <td style={{ fontWeight: 600 }}>{log.rows_written || 0}</td>
                <td style={{ fontSize: 10, color: 'var(--color-text-muted)' }}>{log.ts}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Stats */}
      <div className="flex gap-4">
        <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          Total entries: <strong style={{ color: 'var(--color-text-secondary)' }}>{logs.length}</strong>
        </div>
        <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          Showing: <strong style={{ color: 'var(--color-text-secondary)' }}>{filtered.length}</strong>
        </div>
      </div>
    </div>
  );
}
