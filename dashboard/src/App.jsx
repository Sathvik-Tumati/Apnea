import { useState, useEffect, useCallback } from 'react';
import { api } from './api';
import OverviewTab from './tabs/OverviewTab';
import ArrhythmiaTab from './tabs/ArrhythmiaTab';
import ApneaTab from './tabs/ApneaTab';
import SepsisTab from './tabs/SepsisTab';
import PipelineLogTab from './tabs/PipelineLogTab';

const TABS = [
  { id: 'overview',   label: 'Overview',    icon: '◈' },
  { id: 'arrhythmia', label: 'Arrhythmia',  icon: '♡' },
  { id: 'apnea',      label: 'Apnea',       icon: '◎' },
  { id: 'sepsis',     label: 'Sepsis',      icon: '⚕' },
  { id: 'logs',       label: 'Pipeline Log', icon: '▤' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);

  const fetchSummary = useCallback(async () => {
    try {
      const data = await api.getSummary();
      setSummary(data);
    } catch (err) {
      console.error('Failed to fetch summary:', err);
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSummary();
    const interval = setInterval(fetchSummary, 15000);
    return () => clearInterval(interval);
  }, [fetchSummary]);

  return (
    <div className="flex min-h-screen" style={{ background: 'var(--color-bg-primary)' }}>
      {/* Sidebar Navigation */}
      <aside
        className="flex flex-col w-[220px] min-h-screen border-r"
        style={{
          background: 'var(--color-bg-sidebar)',
          borderColor: 'var(--color-border)',
        }}
      >
        {/* Logo */}
        <div className="px-5 py-5 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-2.5">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect width="28" height="28" rx="7" fill="var(--color-cyan-soft)" stroke="var(--color-cyan)" strokeWidth="1" />
              <polyline
                points="4,14 8,14 11,6 14,22 17,10 20,18 24,14"
                stroke="var(--color-cyan)"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </svg>
            <div>
              <div
                className="text-sm font-semibold"
                style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-primary)' }}
              >
                VitalSign ML
              </div>
              <div
                className="text-[10px]"
                style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-muted)' }}
              >
                v2.0 Dashboard
              </div>
            </div>
          </div>
        </div>

        {/* Nav tabs */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`nav-tab w-full ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span style={{ fontSize: '16px', lineHeight: 1 }}>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Status footer */}
        <div className="px-4 py-3 border-t" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full"
              style={{
                background: summary ? 'var(--color-green)' : 'var(--color-amber)',
                animation: 'glow-pulse 2s ease-in-out infinite',
              }}
            />
            <span
              className="text-[10px] font-medium"
              style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-muted)' }}
            >
              {summary ? 'Pipeline Connected' : 'Connecting…'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 min-w-0 overflow-y-auto h-screen">
        <div className="max-w-[1400px] mx-auto px-6 py-5 space-y-5">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1
                className="text-lg font-semibold"
                style={{ fontFamily: 'var(--font-sans)', color: 'var(--color-text-primary)' }}
              >
                {TABS.find((t) => t.id === activeTab)?.label || 'Dashboard'}
              </h1>
              <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
                Clinical Vital Signs ML Pipeline — Three-Module Analysis
              </p>
            </div>
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-md"
              style={{
                background: 'var(--color-cyan-soft)',
                border: '1px solid var(--color-border)',
              }}
            >
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: 'var(--color-cyan)', animation: 'glow-pulse 2s ease-in-out infinite' }}
              />
              <span
                className="text-xs font-medium"
                style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-cyan)' }}
              >
                {summary ? `${summary.pipeline_stages_run || 0} stages` : '…'}
              </span>
            </div>
          </div>

          {/* Tab content */}
          {activeTab === 'overview'   && <OverviewTab summary={summary} loading={summaryLoading} />}
          {activeTab === 'arrhythmia' && <ArrhythmiaTab />}
          {activeTab === 'apnea'      && <ApneaTab />}
          {activeTab === 'sepsis'     && <SepsisTab />}
          {activeTab === 'logs'       && <PipelineLogTab />}
        </div>
      </main>
    </div>
  );
}
