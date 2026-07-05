import React from 'react';
import { sevBg, sevC } from '../utils/format';

export const CS = {
  contentStyle: { background: '#0c1526', border: '1px solid #2d3f5c', borderRadius: 8, fontSize: 11 },
  labelStyle: { color: '#94a3b8', fontSize: 10 }
};

export function Card({ children, style = {} }) {
  return (
    <div style={{ background: '#0c1526', border: '1px solid #1a2d4d', borderRadius: 10, padding: '14px 18px', ...style }}>
      {children}
    </div>
  );
}

export function Lbl({ children }) {
  return (
    <div style={{ color: '#475569', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>
      {children}
    </div>
  );
}

export function Empty({ msg }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: 200, flexDirection: 'column', gap: 10, color: '#1e3a5f', border: '1px dashed #1a2d4d', borderRadius: 10, margin: 20 }}>
      <span style={{ fontSize: 28 }}>📂</span>
      <span style={{ fontSize: 12 }}>{msg || 'File not loaded'}</span>
    </div>
  );
}

export function St({ label, value, unit, color, mono }) {
  return (
    <div style={{ background: '#0c1526', border: '1px solid #1a2d4d', borderRadius: 8, padding: '12px 16px', flex: 1, minWidth: 90 }}>
      <div style={{ color: '#334155', fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 5 }}>
        {label}
      </div>
      <div style={{ color: color || '#e2e8f0', fontSize: 22, fontWeight: 700, fontFamily: mono ? 'JetBrains Mono' : 'inherit', lineHeight: 1 }}>
        {value ?? '—'}
        {unit && value != null && <span style={{ fontSize: 11, fontWeight: 400, color: '#334155', marginLeft: 3 }}>{unit}</span>}
      </div>
    </div>
  );
}

export function SevBadge({ s }) {
  if (!s) return null;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, background: sevBg(s), color: sevC(s), border: `1px solid ${sevC(s)}40`, borderRadius: 20, padding: '4px 14px', fontSize: 13, fontWeight: 700 }}>
      <span style={{ width: 7, height: 7, borderRadius: '50%', background: sevC(s) }} />
      {s || '—'}
    </span>
  );
}

const TABS = [
  { icon: '📊', label: 'Overview' },
  { icon: '💓', label: 'ECG Viewer' },
  { icon: '🌙', label: 'Sleep Filter' },
  { icon: '⚡', label: 'Models' },
  { icon: '📋', label: 'Segments' }
];

export function TabBar({ active, onSelect }) {
  return (
    <div style={{ display: 'flex', background: '#08111f', borderBottom: '1px solid #1a2d4d', padding: '0 14px', flexShrink: 0 }}>
      {TABS.map((t, i) => (
        <button
          key={i}
          onClick={() => onSelect(i)}
          style={{
            background: 'none',
            border: 'none',
            borderBottom: `2px solid ${active === i ? '#3b82f6' : 'transparent'}`,
            color: active === i ? '#60a5fa' : '#334155',
            padding: '9px 14px',
            fontSize: 11,
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 5,
            transition: 'all 0.15s',
            whiteSpace: 'nowrap'
          }}
        >
          <span>{t.icon}</span>
          <span>{t.label}</span>
        </button>
      ))}
    </div>
  );
}
