import React, { useState, useCallback } from 'react';
import Papa from 'papaparse';


export const FDEFS = [
  { key: 'segments', label: 'ADM_segments.csv', file: 'ADM_segments.csv' },
  { key: 'inferXgb', label: 'infer_results_ADM.csv', file: 'infer_results_ADM.csv' },
  { key: 'inferBL', label: 'bilstm_infer_results.csv', file: 'bilstm_infer_results.csv' },
  { key: 'sleep', label: 'sleep_windows.csv', file: 'sleep_windows.csv' },
  { key: 'summary', label: 'infer_summary.csv', file: 'infer_summary.csv' },
];

const WSRC = `importScripts('https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js');
self.onmessage=function(e){
  const{file,ftype}=e.data; const rows=[]; const ecgMap=Object.create(null); const spo2Map=Object.create(null); let count=0;
  Papa.parse(file,{header:true,dynamicTyping:true,skipEmptyLines:true,
    step:function(r){const row=r.data;count++;
      if(ftype==='segments'){
        const ecg=new Float32Array(3750);
        for(let i=0;i<3750;i++){ecg[i]=row['ecgData['+i+']']||0;delete row['ecgData['+i+']'];}
        ecgMap[String(row.segment_idx)]=ecg;
        if(row['spo2Data[0]']!==undefined){
          const spo2=new Float32Array(30);
          for(let i=0;i<30;i++){spo2[i]=row['spo2Data['+i+']']||0;delete row['spo2Data['+i+']'];}
          spo2Map[String(row.segment_idx)]=spo2;
        }
        rows.push(row);
        if(count%100===0)self.postMessage({type:'progress',n:count});
      }else{rows.push(row);}
    },
    complete:function(){
      if(ftype==='segments'){
        const bufs=Object.values(ecgMap).map(a=>a.buffer).concat(Object.values(spo2Map).map(a=>a.buffer));
        self.postMessage({type:'done',rows,ecgMap,spo2Map},bufs);
      }else{self.postMessage({type:'done',rows});}
    },
    error:function(err){self.postMessage({type:'err',msg:err.message});}
  });
};`;

export function parseCSV(file, ftype, onProg) {
  return new Promise((res, rej) => {
    let worker;
    try {
      const blob = new Blob([WSRC], { type: 'application/javascript' });
      worker = new Worker(URL.createObjectURL(blob));
    } catch (_) {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: r => {
          let ecgMap = null;
          let spo2Map = null;
          if (ftype === 'segments') {
            ecgMap = Object.create(null);
            spo2Map = Object.create(null);
            r.data.forEach(row => {
              const ecg = new Float32Array(3750);
              for (let i = 0; i < 3750; i++) {
                ecg[i] = row['ecgData[' + i + ']'] || 0;
                delete row['ecgData[' + i + ']'];
              }
              ecgMap[String(row.segment_idx)] = ecg;
              if (row['spo2Data[0]'] !== undefined) {
                const spo2 = new Float32Array(30);
                for (let i = 0; i < 30; i++) {
                  spo2[i] = row['spo2Data[' + i + ']'] || 0;
                  delete row['spo2Data[' + i + ']'];
                }
                spo2Map[String(row.segment_idx)] = spo2;
              }
            });
          }
          res({ rows: r.data, ecgMap, spo2Map });
        },
        error: e => rej(e)
      });
      return;
    }
    worker.onmessage = e => {
      if (e.data.type === 'progress') onProg && onProg(e.data.n);
      else if (e.data.type === 'done') { worker.terminate(); res(e.data); }
      else if (e.data.type === 'err') { worker.terminate(); rej(new Error(e.data.msg)); }
    };
    worker.onerror = e => { worker.terminate(); rej(e); };
    worker.postMessage({ file, ftype });
  });
}

export function UploadZone({ st, pg, handle }) {
  const [drag, setDrag] = useState(false);
  
  const onDrop = useCallback(e => {
    e.preventDefault();
    setDrag(false);
    Array.from(e.dataTransfer.files).forEach(f => {
      const n = f.name.toLowerCase();
      if (n.includes('segment')) handle(f, 'segments');
      else if (n.includes('summary')) handle(f, 'summary');
      else if (n.includes('sleep')) handle(f, 'sleep');
      else if (n.includes('bilstm') || n.includes('bi_lstm')) handle(f, 'inferBL');
      else if (n.includes('infer') || n.includes('result')) handle(f, 'inferXgb');
    });
  }, [handle]);
  
  return (
    <div
      onDrop={onDrop}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      style={{
        background: drag ? '#0d1e36' : '#08111f',
        borderBottom: '1px solid #1a2d4d',
        padding: '9px 18px',
        display: 'flex',
        gap: 8,
        flexWrap: 'wrap',
        alignItems: 'center',
        transition: 'background 0.2s',
        flexShrink: 0
      }}
    >
      <span style={{ color: '#1e3a5f', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginRight: 4 }}>
        CSV FILES
      </span>
      {FDEFS.map(({ key, label }) => {
        const s = st[key];
        const col = s === 'ok' ? '#22c55e' : s === 'loading' ? '#3b82f6' : s === 'err' ? '#ef4444' : '#1e3a5f';
        return (
          <label key={key} style={{ cursor: 'pointer' }}>
            <input
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={e => e.target.files[0] && handle(e.target.files[0], key)}
            />
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 5,
              background: col + '12',
              border: `1px solid ${col}30`,
              borderRadius: 6,
              padding: '5px 10px',
              fontSize: 11,
              color: col,
              whiteSpace: 'nowrap'
            }}>
              {s === 'loading' ? <span className="spin">⟳</span> : s === 'ok' ? '✓' : s === 'err' ? '✗' : '+'}
              <span>{label}</span>
              {s === 'loading' && pg[key] > 0 ? <span style={{ color: col + '80', fontSize: 9 }}>{pg[key]}</span> : null}
            </div>
          </label>
        );
      })}
      {drag && <span style={{ color: '#3b82f6', fontSize: 11, marginLeft: 4 }}>Drop CSVs here…</span>}
    </div>
  );
}
