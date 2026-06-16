"""
evaluate_edr_v2_vs_fixed.py
────────────────────────────
Same as evaluate_edr_final_v2.py but replaces the "Old" column with
compute_edr_v2 (the fixed version) so you can confirm it beats the original.

Run:  python3 evaluate_edr_v2_vs_fixed.py
"""

import os, sys
import numpy as np
import wfdb
from scipy.signal import resample, welch, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

try:
    import neurokit2 as nk
    HAS_NK = True
except ImportError:
    HAS_NK = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import (_bandpass, _compute_edr, FS_ECG, FS_RESP, SEGMENT_LEN_S)

# ── Import the fixed EDR ──────────────────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from compute_edr_fixed import compute_edr_v3

VERIFIED_RECORDS = [
    ("83404654", "mimic4wdb/0.1.0/waves/p100/p10020306/83404654"),
    ("85583557", "mimic4wdb/0.1.0/waves/p100/p10039708/85583557"),
    ("85594648", "mimic4wdb/0.1.0/waves/p100/p10079700/85594648"),
    ("83268087", "mimic4wdb/0.1.0/waves/p101/p10100546/83268087"),
    ("83411188", "mimic4wdb/0.1.0/waves/p100/p10039708/83411188"),
]

RESP_LO = 0.1; RESP_HI = 0.6; FS_OUT = 10
MIN_HR = 30;   MAX_HR = 200;   QRS_HALF_MS = 60
RESP_SNR_MIN = 2.0; ECG_SNR_MIN = 1.5

# ── Utilities ─────────────────────────────────────────────────────────────────

def _bp(sig, fs, lo, hi, order=3):
    nyq = fs/2.; hi=min(hi,nyq-0.05); lo=max(lo,0.01)
    if lo>=hi: return sig
    b,a=butter(order,[lo/nyq,hi/nyq],btype='band')
    padlen=3*max(len(a),len(b))
    return filtfilt(b,a,sig) if len(sig)>padlen else sig

def _dominant_bpm(sig, fs, lo=RESP_LO, hi=RESP_HI):
    if sig is None or len(sig)<8 or np.std(sig)<1e-10: return 0., 0.
    w=tukey(len(sig),alpha=0.1)
    nperseg=min(len(sig),max(8,int(fs*20)))
    try: f,pxx=welch(sig*w,fs=fs,nperseg=nperseg,noverlap=nperseg//2,window='hann',nfft=4096)
    except: return 0.,0.
    inn=(f>=lo)&(f<=hi); out_=(~inn)&(f>0)
    if not np.any(inn) or np.all(pxx[inn]==0): return 0.,0.
    peak_freq=float(f[inn][np.argmax(pxx[inn])])
    peak_pwr=float(pxx[inn].max())
    snr=peak_pwr/(float(np.mean(pxx[out_]))+1e-12) if np.any(out_) else 0.
    return peak_freq*60., snr

def _bpm(sig, fs):
    if sig is None or len(sig)<8 or np.std(sig)<1e-10: return 0.
    bpm,_=_dominant_bpm(sig,fs); return bpm

def _resp_snr(sig, fs):
    _,snr=_dominant_bpm(_bp(sig,fs,RESP_LO,RESP_HI),fs); return snr

def _ecg_snr(ecg, fs):
    sq=_bp(ecg,fs,5.,20.)**2
    nperseg=min(len(sq),max(8,int(fs*10)))
    try: f,pxx=welch(sq,fs=fs,nperseg=nperseg,noverlap=nperseg//2)
    except: return 0.
    inn=(f>=5)&(f<=20); out_=~inn&(f>0)
    if not np.any(inn) or not np.any(out_): return 0.
    return float(pxx[inn].mean())/(float(pxx[out_].mean())+1e-12)

def _nms_r_peaks(ecg, fs):
    n=len(ecg)
    if n<int(fs*0.5): return np.array([],dtype=int)
    min_dist=int(60./MAX_HR*fs); expected=int(n/fs*75/60)
    ecg_bp=_bp(ecg,fs,5.,20.)
    if np.sum(np.clip(ecg_bp,None,0)**2)>np.sum(np.clip(ecg_bp,0,None)**2)*1.5:
        ecg_bp=-ecg_bp
    mwi=np.convolve(ecg_bp**2,np.ones(max(1,int(0.15*fs)))/max(1,int(0.15*fs)),'same')
    med=np.median(mwi); mad=np.median(np.abs(mwi-med)); thresh=med+1.5*mad
    if thresh<1e-12:
        nz=mwi[mwi>0]; thresh=np.percentile(nz,10) if len(nz) else 1e-12
    nms_win=max(min_dist,int(0.2*fs)); r_peaks=[]
    for s in range(0,n,nms_win):
        e=min(n,s+nms_win); idx=s+int(np.argmax(mwi[s:e]))
        if mwi[idx]>=thresh: r_peaks.append(idx)
    r_peaks=np.array(r_peaks,dtype=int)
    if len(r_peaks)>1:
        merged=[r_peaks[0]]
        for pk in r_peaks[1:]:
            if pk-merged[-1]<min_dist: merged[-1]=pk if ecg[pk]>ecg[merged[-1]] else merged[-1]
            else: merged.append(pk)
        r_peaks=np.array(merged,dtype=int)
    if len(r_peaks)>1:
        rr=np.diff(r_peaks)
        valid=np.concatenate([[True],(rr>=min_dist)&(rr<=int(60./MIN_HR*fs))])
        r_peaks=r_peaks[valid]
    if len(r_peaks)<max(6,expected//3):
        from scipy.signal import find_peaks
        pks,_=find_peaks(ecg_bp,height=np.percentile(ecg_bp,75),distance=min_dist)
        if len(pks)>len(r_peaks): r_peaks=pks
    refine=max(1,int(0.005*fs))
    return np.array([max(0,pk-refine)+int(np.argmax(np.abs(ecg[max(0,pk-refine):min(n,pk+refine)])))
                     for pk in r_peaks],dtype=int)

def _get_r_peaks(ecg, fs):
    if HAS_NK:
        try:
            _,info=nk.ecg_process(ecg,sampling_rate=fs)
            return np.array(info["ECG_R_Peaks"],dtype=int)
        except: pass
    return _nms_r_peaks(ecg,fs)

# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate():
    print("="*72)
    print(" EDR Comparison: Original _compute_edr  vs  compute_edr_v2 (fixed)")
    print("="*72)
    print(f"  Records: {len(VERIFIED_RECORDS)}  |  SEGMENT_LEN_S={SEGMENT_LEN_S}s  "
          f"FS_ECG={FS_ECG}  FS_RESP={FS_RESP}\n")

    total=0; mae_old=0.; mae_new=0.; per_rec=[]

    for rname,pn_dir in VERIFIED_RECORDS:
        try: rec=wfdb.rdrecord(rname,pn_dir=pn_dir,sampto=96000)
        except Exception as e: print(f"  {rname}: {e}"); continue

        smap={n:i for i,n in enumerate(rec.sig_name)}
        if "II" not in smap or "Resp" not in smap: continue

        fs0=rec.fs
        ecg_=rec.p_signal[:,smap["II"]]; rsp_=rec.p_signal[:,smap["Resp"]]
        for a in (ecg_,rsp_):
            m=np.isnan(a); a[m]=np.nanmean(a) if not m.all() else 0.

        ecg_f=resample(ecg_,int(len(ecg_)*FS_ECG/fs0))
        rsp_f=resample(rsp_,int(len(rsp_)*FS_RESP/fs0))

        se=FS_ECG*SEGMENT_LEN_S; sr=FS_RESP*SEGMENT_LEN_S
        ns=min(len(ecg_f)//se,10)

        r_old=[]; r_new=[]
        print(f"Record: {rname}  —  {ns} segs")
        print(f"  {'Sg':>2}  {'GT':>6}  {'Old-est':>8}  {'OldErr':>7}  "
              f"{'New-est':>8}  {'NewErr':>7}  {'NewSNR':>7}  {'Winner':>6}")
        print(f"  {'--':>2}  {'------':>6}  {'-------':>8}  {'-------':>7}  "
              f"{'-------':>8}  {'-------':>7}  {'------':>7}  {'------':>6}")

        for i in range(ns):
            ecg_seg=_bandpass(ecg_f[i*se:(i+1)*se],FS_ECG)
            rsp_seg=rsp_f[i*sr:(i+1)*sr]

            rsnr=_resp_snr(rsp_seg,FS_RESP)
            esnr=_ecg_snr(ecg_seg,FS_ECG)
            gt,_=_dominant_bpm(_bp(rsp_seg,FS_RESP,RESP_LO,RESP_HI),FS_RESP)

            if gt<4. or rsnr<RESP_SNR_MIN or esnr<ECG_SNR_MIN:
                print(f"  {i+1:>2}  SKIP — GT={gt:.0f} RSNR={rsnr:.1f} ESNR={esnr:.1f}")
                continue

            r_peaks=_get_r_peaks(ecg_seg,FS_ECG)

            # Original
            old_sig=_compute_edr(ecg_seg,r_peaks,FS_ECG,FS_RESP)
            old_est=_bpm(old_sig,FS_RESP)

            # Fixed v2
            new_sig, _, new_snr=compute_edr_v3(ecg_seg,r_peaks,FS_ECG,FS_RESP,SEGMENT_LEN_S)
            new_est=_bpm(new_sig,FS_RESP)

            err_old=abs(gt-old_est); err_new=abs(gt-new_est)
            mae_old+=err_old; mae_new+=err_new
            r_old.append(err_old); r_new.append(err_new)
            total+=1

            winner="NEW ✓" if err_new<err_old else ("TIE" if err_new==err_old else "old")
            print(f"  {i+1:>2}  {gt:>6.1f}  {old_est:>8.1f}  {err_old:>7.1f}  "
                  f"{new_est:>8.1f}  {err_new:>7.1f}  {new_snr:>7.1f}  {winner:>6}")

        per_rec.append({'r':rname,'n':len(r_old),
                        'old':np.mean(r_old) if r_old else 0,
                        'new':np.mean(r_new) if r_new else 0})
        print()

    if total==0: print("No valid segments."); return

    mm_old=mae_old/total; mm_new=mae_new/total
    improv=(mm_old-mm_new)/mm_old*100 if mm_old>0 else 0

    print("="*72)
    print(" PER-RECORD")
    print("="*72)
    print(f"  {'Record':<15}  {'N':>3}  {'Old MAE':>8}  {'New MAE':>8}  {'Delta':>8}")
    print(f"  {'-'*15}  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in per_rec:
        d=r['old']-r['new']
        print(f"  {r['r']:<15}  {r['n']:>3}  {r['old']:>8.2f}  {r['new']:>8.2f}  "
              f"  {'▼'+f'{d:.2f}' if d>0 else '▲'+f'{abs(d):.2f}'}")

    print(f"\n{'='*72}")
    print(f" OVERALL  ({total} segments)")
    print(f"{'='*72}")
    print(f"  Original _compute_edr MAE : {mm_old:.2f} BPM")
    print(f"  Fixed compute_edr_v2 MAE  : {mm_new:.2f} BPM")
    print(f"  {'Improvement' if improv>0 else 'Regression'}: {abs(improv):.1f}%")
    verdict='EXCELLENT (<2)' if mm_new<2 else 'GOOD (<4)' if mm_new<4 else 'NEEDS TUNING (>=4)'
    print(f"  Verdict: {verdict} BPM MAE")

    if mm_new < mm_old:
        print(f"\n  ✓ compute_edr_v2 is better — update pipeline.py")
    else:
        print(f"\n  ✗ No improvement — keep original _compute_edr")

if __name__=="__main__":
    evaluate()