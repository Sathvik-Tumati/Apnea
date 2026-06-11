"""
compute_edr_v3.py — Drop-in replacement for _compute_edr in pipeline.py
────────────────────────────────────────────────────────────────────────
Target MAE: 3.0–4.0 BPM on MIMIC-IV (vs 5.18 original)

Key improvements:
  1. Internal EDR at 10 Hz (vs 4 Hz) → 3× better PSD resolution
  2. Parabolic interpolation around PSD peak → ~0.3 BPM effective resolution
  3. Autocorrelation cross-check → harmonic confusion correction
  4. Adaptive AC weight: higher at low BPM where harmonics are in-band
  5. SNR-based engine selection (original vs improved)
  6. Returns (signal, bpm, quality) — bpm replaces downstream Welch in features
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch, resample
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

FS_INTERNAL = 10   # Hz — internal computation rate


# ── utilities ─────────────────────────────────────────────────────────────────

def _bp(sig, fs, lo=0.1, hi=0.6, order=3):
    nyq=fs/2.; hi=min(hi,nyq-0.05); lo=max(lo,0.01)
    if lo>=hi: return sig
    b,a=butter(order,[lo/nyq,hi/nyq],btype='band')
    padlen=3*max(len(a),len(b))
    return filtfilt(b,a,sig) if len(sig)>padlen else sig


def _interp(times, values, fs_out, seg_len_s):
    if len(times)<4: return None
    t=np.arange(0,seg_len_s,1./fs_out)
    try:
        return interp1d(times,values,kind='linear',
                        bounds_error=False,fill_value='extrapolate')(t)
    except Exception:
        return None


def _detrend(times, values):
    return values - np.polyval(np.polyfit(times,values,1),times)


def _snr(sig, fs, lo=0.1, hi=0.6):
    if sig is None or len(sig)<8 or np.std(sig)<1e-10: return 0.
    nperseg=min(len(sig),max(8,int(fs*20)))
    try: f,pxx=welch(sig,fs=fs,nperseg=nperseg,noverlap=nperseg//2)
    except: return 0.
    inn=(f>=lo)&(f<=hi); out=~inn&(f>0)
    if not np.any(inn) or not np.any(out): return 0.
    return float(pxx[inn].max())/(float(np.mean(pxx[out]))+1e-12)


# ── precision frequency estimator ────────────────────────────────────────────

def _estimate_bpm(sig, fs, lo=0.1, hi=0.6):
    """
    Returns (bpm, snr).

    Pipeline:
    1. Welch PSD with nfft=8192 for fine frequency grid
    2. Parabolic interpolation for sub-bin accuracy (~0.3 BPM)
    3. Autocorrelation cross-check for harmonic confusion
    4. Smart correction: test half/double if PSD and AC disagree > 3 BPM
    5. Adaptive blend: weight AC higher at low rates (< 14 BPM)
    """
    if sig is None or len(sig)<16 or np.std(sig)<1e-10: return 0.,0.

    # Welch PSD
    w=tukey(len(sig),alpha=0.15)
    nperseg=min(len(sig),max(8,int(fs*20)))
    try:
        f,pxx=welch(sig*w,fs=fs,nperseg=nperseg,noverlap=nperseg//2,
                    window='hann',nfft=8192)
    except:
        return 0.,0.

    inn=(f>=lo)&(f<=hi); out_=~inn&(f>0)
    if not np.any(inn) or np.all(pxx[inn]==0): return 0.,0.

    pxx_inn=pxx[inn]; f_inn=f[inn]
    idx=int(np.argmax(pxx_inn))
    snr=float(pxx_inn[idx])/(float(np.mean(pxx[out_]))+1e-12)

    # Parabolic interpolation
    if 0<idx<len(pxx_inn)-1:
        a_,b_,c_=pxx_inn[idx-1],pxx_inn[idx],pxx_inn[idx+1]
        denom=a_-2*b_+c_
        p=0.5*(a_-c_)/(denom+1e-12) if abs(denom)>1e-12 else 0.
        df=f_inn[1]-f_inn[0] if len(f_inn)>1 else 0.
        psd_hz=f_inn[idx]+np.clip(p,-0.5,0.5)*df
    else:
        psd_hz=f_inn[idx]
    psd_bpm=float(psd_hz)*60.

    # Autocorrelation
    sig_z=sig-np.mean(sig)
    ac=np.correlate(sig_z,sig_z,mode='full'); ac=ac[len(ac)//2:]
    if ac[0]>0: ac/=ac[0]
    min_lag=max(1,int(fs*60./(hi*60))); max_lag=min(len(ac)-1,int(fs*60./(lo*60)))
    ac_bpm=psd_bpm
    if max_lag>min_lag and len(ac[min_lag:max_lag])>0:
        pl=min_lag+int(np.argmax(ac[min_lag:max_lag]))
        if pl>0: ac_bpm=np.clip(60./(pl/fs), lo*60, hi*60)

    # Harmonic correction
    diff=abs(psd_bpm-ac_bpm)
    if diff>3.0:
        half=psd_bpm/2.
        double=psd_bpm*2.
        if lo*60<=half<=hi*60 and abs(half-ac_bpm)<diff:
            psd_bpm=half
        elif lo*60<=double<=hi*60 and abs(double-ac_bpm)<diff*0.5:
            psd_bpm=double

    # Adaptive blend
    psd_bpm=np.clip(psd_bpm,lo*60,hi*60)
    w_ac=0.50 if psd_bpm<14 else 0.25
    final=np.clip((1-w_ac)*psd_bpm+w_ac*ac_bpm, lo*60, hi*60)
    return float(final), float(snr)


# ── EDR engines (both at FS_INTERNAL Hz) ─────────────────────────────────────

def _engine_a(ecg, r_peaks, fs_ecg, seg_len_s):
    """Engine A: original logic at FS_INTERNAL Hz."""
    if len(r_peaks)<8: return None
    t=r_peaks/float(fs_ecg)
    qw=max(1,int(0.06*fs_ecg))

    areas=np.array([np.sum(np.abs(ecg[max(0,r-qw):min(len(ecg),r+qw)]))
                    for r in r_peaks],dtype=float)
    m3r=_interp(t,_detrend(t,areas),FS_INTERNAL,seg_len_s)
    if m3r is None: return None
    m3=_bp(m3r,FS_INTERNAL)

    beats=[ecg[r-qw:r+qw] for r in r_peaks if r-qw>=0 and r+qw<=len(ecg)]
    wl=2*qw; beats=[b for b in beats if len(b)==wl]
    m4=m3.copy()
    if len(beats)>=8:
        X=np.array(beats,dtype=float); X-=X.mean(axis=0,keepdims=True)
        try:
            U,S,_=np.linalg.svd(X,full_matrices=False)
            bt=t[:len(beats)]
            ps=_interp(bt,_detrend(bt,U[:,0]*S[0]),FS_INTERNAL,seg_len_s)
            if ps is not None: m4=_bp(ps,FS_INTERNAL)
        except np.linalg.LinAlgError: pass

    ml=min(len(m3),len(m4))
    def _n(s): std=np.std(s); return (s-np.mean(s))/(std+1e-9) if std>1e-9 else s
    return _bp(np.median(np.vstack([_n(m3[:ml]),_n(m4[:ml])]),axis=0),FS_INTERNAL)


def _engine_b(ecg, r_peaks, fs_ecg, seg_len_s):
    """Engine B: bandpass-first, wider PCA window."""
    if len(r_peaks)<8: return None
    t=r_peaks/float(fs_ecg)

    qwa=max(1,int(0.060*fs_ecg))
    areas=np.array([np.sum(np.abs(ecg[max(0,r-qwa):min(len(ecg),r+qwa)]))
                    for r in r_peaks],dtype=float)
    m3r=_interp(t,_detrend(t,areas),FS_INTERNAL,seg_len_s)
    if m3r is None: return None
    m3=_bp(m3r,FS_INTERNAL)

    qwp=max(1,int(0.080*fs_ecg))
    beats,btimes=[],[]
    for r in r_peaks:
        s,e=r-qwp,r+qwp
        if s>=0 and e<=len(ecg): beats.append(ecg[s:e]); btimes.append(r/float(fs_ecg))
    wl=2*qwp; beats=[b for b in beats if len(b)==wl]
    m4=m3.copy()
    if len(beats)>=8:
        X=np.array(beats,dtype=float); X-=X.mean(axis=0,keepdims=True)
        try:
            U,S,_=np.linalg.svd(X,full_matrices=False)
            bt=np.array(btimes[:len(beats)])
            ps=_interp(bt,_detrend(bt,U[:,0]*S[0]),FS_INTERNAL,seg_len_s)
            if ps is not None: m4=_bp(ps,FS_INTERNAL)
        except np.linalg.LinAlgError: pass

    ml=min(len(m3),len(m4))
    def _n(s): std=np.std(s); return (s-np.mean(s))/(std+1e-9) if std>1e-9 else s
    return _bp(np.median(np.vstack([_n(m3[:ml]),_n(m4[:ml])]),axis=0),FS_INTERNAL)


# ── public API ────────────────────────────────────────────────────────────────

def compute_edr_v3(ecg: np.ndarray,
                   r_peaks: np.ndarray,
                   fs_ecg: int,
                   fs_resp: int = 4,
                   seg_len_s: int = 30) -> tuple:
    """
    Returns
    -------
    edr_signal : np.ndarray  shape (seg_len_s * fs_resp,)
    bpm        : float       precision respiratory rate estimate
    quality    : float       in-band SNR (>= 1.5 reliable)
    """
    out_len=seg_len_s*fs_resp
    sa=_engine_a(ecg,r_peaks,fs_ecg,seg_len_s)
    sb=_engine_b(ecg,r_peaks,fs_ecg,seg_len_s)
    snr_a=_snr(sa,FS_INTERNAL) if sa is not None else 0.
    snr_b=_snr(sb,FS_INTERNAL) if sb is not None else 0.
    best=sb if snr_b>=snr_a else sa
    if best is None: return np.zeros(out_len),0.,0.

    bpm,quality=_estimate_bpm(best,FS_INTERNAL)
    edr_out=resample(best,out_len) if fs_resp!=FS_INTERNAL else best[:out_len]
    return edr_out,float(bpm),float(quality)


def _compute_edr_v3(ecg,r_peaks,fs_ecg,fs_resp=4):
    """Signal-only drop-in for _compute_edr."""
    sig,_,_=compute_edr_v3(ecg,r_peaks,fs_ecg,fs_resp)
    return sig


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__=="__main__":
    np.random.seed(42)
    print("="*52)
    print(" compute_edr_v3  —  self-test")
    print("="*52)
    fs_ecg=125; fs_resp=4; seg_s=30
    t=np.arange(0,seg_s,1/fs_ecg)
    errs=[]
    for true_bpm in [9,12,15,18,21,24,27]:
        resp_f=true_bpm/60.
        rp=(np.arange(0.1,seg_s,60./75)*fs_ecg).astype(int)
        rp=rp[rp<len(t)]
        ecg=np.random.randn(len(t))*0.05
        for r in rp:
            amp=1.+0.35*np.sin(2*np.pi*resp_f*r/fs_ecg)+0.15*np.sin(4*np.pi*resp_f*r/fs_ecg)
            for off,sc in [(-1,.3),(0,1.),(1,.3)]:
                if 0<=r+off<len(ecg): ecg[r+off]+=amp*sc
        _,bpm,q=compute_edr_v3(ecg,rp,fs_ecg,fs_resp,seg_s)
        err=abs(true_bpm-bpm); errs.append(err)
        print(f"  {'✓' if err<=2 else '~' if err<=4 else '✗'}"
              f"  True={true_bpm:5.1f}  Est={bpm:5.1f}  Err={err:.2f}  SNR={q:.0f}")
    print(f"\n  Synthetic MAE: {np.mean(errs):.2f} BPM")
    print(f"\nPipeline integration:")
    print("  from compute_edr_v3 import compute_edr_v3")
    print("  edr_sig, resp_bpm, edr_q = compute_edr_v3(ecg_seg, r_peaks, FS_ECG, FS_RESP, SEGMENT_LEN_S)")
    print("  feats['resp_rate_bpm']  = resp_bpm   # precision estimate, skip downstream Welch")
    print("  feats['edr_quality_ok'] = int(edr_q >= 1.5)")
    print("  feats['edr_snr']        = edr_q      # let LSTM learn when to trust EDR")
