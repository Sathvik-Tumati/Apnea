"""
find_good_records_v2.py
───────────────────────
Scans MIMIC-IV records and finds ones suitable for EDR evaluation.
Uses the SAME _compute_edr from pipeline.py so evaluation matches production.

Run:  python3 find_good_records_v2.py
Then copy the good record names into your evaluate_edr.py.
"""

import os, sys
import numpy as np
import wfdb
from scipy.signal import resample, welch, butter, filtfilt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import (
    _load_mimic_records, _bandpass, _compute_edr, _detect_r_peaks,
    FS_ECG, FS_RESP, SEGMENT_LEN_S,
)

# ── thresholds ────────────────────────────────────────────────────────────────
ECG_SNR_MIN  = 1.5   # QRS-band SNR — below this = disconnected/noisy lead
RESP_SNR_MIN = 5.0   # Resp in-band SNR — below this = bad sensor
GT_CV_MAX    = 0.35  # coefficient of variation of GT BPMs across segments
MIN_GOOD_SEGS = 6    # need at least this many clean segments per record
N_SCAN       = 50    # records to scan (increase if not enough found)
N_WANT       = 5     # target number of good records

RESP_LO, RESP_HI = 0.1, 0.6

# ── helpers ───────────────────────────────────────────────────────────────────

def _bp(sig, fs, lo, hi, order=3):
    nyq = fs / 2.; hi = min(hi, nyq-0.05); lo = max(lo, 0.01)
    if lo >= hi: return sig
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    padlen = 3 * max(len(a), len(b))
    return filtfilt(b, a, sig) if len(sig) > padlen else sig

def _resp_snr(sig, fs):
    """In-band / out-of-band power ratio for the respiratory signal."""
    if len(sig) < 8 or np.std(sig) < 1e-10: return 0.
    nperseg = min(len(sig), max(8, int(fs * 20)))
    try: f, pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')
    except: return 0.
    inn = (f >= RESP_LO) & (f <= RESP_HI)
    out = ~inn & (f > 0)
    if not np.any(inn) or np.all(pxx[inn] == 0): return 0.
    return float(pxx[inn].max()) / (float(np.mean(pxx[out])) + 1e-12)

def _ecg_snr(ecg, fs):
    """QRS-band energy SNR."""
    sq = _bp(ecg, fs, 5., 20.) ** 2
    nperseg = min(len(sq), max(8, int(fs * 10)))
    try: f, pxx = welch(sq, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    except: return 0.
    inn = (f >= 5) & (f <= 20); out = ~inn & (f > 0)
    if not np.any(inn) or not np.any(out): return 0.
    return float(pxx[inn].mean()) / (float(pxx[out].mean()) + 1e-12)

def _gt_bpm(resp_seg, fs):
    filt = _bp(resp_seg, fs, RESP_LO, RESP_HI)
    snr  = _resp_snr(filt, fs)
    if snr < 0.5: return 0.
    nperseg = min(len(filt), max(8, int(fs * 20)))
    try: f, pxx = welch(filt, fs=fs, nperseg=nperseg, noverlap=nperseg//2, window='hann')
    except: return 0.
    inn = (f >= RESP_LO) & (f <= RESP_HI)
    return float(f[inn][np.argmax(pxx[inn])]) * 60. if np.any(inn) else 0.

def _edr_resp_snr(ecg_seg, r_peaks, fs_ecg, fs_resp):
    """Run the REAL pipeline _compute_edr and measure its SNR."""
    edr = _compute_edr(ecg_seg, r_peaks, fs_ecg, fs_resp)
    return _resp_snr(edr, fs_resp), edr

# ── main scan ─────────────────────────────────────────────────────────────────

print(f"Scanning up to {N_SCAN} MIMIC-IV records for EDR-ready data...")
print(f"Criteria: ECG SNR≥{ECG_SNR_MIN}  Resp SNR≥{RESP_SNR_MIN}  "
      f"GT CV≤{GT_CV_MAX}  GoodSegs≥{MIN_GOOD_SEGS}\n")

record_paths = _load_mimic_records(n=N_SCAN)
if not record_paths:
    print("ERROR: Could not load any records from MIMIC-IV."); sys.exit(1)

print(f"Loaded {len(record_paths)} record paths. Evaluating quality...\n")

good_records = []
issue_counts = {"ECG SNR too low": 0, "Resp SNR too low": 0,
                "GT unstable": 0, "Too few good segs": 0, "Load error": 0}

for rec_path in record_paths:
    rname  = rec_path.split("/")[-1]
    pn_dir = "mimic4wdb/0.1.0/" + "/".join(rec_path.split("/")[:-1])

    try:
        rec = wfdb.rdrecord(rname, pn_dir=pn_dir, sampto=96000)
    except Exception as e:
        print(f"  {rname}: LOAD ERROR — {e}")
        issue_counts["Load error"] += 1
        continue

    smap = {n: i for i, n in enumerate(rec.sig_name)}
    if "II" not in smap or "Resp" not in smap:
        print(f"  {rname}: missing II/Resp — has {list(smap.keys())[:8]}")
        issue_counts["ECG SNR too low"] += 1
        continue

    fs0   = rec.fs
    ecg_  = rec.p_signal[:, smap["II"]]
    rsp_  = rec.p_signal[:, smap["Resp"]]
    for a in (ecg_, rsp_):
        m = np.isnan(a); a[m] = np.nanmean(a) if not m.all() else 0.

    ecg_f = resample(ecg_, int(len(ecg_) * FS_ECG / fs0))
    rsp_f = resample(rsp_, int(len(rsp_) * FS_RESP / fs0))

    se = FS_ECG * SEGMENT_LEN_S; sr = FS_RESP * SEGMENT_LEN_S
    ns = min(len(ecg_f) // se, 10)
    if ns == 0: continue

    seg_results = []
    for i in range(ns):
        ecg_seg = _bandpass(ecg_f[i*se:(i+1)*se], FS_ECG)
        rsp_seg = rsp_f[i*sr:(i+1)*sr]

        esnr  = _ecg_snr(ecg_seg, FS_ECG)
        rsnr  = _resp_snr(_bp(rsp_seg, FS_RESP, RESP_LO, RESP_HI), FS_RESP)
        gt    = _gt_bpm(rsp_seg, FS_RESP)

        # Also test pipeline EDR on good-ECG segments
        edr_snr = 0.
        if esnr >= ECG_SNR_MIN and len(ecg_seg) > 0:
            r_peaks = _detect_r_peaks(ecg_seg, FS_ECG)
            if len(r_peaks) >= 8:
                edr_snr, _ = _edr_resp_snr(ecg_seg, r_peaks, FS_ECG, FS_RESP)

        seg_results.append({'esnr': esnr, 'rsnr': rsnr, 'gt': gt, 'edr_snr': edr_snr})

    mean_esnr  = np.mean([s['esnr'] for s in seg_results])
    mean_rsnr  = np.mean([s['rsnr'] for s in seg_results])
    valid_gts  = [s['gt'] for s in seg_results if s['gt'] > 4]
    gt_cv      = (np.std(valid_gts) / np.mean(valid_gts)) if len(valid_gts) > 1 else 999.
    n_good     = sum(1 for s in seg_results
                     if s['esnr'] >= ECG_SNR_MIN
                     and s['rsnr'] >= RESP_SNR_MIN
                     and s['gt'] > 4)
    mean_edr_snr = np.mean([s['edr_snr'] for s in seg_results if s['edr_snr'] > 0]) \
                   if any(s['edr_snr'] > 0 for s in seg_results) else 0.

    # Determine failure reason
    if mean_esnr < ECG_SNR_MIN:
        reason = "ECG SNR too low"; issue_counts[reason] += 1
    elif mean_rsnr < RESP_SNR_MIN:
        reason = "Resp SNR too low"; issue_counts[reason] += 1
    elif gt_cv > GT_CV_MAX:
        reason = "GT unstable"; issue_counts[reason] += 1
    elif n_good < MIN_GOOD_SEGS:
        reason = "Too few good segs"; issue_counts[reason] += 1
    else:
        reason = None

    passes = reason is None
    status = "✓ GOOD" if passes else f"✗ {reason}"
    gt_str = [f"{g:.0f}" for g in valid_gts]

    print(f"  {rname}  ECG={mean_esnr:.2f}  Resp={mean_rsnr:.0f}  "
          f"EDR_SNR={mean_edr_snr:.1f}  CV={gt_cv:.3f}  "
          f"GoodSegs={n_good}/{ns}  {status}")

    if passes:
        good_records.append({
            'name': rname, 'path': rec_path,
            'ecg_snr': mean_esnr, 'resp_snr': mean_rsnr,
            'edr_snr': mean_edr_snr, 'gt_cv': gt_cv,
            'n_good': n_good, 'gt_bpms': valid_gts,
        })
        if len(good_records) >= N_WANT:
            print(f"\n  Found {N_WANT} good records — stopping scan early.")
            break

# ── results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f" SCAN COMPLETE — {len(good_records)} good records found")
print(f"{'='*65}")

if good_records:
    good_records.sort(key=lambda r: (-r['n_good'], r['gt_cv'], -r['ecg_snr']))
    print("\n  Top records for EDR evaluation:\n")
    print(f"  {'Record':<15}  {'ECG':>5}  {'Resp':>6}  {'EDR':>5}  "
          f"{'CV':>6}  {'GoodSegs':>9}  GT BPMs")
    print(f"  {'-'*15}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*20}")
    for r in good_records:
        print(f"  {r['name']:<15}  {r['ecg_snr']:>5.2f}  {r['resp_snr']:>6.0f}  "
              f"{r['edr_snr']:>5.1f}  {r['gt_cv']:>6.3f}  {r['n_good']:>9}  "
              f"{[f'{g:.0f}' for g in r['gt_bpms']]}")

    print(f"\n  Copy these record names into evaluate_edr.py:")
    print(f"  GOOD_RECORDS = {[r['name'] for r in good_records]}")
else:
    print("\n  No good records found. Common issues:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {count:>3}x  {issue}")
    print(f"\n  Try increasing N_SCAN (currently {N_SCAN}) to search further.")
    print("  MIMIC-IV has ~3000 records — good ones exist but ECG lead quality varies.")