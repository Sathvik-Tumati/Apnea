"""
edf_to_pipeline.py
==================
Convert EDF/BDF wearable recordings to CSV/JSON for the apnea pipeline.

Designed for devices with:
  ECG_CH_A  (244.1 Hz) → resampled to 125 Hz
  RESP      (15.3 Hz)  → resampled to 4 Hz
  HR        (2.0 Hz)   → used for pseudo-labelling fallback
  RR        (2.0 Hz)   → breath rate in breaths/min (if present)

Usage
-----
  # Inspect channels
  python edf_to_pipeline.py --input file.edf --inspect

  # Convert first 2 hours, CSV output
  python edf_to_pipeline.py --input file.edf --mode csv --max-duration-s 7200

  # Convert with pseudo-labels from RESP channel (no annotations)
  python edf_to_pipeline.py --input file.edf --mode csv --label-mode resp

  # Convert all EDFs in a folder, JSON, first hour each
  python edf_to_pipeline.py --input ./edf_files/ --mode json --max-duration-s 3600

  # Treat as unlabelled inference data (label=-1 for all segments)
  python edf_to_pipeline.py --input file.edf --mode csv --label-mode none
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import pyedflib
    HAS_PYEDF = True
except ImportError:
    HAS_PYEDF = False

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from scipy.signal import resample as scipy_resample, butter, filtfilt, find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if not HAS_SCIPY:
    print("ERROR: scipy required.  pip install scipy")
    sys.exit(1)
if not HAS_PYEDF and not HAS_MNE:
    print("ERROR: pip install pyedflib   (or: pip install mne)")
    sys.exit(1)

# ── Pipeline constants (must match pipeline.py) ───────────────────────────────
FS_ECG  = 125   # Hz
FS_RESP = 4     # Hz
SEG_LEN = 30    # seconds

SPE = FS_ECG  * SEG_LEN   # 3750 ECG samples per segment
SPR = FS_RESP * SEG_LEN   # 120 RESP samples per segment

# ── Channel heuristics ────────────────────────────────────────────────────────
ECG_NAMES  = {"ecg_ch_a", "ecg_ch_b", "ecg", "ekg", "ii", "lead ii"}
RESP_NAMES = {"resp", "respiration", "thorax", "abdomen", "chest", "flow",
              "airflow", "respbelt", "resp effort"}
HR_NAMES   = {"hr", "heart rate", "pulse rate"}
RR_NAMES   = {"rr", "rr_interval", "breath rate", "br"}


# ══════════════════════════════════════════════════════════════════════════════
#  EDF / BDF READER
# ══════════════════════════════════════════════════════════════════════════════

class EDFReader:
    """Streaming-safe wrapper: reads one channel at a time to avoid OOM."""

    def __init__(self, path: str):
        self.path = path
        self._labels: List[str] = []
        self._fs: Dict[str, float] = {}
        self._n_samples: Dict[str, int] = {}
        self._annotations: List[Tuple[float, float, str]] = []
        self._duration_s: float = 0.0
        self._file_duration_from_header: float = 0.0

        if HAS_PYEDF:
            self._open_pyedf()
        else:
            self._open_mne()

    def _open_pyedf(self):
        self._f = pyedflib.EdfReader(self.path)
        self._labels = self._f.getSignalLabels()
        for i, lbl in enumerate(self._labels):
            self._fs[lbl]       = float(self._f.getSampleFrequency(i))
            self._n_samples[lbl] = int(self._f.getNSamples()[i])
        self._file_duration_from_header = self._f.getFileDuration()
        self._duration_s = self._file_duration_from_header
        try:
            anns = self._f.readAnnotations()
            if anns and len(anns[0]) > 0:
                for onset, dur, text in zip(*anns):
                    self._annotations.append((float(onset), float(dur), str(text).strip()))
        except Exception:
            pass

    def _open_mne(self):
        self._raw = mne.io.read_raw_edf(self.path, preload=False, verbose=False)
        for ch in self._raw.ch_names:
            self._labels.append(ch)
            self._fs[ch] = float(self._raw.info["sfreq"])
            self._n_samples[ch] = int(self._raw.n_times)
        self._duration_s = float(self._raw.times[-1])
        for ann in self._raw.annotations:
            self._annotations.append(
                (float(ann["onset"]), float(ann["duration"]), str(ann["description"]))
            )

    @property
    def channels(self) -> List[str]:
        return list(self._labels)

    def get_fs(self, channel: str) -> float:
        return self._fs[channel]

    def get_n_samples(self, channel: str) -> int:
        return self._n_samples[channel]

    def read_channel(self, channel: str, max_samples: Optional[int] = None) -> np.ndarray:
        """Read one channel, optionally capped to max_samples (avoids OOM)."""
        if HAS_PYEDF:
            idx = self._labels.index(channel)
            n   = self._n_samples[channel]
            if max_samples:
                n = min(n, max_samples)
            sig = self._f.readSignal(idx, start=0, n=n, digital=False)
        else:
            data, _ = self._raw[channel]
            sig = data[0]
            if max_samples:
                sig = sig[:max_samples]
        return sig.astype(np.float32)

    def annotations(self) -> List[Tuple[float, float, str]]:
        return self._annotations

    def duration_s(self) -> float:
        return self._duration_s

    def close(self):
        try:
            if HAS_PYEDF:
                self._f.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  CHANNEL AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _best_match(channels: List[str], name_set: set) -> Optional[str]:
    for ch in channels:
        if ch.lower().strip() in name_set:
            return ch
    for ch in channels:
        cl = ch.lower()
        if any(n in cl for n in name_set):
            return ch
    return None


def auto_detect(channels: List[str], hints: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    return {
        "ecg":  hints.get("ecg")  or _best_match(channels, ECG_NAMES),
        "resp": hints.get("resp") or _best_match(channels, RESP_NAMES),
        "hr":   hints.get("hr")   or _best_match(channels, HR_NAMES),
        "rr":   hints.get("rr")   or _best_match(channels, RR_NAMES),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _resample(sig: np.ndarray, fs_orig: float, fs_target: int) -> np.ndarray:
    if abs(fs_orig - fs_target) < 0.5:
        return sig.astype(np.float32)
    n_out = int(len(sig) * fs_target / fs_orig)
    return scipy_resample(sig, n_out).astype(np.float32)


def _bandpass(sig: np.ndarray, fs: int,
              lo: float = 0.5, hi: float = 40.0, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    hi  = min(hi, nyq - 0.1)
    lo  = min(lo, hi - 0.1)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def _fill_nan(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    mask = ~np.isfinite(arr)
    if mask.any():
        fv = float(np.nanmean(arr)) if not mask.all() else fill
        arr = arr.copy()
        arr[mask] = fv
    return arr


def _clip_outliers(sig: np.ndarray, n_std: float = 6.0) -> np.ndarray:
    """Clip extreme spikes (common in RR_INTERVAL channels)."""
    m, s = np.median(sig), np.std(sig)
    return np.clip(sig, m - n_std * s, m + n_std * s).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PSEUDO-LABELLING  (no annotations available)
# ══════════════════════════════════════════════════════════════════════════════

def _resp_apnea_flag(resp_seg: np.ndarray, fs: int = FS_RESP) -> bool:
    """
    Detect respiratory suppression in a 30-second RESP segment.
    Uses the same logic as pipeline._resp_flag_gt().
    """
    if len(resp_seg) < fs * 10:
        return False
    resp_std = np.std(resp_seg)
    if resp_std < 0.02:          # flat / disconnected sensor
        return False
    prominence = max(0.02, 0.20 * resp_std)
    min_dist   = int(fs * 1.67)  # max 36 breaths/min
    peaks, props = find_peaks(resp_seg, distance=min_dist, prominence=prominence)
    if len(peaks) < 2:
        return resp_std > 0.05   # real signal but no breathing
    amps = props["prominences"]
    low  = amps < 0.30 * np.percentile(amps, 75)
    run = cur = 0
    for v in low:
        if v:
            cur += 1; run = max(run, cur)
        else:
            cur = 0
    return run >= 3


def pseudo_label_resp(
    resp_125hz: np.ndarray,          # full RESP signal already resampled to FS_RESP
    n_segs: int,
) -> Tuple[np.ndarray, str]:
    """
    Build binary labels from RESP channel suppression.
    Returns (labels array, method_name).
    """
    labels = np.zeros(n_segs, dtype=int)
    for i in range(n_segs):
        seg = resp_125hz[i * SPR: (i + 1) * SPR]
        if _resp_apnea_flag(seg, FS_RESP):
            labels[i] = 1
    return labels, "resp_suppression"


# ══════════════════════════════════════════════════════════════════════════════
#  CORE CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

def process_edf(
    path: str,
    channel_hints: Dict[str, Optional[str]],
    max_duration_s: Optional[int],
    label_mode: str,        # "annotation" | "resp" | "none"
) -> Optional[Dict]:

    print(f"\n  Reading: {os.path.basename(path)}")
    try:
        reader = EDFReader(path)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    cmap = auto_detect(reader.channels, channel_hints)
    print(f"  Channel map: {cmap}")

    if not cmap["ecg"]:
        print("  ERROR: No ECG channel found — skipping")
        reader.close()
        return None

    # ── Determine how many samples to read ───────────────────────────────────
    fs_ecg = reader.get_fs(cmap["ecg"])
    total_ecg_samples = reader.get_n_samples(cmap["ecg"])

    if max_duration_s:
        max_ecg_samples = int(max_duration_s * fs_ecg)
        if max_ecg_samples < total_ecg_samples:
            actual_duration_s = max_duration_s
            print(f"  Capping to {max_duration_s}s "
                  f"({max_duration_s/3600:.1f}h) of {reader.duration_s()/3600:.1f}h total")
        else:
            max_ecg_samples = None
            actual_duration_s = reader.duration_s()
    else:
        max_ecg_samples = None
        actual_duration_s = reader.duration_s()

    # ── Load & resample ECG ───────────────────────────────────────────────────
    print("  Loading ECG ...", end=" ", flush=True)
    raw_ecg = reader.read_channel(cmap["ecg"], max_samples=max_ecg_samples)
    raw_ecg = _fill_nan(_clip_outliers(raw_ecg))
    ecg     = _bandpass(_resample(raw_ecg, fs_ecg, FS_ECG), FS_ECG)
    print(f"done ({len(ecg):,} samples @ {FS_ECG} Hz)")
    del raw_ecg

    # ── Load & resample RESP ──────────────────────────────────────────────────
    resp_full = None
    if cmap["resp"]:
        fs_resp = reader.get_fs(cmap["resp"])
        max_resp = int(actual_duration_s * fs_resp) if max_duration_s else None
        print("  Loading RESP ...", end=" ", flush=True)
        raw_resp  = reader.read_channel(cmap["resp"], max_samples=max_resp)
        raw_resp  = _fill_nan(_clip_outliers(raw_resp))
        resp_full = _resample(raw_resp, fs_resp, FS_RESP)
        print(f"done ({len(resp_full):,} samples @ {FS_RESP} Hz)")
        del raw_resp

    # ── Load HR (lightweight — for metadata only) ─────────────────────────────
    hr_full = None
    if cmap["hr"]:
        fs_hr = reader.get_fs(cmap["hr"])
        max_hr = int(actual_duration_s * fs_hr) if max_duration_s else None
        raw_hr = reader.read_channel(cmap["hr"], max_samples=max_hr)
        hr_full = _fill_nan(raw_hr)

    reader.close()

    # ── Segment count ─────────────────────────────────────────────────────────
    n_segs = len(ecg) // SPE
    if n_segs == 0:
        print("  WARNING: Too short for a single 30-s segment")
        return None
    print(f"  Segments: {n_segs} × 30s")

    # ── Labels ────────────────────────────────────────────────────────────────
    annotations = reader.annotations() if hasattr(reader, '_annotations') else []
    # Re-filter to within our time window
    if max_duration_s:
        annotations = [(o, d, t) for o, d, t in annotations if o < max_duration_s]

    if label_mode == "annotation" and annotations:
        labels = _annotations_to_labels(annotations, n_segs)
        label_source = "edf_annotation"
        print(f"  Labels from annotations: {int(labels.sum())} apnea / {n_segs - int(labels.sum())} normal")
    elif label_mode == "resp" and resp_full is not None:
        labels, label_source = pseudo_label_resp(resp_full, n_segs)
        print(f"  Pseudo-labels (RESP suppression): {int(labels.sum())} apnea / {n_segs - int(labels.sum())} normal")
    else:
        labels       = np.full(n_segs, -1, dtype=int)  # -1 = unlabelled
        label_source = "none"
        print("  Labels: none (unlabelled inference mode)")

    # ── Per-segment slicing ───────────────────────────────────────────────────
    hr_per_seg = 2 * SEG_LEN  # samples of HR at 2 Hz per segment

    segments = []
    for i in range(n_segs):
        seg: Dict = {
            "segment_idx":  i,
            "onset_s":      i * SEG_LEN,
            "true_label":   int(labels[i]),
            "label_source": label_source,
            "has_resp":     resp_full is not None,
            "has_hr":       hr_full   is not None,
            # Modality flags for pipeline compatibility
            "has_spo2":     0,
            "has_abp":      0,
            "has_resp_gt":  int(resp_full is not None),
        }

        # ECG
        seg["ecg"] = ecg[i * SPE: (i + 1) * SPE].tolist()

        # RESP
        if resp_full is not None and (i + 1) * SPR <= len(resp_full):
            seg["resp"] = resp_full[i * SPR: (i + 1) * SPR].tolist()
        else:
            seg["resp"] = [0.0] * SPR

        # HR summary (mean over 30-s window, from 2 Hz signal)
        if hr_full is not None:
            s = i * hr_per_seg
            e = s + hr_per_seg
            hr_slice = hr_full[s: e] if e <= len(hr_full) else hr_full[s:]
            valid = hr_slice[(hr_slice > 20) & (hr_slice < 220)]  # filter invalid
            seg["hr_mean"] = float(np.mean(valid)) if len(valid) > 0 else 0.0
        else:
            seg["hr_mean"] = 0.0

        segments.append(seg)

    n_labelled = int((labels >= 0).sum())
    n_apnea    = int((labels == 1).sum())

    return {
        "meta": {
            "source_file":    os.path.abspath(path),
            "device":         "wearable_ecg_resp",
            "total_duration_s": reader.duration_s(),
            "converted_duration_s": actual_duration_s,
            "n_segments":     n_segs,
            "n_labelled":     n_labelled,
            "n_apnea":        n_apnea,
            "n_normal":       int((labels == 0).sum()),
            "label_source":   label_source,
            "channel_map":    cmap,
            "fs_ecg":         FS_ECG,
            "fs_resp":        FS_RESP,
            "segment_len_s":  SEG_LEN,
        },
        "segments": segments,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ANNOTATION → LABELS
# ══════════════════════════════════════════════════════════════════════════════

APNEA_TOKENS = {"OA", "CA", "MA", "X", "A", "H", "HA", "APNEA", "HYPOPNEA",
                "OBSTRUCTIVE", "CENTRAL", "MIXED"}

def _annotations_to_labels(annotations, n_segs: int) -> np.ndarray:
    labels = np.zeros(n_segs, dtype=int)
    for onset, duration, text in annotations:
        tokens = set(text.upper().replace("/", " ").split())
        if not APNEA_TOKENS.intersection(tokens):
            continue
        event_end = onset + max(duration, 10.0)
        seg_s = max(0, int(onset // SEG_LEN) - 1)
        seg_e = min(n_segs - 1, int(event_end // SEG_LEN) + 1)
        for si in range(seg_s, seg_e + 1):
            ss, se = si * SEG_LEN, (si + 1) * SEG_LEN
            if onset < se and event_end > ss:
                labels[si] = 1
    return labels


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT WRITERS
# ══════════════════════════════════════════════════════════════════════════════

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def write_csv(result: Dict, out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    segs = result["segments"]
    written = []

    for channel in ["ecg", "resp"]:
        rows = []
        for seg in segs:
            if channel not in seg:
                continue
            row = {
                "segment_idx":  seg["segment_idx"],
                "onset_s":      seg["onset_s"],
                "true_label":   seg["true_label"],
                "label_source": seg.get("label_source", ""),
                "hr_mean":      seg.get("hr_mean", 0.0),
                "has_resp_gt":  seg.get("has_resp_gt", 0),
            }
            for j, v in enumerate(seg[channel]):
                row[f"t{j}"] = round(float(v), 6)
            rows.append(row)
        if not rows:
            continue
        p = os.path.join(out_dir, f"{stem}_{channel}.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        written.append(p)
        print(f"    → {p}  ({len(rows)} segments)")

    # Compact summary (no raw signal — useful for quick inspection)
    summary = [{
        "segment_idx":  s["segment_idx"],
        "onset_s":      s["onset_s"],
        "true_label":   s["true_label"],
        "label_source": s.get("label_source", ""),
        "hr_mean":      s.get("hr_mean", 0.0),
        "has_resp_gt":  s.get("has_resp_gt", 0),
        "has_spo2":     s.get("has_spo2", 0),
        "has_abp":      s.get("has_abp", 0),
    } for s in segs]
    sp = os.path.join(out_dir, f"{stem}_summary.csv")
    pd.DataFrame(summary).to_csv(sp, index=False)
    written.append(sp)
    print(f"    → {sp}  (summary, no raw signals)")
    return written


def write_json(result: Dict, out_dir: str, stem: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{stem}.json")
    with open(p, "w") as f:
        json.dump(result, f, cls=_NpEncoder, separators=(",", ":"))
    print(f"    → {p}  ({os.path.getsize(p)/1024/1024:.1f} MB)")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  INSPECT MODE
# ══════════════════════════════════════════════════════════════════════════════

def inspect_edf(path: str):
    print(f"\n{'='*60}")
    print(f"  {os.path.basename(path)}")
    print(f"{'='*60}")
    try:
        reader = EDFReader(path)
    except Exception as e:
        print(f"  ERROR: {e}"); return

    dur = reader.duration_s()
    print(f"  Duration : {dur:.0f}s  ({dur/3600:.1f}h)")
    print(f"  Channels ({len(reader.channels)}):")
    for ch in reader.channels:
        fs  = reader.get_fs(ch)
        n   = reader.get_n_samples(ch)
        # Read tiny slice for range
        try:
            tiny = reader.read_channel(ch, max_samples=min(n, 10000))
            rng  = f"[{tiny.min():.3f}, {tiny.max():.3f}]"
        except Exception:
            rng = "[?]"
        print(f"    {ch:<32} fs={fs:6.1f}Hz  samples={n:>12,}  range={rng}")

    anns = reader.annotations()
    if anns:
        print(f"\n  Annotations ({len(anns)}):")
        for o, d, t in anns[:10]:
            print(f"    t={o:.0f}s  dur={d:.0f}s  label={t!r}")
    else:
        print("\n  Annotations: none  (use --label-mode resp for pseudo-labels)")

    cmap = auto_detect(reader.channels, {})
    print("\n  Auto-detected mapping:")
    for k, v in cmap.items():
        print(f"    {k:<8} → {v or 'NOT FOUND'}")

    n_segs_possible = int(dur // SEG_LEN)
    print(f"\n  Max 30-s segments from full file : {n_segs_possible:,}")
    print(f"  Suggested --max-duration-s values:")
    for h in [1, 2, 4, 8]:
        s = h * 3600
        if s <= dur:
            print(f"    {h}h → {s}s  ({int(s//SEG_LEN)} segments)")
    reader.close()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _collect(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(p.rglob("*.edf")) + sorted(p.rglob("*.EDF")) + \
                sorted(p.rglob("*.bdf")) + sorted(p.rglob("*.BDF"))
        return [str(f) for f in files]
    print(f"ERROR: {input_path} not found"); sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Convert EDF/BDF wearable recordings for the apnea pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--input",  "-i", required=True, help="EDF/BDF file or folder")
    ap.add_argument("--inspect", action="store_true", help="Print channel info and exit")
    ap.add_argument("--mode",   choices=["csv","json","both"], default="csv")
    ap.add_argument("--out-dir", "-o", default="./converted")
    ap.add_argument("--max-duration-s", type=int, default=None,
                    help="Seconds to convert per file (e.g. 7200 = 2h). "
                         "REQUIRED for large files to avoid OOM.")
    ap.add_argument("--label-mode", choices=["annotation","resp","none"], default="resp",
                    help="annotation: use EDF+ annotations; "
                         "resp: pseudo-label from RESP suppression (default); "
                         "none: all segments unlabelled (label=-1)")
    # Channel overrides
    ap.add_argument("--ecg-channel",  default=None)
    ap.add_argument("--resp-channel", default=None)
    ap.add_argument("--hr-channel",   default=None)
    ap.add_argument("--rr-channel",   default=None)
    args = ap.parse_args()

    files = _collect(args.input)
    print(f"Found {len(files)} file(s)")

    if args.inspect:
        for f in files:
            inspect_edf(f)
        return

    if not args.max_duration_s:
        print("\nWARNING: --max-duration-s not set.")
        print("  Your files are ~4-8 days long (100M+ samples).")
        print("  This will use ~8GB+ RAM and take a very long time.")
        print("  Recommended: --max-duration-s 7200  (2 hours)")
        ans = input("  Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted. Re-run with e.g. --max-duration-s 7200")
            sys.exit(0)

    hints = {
        "ecg":  args.ecg_channel,
        "resp": args.resp_channel,
        "hr":   args.hr_channel,
        "rr":   args.rr_channel,
    }

    total_segs = 0
    for path in files:
        stem   = Path(path).stem
        result = process_edf(path, hints, args.max_duration_s, args.label_mode)
        if result is None:
            continue
        if args.mode in ("csv", "both"):
            write_csv(result, args.out_dir, stem)
        if args.mode in ("json", "both"):
            write_json(result, args.out_dir, stem)
        total_segs += result["meta"]["n_segments"]
        print(f"  Meta: {json.dumps(result['meta'], indent=4, cls=_NpEncoder)}")

    print(f"\n✓ Done. Total segments: {total_segs}")
    print(f"  Output: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()