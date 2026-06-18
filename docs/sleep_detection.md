# ECG-Based Sleep Detection

`automation/ecg_sleep_filter.py` detects sleep windows from continuous ECG data pulled from MongoDB, without requiring EEG, accelerometers, or skin temperature. It runs automatically inside `mongo_infer.py` before the segment CSV is built, ensuring only sleep-time ECG is fed to the apnea model.

---

## Why Sleep Filter Before Inference?

Hospital wearable recordings often span 8–12+ hours, covering both wake and sleep periods. Running apnea inference on the full recording would:

- Inflate the AHI denominator (including daytime hours with no actual sleep)
- Introduce false-positive apnea detections from events like breath-holding, talking, or walking
- Dilute the model's sensitivity to true nocturnal apnea events

By isolating sleep epochs first, the AHI proxy becomes more accurate and the model operates in the physiological regime it was trained for.

---

## Algorithm

Sleep detection uses three layers of evidence, applied in order:

### 1. Time-of-Day Gate (IST 9pm – 9am)
All segments outside this window are immediately excluded as candidates. This is a free, zero-cost gate that removes daytime data with no computation.

- Handles midnight wrapping: `hour >= 21 OR hour < 9` (IST)
- Conservative: if a timestamp is missing or unparseable, the segment is **not** gated out

### 2. HR and SDNN Scoring (Per-Segment HRV)
For each 30-second ECG segment inside the time window, we compute:
- **Mean HR** (bpm): From R-peaks detected by neurokit2 (scipy fallback)
- **SDNN** (ms): Standard deviation of R-R intervals

Thresholds are calculated from the nighttime segments of that specific recording:
- `hr_sleep = (mean_hr < p45 of nighttime HRs)` — lower HR = sleep candidate
- `sdnn_sleep = (sdnn > p55 of nighttime SDNNs)` — higher SDNN = parasympathetic dominance = sleep

**Sleep score** per segment:
```
score = 0.5 × hr_sleep + 0.5 × sdnn_sleep
score × in_sleep_hours  (zero out daytime even if HR/SDNN match)
```

A 5-epoch median filter smooths the score to reduce transient artifacts.

### 3. Minimum Duration Enforcement
After scoring, run-length processing enforces two rules:
1. **Gap bridging**: Wake gaps ≤ 3 minutes (6 epochs) flanked by sleep on both sides are converted to sleep (accounts for brief arousals without re-scoring)
2. **Minimum bout**: Sleep bouts shorter than 20 minutes (40 epochs) are discarded (prevents classifying short naps or recording artifacts as sleep)

---

## Integration with `mongo_infer.py`

```
ECG signal (full recording)
         │
         ▼
detect_sleep_segments()
  ├── HRV features per 30s epoch
  ├── IST time gate
  ├── HR/SDNN percentile scoring
  └── Run-length min-duration filter
         │
         ▼
sleep_idxs = [epoch indices where is_sleep == 1]
         │
         ▼
ECG trimmed to sleep epochs only
    → passed to build_segment_csv()
    → timestamps from sleep_df (not re-derived — avoids edge-packet drift)
         │
         ▼
 {admissionId}_sleep_windows.csv saved to infer_output/
```

**Fallback behaviour:** If fewer than `MIN_SEGMENTS` (11) sleep epochs are detected, the full unfiltered recording is used and a warning is logged. This prevents inference failure on short or daytime-only recordings.

---

## Output DataFrame Schema

`detect_sleep_segments()` returns one row per 30-second epoch:

| Column | Type | Description |
|---|---|---|
| `segment_idx` | int | 0-indexed position in the full ECG signal |
| `timestamp_utc` | datetime | Wall-clock timestamp (UTC) resolved via cumulative document lookup |
| `timestamp_ist` | str | IST-converted display string |
| `hour_ist` | int | IST hour (−1 if timestamp unavailable) |
| `mean_hr` | float | Mean heart rate (bpm), 0 if < 3 R-peaks detected |
| `sdnn` | float | SDNN (ms), 0 on failure |
| `n_peaks` | int | Number of R-peaks detected |
| `in_sleep_hours` | bool | True if within IST 9pm–9am gate |
| `hr_sleep` | bool | True if HR below nighttime p45 |
| `sdnn_sleep` | bool | True if SDNN above nighttime p55 |
| `sleep_score` | float | Smoothed score 0–1 |
| `is_sleep` | int | 1 = sleep, 0 = wake (after min-duration filter) |
| `sleep_window_id` | int | Contiguous sleep bout ID (−1 = wake) |

---

## Standalone CLI Usage

```bash
# Summary only — no files written
python automation/ecg_sleep_filter.py \
    --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
    --stats

# Save sleep windows CSV
python automation/ecg_sleep_filter.py \
    --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
    --out infer_output/ADM1819906487/sleep_windows.csv

# Save CSV + diagnostic 4-panel plot
python automation/ecg_sleep_filter.py \
    --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
    --out infer_output/ADM1819906487/sleep_windows.csv \
    --plot

# Disable IST gate (useful for short test recordings at non-standard hours)
python automation/ecg_sleep_filter.py \
    --csv infer_output/ADM1819906487/ADM1819906487_segments.csv \
    --no-time-gate --stats
```

**CLI requires a segment CSV with `ecgData[0]` … `ecgData[3749]` columns** (as produced by `mongo_infer.py --dry-run`).

---

## Typical Output (Real Admission Example)

```
[SLEEP] Extracting HRV features from 1027 segments ...
[SLEEP] Time gate (IST 21:00–9:00): 673 / 1027 segments in window, 354 excluded
[SLEEP] Thresholds — HR < 98.4 bpm (p45),  SDNN > 24.6 ms (p55)
[SLEEP] Result: 601 / 1027 segments classified as sleep (59%)  = 5.0 hours  across 6 window(s)
[SLEEP]   Window 0: 97 segments (48 min)  2026-05-01 15:30 → 15:52 UTC
[SLEEP]   Window 1: 171 segments (86 min)  2026-05-01 16:00 → 17:19 UTC
[SLEEP]   Window 2: 145 segments (72 min)  2026-05-01 17:23 → 18:39 UTC
...
[SLEEP] Kept 601 / 1027 segments after sleep filter
```

---

## Limitations

- **No EEG ground truth**: Sleep staging (N1/N2/N3/REM) is not possible from ECG alone. The detector identifies "likely sleeping" epochs, not precise sleep stages.
- **IST-only time gate**: If patients are in a different timezone or have highly irregular sleep schedules, the fixed IST gate may exclude valid sleep or include daytime naps. Adjust `SLEEP_HOUR_START`/`SLEEP_HOUR_END` in the file constants if needed.
- **HR threshold depends on recording length**: Very short recordings (< 30 minutes nighttime) fall back to computing thresholds from all valid segments, which may be less accurate.
- **neurokit2 strongly recommended**: The scipy R-peak fallback is significantly less accurate at high heart rates or noisy ECG.
