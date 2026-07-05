# apnea_validator.py — Technical Reference

`apnea_validator.py` is the physiological plausibility validator — a rule-based second-layer
verification system that runs after the ML pipeline and checks whether each predicted apnea
event has real cardiorespiratory support in the data.

> [!IMPORTANT]
> The validator **does not replace** the model. It is a post-processing filter. Events that
> score below the CONFIRMED threshold are not necessarily wrong — they may have been missed
> due to sensor dropout or a short recording. Always check the `data_completeness` flag
> before discarding an event.

---

## Why It Exists

Machine learning models trained on clinical datasets (MIMIC-IV, SLPDB) are highly sensitive
and will occasionally flag noise, motion artifact, or benign sleep irregularities as apnea.
The validator cross-checks each model prediction against four independent physiological
criteria derived from the ECG and SpO2 signals already present in the inference CSV. No
additional data collection is required.

---

## Execution Flow

The validator runs in 7 sequential passes over the predicted apnea events:

```
PASS 1 — Score each event independently (all four criteria)
PASS 2 — Apply cluster-level recovery surge bonus (mutates total_score)
PASS 3 — Annotate data completeness flags (mutates data_completeness)
PASS 4 — Print verbose per-event lines (only if --verbose)
PASS 5 — Aggregate verdicts and compute validated_ahi (lower bound)
PASS 6 — Compute AHI range (upper bound) and log severity flag
PASS 7 — Log method_used breakdown for calibration
```

All seven passes run inside `validate_admission()`. Verbose output is printed **after**
the cluster bonus is applied, so it always reflects the final score.

---

## Scoring Criteria

Total score: **0–100 points**

### Criterion 1 — Primary (Combined, 40 pts)

The primary criterion is **dual-mechanism**: both sub-mechanisms are computed for every
event, and the higher score is used. This is implemented in `_score_primary_combined()`.

#### Sub-mechanism A: HR/RespRate Ratio (`_score_hr_resp_ratio`)

During apnea, the respiratory rate falls toward zero (or very low) while heart rate
remains elevated due to sympathetic arousal. The ratio `HR / RespRate` rises sharply.

```
baseline_ratio = mean(HR / max(RespRate, 2.0)) over up to 3 pre-event segments
event_ratio    = HR_event / max(RespRate_event, 2.0)
increase_pct   = 100 × (event_ratio - baseline_ratio) / baseline_ratio
```

Scoring:
- increase_pct <= 0             → 0 pts
- 0 < increase_pct < 15%        → partial credit: 40 × 0.4 × (pct / 15)
- 15% <= increase_pct < 80%     → 40 × (0.5 + 0.5 × (pct - 15) / 65)
- increase_pct >= 80%           → full 40 pts

**Availability guard**: Requires `MIN_VALID_BASELINE_SEGS = 2` clean pre-event segments
with valid `resp_rate_bpm`. Returns `None` (not computable) if unavailable.

#### Sub-mechanism B: HR Drop/Surge (`_score_hr_drop_surge`)

Classic vagal bradycardia during apnea followed by sympathetic tachycardia at termination.

```
hr_drop  = baseline_hr - event_hr          (score if >= 3.0 bpm)
hr_surge = post_event_hr - event_hr        (score if >= 4.0 bpm)

drop_score  = 40 × 0.6 × min(hr_drop / 20, 1.0)
surge_score = 40 × 0.4 × min(hr_surge / 20, 1.0)
```

Always computable as long as `ecg_hr_bpm` (or `mean_hr` / `overall_hr_bpm`) exists
for the event and its baseline window. Returns `hr_drop_bpm = None` if baseline HR
is unavailable.

#### Selection logic (`_score_primary_combined`)

```python
final_score = max(ratio_score, hr_drop_score)

if ratio_score > hr_drop_score:   method_used = "ratio"
elif hr_drop_score > ratio_score: method_used = "hr_drop"
elif ratio_score > 0:             method_used = "both"   # equal and nonzero
else:                             method_used = "neither"
```

The `method_used` field is written to both the log and the output CSV. Both ratio
values (`ratio_baseline`, `ratio_event`, `ratio_increase_pct`) and the HR drop
value (`hr_drop_bpm`) are always shown in verbose output regardless of which mechanism won.

---

### Criterion 2 — SpO2 Desaturation (`_score_spo2_with_availability`, 35 pts)

Checks for a blood oxygen drop >= `MIN_SPO2_DROP = 3.0%` within `SPO2_LAG_SEGS = 2`
segments after the event (accounting for the ~10–30 s circulation lag from apnea to
peripheral SpO2 drop).

Checks **both** `spo2_min` (captures transient drops < 30 s) and `spo2_mean` (captures
sustained drops). The **larger** drop across both columns and all lag segments is used.

**Baseline computation:**
1. Mean `spo2_mean` from `has_spo2=1` segments in the 3 segments immediately before the event.
2. Falls back to global 90th percentile of all `has_spo2=1` segments in the admission if no
   pre-event SpO2 is available (e.g. event is near the start of the recording).
3. Returns `(0.0, False, 0.0, False)` if no SpO2 data exists anywhere in the admission.

Partial credit is awarded for drops between 1% and the 3% threshold.

`spo2_available` is set `True` if at least one `has_spo2=1` row existed in either the
baseline window or the event+lag window. This flag drives the dynamic reweighting.

---

### Criterion 3 — Respiratory Suppression (`_score_resp`, 15 pts)

Uses EDR-derived `resp_amplitude_mean` and `resp_rate_variability`.

```
local_baseline = median(resp_amplitude_mean) over up to 10 pre-event segments
drop_frac      = 1 - (event_amp / baseline_amp)
suppressed     = drop_frac >= 0.15
```

Score composition:
- Amplitude drop: `15 × 0.7 × min(drop_frac / 0.4, 1.0)` if suppressed
- Variability increase: `15 × 0.3` if `event_rrv > baseline_rrv × 1.2`

The global resp baseline (`global_resp_baseline`) is computed from the **admission-filtered**
DataFrame median before the event loop starts, ensuring it reflects this patient's recording.

---

### Criterion 4 — HR Dip-Surge Fallback (`_score_dip_surge`, 10 pts)

A corroborating secondary check for the bradycardia-to-tachycardia pattern. Uses the same
HR baseline as Criterion 1 Sub-mechanism B but contributes to a different score bucket and
uses a softer scaling (15 bpm = full credit vs 20 bpm in Criterion 1).

This criterion is separate from Criterion 1 so that the dip-surge signal can contribute
independently even when it does not win the `max()` comparison in the primary criterion.

---

## Dynamic SpO2 Reweighting (`_combine_scores`)

When `spo2_available = False` (SpO2 sensor was disconnected for both the baseline and
the event window), the 35 SpO2 points are redistributed proportionally across the
remaining three criteria:

```
other_sum = W_HR_RR + W_RESP + W_DIP_SURGE   # = 65

hr_effective   = W_HR_RR     + W_SPO2 × (W_HR_RR     / 65)  # → ~61.5
resp_effective = W_RESP      + W_SPO2 × (W_RESP      / 65)  # → ~23.1
dip_effective  = W_DIP_SURGE + W_SPO2 × (W_DIP_SURGE / 65)  # → ~15.4

total = sc1 × (hr_eff / 40) + sc3 × (resp_eff / 15) + sc4 × (dip_eff / 10)
```

This ensures the maximum achievable score stays at 100 even when SpO2 is completely
absent, so events in no-SpO2 windows are not unfairly penalised.

---

## Cluster-Level Recovery Surge Bonus (`_apply_cluster_surge_bonus`)

Apnea frequently occurs in dense back-to-back clusters. Within a cluster, the heart
does not have time to recover between events, so individual per-event HR surges are
absent. This causes cluster members to score low on Criterion 1 even when the cluster
as a whole is clearly physiologically supported.

**Cluster definition (`_find_clusters`):** Consecutive predicted segments separated by
at most `CLUSTER_GAP_TOLERANCE = 1` non-predicted segment are treated as one cluster.

**Surge search (`_cluster_recovery_surge`):** Looks for an HR surge >= `MIN_HR_SURGE`
in the first `MAX_SURGE_SEARCH = 5` clean segments after the cluster ends, measured
against the HR at the segment before the cluster started.

**Bonus application:**
- Bonus: `+CLUSTER_BONUS = 8` points per cluster member
- Cap: `CLUSTER_BONUS_CAP = 59` — the bonus alone can **never** produce a CONFIRMED verdict
- Only applied if `total_score > 0` (will not manufacture support from nothing)
- Single-event "clusters" (length 1) are skipped — per-event surge is already checked in Criterion 1

The bonus is applied in **PASS 2**, before verbose logging, so logged scores always match final verdicts.

---

## Data Completeness Annotation (`_annotate_data_completeness`)

Events that score below the CONFIRMED threshold (60) receive a `data_completeness` tag
explaining *why* they failed, which is critical for clinical review:

| Flag | Condition |
|---|---|
| `complete` | SpO2 was available AND recording had >=5 segments after cluster end |
| `insufficient_spo2` | `spo2_available = False` for this event |
| `insufficient_trailing` | `cluster_end + MAX_SURGE_SEARCH >= len(df)` (recording ended too soon) |
| `insufficient_both` | Both SpO2 and trailing window were unavailable |

These flags do **not** change the verdict or AHI. They are diagnostic annotations only.

---

## AHI Range Reporting (`_report_ahi_range`)

Because data-incomplete events may be real events the validator couldn't check, AHI is
reported as a range rather than a single number:

```
lower_ahi = (n_CONFIRMED + n_PROBABLE) / duration_hours
upper_ahi = (n_CONFIRMED + n_PROBABLE + n_flagged_incomplete) / duration_hours

n_flagged_incomplete = count of (UNCONFIRMED | UNCERTAIN) events
                       where data_completeness != "complete"
```

The validator checks whether the range straddles a clinical severity threshold:

- `lower_ahi < 5 <= upper_ahi`  → logs FLAG: PROBABLE MILD APNEA — manual review recommended
- `lower_ahi >= 5`              → apnea confirmed regardless of flagged events
- Both bounds `< 5`             → likely normal, review flagged segments if clinically indicated

The `upper_ahi` value is written to `AdmissionValidation.upper_ahi` and upserted to
Supabase as `validated_ahi_upper`.

---

## Method Breakdown Reporting (`_log_method_breakdown`)

At the end of every run (PASS 7), the validator logs the distribution of `method_used`
across all predicted events:

```
------------------------------------------------------------
  Primary criterion method_used breakdown:
    N events used: ratio
    N events used: hr_drop
    N events used: both
    N events used: neither
------------------------------------------------------------
```

This output is intended for **cross-admission calibration**. If `"neither"` dominates
consistently across multiple admissions, the primary criterion thresholds or the HR
column selection logic may need adjustment.

---

## Verbose Output Format

When run with `--verbose`, one line is logged per predicted event after PASS 2:

```
seg=NNN  [verdict]  score=XX.X
  C1=method_used [ratio:BL→EV(+PCT%) | HRdrop:↓X.Xbpm]
  SpO2=[checkmark/x](↓X.X%,avail=[checkmark/x])  Resp=[checkmark/x]  DipSurge=[checkmark/x]
  prob=X.XX  [DATA:flag_if_not_complete]
```

Both the ratio values and the HR drop value are shown **regardless of which mechanism won**,
so you can see all the evidence even when only one contributed the score.

---

## Output CSV Columns (appended by `save_validated_csv`)

| Column | Type | Description |
|---|---|---|
| `validation_score` | float | Final 0–100 score |
| `validation_verdict` | str | `CONFIRMED` / `PROBABLE` / `UNCERTAIN` / `UNCONFIRMED` |
| `method_used` | str | Which C1 mechanism won: `ratio`, `hr_drop`, `both`, `neither` |
| `ratio_score` | float | Score from the HR/RespRate ratio sub-mechanism (0–40) |
| `ratio_baseline` | float | Pre-event HR/RespRate baseline ratio (NaN if not computable) |
| `ratio_event` | float | HR/RespRate ratio at the event segment (NaN if not computable) |
| `ratio_increase_pct` | float | % increase in ratio vs. baseline (NaN if not computable) |
| `hr_drop_score` | float | Score from the HR drop/surge sub-mechanism (0–40) |
| `hr_drop_bpm` | float | Actual HR drop in bpm (NaN if baseline HR unavailable) |
| `hr_dropped` | bool | True if HR fell >= MIN_HR_DIP (3.0 bpm) from baseline |
| `hr_surged` | bool | True if post-event HR rose >= MIN_HR_SURGE (4.0 bpm) |
| `spo2_drop_pct` | float | SpO2 drop magnitude in % |
| `spo2_available` | bool | True if SpO2 data existed for this event window |
| `resp_suppressed` | bool | True if resp amplitude dropped >= 15% below baseline |
| `hr_dip_confirmed` | bool | True if C4 fallback dip confirmed |
| `hr_surge_confirmed` | bool | True if C4 fallback surge confirmed |
| `cluster_bonus_applied` | bool | True if cluster-level recovery surge bonus was added |
| `data_completeness` | str | `complete` / `insufficient_spo2` / `insufficient_trailing` / `insufficient_both` |

---

## Key Constants

| Constant | Value | Description |
|---|---|---|
| `SEGMENT_LEN_S` | 30 | Seconds per segment |
| `SPO2_LAG_SEGS` | 2 | SpO2 drop can lag the ECG event by up to 2 segments (~60 s) |
| `HR_RR_WINDOW` | 3 | Pre-event baseline window (segments) |
| `MIN_SPO2_DROP` | 3.0% | Minimum desaturation to count for C2 |
| `MIN_HR_DIP` | 3.0 bpm | Minimum HR drop to count (lowered from 5 bpm) |
| `MIN_HR_SURGE` | 4.0 bpm | Minimum HR surge to count (lowered from 5 bpm) |
| `MIN_RESP_DROP` | 0.15 | Minimum fractional resp amplitude drop for C3 |
| `MIN_RESP_RATE_FOR_RATIO` | 2.0 bpm | Floor applied to RespRate in ratio to avoid divide-by-near-zero |
| `MIN_RATIO_INCREASE_PCT` | 15.0% | Ratio increase threshold for C1A confirmation |
| `FULL_CREDIT_RATIO_INCREASE_PCT` | 80.0% | Ratio increase for full C1A credit |
| `MIN_VALID_BASELINE_SEGS` | 2 | Minimum pre-event baseline segments required for C1A |
| `CLUSTER_GAP_TOLERANCE` | 1 | Max non-apnea segments allowed inside a cluster |
| `CLUSTER_BONUS` | 8 pts | Bonus applied per cluster member when surge found |
| `CLUSTER_BONUS_CAP` | 59 | Score ceiling for cluster bonus (cannot produce CONFIRMED) |
| `MAX_SURGE_SEARCH` | 5 | Segments to search past cluster end for recovery surge |
| `CONFIRMED` | 60 | Verdict threshold |
| `PROBABLE` | 40 | Verdict threshold |
| `UNCERTAIN` | 20 | Verdict threshold |

---

## Supabase Schema (validation columns)

```sql
-- Add to apnea_results after initial table creation:
ALTER TABLE apnea_results
    ADD COLUMN IF NOT EXISTS validated_ahi             FLOAT,        -- lower bound (CONFIRMED + PROBABLE)
    ADD COLUMN IF NOT EXISTS validated_ahi_upper       FLOAT,        -- upper bound (includes data-incomplete events)
    ADD COLUMN IF NOT EXISTS validation_confirmed      INT,
    ADD COLUMN IF NOT EXISTS validation_probable       INT,
    ADD COLUMN IF NOT EXISTS validation_uncertain      INT,
    ADD COLUMN IF NOT EXISTS validation_unconfirmed    INT,
    ADD COLUMN IF NOT EXISTS validation_mean_score     FLOAT,
    ADD COLUMN IF NOT EXISTS physiologically_supported BOOLEAN;      -- true if validated_ahi >= 5
```

---

## CLI Reference

```
usage: apnea_validator.py [-h] --infer INFER [--admission ADMISSION]
                          [--out OUT] [--write-supabase] [--verbose]

arguments:
  --infer INFER         Path to infer_results_*.csv from infer.py (required)
  --admission ID        Filter to a single admission ID within the CSV
  --out PATH            Output path for validated CSV
                        (default: replaces .csv with _validated.csv)
  --write-supabase      Upsert validated_ahi, validated_ahi_upper, and
                        verdict counts to the apnea_results Supabase table
  --verbose             Print per-event score breakdown after validation
```

**Examples:**

```bash
# Validate a single admission, auto-save validated CSV
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv

# Verbose mode — see per-event method, ratio values, HR drop, SpO2 drop
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --verbose

# Explicit output path + Supabase write
python apnea_validator.py \
    --infer  infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --out    infer_output/ADM1819906487/ADM1819906487_validated.csv \
    --write-supabase

# Filter to one admission in a multi-admission CSV
python apnea_validator.py \
    --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
    --admission ADM1819906487
```

---

## SpO2-Aware AHI Breakdown (companion script)

`spo2_split_ahi.py` is a standalone diagnostic companion that splits the validated CSV
into contiguous `has_spo2=1` and `has_spo2=0` windows and reports raw + validated AHI
separately for each. This is useful when SpO2 dropout coincides with apnea clusters,
making a single blended AHI misleading.

```bash
python spo2_split_ahi.py \
    --validated infer_output/ADM1819906487/ADM1819906487_validated.csv
```

Input: the validated CSV produced by `apnea_validator.py`. Does not write any files.

---

*See also: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) Section 4 for operational usage,
[README.md](../README.md) for the full output column reference.*
