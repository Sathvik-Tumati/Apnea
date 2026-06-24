"""
apnea_validator.py
==================
Physiological plausibility validator — second-layer verification of
model-predicted apnea events using rule-based cardiorespiratory criteria.

After the model flags a segment as apnea, this script asks:
  "Does the physiology actually support this prediction?"

Three validation criteria (in order of weight):
─────────────────────────────────────────────────
  1. HR/RR RATIO PATTERN  (primary)
     During apnea: RR interval increases → HR decreases → ratio drops
     At termination: RR decreases sharply → HR spikes → ratio surges
     We compute the ratio per segment and look for the dip-surge pattern.

  2. SpO2 DESATURATION CONFIRMATION  (strong supporting)
     SpO2 should drop ≥3% within ±2 segments of the predicted event.
     Delayed by ~15-30s due to circulation lag — hence the ±2 window.

  3. RESPIRATORY SUPPRESSION  (supporting)
     EDR-derived resp amplitude should drop during the event.
     Resp rate variability should increase (irregular breathing pattern).

  4. HR DIP-SURGE PATTERN  (fallback / corroborating)
     Classic bradycardia during apnea → tachycardia at termination.
     Used as a tiebreaker when other criteria are borderline.

Scoring
───────
  Each criterion contributes a confidence score:
    HR/RR pattern      → 0–40 points
    SpO2 desaturation  → 0–35 points
    Resp suppression   → 0–15 points
    HR dip-surge       → 0–10 points
  Total: 0–100

  Verdict:
    ≥ 60  → CONFIRMED    (strong physiological support)
    40–59 → PROBABLE     (partial support — review recommended)
    20–39 → UNCERTAIN    (weak support — may be artifact)
    < 20  → UNCONFIRMED  (no physiological support — likely false positive)

Usage
─────
  # Validate predictions for one admission
  python apnea_validator.py \
      --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv

  # With verbose per-event breakdown
  python apnea_validator.py \
      --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
      --verbose

  # Write validated results back to a CSV
  python apnea_validator.py \
      --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
      --out   infer_output/ADM1819906487/validated_results.csv

  # Update Supabase with validation results
  python apnea_validator.py \
      --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
      --write-supabase
"""

import argparse
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SEGMENT_LEN_S  = 30          # seconds per segment
SPO2_LAG_SEGS  = 2           # SpO2 drop can lag the ECG event by up to 2 segments
HR_RR_WINDOW   = 3           # segments on each side for baseline computation
MIN_SPO2_DROP  = 3.0         # % — minimum desaturation to count
MIN_HR_DIP     = 5.0         # bpm — minimum HR drop during event
MIN_HR_SURGE   = 5.0         # bpm — minimum HR rise at termination
MIN_RESP_DROP  = 0.15        # fractional drop in resp amplitude
RR_RATIO_DIP   = 0.85        # HR/RR ratio must drop to ≤85% of baseline

# Scoring weights (sum to 100)
W_HR_RR   = 40
W_SPO2    = 35
W_RESP    = 15
W_DIP_SURGE = 10

# Verdict thresholds
CONFIRMED    = 60
PROBABLE     = 40
UNCERTAIN    = 20


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventValidation:
    """Validation result for a single predicted apnea event (segment)."""
    segment_idx:       int
    timestamp:         str
    apnea_prob:        float

    # Raw signals at event
    hr_at_event:       float = 0.0
    rr_mean_at_event:  float = 0.0
    spo2_at_event:     float = 0.0
    resp_amp_at_event: float = 0.0

    # Criterion scores (0–max_weight each)
    score_hr_rr:       float = 0.0
    score_spo2:        float = 0.0
    score_resp:        float = 0.0
    score_dip_surge:   float = 0.0

    # Detail flags
    hr_rr_ratio_dropped:   bool  = False
    hr_rr_ratio_surged:    bool  = False
    spo2_drop_confirmed:   bool  = False
    spo2_drop_magnitude:   float = 0.0
    resp_suppressed:       bool  = False
    hr_dip_confirmed:      bool  = False
    hr_surge_confirmed:    bool  = False

    # Final
    total_score:   float = 0.0
    verdict:       str   = "UNCONFIRMED"
    verdict_short: str   = "✗"


@dataclass
class AdmissionValidation:
    """Aggregated validation result for one admission."""
    admission_id:       str
    total_predicted:    int   = 0
    confirmed:          int   = 0
    probable:           int   = 0
    uncertain:          int   = 0
    unconfirmed:        int   = 0
    mean_score:         float = 0.0
    validated_ahi:      float = 0.0    # AHI using only CONFIRMED + PROBABLE events
    original_ahi:       float = 0.0
    duration_min:       float = 0.0
    events:             List[EventValidation] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 1 — HR/RR RATIO PATTERN
# ══════════════════════════════════════════════════════════════════════════════

def _compute_hr_rr_ratio(
    df: pd.DataFrame,
    seg_i: int,
    window: int = HR_RR_WINDOW,
) -> Tuple[float, float, float]:
    """
    Compute HR/RR ratio at the event segment and compare to pre-event baseline.

    HR/RR ratio = mean_hr / rr_mean  (both in their natural units)
    During apnea: RR increases, HR decreases → ratio drops
    At termination: RR decreases, HR spikes → ratio surges

    Returns (baseline_ratio, event_ratio, post_ratio).
    """
    n = len(df)

    def _ratio_at(idx: int) -> float:
        if idx < 0 or idx >= n:
            return np.nan
        row = df.iloc[idx]
        hr  = float(row.get("mean_hr", 0) or row.get("ecg_hr_bpm", 0) or 0)
        rr  = float(row.get("rr_mean", 0) or 0)
        if hr <= 0 or rr <= 0:
            return np.nan
        return hr / (rr / 1000.0)   # normalise RR from ms to seconds

    # Baseline: mean ratio over the window before the event
    pre_ratios = [_ratio_at(i) for i in range(
        max(0, seg_i - window), seg_i)]
    pre_ratios = [r for r in pre_ratios if np.isfinite(r)]
    baseline   = float(np.mean(pre_ratios)) if pre_ratios else np.nan

    event_ratio = _ratio_at(seg_i)

    # Post-event: first segment after event (termination surge)
    post_ratio = _ratio_at(seg_i + 1)

    return baseline, event_ratio, post_ratio


def _score_hr_rr(
    baseline: float,
    event:    float,
    post:     float,
) -> Tuple[float, bool, bool]:
    """
    Score the HR/RR criterion.
    Returns (score, ratio_dropped, ratio_surged).
    """
    if not np.isfinite(baseline) or not np.isfinite(event):
        return 0.0, False, False

    score       = 0.0
    dropped     = False
    surged      = False
    drop_frac   = event / (baseline + 1e-9)
    surge_frac  = (post  / (baseline + 1e-9)) if np.isfinite(post) else np.nan

    # Ratio must drop to ≤ RR_RATIO_DIP × baseline during event
    if drop_frac <= RR_RATIO_DIP:
        dropped  = True
        # Score scales with depth of drop: deeper drop = higher score
        depth    = max(0.0, 1.0 - drop_frac)          # 0 → 1
        score   += W_HR_RR * 0.6 * min(depth / 0.3, 1.0)

    # Ratio should surge post-event (termination arousal)
    if np.isfinite(surge_frac) and surge_frac >= 1.05:
        surged  = True
        surge   = surge_frac - 1.0
        score  += W_HR_RR * 0.4 * min(surge / 0.15, 1.0)

    return min(score, W_HR_RR), dropped, surged


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 2 — SpO2 DESATURATION CONFIRMATION
# ══════════════════════════════════════════════════════════════════════════════

def _score_spo2(
    df:    pd.DataFrame,
    seg_i: int,
    lag:   int = SPO2_LAG_SEGS,
) -> Tuple[float, bool, float]:
    """
    Check for SpO2 desaturation ≥ MIN_SPO2_DROP% within the lag window.
    SpO2 drop lags the event by up to `lag` segments (~15-30s).

    Returns (score, confirmed, drop_magnitude).
    """
    n = len(df)

    # Pre-event SpO2 baseline (up to 3 segments before)
    pre_spo2 = []
    for i in range(max(0, seg_i - 3), seg_i):
        v = float(df.iloc[i].get("spo2_mean", np.nan) or np.nan)
        if np.isfinite(v) and v > 50:
            pre_spo2.append(v)
    if not pre_spo2:
        return 0.0, False, 0.0
    baseline_spo2 = float(np.mean(pre_spo2))

    # Look for drop in event window + lag
    window_spo2 = []
    for i in range(seg_i, min(n, seg_i + lag + 1)):
        v = float(df.iloc[i].get("spo2_mean", np.nan) or np.nan)
        if np.isfinite(v) and v > 50:
            window_spo2.append(v)

    if not window_spo2:
        return 0.0, False, 0.0

    min_spo2   = float(np.min(window_spo2))
    drop       = baseline_spo2 - min_spo2
    confirmed  = drop >= MIN_SPO2_DROP

    if not confirmed:
        # Partial credit for smaller drops (1–3%)
        score = W_SPO2 * max(0.0, (drop - 1.0) / (MIN_SPO2_DROP - 1.0)) * 0.4
    else:
        # Full credit, scaled by severity: 3% → base, 6%+ → full
        score = W_SPO2 * (0.6 + 0.4 * min((drop - MIN_SPO2_DROP) / 3.0, 1.0))

    return min(score, W_SPO2), confirmed, round(drop, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 3 — RESPIRATORY SUPPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def _score_resp(
    df:                  pd.DataFrame,
    seg_i:               int,
    global_resp_baseline: float = 0.0,
) -> Tuple[float, bool]:
    """
    Check for suppressed respiratory amplitude and elevated variability.
    EDR-derived features: resp_amplitude_mean, resp_rate_variability.

    Returns (score, suppressed).
    """
    n = len(df)

    def _get(col, idx):
        if idx < 0 or idx >= n:
            return np.nan
        return float(df.iloc[idx].get(col, np.nan) or np.nan)

    # Pre-event baseline resp amplitude
    pre_amp = [_get("resp_amplitude_mean", i)
           for i in range(max(0, seg_i - 10), seg_i)]
    pre_amp = [v for v in pre_amp if np.isfinite(v) and v > 0]
    if pre_amp:
        local_baseline = float(np.median(pre_amp))
        baseline_amp   = max(local_baseline, global_resp_baseline * 0.5)
    else:
        baseline_amp   = global_resp_baseline

    # FIX: Bug 1 - fetch event_amp that was missing
    event_amp = _get("resp_amplitude_mean", seg_i)

    if not np.isfinite(event_amp) or baseline_amp <= 0:
        return 0.0, False

    drop_frac  = 1.0 - (event_amp / (baseline_amp + 1e-9))
    suppressed = drop_frac >= MIN_RESP_DROP

    # Also check resp rate variability — should be elevated during apnea
    event_rrv  = _get("resp_rate_variability", seg_i)
    pre_rrv    = [_get("resp_rate_variability", i)
                  for i in range(max(0, seg_i - HR_RR_WINDOW), seg_i)]
    pre_rrv    = [v for v in pre_rrv if np.isfinite(v)]
    baseline_rrv = float(np.mean(pre_rrv)) if pre_rrv else np.nan

    rrv_elevated = (
        np.isfinite(event_rrv) and np.isfinite(baseline_rrv)
        and event_rrv > baseline_rrv * 1.2
    )

    score = 0.0
    if suppressed:
        score += W_RESP * 0.7 * min(drop_frac / 0.4, 1.0)
    if rrv_elevated:
        score += W_RESP * 0.3

    return min(score, W_RESP), suppressed


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 4 — HR DIP-SURGE PATTERN (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _score_dip_surge(
    df:    pd.DataFrame,
    seg_i: int,
) -> Tuple[float, bool, bool]:
    """
    Check for classic bradycardia during apnea → tachycardia at termination.

    Returns (score, dip_confirmed, surge_confirmed).
    """
    n = len(df)

    def _hr(idx):
        if idx < 0 or idx >= n:
            return np.nan
        row = df.iloc[idx]
        v = float(row.get("mean_hr", 0) or row.get("ecg_hr_bpm", 0) or 0)
        return v if v > 20 else np.nan

    # Pre-event baseline HR
    pre_hr = [_hr(i) for i in range(max(0, seg_i - HR_RR_WINDOW), seg_i)]
    pre_hr = [v for v in pre_hr if np.isfinite(v)]
    if not pre_hr:
        return 0.0, False, False
    baseline_hr = float(np.mean(pre_hr))

    event_hr = _hr(seg_i)
    post_hr  = _hr(seg_i + 1)

    dip_confirmed   = False
    surge_confirmed = False
    score = 0.0

    if np.isfinite(event_hr):
        dip = baseline_hr - event_hr
        if dip >= MIN_HR_DIP:
            dip_confirmed = True
            score += W_DIP_SURGE * 0.5 * min(dip / 15.0, 1.0)

    if np.isfinite(post_hr) and np.isfinite(event_hr):
        surge = post_hr - event_hr
        if surge >= MIN_HR_SURGE:
            surge_confirmed = True
            score += W_DIP_SURGE * 0.5 * min(surge / 15.0, 1.0)

    return min(score, W_DIP_SURGE), dip_confirmed, surge_confirmed


# ══════════════════════════════════════════════════════════════════════════════
#  VERDICT
# ══════════════════════════════════════════════════════════════════════════════

def _verdict(score: float) -> Tuple[str, str]:
    if score >= CONFIRMED:
        return "CONFIRMED", "✓"
    elif score >= PROBABLE:
        return "PROBABLE", "~"
    elif score >= UNCERTAIN:
        return "UNCERTAIN", "?"
    else:
        return "UNCONFIRMED", "✗"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

def validate_admission(
    infer_csv:    str,
    admission_id: Optional[str] = None,
    verbose:      bool = False,
) -> AdmissionValidation:
    """
    Load inference results CSV and validate each predicted apnea segment.
    Returns an AdmissionValidation with per-event breakdowns.
    """
    df = pd.read_csv(infer_csv, low_memory=False)

    # FIX: Bug 2 - Apply admission filter BEFORE computing global baseline
    if admission_id and "admission_id" in df.columns:
        df = df[df["admission_id"] == admission_id].reset_index(drop=True)

    # Compute global baseline AFTER filtering (not before)
    global_resp_baseline = float(
        df["resp_amplitude_mean"].replace(0, np.nan).median()
    )

    adm_id = str(df["admission_id"].iloc[0]) if "admission_id" in df.columns else "UNKNOWN"

    # Duration
    dur_min = len(df) * SEGMENT_LEN_S / 60.0
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
        if len(ts) >= 2:
            wall = (ts.max() - ts.min()).total_seconds() / 60.0
            if SEGMENT_LEN_S / 60.0 * len(df) * 0.5 < wall < 1440:
                dur_min = wall

    # Predicted apnea rows
    predicted_mask = df["apnea_pred"] == 1.0
    predicted_idxs = df.index[predicted_mask].tolist()
    n_predicted    = len(predicted_idxs)

    original_ahi = n_predicted / max(dur_min / 60.0, 1e-6)

    logger.info("[VALIDATE] %s — %d predicted apnea events over %.1f min (AHI=%.1f)",
                adm_id, n_predicted, dur_min, original_ahi)

    if n_predicted == 0:
        logger.info("[VALIDATE] No apnea predictions to validate.")
        return AdmissionValidation(
            admission_id=adm_id,
            duration_min=round(dur_min, 1),
            original_ahi=round(original_ahi, 1),
        )

    events: List[EventValidation] = []

    for seg_i in predicted_idxs:
        row = df.iloc[seg_i]
        ev  = EventValidation(
            segment_idx      = int(row.get("segment_idx", seg_i)),
            timestamp        = str(row.get("timestamp", "")),
            apnea_prob       = float(row.get("apnea_prob", 0) or 0),
            hr_at_event      = float(row.get("mean_hr", 0) or row.get("ecg_hr_bpm", 0) or 0),
            rr_mean_at_event = float(row.get("rr_mean", 0) or 0),
            spo2_at_event    = float(row.get("spo2_mean", 0) or 0),
            resp_amp_at_event= float(row.get("resp_amplitude_mean", 0) or 0),
        )

        # ── Criterion 1: HR/RR ratio ──────────────────────────────────────────
        baseline_ratio, event_ratio, post_ratio = _compute_hr_rr_ratio(df, seg_i)
        sc1, dropped, surged = _score_hr_rr(baseline_ratio, event_ratio, post_ratio)
        ev.score_hr_rr         = round(sc1, 2)
        ev.hr_rr_ratio_dropped = dropped
        ev.hr_rr_ratio_surged  = surged

        # ── Criterion 2: SpO2 desaturation ───────────────────────────────────
        sc2, spo2_confirmed, spo2_drop = _score_spo2(df, seg_i)
        ev.score_spo2            = round(sc2, 2)
        ev.spo2_drop_confirmed   = spo2_confirmed
        ev.spo2_drop_magnitude   = spo2_drop

        # ── Criterion 3: Respiratory suppression ─────────────────────────────
        # FIX: Bug 2 - Pass global_resp_baseline to _score_resp
        sc3, resp_suppressed = _score_resp(df, seg_i, global_resp_baseline)
        ev.score_resp        = round(sc3, 2)
        ev.resp_suppressed   = resp_suppressed

        # ── Criterion 4: HR dip-surge (fallback) ─────────────────────────────
        sc4, dip, surge = _score_dip_surge(df, seg_i)
        ev.score_dip_surge    = round(sc4, 2)
        ev.hr_dip_confirmed   = dip
        ev.hr_surge_confirmed = surge

        # ── Total ─────────────────────────────────────────────────────────────
        ev.total_score  = round(sc1 + sc2 + sc3 + sc4, 2)
        ev.verdict, ev.verdict_short = _verdict(ev.total_score)

        events.append(ev)

        if verbose:
            logger.info(
                "  seg=%3d  %s  score=%.0f  "
                "HR/RR=%s(%.1f)  SpO2=%s(↓%.1f%%)  Resp=%s  DipSurge=%s  prob=%.2f",
                ev.segment_idx,
                f"{ev.verdict_short} {ev.verdict:<12}",
                ev.total_score,
                "✓" if dropped else "✗", ev.score_hr_rr,
                "✓" if spo2_confirmed else "✗", spo2_drop,
                "✓" if resp_suppressed else "✗",
                "✓" if (dip or surge) else "✗",
                ev.apnea_prob,
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    verdicts    = [e.verdict for e in events]
    n_confirmed = verdicts.count("CONFIRMED")
    n_probable  = verdicts.count("PROBABLE")
    n_uncertain = verdicts.count("UNCERTAIN")
    n_unconf    = verdicts.count("UNCONFIRMED")
    mean_score  = float(np.mean([e.total_score for e in events]))

    # Validated AHI: count only CONFIRMED + PROBABLE events
    validated_events = n_confirmed + n_probable
    validated_ahi    = validated_events / max(dur_min / 60.0, 1e-6)

    result = AdmissionValidation(
        admission_id    = adm_id,
        total_predicted = n_predicted,
        confirmed       = n_confirmed,
        probable        = n_probable,
        uncertain       = n_uncertain,
        unconfirmed     = n_unconf,
        mean_score      = round(mean_score, 1),
        validated_ahi   = round(validated_ahi, 1),
        original_ahi    = round(original_ahi, 1),
        duration_min    = round(dur_min, 1),
        events          = events,
    )

    _log_summary(result)
    return result


def _log_summary(r: AdmissionValidation) -> None:
    logger.info("=" * 60)
    logger.info("  VALIDATION SUMMARY — %s", r.admission_id)
    logger.info("=" * 60)
    logger.info("  Predicted events   : %d", r.total_predicted)
    logger.info("  ✓ CONFIRMED        : %d  (strong physiological support)",  r.confirmed)
    logger.info("  ~ PROBABLE         : %d  (partial support)",               r.probable)
    logger.info("  ? UNCERTAIN        : %d  (weak support — review)",         r.uncertain)
    logger.info("  ✗ UNCONFIRMED      : %d  (no physiological support)",      r.unconfirmed)
    logger.info("  Mean score         : %.1f / 100", r.mean_score)
    logger.info("  Original AHI       : %.1f /hr", r.original_ahi)
    logger.info("  Validated AHI      : %.1f /hr  (confirmed + probable only)", r.validated_ahi)

    delta = r.original_ahi - r.validated_ahi
    if delta > 1.0:
        logger.warning(
            "  ⚠ AHI reduced by %.1f after validation — "
            "%d model predictions had no physiological support",
            delta, r.unconfirmed + r.uncertain)
    elif r.validated_ahi >= 5.0:
        logger.info("  ✓ Validated AHI ≥ 5 — apnea physiologically supported")
    else:
        logger.info("  ✓ Validated AHI < 5 — Normal after validation")
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_validated_csv(
    infer_csv: str,
    result:    AdmissionValidation,
    out_path:  str,
) -> None:
    """
    Write the original inference CSV with validation columns appended.
    New columns: validation_score, validation_verdict, spo2_drop_magnitude,
                 hr_rr_dropped, hr_rr_surged, resp_suppressed, hr_dip, hr_surge
    """
    df = pd.read_csv(infer_csv, low_memory=False)

    # Build lookup by segment_idx
    ev_lookup = {e.segment_idx: e for e in result.events}

    def _val(seg_i: int, attr: str, default):
        e = ev_lookup.get(seg_i)
        return getattr(e, attr, default) if e else default

    df["validation_score"]    = df["segment_idx"].apply(
        lambda i: _val(i, "total_score", np.nan))
    df["validation_verdict"]  = df["segment_idx"].apply(
        lambda i: _val(i, "verdict", ""))
    df["spo2_drop_pct"]       = df["segment_idx"].apply(
        lambda i: _val(i, "spo2_drop_magnitude", np.nan))
    df["hr_rr_dropped"]       = df["segment_idx"].apply(
        lambda i: _val(i, "hr_rr_ratio_dropped", False))
    df["hr_rr_surged"]        = df["segment_idx"].apply(
        lambda i: _val(i, "hr_rr_ratio_surged", False))
    df["resp_suppressed"]     = df["segment_idx"].apply(
        lambda i: _val(i, "resp_suppressed", False))
    df["hr_dip_confirmed"]    = df["segment_idx"].apply(
        lambda i: _val(i, "hr_dip_confirmed", False))
    df["hr_surge_confirmed"]  = df["segment_idx"].apply(
        lambda i: _val(i, "hr_surge_confirmed", False))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("[OUT] Validated CSV → %s", out_path)


def write_validation_to_supabase(result: AdmissionValidation) -> None:
    """
    Update the apnea_results row in Supabase with validation fields.

    Run this SQL once in Supabase editor first:
    ─────────────────────────────────────────────
    ALTER TABLE apnea_results
      ADD COLUMN IF NOT EXISTS validated_ahi        FLOAT,
      ADD COLUMN IF NOT EXISTS validation_confirmed  INT,
      ADD COLUMN IF NOT EXISTS validation_probable   INT,
      ADD COLUMN IF NOT EXISTS validation_uncertain  INT,
      ADD COLUMN IF NOT EXISTS validation_unconfirmed INT,
      ADD COLUMN IF NOT EXISTS validation_mean_score FLOAT,
      ADD COLUMN IF NOT EXISTS physiologically_supported BOOLEAN;
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logger.error("[SUPABASE] SUPABASE_URL / SUPABASE_KEY not set")
        return

    try:
        from supabase import create_client
        client = create_client(url, key)
    except ImportError:
        logger.error("[SUPABASE] pip install supabase")
        return

    record = {
        "admission_id":              result.admission_id,
        "validated_ahi":             result.validated_ahi,
        "validation_confirmed":      result.confirmed,
        "validation_probable":       result.probable,
        "validation_uncertain":      result.uncertain,
        "validation_unconfirmed":    result.unconfirmed,
        "validation_mean_score":     result.mean_score,
        "physiologically_supported": result.validated_ahi >= 5.0,
    }
    try:
        (client.table("apnea_results")
         .upsert(record, on_conflict="admission_id")
         .execute())
        logger.info("[SUPABASE] Validation upserted for %s", result.admission_id)
    except Exception as e:
        logger.error("[SUPABASE] Failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physiological plausibility validator for apnea predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--infer",          required=True,
                   help="infer_results_*.csv from infer.py")
    p.add_argument("--admission",      default=None,
                   help="Filter to one admission ID")
    p.add_argument("--out",            default=None,
                   help="Output CSV with validation columns appended")
    p.add_argument("--write-supabase", action="store_true",
                   help="Update Supabase apnea_results with validation fields")
    p.add_argument("--verbose",        action="store_true",
                   help="Print per-event breakdown")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    result = validate_admission(
        infer_csv    = args.infer,
        admission_id = args.admission,
        verbose      = args.verbose,
    )

    if args.out:
        save_validated_csv(args.infer, result, args.out)
    else:
        # Default: save alongside the input file
        default_out = args.infer.replace(".csv", "_validated.csv")
        save_validated_csv(args.infer, result, default_out)

    if args.write_supabase:
        write_validation_to_supabase(result)


if __name__ == "__main__":
    main()