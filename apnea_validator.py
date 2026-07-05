"""
apnea_validator.py
==================
Physiological plausibility validator — second-layer verification of
model-predicted apnea events using rule-based cardiorespiratory criteria.

After the model flags a segment as apnea, this script asks:
  "Does the physiology actually support this prediction?"

Four validation criteria (in order of weight):
─────────────────────────────────────────────────
  1. PRIMARY CRITERION (combined, 40 pts)
     Computes BOTH the HR/RespRate ratio increase AND the HR drop/surge
     pattern whenever possible, and takes the MAX score. This allows the
     validator to capture the most salient physiological evidence:
     
     - HR/RespRate ratio: During apnea, HR rises (sympathetic arousal)
       while respiratory rate falls toward zero. The ratio increases.
     
     - HR drop/surge: Classic vagal bradycardia during apnea followed
       by sympathetic tachycardia at termination.
     
     The method_used field tracks which mechanism actually contributed
     the score ("ratio", "hr_drop", "both", or "neither") for auditing
     and calibration across admissions.

  2. SpO2 DESATURATION CONFIRMATION  (strong supporting, 35 pts)
     Checks BOTH spo2_min (transient drops within segment) and spo2_mean.
     Drop ≥3% within ±2 segments (circulation lag). Falls back to global
     baseline when no pre-event SpO2 is available.

  3. RESPIRATORY SUPPRESSION  (supporting, 15 pts)
     EDR-derived resp amplitude should drop during the event.
     Resp rate variability should increase (irregular breathing pattern).

  4. HR DIP-SURGE PATTERN  (fallback / corroborating, 10 pts)
     Classic bradycardia during apnea → tachycardia at termination.
     Used as a tiebreaker when other criteria are borderline.

Scoring
───────
  Each criterion contributes a confidence score:
    Primary (max of ratio and HR drop/surge) → 0–40 points
    SpO2 desaturation  → 0–35 points
    Resp suppression   → 0–15 points
    HR dip-surge       → 0–10 points
  Total: 0–100

  When SpO2 is completely unavailable (has_spo2==0 for baseline AND event),
  the 35 SpO2 points are redistributed proportionally across the other
  three criteria, ensuring the maximum possible score remains 100.

  CLUSTER-LEVEL RECOVERY SURGE BONUS (CORROBORATING, CAPPED):
    Detects clusters of consecutive apnea predictions and looks for an
    HR surge at the FIRST clean segment after the cluster ends (the real
    recovery point). If found, each segment in the cluster gets a small
    bonus (CLUSTER_BONUS=8) added on top, capped at 59 so the bonus
    alone can never push a segment past PROBABLE. This helps segments
    with partial independent support (e.g., score=32) cross into
    PROBABLE territory, but prevents isolated false positives with zero
    support from being confirmed.

  DATA COMPLETENESS FLAGGING:
    Segments that end up UNCONFIRMED or UNCERTAIN are annotated with
    whether the validator had complete data to check them:
      - "complete": both SpO2 and recovery window were available
      - "insufficient_trailing": recording ended before recovery surge could be checked
      - "insufficient_spo2": SpO2 was unavailable for this segment
      - "insufficient_both": both issues present
    These flags do NOT affect verdict or AHI; they help distinguish
    genuine unsupported predictions from cases where the validator
    couldn't fully check.

  AHI RANGE REPORTING (TEMPORARY):
    Reports validated AHI as a range instead of a single number:
      lower_ahi = (CONFIRMED + PROBABLE) / duration_hours
      upper_ahi = (CONFIRMED + PROBABLE + flagged_incomplete) / duration_hours
    This prevents silent under-reporting when many events are flagged
    for review due to data incompleteness.

  Verdict:
    ≥ 60  → CONFIRMED    (strong physiological support)
    40–59 → PROBABLE     (partial support — review recommended)
    20–39 → UNCERTAIN    (weak support — may be artifact)
    < 20  → UNCONFIRMED  (no physiological support — likely false positive)

Changes from previous version
──────────────────────────────
  - PRIMARY CRITERION now computes BOTH HR/RespRate ratio and HR drop/surge,
    taking the max score and tracking which method contributed.
  - SpO2 criterion now checks spo2_min as well as spo2_mean.
  - SpO2 baseline falls back to global 90th-pct when no pre-event SpO2.
  - MIN_HR_DIP lowered 5→3 bpm, MIN_HR_SURGE 5→4 bpm.
  - global_resp_baseline is computed after admission filtering.
  - Dynamic SpO2 reweighting redistributes weight when unavailable.
  - Cluster-level recovery surge bonus with cap at 59.
  - Verbose logging runs AFTER cluster bonus application.
  - Data completeness flags distinguish genuine from incomplete.
  - AHI reported as a range (temporary fix).
  - Method_used breakdown added to summary for calibration.
  - Verbose display fixes: score shows 1 decimal place to match verdict.
  - Verbose display shows both ratio and HR drop values when computable,
    regardless of which mechanism "won".

Usage
─────
  python apnea_validator.py \
      --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv

  python apnea_validator.py \
      --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
      --verbose

  python apnea_validator.py \
      --infer infer_output/ADM1819906487/infer_results_ADM1819906487.csv \
      --out   infer_output/ADM1819906487/validated_results.csv

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
MIN_HR_DIP     = 3.0         # bpm — lowered from 5.0; sleep HR is 50-60 bpm
MIN_HR_SURGE   = 4.0         # bpm — lowered from 5.0
MIN_RESP_DROP  = 0.15        # fractional drop in resp amplitude

# Primary criterion (combined) constants
MIN_RESP_RATE_FOR_RATIO = 2.0   # bpm -- floor to avoid divide-by-near-zero blowups
MIN_RATIO_INCREASE_PCT = 15.0   # % increase in HR/RespRate ratio to count as "confirmed"
FULL_CREDIT_RATIO_INCREASE_PCT = 80.0  # % increase for full score credit
MIN_VALID_BASELINE_SEGS = 2     # need at least this many clean segments to trust a baseline

# Scoring weights (sum to 100)
W_HR_RR     = 40
W_SPO2      = 35
W_RESP      = 15
W_DIP_SURGE = 10

# Cluster surge bonus
CLUSTER_GAP_TOLERANCE = 1     # max consecutive non-apnea segments allowed
                               # inside a cluster before it's considered ended
CLUSTER_BONUS = 8             # points added if cluster-level surge found
CLUSTER_BONUS_CAP = 59        # bonus alone can never push score above this
MAX_SURGE_SEARCH = 5          # segments to search for recovery surge

# Verdict thresholds
CONFIRMED  = 60
PROBABLE   = 40
UNCERTAIN  = 20


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EventValidation:
    """Validation result for a single predicted apnea event (segment)."""
    segment_idx:        int
    timestamp:          str
    apnea_prob:         float

    # Raw signals at event
    hr_at_event:        float = 0.0
    rr_mean_at_event:   float = 0.0
    spo2_at_event:      float = 0.0
    resp_amp_at_event:  float = 0.0

    # Criterion scores (0–max_weight each)
    score_hr_rr:        float = 0.0
    score_spo2:         float = 0.0
    score_resp:         float = 0.0
    score_dip_surge:    float = 0.0

    # Primary criterion details (combined ratio + HR drop)
    method_used:           str   = "neither"  # "ratio", "hr_drop", "both", "neither"
    ratio_score:           float = 0.0
    ratio_baseline:        Optional[float] = None
    ratio_event:           Optional[float] = None
    ratio_increase_pct:    Optional[float] = None
    hr_drop_score:         float = 0.0
    
    # Detail flags (from HR drop/surge mechanism)
    hr_dropped:            bool  = False   # HR dropped ≥ MIN_HR_DIP from baseline
    hr_surged:             bool  = False   # HR surged ≥ MIN_HR_SURGE after event
    hr_drop_bpm:           Optional[float] = None  # None = not computable; else actual bpm drop (clipped at 0)
    
    spo2_drop_confirmed:   bool  = False
    spo2_drop_magnitude:   float = 0.0
    spo2_available:        bool  = False  # True if SpO2 data existed for this segment
    resp_suppressed:       bool  = False
    hr_dip_confirmed:      bool  = False
    hr_surge_confirmed:    bool  = False
    cluster_bonus_applied: bool  = False  # True if cluster surge bonus was added
    data_completeness:     str   = "complete"  # complete | insufficient_trailing | insufficient_spo2 | insufficient_both

    # Final
    total_score:   float = 0.0
    verdict:       str   = "UNCONFIRMED"
    verdict_short: str   = "✗"


@dataclass
class AdmissionValidation:
    """Aggregated validation result for one admission."""
    admission_id:    str
    total_predicted: int   = 0
    confirmed:       int   = 0
    probable:        int   = 0
    uncertain:       int   = 0
    unconfirmed:     int   = 0
    mean_score:      float = 0.0
    validated_ahi:   float = 0.0  # lower bound (confirmed + probable only)
    upper_ahi:       float = 0.0  # upper bound (including flagged incomplete)
    original_ahi:    float = 0.0
    duration_min:    float = 0.0
    events:          List[EventValidation] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 1 — COMBINED PRIMARY (ratio + HR drop/surge, take max)
# ══════════════════════════════════════════════════════════════════════════════

def _get_hr(df: pd.DataFrame, idx: int) -> float:
    """
    Return HR in bpm at DataFrame row `idx`.
    Tries ecg_hr_bpm first (computed fresh in infer.py), then mean_hr
    (from feature extraction), then overall_hr_bpm (from device).
    Returns nan if none are valid.
    """
    if idx < 0 or idx >= len(df):
        return np.nan
    row = df.iloc[idx]
    for col in ["ecg_hr_bpm", "mean_hr", "overall_hr_bpm"]:
        v = row.get(col, None)
        if v is not None:
            try:
                fv = float(v)
                if fv > 20:
                    return fv
            except (TypeError, ValueError):
                pass
    return np.nan


def _get_resp_rate(df: pd.DataFrame, idx: int) -> float:
    """EDR-derived breathing rate in bpm. Returns nan if missing/invalid."""
    if idx < 0 or idx >= len(df):
        return np.nan
    row = df.iloc[idx]
    v = row.get("resp_rate_bpm", None)
    try:
        fv = float(v)
        if fv >= 0:
            return fv
    except (TypeError, ValueError):
        pass
    return np.nan


def _hr_resp_ratio(hr: float, resp_rate: float, floor: float = MIN_RESP_RATE_FOR_RATIO) -> float:
    """HR / RespRate, with RespRate floored to avoid blow-up when
    breathing rate approaches zero during a true apnea."""
    if not np.isfinite(hr) or not np.isfinite(resp_rate):
        return np.nan
    return hr / max(resp_rate, floor)


def _compute_baseline_ratio(df: pd.DataFrame, seg_i: int, window: int = HR_RR_WINDOW) -> Tuple[float, int]:
    """
    Mean HR/RespRate ratio over up to `window` clean segments immediately
    before seg_i. Returns (baseline_ratio, n_valid_segments_used).
    """
    ratios = []
    for i in range(max(0, seg_i - window), seg_i):
        hr = _get_hr(df, i)
        rr = _get_resp_rate(df, i)
        r = _hr_resp_ratio(hr, rr)
        if np.isfinite(r):
            ratios.append(r)
    if not ratios:
        return np.nan, 0
    return float(np.mean(ratios)), len(ratios)


def _score_hr_resp_ratio(df: pd.DataFrame, seg_i: int) -> Optional[Dict]:
    """
    PRIMARY criterion mechanism 1: score based on the increase in HR/RespRate ratio
    during the event relative to a pre-event baseline.

    Returns a dict with score and diagnostic values, or None if the ratio
    cannot be reliably computed.
    """
    baseline_ratio, n_valid = _compute_baseline_ratio(df, seg_i)
    if not np.isfinite(baseline_ratio) or n_valid < MIN_VALID_BASELINE_SEGS:
        return None

    event_hr = _get_hr(df, seg_i)
    event_resp = _get_resp_rate(df, seg_i)
    event_ratio = _hr_resp_ratio(event_hr, event_resp)
    if not np.isfinite(event_ratio):
        return None

    increase_pct = 100.0 * (event_ratio - baseline_ratio) / baseline_ratio
    confirmed = increase_pct >= MIN_RATIO_INCREASE_PCT

    if increase_pct <= 0:
        score = 0.0
    elif not confirmed:
        # partial credit below the confirmation threshold
        score = W_HR_RR * 0.4 * (increase_pct / MIN_RATIO_INCREASE_PCT)
    else:
        # full credit scaling between confirmation threshold and "full credit" threshold
        span = max(FULL_CREDIT_RATIO_INCREASE_PCT - MIN_RATIO_INCREASE_PCT, 1e-6)
        frac = min((increase_pct - MIN_RATIO_INCREASE_PCT) / span, 1.0)
        score = W_HR_RR * (0.5 + 0.5 * frac)

    score = min(max(score, 0.0), float(W_HR_RR))

    return {
        "score": round(score, 2),
        "baseline_ratio": round(baseline_ratio, 3),
        "event_ratio": round(event_ratio, 3),
        "increase_pct": round(increase_pct, 1),
    }


# ── HR drop/surge mechanism (always computed) ──────────────────────────────

def _compute_hr_baseline(df: pd.DataFrame, seg_i: int, window: int = HR_RR_WINDOW) -> float:
    """Mean HR over the `window` segments immediately before seg_i."""
    pre = [_get_hr(df, i)
           for i in range(max(0, seg_i - window), seg_i)]
    valid = [v for v in pre if np.isfinite(v)]
    return float(np.mean(valid)) if valid else np.nan


def _score_hr_drop_surge(df: pd.DataFrame, seg_i: int) -> Dict:
    """
    PRIMARY criterion mechanism 2: original HR drop/surge logic.
    Always computable as long as baseline and event HR exist.
    """
    baseline_hr = _compute_hr_baseline(df, seg_i)
    event_hr = _get_hr(df, seg_i)
    post_hr = _get_hr(df, seg_i + 1)

    if not np.isfinite(baseline_hr) or not np.isfinite(event_hr):
        return {
            "score": 0.0,
            "hr_dropped": False,
            "hr_surged": False,
            "hr_drop_bpm": None,  # None means "not computable"
        }

    score = 0.0
    dropped = False
    surged = False

    hr_drop = baseline_hr - event_hr
    if hr_drop >= MIN_HR_DIP:
        dropped = True
        score += W_HR_RR * 0.6 * min(hr_drop / 20.0, 1.0)

    if np.isfinite(post_hr):
        hr_surge = post_hr - event_hr
        if hr_surge >= MIN_HR_SURGE:
            surged = True
            score += W_HR_RR * 0.4 * min(hr_surge / 20.0, 1.0)

    return {
        "score": round(min(score, float(W_HR_RR)), 2),
        "hr_dropped": dropped,
        "hr_surged": surged,
        "hr_drop_bpm": round(max(hr_drop, 0.0), 2),  # real computed value
    }


# ── Combined entry point ────────────────────────────────────────────────────

def _score_primary_combined(df: pd.DataFrame, seg_i: int) -> Tuple[float, Dict]:
    """
    Computes both mechanisms, takes the max, and reports method_used so
    you can track method_used vs apnea_label across admissions and
    calibrate by trial and error.
    """
    ratio_result = _score_hr_resp_ratio(df, seg_i)
    ratio_score = ratio_result["score"] if ratio_result else 0.0

    hr_drop_result = _score_hr_drop_surge(df, seg_i)
    hr_drop_score = hr_drop_result["score"]

    if ratio_score > hr_drop_score:
        method_used = "ratio"
    elif hr_drop_score > ratio_score:
        method_used = "hr_drop"
    elif ratio_score > 0:  # equal and nonzero
        method_used = "both"
    else:
        method_used = "neither"

    final_score = max(ratio_score, hr_drop_score)

    info = {
        "method_used": method_used,
        "ratio_score": ratio_score,
        "hr_drop_score": hr_drop_score,
        "hr_dropped": hr_drop_result["hr_dropped"],
        "hr_surged": hr_drop_result["hr_surged"],
        "hr_drop_bpm": hr_drop_result["hr_drop_bpm"],
    }
    if ratio_result:
        info["ratio_baseline"] = ratio_result["baseline_ratio"]
        info["ratio_event"] = ratio_result["event_ratio"]
        info["ratio_increase_pct"] = ratio_result["increase_pct"]
    else:
        info["ratio_baseline"] = None
        info["ratio_event"] = None
        info["ratio_increase_pct"] = None

    return final_score, info


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 2 — SpO2 DESATURATION CONFIRMATION  (with availability tracking)
# ══════════════════════════════════════════════════════════════════════════════

def _score_spo2_with_availability(
    df:    pd.DataFrame,
    seg_i: int,
    lag:   int = SPO2_LAG_SEGS,
) -> Tuple[float, bool, float, bool]:
    """
    Check for SpO2 desaturation ≥ MIN_SPO2_DROP% within the lag window.

    Checks BOTH spo2_min (captures transient drops within a 30s segment)
    and spo2_mean (captures sustained drops). Takes the larger of the two.

    Baseline is computed from has_spo2=1 segments before the event.
    Falls back to the global 90th-pct baseline when no pre-event SpO2
    is available (e.g. event occurs early in the recording).

    Returns (score, confirmed, drop_magnitude, spo2_available).
    spo2_available is True if at least one has_spo2==1 row existed in
    the baseline OR event+lag window.
    """
    n = len(df)

    # ── Pre-event baseline ────────────────────────────────────────────────────
    pre_spo2 = []
    baseline_has_spo2 = False
    for i in range(max(0, seg_i - 3), seg_i):
        row = df.iloc[i]
        if int(float(row.get("has_spo2", 0) or 0)) == 0:
            continue
        baseline_has_spo2 = True
        v = float(row.get("spo2_mean", np.nan) or np.nan)
        if np.isfinite(v) and v > 50:
            pre_spo2.append(v)

    if pre_spo2:
        baseline_spo2 = float(np.mean(pre_spo2))
    else:
        # Fall back to global 90th-pct of has_spo2=1 segments
        spo2_rows = df[df["has_spo2"].astype(float) == 1]["spo2_mean"]
        if len(spo2_rows) > 0:
            baseline_spo2 = float(spo2_rows.quantile(0.90))
        else:
            return 0.0, False, 0.0, False

    # ── Event window + lag ────────────────────────────────────────────────────
    best_drop = 0.0
    event_has_spo2 = False
    for i in range(seg_i, min(n, seg_i + lag + 1)):
        row = df.iloc[i]
        if int(float(row.get("has_spo2", 0) or 0)) == 0:
            continue
        event_has_spo2 = True
        # Check both min and mean — take whichever gives the larger drop
        for col in ["spo2_min", "spo2_mean"]:
            v = float(row.get(col, np.nan) or np.nan)
            if np.isfinite(v) and v > 50:
                drop = baseline_spo2 - v
                if drop > best_drop:
                    best_drop = drop

    spo2_available = baseline_has_spo2 or event_has_spo2

    if not spo2_available or baseline_spo2 is None:
        return 0.0, False, 0.0, False

    confirmed = best_drop >= MIN_SPO2_DROP

    if best_drop <= 0.0:
        return 0.0, False, 0.0, True
    elif not confirmed:
        # Partial credit for 1–3% drops
        score = W_SPO2 * max(0.0, (best_drop - 1.0) / (MIN_SPO2_DROP - 1.0)) * 0.4
    else:
        # Full credit, scaled by severity: 3% → base, 6%+ → full
        score = W_SPO2 * (0.6 + 0.4 * min((best_drop - MIN_SPO2_DROP) / 3.0, 1.0))

    return min(score, float(W_SPO2)), confirmed, round(best_drop, 2), True


def _combine_scores(
    sc1: float,
    sc2: float,
    sc3: float,
    sc4: float,
    spo2_available: bool,
) -> float:
    """
    Combine the four raw criterion sub-scores into a 0-100 total,
    redistributing SpO2's weight across the other three criteria when
    SpO2 was structurally unavailable for this segment.

    sc1 = Primary criterion score (max of ratio and HR drop) (0 - W_HR_RR)
    sc2 = SpO2 score                (0 - W_SPO2, always 0.0 if unavailable)
    sc3 = Respiratory score        (0 - W_RESP)
    sc4 = HR dip-surge score       (0 - W_DIP_SURGE)
    """
    if spo2_available:
        return round(sc1 + sc2 + sc3 + sc4, 2)

    # Redistribute SpO2's 35 points proportionally across the other 3 criteria
    other_weight_sum = W_HR_RR + W_RESP + W_DIP_SURGE  # 65
    hr_eff   = W_HR_RR     + W_SPO2 * (W_HR_RR     / other_weight_sum)
    resp_eff = W_RESP      + W_SPO2 * (W_RESP      / other_weight_sum)
    dip_eff  = W_DIP_SURGE + W_SPO2 * (W_DIP_SURGE / other_weight_sum)

    sc1_scaled = sc1 * (hr_eff   / W_HR_RR)
    sc3_scaled = sc3 * (resp_eff / W_RESP)
    sc4_scaled = sc4 * (dip_eff  / W_DIP_SURGE)

    return round(sc1_scaled + sc3_scaled + sc4_scaled, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 3 — RESPIRATORY SUPPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def _score_resp(
    df:                   pd.DataFrame,
    seg_i:                int,
    global_resp_baseline: float = 0.0,
) -> Tuple[float, bool]:
    """
    Check for suppressed respiratory amplitude and elevated variability.
    EDR-derived features: resp_amplitude_mean, resp_rate_variability.

    Returns (score, suppressed).
    """
    n = len(df)

    def _get(col: str, idx: int) -> float:
        if idx < 0 or idx >= n:
            return np.nan
        v = df.iloc[idx].get(col, np.nan)
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    # Pre-event baseline resp amplitude (local median over up to 10 segs)
    pre_amp = [_get("resp_amplitude_mean", i)
               for i in range(max(0, seg_i - 10), seg_i)]
    pre_amp = [v for v in pre_amp if np.isfinite(v) and v > 0]

    if pre_amp:
        local_baseline = float(np.median(pre_amp))
        baseline_amp   = max(local_baseline, global_resp_baseline * 0.5)
    else:
        baseline_amp = global_resp_baseline

    event_amp = _get("resp_amplitude_mean", seg_i)

    if not np.isfinite(event_amp) or baseline_amp <= 0:
        return 0.0, False

    drop_frac  = 1.0 - (event_amp / (baseline_amp + 1e-9))
    suppressed = drop_frac >= MIN_RESP_DROP

    # Resp rate variability — should be elevated during apnea
    event_rrv = _get("resp_rate_variability", seg_i)
    pre_rrv   = [_get("resp_rate_variability", i)
                 for i in range(max(0, seg_i - HR_RR_WINDOW), seg_i)]
    pre_rrv   = [v for v in pre_rrv if np.isfinite(v)]
    baseline_rrv = float(np.mean(pre_rrv)) if pre_rrv else np.nan

    rrv_elevated = (
        np.isfinite(event_rrv)
        and np.isfinite(baseline_rrv)
        and event_rrv > baseline_rrv * 1.2
    )

    score = 0.0
    if suppressed:
        score += W_RESP * 0.7 * min(drop_frac / 0.4, 1.0)
    if rrv_elevated:
        score += W_RESP * 0.3

    return min(score, float(W_RESP)), suppressed


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION 4 — HR DIP-SURGE PATTERN  (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _score_dip_surge(
    df:    pd.DataFrame,
    seg_i: int,
) -> Tuple[float, bool, bool]:
    """
    Check for classic bradycardia during apnea → tachycardia at termination.
    This is a corroborating criterion; criterion 1 is the primary HR check.

    Returns (score, dip_confirmed, surge_confirmed).
    """
    baseline_hr = _compute_hr_baseline(df, seg_i)
    event_hr    = _get_hr(df, seg_i)
    post_hr     = _get_hr(df, seg_i + 1)

    if not np.isfinite(baseline_hr):
        return 0.0, False, False

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

    return min(score, float(W_DIP_SURGE)), dip_confirmed, surge_confirmed


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER-LEVEL RECOVERY SURGE BONUS
# ══════════════════════════════════════════════════════════════════════════════

def _find_clusters(df: pd.DataFrame, predicted_idxs: List[int],
                   gap_tolerance: int = CLUSTER_GAP_TOLERANCE) -> List[List[int]]:
    """
    Group predicted apnea segment indices into clusters, where segments
    separated by up to `gap_tolerance` non-predicted segments are treated
    as the same cluster.

    Returns a list of clusters, each a sorted list of segment indices.
    """
    if not predicted_idxs:
        return []
    idxs = sorted(predicted_idxs)
    clusters = [[idxs[0]]]
    for i in idxs[1:]:
        if i - clusters[-1][-1] <= gap_tolerance + 1:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _cluster_recovery_surge(df: pd.DataFrame, cluster: List[int],
                            max_search: int = MAX_SURGE_SEARCH) -> Tuple[bool, float]:
    """
    Look for an HR surge at the first clean (non-apnea-predicted)
    segment after the cluster ends. Searches up to `max_search`
    segments past the cluster end in case of a 1-segment gap that's
    itself still elevated.

    Returns (surge_found: bool, surge_bpm: float)
    """
    cluster_end = cluster[-1]
    pre_cluster_start = cluster[0]

    baseline_hr = _get_hr(df, pre_cluster_start - 1)
    if not np.isfinite(baseline_hr):
        # fall back to mean HR over up to 3 segments before cluster start
        pre = [_get_hr(df, i) for i in range(max(0, pre_cluster_start - 3),
                                               pre_cluster_start)]
        valid = [v for v in pre if np.isfinite(v)]
        baseline_hr = float(np.mean(valid)) if valid else np.nan

    best_surge = 0.0
    for offset in range(1, max_search + 1):
        idx = cluster_end + offset
        if idx >= len(df):
            break
        hr = _get_hr(df, idx)
        if np.isfinite(hr) and np.isfinite(baseline_hr):
            surge = hr - baseline_hr
            best_surge = max(best_surge, surge)

    return best_surge >= MIN_HR_SURGE, round(best_surge, 2)


def _apply_cluster_surge_bonus(df: pd.DataFrame, events: List[EventValidation]) -> None:
    """
    Mutates `events` in place: adds a capped corroborating bonus to
    every event in a cluster where a genuine post-cluster recovery
    surge was detected.
    """
    predicted_idxs = [e.segment_idx for e in events]
    clusters = _find_clusters(df, predicted_idxs)
    ev_by_seg = {e.segment_idx: e for e in events}

    for cluster in clusters:
        if len(cluster) < 2:
            continue  # single isolated events already handled by per-segment surge check

        surge_found, surge_bpm = _cluster_recovery_surge(df, cluster)
        if not surge_found:
            continue

        for seg in cluster:
            ev = ev_by_seg.get(seg)
            if ev is None:
                continue
            boosted = min(ev.total_score + CLUSTER_BONUS, CLUSTER_BONUS_CAP)
            # Never let the bonus alone manufacture a score out of nothing;
            # only apply if the segment already has SOME independent support
            if ev.total_score > 0:
                ev.total_score = round(boosted, 2)
                ev.cluster_bonus_applied = True
                # re-derive verdict using the same thresholds
                if ev.total_score >= CONFIRMED:
                    ev.verdict, ev.verdict_short = "CONFIRMED", "✓"
                elif ev.total_score >= PROBABLE:
                    ev.verdict, ev.verdict_short = "PROBABLE", "~"
                elif ev.total_score >= UNCERTAIN:
                    ev.verdict, ev.verdict_short = "UNCERTAIN", "?"
                else:
                    ev.verdict, ev.verdict_short = "UNCONFIRMED", "✗"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA COMPLETENESS ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def _annotate_data_completeness(df: pd.DataFrame, events: List[EventValidation],
                                max_search: int = MAX_SURGE_SEARCH) -> None:
    """
    Mutates `events` in place, setting data_completeness on each based
    on (a) whether the recording had enough trailing segments after this
    event's cluster to search for a recovery surge, and (b) whether
    SpO2 was available for this specific segment.
    """
    n = len(df)
    predicted_idxs = [e.segment_idx for e in events]
    clusters = _find_clusters(df, predicted_idxs)
    ev_by_seg = {e.segment_idx: e for e in events}

    # map segment -> which cluster it belongs to, and that cluster's end
    cluster_end_for_seg = {}
    for cluster in clusters:
        for seg in cluster:
            cluster_end_for_seg[seg] = cluster[-1]

    for ev in events:
        seg = ev.segment_idx
        cluster_end = cluster_end_for_seg.get(seg, seg)

        trailing_insufficient = (cluster_end + max_search) >= n

        # spo2_available should have been set by _score_spo2_with_availability
        spo2_insufficient = not ev.spo2_available

        if trailing_insufficient and spo2_insufficient:
            ev.data_completeness = "insufficient_both"
        elif trailing_insufficient:
            ev.data_completeness = "insufficient_trailing"
        elif spo2_insufficient:
            ev.data_completeness = "insufficient_spo2"
        else:
            ev.data_completeness = "complete"


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


def _print_verbose_line(ev: EventValidation) -> None:
    """
    Verbose line printer, called AFTER cluster bonus + data-completeness
    annotation so output matches the final summary.
    """
    tag = "" if ev.data_completeness == "complete" else f"  [DATA:{ev.data_completeness}]"
    
    # Show actual computed values whenever available, regardless of
    # which mechanism "won" via max() -- method_used still tells you
    # which one contributed the score, but both are worth seeing.
    if ev.ratio_baseline is not None and ev.ratio_event is not None:
        sign = "+" if ev.ratio_increase_pct >= 0 else ""
        ratio_str = f"ratio:{ev.ratio_baseline:.1f}→{ev.ratio_event:.1f}({sign}{ev.ratio_increase_pct:.0f}%)"
    else:
        ratio_str = "ratio:NA"
    
    if ev.hr_drop_bpm is not None:
        hr_str = f"HRdrop:↓{ev.hr_drop_bpm:.1f}bpm"
    else:
        hr_str = "HRdrop:NA"
    
    method_str = f"{ev.method_used} [{ratio_str} | {hr_str}]"
    
    logger.info(
        "  seg=%3d  %s  score=%.1f  "
        "C1=%s  SpO2=%s(↓%.1f%%,avail=%s)  Resp=%s  DipSurge=%s  prob=%.2f%s",
        ev.segment_idx,
        f"{ev.verdict_short} {ev.verdict:<12}",
        ev.total_score,
        method_str,
        "✓" if ev.spo2_drop_confirmed else "✗", ev.spo2_drop_magnitude,
        "✓" if ev.spo2_available else "✗",
        "✓" if ev.resp_suppressed else "✗",
        "✓" if (ev.hr_dip_confirmed or ev.hr_surge_confirmed) else "✗",
        ev.apnea_prob,
        tag
    )


# ══════════════════════════════════════════════════════════════════════════════
#  AHI RANGE REPORTING (TEMPORARY)
# ══════════════════════════════════════════════════════════════════════════════

def _report_ahi_range(result: AdmissionValidation, events: List[EventValidation]) -> Dict:
    """
    Reports validated AHI as a range instead of a single number, so an
    admission with a lot of data-incomplete (flagged-for-review) events
    doesn't silently under-report severity.

    Returns dict with lower_ahi, upper_ahi, n_flagged_incomplete.
    """
    duration_hr = max(result.duration_min / 60.0, 1e-6)

    n_confirmed_probable = result.confirmed + result.probable
    n_flagged_incomplete = sum(
        1 for e in events
        if e.verdict in ("UNCONFIRMED", "UNCERTAIN")
        and getattr(e, "data_completeness", "complete") != "complete"
    )

    lower_ahi = round(n_confirmed_probable / duration_hr, 1)
    upper_ahi = round((n_confirmed_probable + n_flagged_incomplete) / duration_hr, 1)

    logger.info("=" * 60)
    logger.info("  AHI RANGE (TEMPORARY REPORTING)")
    logger.info("=" * 60)
    logger.info("  Validated AHI (confirmed): %s /hr", lower_ahi)
    logger.info("  Validated AHI (if all %d flagged segments", n_flagged_incomplete)
    logger.info("    turn out real):          %s /hr", upper_ahi)
    logger.info("  Reported range:            %s - %s /hr", lower_ahi, upper_ahi)

    if lower_ahi < 5.0 <= upper_ahi:
        logger.info("  >>> FLAG: PROBABLE MILD APNEA -- upper bound crosses AHI=5")
        logger.info("      threshold. Manual review of flagged segments recommended")
        logger.info("      before finalizing severity classification.")
    elif lower_ahi >= 5.0:
        logger.info("  >>> Lower bound already >= 5 -- apnea confirmed regardless")
        logger.info("      of flagged-segment outcome.")
    else:
        logger.info("  >>> Both bounds < 5 -- likely normal, but review flagged")
        logger.info("      segments if clinically indicated.")
    logger.info("=" * 60)

    return {
        "lower_ahi": lower_ahi,
        "upper_ahi": upper_ahi,
        "n_flagged_incomplete": n_flagged_incomplete
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD BREAKDOWN REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _log_method_breakdown(events: List[EventValidation]) -> None:
    """
    Prints method_used distribution for calibration across admissions.
    """
    counts = {"ratio": 0, "hr_drop": 0, "both": 0, "neither": 0}
    for e in events:
        m = getattr(e, "method_used", "neither")
        counts[m] = counts.get(m, 0) + 1

    logger.info("-" * 60)
    logger.info("  Primary criterion method_used breakdown:")
    for m in ("ratio", "hr_drop", "both", "neither"):
        logger.info("    %3d events used: %s", counts[m], m)
    logger.info("-" * 60)


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

    # Filter to one admission BEFORE computing any baselines
    if admission_id and "admission_id" in df.columns:
        df = df[df["admission_id"] == admission_id].reset_index(drop=True)

    # Global resp baseline computed on the filtered admission
    global_resp_baseline = float(
        df["resp_amplitude_mean"].replace(0, np.nan).median()
        if "resp_amplitude_mean" in df.columns else 0.0
    )
    if not np.isfinite(global_resp_baseline):
        global_resp_baseline = 0.0

    adm_id = (str(df["admission_id"].iloc[0])
               if "admission_id" in df.columns and len(df) > 0
               else "UNKNOWN")

    # Duration — prefer wall-clock timestamps
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
    original_ahi   = n_predicted / max(dur_min / 60.0, 1e-6)

    logger.info("[VALIDATE] %s — %d predicted apnea events over %.1f min (AHI=%.1f)",
                adm_id, n_predicted, dur_min, original_ahi)

    if n_predicted == 0:
        logger.info("[VALIDATE] No apnea predictions to validate.")
        return AdmissionValidation(
            admission_id=adm_id,
            duration_min=round(dur_min, 1),
            original_ahi=round(original_ahi, 1),
        )

    # ── PASS 1: Compute scores for all events (silently) ─────────────────────
    events: List[EventValidation] = []

    for seg_i in predicted_idxs:
        row = df.iloc[seg_i]
        ev  = EventValidation(
            segment_idx       = int(row.get("segment_idx", seg_i)),
            timestamp         = str(row.get("timestamp", "")),
            apnea_prob        = float(row.get("apnea_prob", 0) or 0),
            hr_at_event       = float(row.get("ecg_hr_bpm", 0) or
                                      row.get("mean_hr", 0) or 0),
            rr_mean_at_event  = float(row.get("rr_mean", 0) or 0),
            spo2_at_event     = float(row.get("spo2_mean", 0) or 0),
            resp_amp_at_event = float(row.get("resp_amplitude_mean", 0) or 0),
        )

        # ── Criterion 1: Combined primary (ratio + HR drop, take max) ──────
        sc1, c1_info = _score_primary_combined(df, seg_i)
        ev.score_hr_rr = round(sc1, 2)
        ev.method_used = c1_info["method_used"]
        ev.ratio_score = c1_info["ratio_score"]
        ev.ratio_baseline = c1_info.get("ratio_baseline")
        ev.ratio_event = c1_info.get("ratio_event")
        ev.ratio_increase_pct = c1_info.get("ratio_increase_pct")
        ev.hr_drop_score = c1_info["hr_drop_score"]
        ev.hr_dropped = c1_info.get("hr_dropped", False)
        ev.hr_surged = c1_info.get("hr_surged", False)
        ev.hr_drop_bpm = c1_info.get("hr_drop_bpm", None)

        # ── Criterion 2: SpO2 desaturation (with availability) ──────────────
        sc2, spo2_confirmed, spo2_drop, spo2_available = _score_spo2_with_availability(df, seg_i)
        ev.score_spo2          = round(sc2, 2)
        ev.spo2_drop_confirmed = spo2_confirmed
        ev.spo2_drop_magnitude = spo2_drop
        ev.spo2_available      = spo2_available

        # ── Criterion 3: Respiratory suppression ─────────────────────────────
        sc3, resp_suppressed = _score_resp(df, seg_i, global_resp_baseline)
        ev.score_resp    = round(sc3, 2)
        ev.resp_suppressed = resp_suppressed

        # ── Criterion 4: HR dip-surge (fallback) ─────────────────────────────
        sc4, dip, surge = _score_dip_surge(df, seg_i)
        ev.score_dip_surge    = round(sc4, 2)
        ev.hr_dip_confirmed   = dip
        ev.hr_surge_confirmed = surge

        # ── Total with dynamic reweighting ────────────────────────────────────
        ev.total_score = _combine_scores(sc1, sc2, sc3, sc4, spo2_available)
        ev.verdict, ev.verdict_short = _verdict(ev.total_score)

        events.append(ev)

    # ── PASS 2: Apply cluster-level recovery surge bonus ─────────────────────
    _apply_cluster_surge_bonus(df, events)

    # ── PASS 3: Annotate data completeness ───────────────────────────────────
    _annotate_data_completeness(df, events)

    # ── PASS 4: Print verbose lines (if requested) ──────────────────────────
    if verbose:
        for ev in events:
            _print_verbose_line(ev)

    # ── PASS 5: Aggregate ─────────────────────────────────────────────────────
    verdicts    = [e.verdict for e in events]
    n_confirmed = verdicts.count("CONFIRMED")
    n_probable  = verdicts.count("PROBABLE")
    n_uncertain = verdicts.count("UNCERTAIN")
    n_unconf    = verdicts.count("UNCONFIRMED")
    mean_score  = float(np.mean([e.total_score for e in events]))

    # Standard validated AHI (lower bound - confirmed + probable only)
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
        upper_ahi       = round(validated_ahi, 1),  # will be updated below
        original_ahi    = round(original_ahi, 1),
        duration_min    = round(dur_min, 1),
        events          = events,
    )

    _log_summary(result)

    # ── PASS 6: AHI range reporting (temporary) ──────────────────────────────
    ahi_range = _report_ahi_range(result, events)
    result.upper_ahi = ahi_range["upper_ahi"]

    # ── PASS 7: Method breakdown for calibration ─────────────────────────────
    _log_method_breakdown(events)

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

    # ── Data completeness addendum ────────────────────────────────────────────
    unresolved = [e for e in r.events if e.verdict in ("UNCONFIRMED", "UNCERTAIN")]
    if unresolved:
        flagged = [e for e in unresolved if e.data_completeness != "complete"]
        genuine = [e for e in unresolved if e.data_completeness == "complete"]

        logger.info("-" * 60)
        logger.info("  Of %d UNCONFIRMED/UNCERTAIN events:", len(unresolved))
        logger.info("    %d checked with complete data -> genuinely unsupported", len(genuine))
        logger.info("    %d flagged for review -> validator could not fully check:", len(flagged))
        for reason in ("insufficient_trailing", "insufficient_spo2", "insufficient_both"):
            count = sum(1 for e in flagged if e.data_completeness == reason)
            if count:
                label = {
                    "insufficient_trailing": "recording ended before recovery window",
                    "insufficient_spo2": "SpO2 sensor unavailable",
                    "insufficient_both": "SpO2 unavailable AND recording ended early",
                }[reason]
                segs = [e.segment_idx for e in flagged if e.data_completeness == reason]
                logger.info("      - %2d segments (%s): %s", count, label, segs)
        logger.info("-" * 60)

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
    """
    df = pd.read_csv(infer_csv, low_memory=False)
    ev_lookup = {e.segment_idx: e for e in result.events}

    def _val(seg_i: int, attr: str, default):
        e = ev_lookup.get(int(seg_i))
        return getattr(e, attr, default) if e else default

    df["validation_score"]   = df["segment_idx"].apply(
        lambda i: _val(i, "total_score", np.nan))
    df["validation_verdict"] = df["segment_idx"].apply(
        lambda i: _val(i, "verdict", ""))
    df["method_used"]        = df["segment_idx"].apply(
        lambda i: _val(i, "method_used", "neither"))
    df["ratio_score"]        = df["segment_idx"].apply(
        lambda i: _val(i, "ratio_score", 0.0))
    df["ratio_baseline"]     = df["segment_idx"].apply(
        lambda i: _val(i, "ratio_baseline", np.nan))
    df["ratio_event"]        = df["segment_idx"].apply(
        lambda i: _val(i, "ratio_event", np.nan))
    df["ratio_increase_pct"] = df["segment_idx"].apply(
        lambda i: _val(i, "ratio_increase_pct", np.nan))
    df["hr_drop_score"]      = df["segment_idx"].apply(
        lambda i: _val(i, "hr_drop_score", 0.0))
    df["hr_drop_bpm"]        = df["segment_idx"].apply(
        lambda i: _val(i, "hr_drop_bpm", np.nan))
    df["spo2_drop_pct"]      = df["segment_idx"].apply(
        lambda i: _val(i, "spo2_drop_magnitude", np.nan))
    df["spo2_available"]     = df["segment_idx"].apply(
        lambda i: _val(i, "spo2_available", False))
    df["hr_dropped"]         = df["segment_idx"].apply(
        lambda i: _val(i, "hr_dropped", False))
    df["hr_surged"]          = df["segment_idx"].apply(
        lambda i: _val(i, "hr_surged", False))
    df["resp_suppressed"]    = df["segment_idx"].apply(
        lambda i: _val(i, "resp_suppressed", False))
    df["hr_dip_confirmed"]   = df["segment_idx"].apply(
        lambda i: _val(i, "hr_dip_confirmed", False))
    df["hr_surge_confirmed"] = df["segment_idx"].apply(
        lambda i: _val(i, "hr_surge_confirmed", False))
    df["cluster_bonus_applied"] = df["segment_idx"].apply(
        lambda i: _val(i, "cluster_bonus_applied", False))
    df["data_completeness"] = df["segment_idx"].apply(
        lambda i: _val(i, "data_completeness", "complete"))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("[OUT] Validated CSV → %s", out_path)


def write_validation_to_supabase(result: AdmissionValidation) -> None:
    """
    Update the apnea_results row in Supabase with validation fields.
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
        "validated_ahi_upper":       result.upper_ahi,
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

    out_path = args.out or args.infer.replace(".csv", "_validated.csv")
    save_validated_csv(args.infer, result, out_path)

    if args.write_supabase:
        write_validation_to_supabase(result)


if __name__ == "__main__":
    main()