import os
import json
import logging
import random
import numpy as np
import tensorflow as tf
from typing import Dict, List

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_all_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_all_seeds(42)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("apnea_pipeline")

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars to native Python types for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR: str        = os.environ.get("DATA_DIR", "../archive2/")
MIMIC_URL: str       = "https://physionet.org/files/mimic4wdb/0.1.0/"
SLPDB_CACHE_DIR: str = os.path.expanduser("~/.cache/slpdb")
MIMIC_CACHE_DIR: str = os.path.expanduser("~/.cache/mimic4wdb")

FS_MIMIC: int        = 320
FS_ECG: int          = 125
FS_PPG: int          = 120
FS_RESP: int         = 4
SEGMENT_LEN_S: int   = 30
N_MIMIC_RECORDS: int = 60
TIMESTEPS: int       = 10

# SLPDB records and their ECG channel indices
SLPDB_RECORDS: List[str] = [
    "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04",
    "slp14",  "slp16",  "slp32",  "slp37",  "slp41", "slp45",
    "slp48",  "slp59",  "slp60",  "slp61",  "slp66", "slp67x",
]
SLPDB_SIGNAL_MAP: Dict[str, Dict[str, int]] = {
    "default": {"ecg": 0},
    "slp32":   {"ecg": 0},
    "slp41":   {"ecg": 0},
    "slp45":   {"ecg": 0},
    "slp48":   {"ecg": 0},
    "slp66":   {"ecg": 0},
}
SLPDB_APNEA_TOKENS = {"OA", "CA", "MA", "X", "A", "H", "HA"}

# ── Feature columns ───────────────────────────────────────────────────────────
ECG_FEATURE_COLS: List[str] = [
    "rr_mean", "rr_std", "rmssd", "pnn50", "mean_hr", "hr_range", "lf_hf_ratio",
    "resp_rate_bpm", "resp_rate_variability", "flatline_duration_s",
    "resp_amplitude_mean", "resp_amplitude_std",
]
SPO2_FEATURE_COLS: List[str] = [
    "spo2_mean", "spo2_min", "spo2_delta_index", "odi", "t90", "spo2_approx_entropy",
]
ABP_FEATURE_COLS: List[str] = [
    "map_mean", "map_std", "map_variability", "sbp_max", "dbp_min", "pulse_pressure",
]
CROSS_FEATURE_COLS: List[str] = [
    "resp_spo2_lag_s", "ptt_ms", "ecg_resp_coherence",
]
MODALITY_FLAG_COLS: List[str] = [
    "has_spo2",     # 1 = real SpO2, 0 = imputed / missing
    "has_abp",      # 1 = real ABP,  0 = zeros
    "has_resp_gt",  # 1 = GT Resp channel, 0 = EDR fallback
]

APNEA_FEATURE_COLS: List[str] = (
    ECG_FEATURE_COLS
    + SPO2_FEATURE_COLS
    + ABP_FEATURE_COLS
    + CROSS_FEATURE_COLS
    + MODALITY_FLAG_COLS
)
