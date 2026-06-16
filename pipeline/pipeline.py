"""
pipeline/pipeline.py
====================
Multi-source apnea detection pipeline with modality-aware BiLSTM.

Architecture
------------
  - Shared BiLSTM encoder trained on MIMIC-IV + SLPDB simultaneously
  - Modality flags (has_spo2, has_abp, has_resp_gt) injected after encoding
  - 30% modality dropout on MIMIC batches forces ECG-only robustness
  - At inference: flags route prediction automatically based on available signals

Usage
-----
  python pipeline/pipeline.py
  python pipeline/pipeline.py --fresh
  python pipeline/pipeline.py --save-model
  python pipeline/pipeline.py --no-slpdb
  python pipeline/pipeline.py --slpdb-records slp37 slp41 slp66
"""

import argparse
import sys
from pathlib import Path
import warnings
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.db.database import init_db
from pipeline.modules.config import set_all_seeds, logger
from pipeline.modules.train import run_apnea_module

warnings.filterwarnings("ignore")
set_all_seeds(42)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Delete old data/model and re-run.")
    parser.add_argument("--save-model", action="store_true", help="Save the trained .keras and .pkl files.")
    parser.add_argument("--no-slpdb", action="store_true", help="Skip SLPDB completely.")
    parser.add_argument("--slpdb-records", nargs="+", help="Specific SLPDB records to ingest (e.g. slp01a slp66)")
    return parser.parse_args()

def main():
    args = _parse_args()
    if args.fresh:
        import os
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vitals_pipeline.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info("[FRESH] Deleted %s", db_path)
    init_db()
    run_apnea_module(
        save_model=args.save_model,
        skip_slpdb=args.no_slpdb,
        slpdb_records=args.slpdb_records,
    )

if __name__ == "__main__":
    # Ensure TF avoids fully locking GPU mem if needed
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        for d in physical_devices:
            tf.config.experimental.set_memory_growth(d, True)
    except Exception:
        pass
    
    main()
