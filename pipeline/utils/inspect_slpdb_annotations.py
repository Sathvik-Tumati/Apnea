"""
inspect_slpdb_annotations.py
=============================
Inspects the raw .st annotation files from SLPDB to see exactly
what symbols are used, before fixing evaluate_on_slpdb.py.

Usage
-----
python inspect_slpdb_annotations.py
python inspect_slpdb_annotations.py --records slp37 slp41 slp66
"""

import argparse
import os
from collections import Counter
from typing import List

CACHE_DIR = os.path.expanduser("~/.cache/slpdb")

SLPDB_RECORDS = [
    'slp01a','slp01b','slp02a','slp02b','slp03','slp04',
    'slp14', 'slp16', 'slp32', 'slp37', 'slp41','slp45',
    'slp48', 'slp59', 'slp60', 'slp61', 'slp66','slp67x',
]

try:
    import wfdb
    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False
    print("ERROR: pip install wfdb")


def inspect_record(record_name: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {record_name}")
    print(f"{'='*55}")

    # Try local cache first, then stream
    local_path = os.path.join(CACHE_DIR, record_name)
    local_st   = local_path + ".st"

    try:
        if os.path.exists(local_st):
            ann = wfdb.rdann(local_path, 'st')
            print(f"  Source: local cache ({local_st})")
        else:
            ann = wfdb.rdann(record_name, 'st', pn_dir='slpdb/1.0.0')
            print(f"  Source: PhysioNet stream")
    except Exception as exc:
        print(f"  ERROR loading .st: {exc}")
        return

    # Raw annotation attributes
    print(f"\n  ann attributes: {[a for a in dir(ann) if not a.startswith('_')]}")

    # Symbols
    symbols = ann.symbol if hasattr(ann, 'symbol') else []
    samples = ann.sample if hasattr(ann, 'sample') else []

    print(f"\n  Total annotations : {len(symbols)}")
    print(f"  Symbol type       : {type(symbols)}")

    if len(symbols) > 0:
        sym_list = list(symbols) if not isinstance(symbols, list) else symbols
        counts = Counter(sym_list)
        print(f"\n  Symbol counts:")
        for sym, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    repr={repr(sym):<20} str={str(sym):<20} count={cnt}")

        # Show first 20 annotations
        print(f"\n  First 20 annotations (sample, symbol):")
        sam_list = list(samples) if not isinstance(samples, list) else samples
        for i, (s, sym) in enumerate(zip(sam_list[:20], sym_list[:20])):
            print(f"    [{i:3d}]  sample={int(s):8d}  "
                  f"symbol repr={repr(sym):<20}  str={str(sym)!r}")

    # Also check aux_note if present (some WFDB files put labels there)
    if hasattr(ann, 'aux_note') and ann.aux_note:
        aux = list(ann.aux_note)
        aux_nonempty = [(i, a) for i, a in enumerate(aux) if a]
        print(f"\n  aux_note: {len(aux_nonempty)} non-empty entries")
        for i, a in aux_nonempty[:10]:
            print(f"    [{i:3d}]  {repr(a)}")

    # Check subtype / chan / num if present
    for attr in ('subtype', 'chan', 'num'):
        if hasattr(ann, attr):
            vals = getattr(ann, attr)
            if vals is not None and len(vals) > 0:
                unique_vals = set(list(vals)[:100])
                print(f"\n  {attr}: {unique_vals} (unique from first 100)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", default=None,
                        help="Records to inspect (default: first 6)")
    args = parser.parse_args()

    if not HAS_WFDB:
        return

    records = args.records or SLPDB_RECORDS[:6]
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"\nInspecting {len(records)} SLPDB records ...")
    print(f"Cache dir: {CACHE_DIR}")

    for rec in records:
        inspect_record(rec)

    print(f"\n{'='*55}")
    print("  INSPECTION COMPLETE")
    print(f"{'='*55}")
    print("\nNow look at the 'Symbol counts' for each record.")
    print("The apnea symbols will likely be single letters like:")
    print("  'A' = obstructive apnea")
    print("  'O' = obstructive apnea (alternate)")
    print("  'C' = central apnea")
    print("  'H' = hypopnea")
    print("  'X' = obstructive apnea (some records)")
    print("  ' ' = space-padded or empty string — NOT an apnea event")
    print("\nCheck repr() of symbols — hidden spaces or bytes cause mismatch.")

if __name__ == "__main__":
    main()