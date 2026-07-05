"""
FrontierX ECG Importer

Converts the new FrontierX multi-file ECG export format into the
SAME packet format already used by import_old_ecg.py, then uploads
it into the existing `ecg_stream` table.

File format (input):

    ECG_<ADMISSION_ID>_<HHMMSS>.json

    {
        "admissionId": "...",
        "window_start_ms": ...,
        "window_end_ms": ...,
        "ECG_CH_A": [ ... thousands of samples ... ],
        "ECG_CH_B": [ ... ]
    }

Output (unchanged schema, identical to import_old_ecg.py):

    ecg_stream
        id
        admission_id
        packet_no
        utc_timestamp
        ecg_data (JSONB)   # list of 125 samples

Packet numbering is continuous across ALL files belonging to the
same admission (sorted by the HHMMSS in the filename) -- it does
NOT restart per file.

Author: Sathvik
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

###########################################################
# CONFIG
###########################################################

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE = 25

SAMPLES_PER_PACKET = 125

# FrontierX channel A is sampled at 125Hz, same as the old format,
# so one packet of 125 samples == 1 second of recording.
PACKET_DURATION_MS = 1000

FRONTIERX_FILENAME_RE = re.compile(
    r"^ECG_(?P<admission_id>[A-Za-z0-9]+)_(?P<hhmmss>\d{6})\.json$"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("import_frontierx")


###########################################################
# DISCOVERY
###########################################################

def is_frontierx_file(path: Path) -> bool:
    """
    Returns True if a file matches the FrontierX naming convention:

        ECG_<ADMISSION_ID>_<HHMMSS>.json
    """

    return bool(FRONTIERX_FILENAME_RE.match(path.name))


def discover_frontierx_recordings(upload_dir: Path) -> Dict[str, List[Path]]:
    """
    Scans upload_dir for FrontierX ECG files and groups them by
    admission ID, sorted by the HHMMSS timestamp in the filename.

    Returns:
        { admission_id: [file1, file2, file3, ...] }  # time-ordered
    """

    grouped: Dict[str, List[Path]] = {}

    for file in upload_dir.glob("ECG_*.json"):

        match = FRONTIERX_FILENAME_RE.match(file.name)

        if not match:
            continue

        admission_id = match.group("admission_id")

        grouped.setdefault(admission_id, []).append(file)

    for admission_id, files in grouped.items():
        files.sort(key=lambda f: FRONTIERX_FILENAME_RE.match(f.name).group("hhmmss"))

    return grouped


###########################################################
# FILE HELPERS
###########################################################

def load_json(path: Path):

    logger.info("Reading %s", path.name)

    with open(path, "r") as f:
        return json.load(f)


def ms_to_iso(ms: Optional[int]) -> Optional[str]:
    """
    Converts an epoch-millisecond timestamp into the same ISO-8601
    "Z" string format used elsewhere in the project
    (e.g. "2026-06-26T02:37:17.000Z").
    """

    if ms is None:
        return None

    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}Z"


###########################################################
# PACKETIZE ONE FILE
###########################################################

def chunk_samples(samples: List, size: int) -> List[List]:
    """
    Splits a flat list of samples into fixed-size chunks.

    The final, incomplete chunk (if any) is dropped, since the
    inference pipeline expects exactly `size` samples per packet.
    """

    chunks = []

    for i in range(0, len(samples), size):

        chunk = samples[i:i + size]

        if len(chunk) == size:
            chunks.append(chunk)

    return chunks


def build_packets_for_file(
    file_data: dict,
    admission_id: str,
    start_packet_no: int,
) -> List[Dict]:
    """
    Converts a single FrontierX file's ECG_CH_A array into a list
    of packet rows in the OLD format, continuing packet numbering
    from `start_packet_no`.
    """

    samples = file_data.get("ECG_CH_A", [])

    if not samples:
        logger.warning(
            "No ECG_CH_A samples found for admission %s, skipping file",
            admission_id,
        )
        return []

    window_start_ms = file_data.get("window_start_ms")

    chunks = chunk_samples(samples, SAMPLES_PER_PACKET)

    packets = []

    for local_index, chunk in enumerate(chunks):

        packet_no = start_packet_no + local_index

        if window_start_ms is not None:
            packet_ms = window_start_ms + (local_index * PACKET_DURATION_MS)
            utc_timestamp = ms_to_iso(packet_ms)
        else:
            utc_timestamp = None

        packets.append({
            "admission_id": admission_id,
            # packet_no is NOT a real column on ecg_stream -- it's kept
            # here only to preserve correct ordering/numbering while
            # building packets. It's stripped out in upload_packets()
            # before the row is sent to Supabase.
            "packet_no": packet_no,
            "utc_timestamp": utc_timestamp,
            "ecg_data": chunk,
        })

    return packets


###########################################################
# BUILD PACKETS FOR ONE ADMISSION (ALL FILES, IN ORDER)
###########################################################

def build_packets_for_admission(
    admission_id: str,
    files: List[Path],
) -> List[Dict]:
    """
    Reads every file for an admission, in time order, and produces
    one continuous, globally-numbered packet stream.
    """

    all_packets: List[Dict] = []

    next_packet_no = 1

    for file in files:

        file_data = load_json(file)

        packets = build_packets_for_file(
            file_data,
            admission_id,
            next_packet_no,
        )

        all_packets.extend(packets)

        next_packet_no += len(packets)

    return all_packets


###########################################################
# UPLOAD (same shape/behaviour as import_old_ecg.py)
###########################################################

ECG_STREAM_CONFLICT_KEY = "admission_id,utc_timestamp"


def strip_internal_fields(row: Dict) -> Dict:
    """
    packet_no is used internally to keep packets ordered while
    building them, but it is NOT a real column on ecg_stream --
    the table is keyed on (admission_id, utc_timestamp), matching
    import_old_ecg.py. This removes it before upload.
    """

    return {k: v for k, v in row.items() if k != "packet_no"}


def upload_packets(rows: List[Dict]):

    if not rows:
        return

    if supabase is None:
        raise RuntimeError("Supabase client not configured (missing env vars)")

    upload_rows = [strip_internal_fields(row) for row in rows]

    total = len(upload_rows)
    uploaded = 0

    for i in range(0, total, BATCH_SIZE):

        batch = upload_rows[i:i + BATCH_SIZE]

        try:

            (
                supabase
                .table("ecg_stream")
                .upsert(
                    batch,
                    on_conflict=ECG_STREAM_CONFLICT_KEY,
                )
                .execute()
            )

            uploaded += len(batch)

            logger.info("ecg_stream (frontierx) %d/%d", uploaded, total)

        except Exception as e:

            logger.error("Batch failed (ecg_stream): %s", e)

            # Retry row-by-row
            for row in batch:

                try:

                    (
                        supabase
                        .table("ecg_stream")
                        .upsert(
                            row,
                            on_conflict=ECG_STREAM_CONFLICT_KEY,
                        )
                        .execute()
                    )

                except Exception as err:

                    logger.error(
                        "Skipped packet (admission %s, utc %s): %s",
                        row.get("admission_id"),
                        row.get("utc_timestamp"),
                        err,
                    )


###########################################################
# PUBLIC ENTRY POINT
###########################################################

def import_frontierx(upload_dir: Path = UPLOAD_DIR):
    """
    Discovers, packetizes, and uploads all FrontierX ECG recordings
    found in `upload_dir`. Safe to call even if no FrontierX files
    are present (no-op).
    """

    recordings = discover_frontierx_recordings(upload_dir)

    if not recordings:
        logger.info("No FrontierX ECG files found.")
        return

    logger.info("Found %d FrontierX admission(s)", len(recordings))

    for admission_id, files in recordings.items():

        logger.info(
            "Importing FrontierX admission %s (%d file(s))",
            admission_id,
            len(files),
        )

        packets = build_packets_for_admission(admission_id, files)

        logger.info(
            "Generated %d packets for admission %s",
            len(packets),
            admission_id,
        )

        upload_packets(packets)

        logger.info("Finished FrontierX admission %s", admission_id)


###########################################################
# ENTRY POINT (standalone use)
###########################################################

if __name__ == "__main__":

    import_frontierx()