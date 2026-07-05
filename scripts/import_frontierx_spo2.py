"""
FrontierX SpO2 / Pleth Importer

Converts the new FrontierX per-packet SpO2 export format into the
SAME row shapes already produced by import_spo2.py, then uploads
into the existing `spo2_stream` and `pleth_stream` tables (and
optionally `devices`).

File format (input) -- ONE FILE PER PACKET:

    ADM<ADMISSION_ID>_<DEVICE_NAME>_<HHMMSS>_<SEQ>.json

    e.g. ADM1227044863_NISO101_223014_202.json

    {
        "admissionId": "...",
        "epochTime": 1782752412,        # seconds, NOT ms
        "seqNum": 6,
        "seqPart": 1,
        "spo2": {
            "SPO2": 99,
            "PR": 103,
            "PI": 95.0,
            "SPO2_ALL": [...],
            "PR_ALL": [...],
            "PI_ALL": [...]
        },
        "pleth": {
            "PLETH": [...]
        },
        "device": {...},
        "deviceName": "NISO101",
        ...
    }

This is unrelated to the combined-list old format
(ADM<id>_spo2_data.json), which import_spo2.py already handles --
that one is left completely untouched.

Output (unchanged schema, identical row shape to import_spo2.py):

    spo2_stream
        admission_id, utc_timestamp, spo2_value, pulse_rate, pi,
        spo2_data, pulse_data, pi_data

    pleth_stream
        admission_id, utc_timestamp, pleth_data

Author: Sathvik
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
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

SPO2_STREAM_CONFLICT_KEY = "admission_id,utc_timestamp"
PLETH_STREAM_CONFLICT_KEY = "admission_id,utc_timestamp"

# ADM1227044863_NISO101_223014_202.json
FRONTIERX_SPO2_FILENAME_RE = re.compile(
    r"^(?P<admission_id>ADM[A-Za-z0-9]+)_(?P<device>[A-Za-z0-9]+)_"
    r"(?P<hhmmss>\d{6})_(?P<seq>\d+)\.json$"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("import_frontierx_spo2")


###########################################################
# DISCOVERY
###########################################################

def is_frontierx_spo2_file(path: Path) -> bool:
    """
    Returns True if a file matches the FrontierX per-packet SpO2
    naming convention:

        ADM<ADMISSION_ID>_<DEVICE>_<HHMMSS>_<SEQ>.json

    This deliberately does NOT match the old combined-list format
    (ADM<id>_spo2_data.json), since "spo2_data" isn't a 6-digit
    HHMMSS + numeric sequence.
    """

    return bool(FRONTIERX_SPO2_FILENAME_RE.match(path.name))


def discover_frontierx_spo2_recordings(upload_dir: Path) -> Dict[str, List[Path]]:
    """
    Scans upload_dir for FrontierX SpO2 packet files and groups them
    by admission ID, sorted by (HHMMSS, SEQ) from the filename.

    Returns:
        { admission_id: [file1, file2, file3, ...] }  # time-ordered
    """

    grouped: Dict[str, List[Path]] = {}

    for file in upload_dir.glob("ADM*.json"):

        match = FRONTIERX_SPO2_FILENAME_RE.match(file.name)

        if not match:
            continue

        admission_id = match.group("admission_id")

        grouped.setdefault(admission_id, []).append(file)

    for admission_id, files in grouped.items():

        files.sort(
            key=lambda f: (
                FRONTIERX_SPO2_FILENAME_RE.match(f.name).group("hhmmss"),
                int(FRONTIERX_SPO2_FILENAME_RE.match(f.name).group("seq")),
            )
        )

    return grouped


###########################################################
# FILE HELPERS
###########################################################

def load_json(path: Path):

    logger.info("Reading %s", path.name)

    with open(path, "r") as f:
        return json.load(f)


def epoch_seconds_to_iso(epoch_seconds: Optional[int]) -> Optional[str]:
    """
    Converts an epoch-SECOND timestamp (FrontierX's epochTime) into
    the same ISO-8601 "Z" string format used elsewhere in the
    project (e.g. "2026-06-26T02:37:17.000Z").
    """

    if epoch_seconds is None:
        return None

    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)

    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


###########################################################
# ROW BUILDERS (one packet file -> one spo2 row + one pleth row)
###########################################################

def build_spo2_row(packet: dict, admission_id: str, utc_timestamp: Optional[str]) -> Optional[Dict]:

    spo2 = packet.get("spo2", {})

    if not spo2:
        return None

    return {
        "admission_id": admission_id,
        "utc_timestamp": utc_timestamp,
        "spo2_value": spo2.get("SPO2"),
        "pulse_rate": (
            float(spo2["PR"]) if spo2.get("PR") is not None else None
        ),
        "pi": (
            float(spo2["PI"]) if spo2.get("PI") is not None else None
        ),
        "spo2_data": spo2.get("SPO2_ALL", []),
        "pulse_data": spo2.get("PR_ALL", []),
        "pi_data": spo2.get("PI_ALL", []),
    }


def build_pleth_row(packet: dict, admission_id: str, utc_timestamp: Optional[str]) -> Optional[Dict]:

    pleth = packet.get("pleth", {})

    waveform = pleth.get("PLETH", [])

    if not waveform:
        return None

    return {
        "admission_id": admission_id,
        "utc_timestamp": utc_timestamp,
        "pleth_data": waveform,
    }


def build_rows_for_admission(
    admission_id: str,
    files: List[Path],
) -> Dict[str, List[Dict]]:
    """
    Reads every packet file for an admission, in time order, and
    builds the spo2_stream / pleth_stream rows.
    """

    spo2_rows: List[Dict] = []
    pleth_rows: List[Dict] = []

    for file in files:

        packet = load_json(file)

        utc_timestamp = epoch_seconds_to_iso(packet.get("epochTime"))

        spo2_row = build_spo2_row(packet, admission_id, utc_timestamp)

        if spo2_row:
            spo2_rows.append(spo2_row)

        pleth_row = build_pleth_row(packet, admission_id, utc_timestamp)

        if pleth_row:
            pleth_rows.append(pleth_row)

    return {"spo2": spo2_rows, "pleth": pleth_rows}


###########################################################
# DEVICE (optional, mirrors import_spo2.py's upload_device)
###########################################################

def build_device_row(packet: dict) -> Optional[Dict]:

    if not packet:
        return None

    device = packet.get("device", {})

    return {
        "admission_id": packet.get("admissionId"),
        "facility_id": packet.get("facilityId"),
        "patient_name": packet.get("patientName"),
        "patient_id": packet.get("patientId"),
        "age": packet.get("age"),
        "gender": packet.get("gender"),
        "device_id": packet.get("deviceId"),
        "device_name": packet.get("deviceName"),
        "battery": device.get("batteryLevel"),
        "mac_address": device.get("macAddress"),
    }


def upload_device(packet: dict):

    row = build_device_row(packet)

    if not row or supabase is None:
        return

    logger.info("Uploading device (frontierx spo2)...")

    try:

        (
            supabase
            .table("devices")
            .upsert(row, on_conflict="admission_id")
            .execute()
        )

        logger.info("Device uploaded.")

    except Exception as e:

        logger.error("Device upload failed: %s", e)


###########################################################
# UPLOAD (same shape/behaviour as import_spo2.py)
###########################################################

def upload_rows(table: str, rows: List[Dict], conflict_key: str):

    if not rows:
        return

    if supabase is None:
        raise RuntimeError("Supabase client not configured (missing env vars)")

    total = len(rows)
    uploaded = 0

    for i in range(0, total, BATCH_SIZE):

        batch = rows[i:i + BATCH_SIZE]

        try:

            (
                supabase
                .table(table)
                .upsert(batch, on_conflict=conflict_key)
                .execute()
            )

            uploaded += len(batch)

            logger.info("%s (frontierx) %d/%d", table, uploaded, total)

        except Exception as e:

            logger.error("Batch failed (%s): %s", table, e)

            # Retry row-by-row
            for row in batch:

                try:

                    (
                        supabase
                        .table(table)
                        .upsert(row, on_conflict=conflict_key)
                        .execute()
                    )

                except Exception as err:

                    logger.error(
                        "Skipped row (admission %s, utc %s): %s",
                        row.get("admission_id"),
                        row.get("utc_timestamp"),
                        err,
                    )


###########################################################
# PUBLIC ENTRY POINT
###########################################################

def import_frontierx_spo2(upload_dir: Path = UPLOAD_DIR):
    """
    Discovers, converts, and uploads all FrontierX SpO2/Pleth packet
    files found in `upload_dir`. Safe to call even if no FrontierX
    SpO2 files are present (no-op).
    """

    recordings = discover_frontierx_spo2_recordings(upload_dir)

    if not recordings:
        logger.info("No FrontierX SpO2 files found.")
        return

    logger.info("Found %d FrontierX SpO2 admission(s)", len(recordings))

    for admission_id, files in recordings.items():

        logger.info(
            "Importing FrontierX SpO2 admission %s (%d file(s))",
            admission_id,
            len(files),
        )

        # Use the first packet to populate the devices table.
        first_packet = load_json(files[0])
        upload_device(first_packet)

        rows = build_rows_for_admission(admission_id, files)

        logger.info(
            "Generated %d SpO2 rows / %d Pleth rows for admission %s",
            len(rows["spo2"]),
            len(rows["pleth"]),
            admission_id,
        )

        upload_rows("spo2_stream", rows["spo2"], SPO2_STREAM_CONFLICT_KEY)
        upload_rows("pleth_stream", rows["pleth"], PLETH_STREAM_CONFLICT_KEY)

        logger.info("Finished FrontierX SpO2 admission %s", admission_id)


###########################################################
# ENTRY POINT (standalone use)
###########################################################

if __name__ == "__main__":

    import_frontierx_spo2()