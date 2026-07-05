"""
Supabase Medical Data Importer

Imports:

    ECG
    SPO2
    Pleth

from exported MongoDB JSON files.

Uses utc_timestamp as the canonical ordering key.

Author: Sathvik
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from supabase import create_client, Client
from import_frontierx import import_frontierx, is_frontierx_file
from import_frontierx_spo2 import import_frontierx_spo2

###########################################################
# CONFIG
###########################################################

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL missing")

if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY missing")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

# HTTP inserts are happier with smaller batches
BATCH_SIZE = 25

###########################################################
# JSON HELPERS
###########################################################

def mongo_date(value):

    """
    Converts

    {"$date":"2026-06-26T02:37:17.000Z"}

    into

    2026-06-26T02:37:17.000Z
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return value.get("$date")

    return value


def mongo_long(value):

    if value is None:
        return None

    if isinstance(value, dict):
        if "$numberLong" in value:
            return int(value["$numberLong"])

    return value


###########################################################
# FILE HELPERS
###########################################################

def load_json(path: Path):

    print(f"\nReading {path.name}")

    with open(path, "r") as f:
        return json.load(f)


def discover_recordings():

    """
    Finds pairs like

    ADM123.json

    ADM123_spo2_data.json
    """

    recordings = {}

    for file in UPLOAD_DIR.glob("*.json"):

        stem = file.stem

        if stem.endswith("_spo2_data"):

            admission = stem.replace("_spo2_data", "")

            recordings.setdefault(admission, {})

            recordings[admission]["spo2"] = file

        else:

            recordings.setdefault(stem, {})

            recordings[stem]["ecg"] = file

    return recordings


###########################################################
# UPSERT
###########################################################

def upload_rows(table: str, rows: List[Dict]):

    """
    Generic uploader.

    Uses

        admission_id
        utc_timestamp

    as the unique key.
    """

    if not rows:
        return

    uploaded = 0

    total = len(rows)

    for i in range(0, total, BATCH_SIZE):

        batch = rows[i:i+BATCH_SIZE]

        try:

            (
                supabase
                .table(table)
                .upsert(
                    batch,
                    on_conflict="admission_id,utc_timestamp"
                )
                .execute()
            )

            uploaded += len(batch)

            print(
                f"{table:<15}"
                f"{uploaded}/{total}"
            )

        except Exception as e:

            print(f"\nBatch failed ({table})")

            print(e)

            #
            # Retry row-by-row
            #

            for row in batch:

                try:

                    (
                        supabase
                        .table(table)
                        .upsert(
                            row,
                            on_conflict="admission_id,utc_timestamp"
                        )
                        .execute()
                    )

                except Exception as err:

                    print(
                        "\nSkipped row:",
                        row.get("utc_timestamp")
                    )

                    print(err)

###########################################################
# DEVICE
###########################################################

def upload_device(spo2_docs):

    if not spo2_docs:
        return

    first = spo2_docs[0]

    row = {

        "admission_id": first.get("admissionId"),

        "facility_id": first.get("facilityId"),

        "patient_name": first.get("patientName"),

        "patient_id": first.get("patientId"),

        "age": first.get("age"),

        "gender": first.get("gender"),

        "device_id": first.get("deviceId"),

        "device_name": first.get("deviceName"),

        "battery": (
            first.get("device", {})
                 .get("batteryLevel")
        ),

        "mac_address": (
            first.get("device", {})
                 .get("macAddress")
        )

    }

    print("\nUploading device...")

    try:

        (
            supabase
            .table("devices")
            .upsert(
                row,
                on_conflict="admission_id"
            )
            .execute()
        )

        print("Device uploaded.")

    except Exception as e:

        print(e)


###########################################################
# ECG
###########################################################

def upload_ecg(ecg_docs):

    if not ecg_docs:
        return

    print("\nPreparing ECG...")

    rows = []

    for packet in ecg_docs:

        ####################################################
        # ECG values
        ####################################################

        ecg = packet.get("value", [])

        #
        # Mongo export stores ECG as
        #
        # value:
        # [
        #    [125 samples]
        # ]
        #

        if isinstance(ecg, list):

            if len(ecg):

                if isinstance(ecg[0], list):

                    ecg = ecg[0]

        ####################################################
        # Build row
        ####################################################

        rows.append({

            "admission_id":

                packet.get("admissionId"),

            "utc_timestamp":

                mongo_date(
                    packet.get("utcTimestamp")
                ),

            "ecg_data":

                ecg

        })

    print(
        f"Uploading {len(rows)} ECG packets..."
    )

    upload_rows(
        "ecg_stream",
        rows
    )

    print("ECG upload complete.")

###########################################################
# SPO2 + PLETH
###########################################################

def upload_spo2_and_pleth(spo2_docs):

    if not spo2_docs:
        return

    print("\nPreparing SpO2 & Pleth...")

    spo2_rows = []
    pleth_rows = []

    for packet in spo2_docs:

        timestamp = mongo_date(
            packet.get("utcTimestamp")
        )

        admission = packet.get("admissionId")

        ####################################################
        # SPO2
        ####################################################

        spo2 = packet.get("spo2", {})

        if spo2:

            spo2_rows.append({

                "admission_id":
                    admission,

                "utc_timestamp":
                    timestamp,

                "spo2_value":
                    spo2.get("spo2"),

                "pulse_rate":
                    float(spo2.get("pulseRate"))
                    if spo2.get("pulseRate") is not None
                    else None,

                "pi":
                    float(spo2.get("pi"))
                    if spo2.get("pi") is not None
                    else None,

                #
                # Optional waveform arrays
                #

                "spo2_data":
                    spo2.get("spo2Data", []),

                "pulse_data":
                    spo2.get("prAllData", []),

                "pi_data":
                    spo2.get("piAllData", [])

            })

        ####################################################
        # PLETH
        ####################################################

        pleth = packet.get("pleth", {})

        if pleth:

            waveform = pleth.get("rawData", [])

            if waveform:

                pleth_rows.append({

                    "admission_id":
                        admission,

                    "utc_timestamp":
                        timestamp,

                    "pleth_data":
                        waveform

                })

    ########################################################
    # Upload SPO2
    ########################################################

    print(
        f"\nUploading {len(spo2_rows)} SpO2 packets..."
    )

    upload_rows(
        "spo2_stream",
        spo2_rows
    )

    ########################################################
    # Upload Pleth
    ########################################################

    #
    # Pleth packets are much larger than ECG.
    #
    # Upload one row at a time to avoid REST timeouts.
    #

    print(
        f"\nUploading {len(pleth_rows)} Pleth packets..."
    )

    uploaded = 0

    total = len(pleth_rows)

    for row in pleth_rows:

        try:

            (
                supabase
                .table("pleth_stream")
                .upsert(
                    row,
                    on_conflict="admission_id,utc_timestamp"
                )
                .execute()
            )

            uploaded += 1

            if uploaded % 25 == 0 or uploaded == total:

                print(
                    f"pleth_stream   {uploaded}/{total}"
                )

        except Exception as e:

            print(
                "\nSkipped pleth packet:",
                row["utc_timestamp"]
            )

            print(e)

    print("\nSpO2 upload complete.")
    print("Pleth upload complete.")

###########################################################
# IMPORT ONE ADMISSION
###########################################################

def import_admission(admission, files):

    print("\n" + "=" * 70)
    print(f"Importing Admission : {admission}")
    print("=" * 70)

    #######################################################
    # Load ECG
    #######################################################

    ecg_docs = []

    if "ecg" in files:

        ecg_docs = load_json(files["ecg"])

        print(f"ECG packets : {len(ecg_docs)}")

    #######################################################
    # Load SPO2
    #######################################################

    spo2_docs = []

    if "spo2" in files:

        spo2_docs = load_json(files["spo2"])

        print(f"SPO2 packets : {len(spo2_docs)}")

    #######################################################
    # Device
    #######################################################

    if spo2_docs:

        upload_device(spo2_docs)

    #######################################################
    # ECG
    #######################################################

    if ecg_docs:

        upload_ecg(ecg_docs)

    #######################################################
    # SPO2 + Pleth
    #######################################################

    if spo2_docs:

        upload_spo2_and_pleth(spo2_docs)

    print(f"\nFinished {admission}")


###########################################################
# MAIN
###########################################################

def main():

    import_frontierx(UPLOAD_DIR)
    import_frontierx_spo2(UPLOAD_DIR)

    print("=" * 70)
    print("SUPABASE MEDICAL DATA IMPORTER")
    print("=" * 70)

    if not UPLOAD_DIR.exists():

        print("uploads/ folder not found.")

        return

    recordings = discover_recordings()

    if len(recordings) == 0:

        print("No JSON recordings found.")

        return

    print(f"\nFound {len(recordings)} admission(s)\n")

    success = 0
    failed = 0

    for admission, files in recordings.items():

        try:

            import_admission(
                admission,
                files
            )

            success += 1

        except Exception as e:

            failed += 1

            print("\n")
            print("=" * 70)
            print("IMPORT FAILED")
            print("=" * 70)
            print(admission)
            print(e)

    print("\n")
    print("=" * 70)
    print("IMPORT SUMMARY")
    print("=" * 70)

    print(f"Successful : {success}")
    print(f"Failed     : {failed}")

    print("\nDone.")


###########################################################
# ENTRY POINT
###########################################################

if __name__ == "__main__":

    main()