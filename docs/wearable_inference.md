# Wearable Device Inference Guide

The true test of an apnea model is taking consumer wearable data (which usually only has a single noisy ECG lead and maybe a PPG sensor) and producing clinical-grade predictions.

This project includes a dedicated toolchain for this exact purpose:
1. `edf_to_pipeline.py`: Converts raw `.edf` files into segmented CSVs.
2. `sleep_filter.py`: Discards daytime / awake periods.
3. `edf_test_loader.py`: Runs the model on the filtered data.

## Step 1: Convert EDF to Pipeline Format
Assuming you have an EDF file from an Apple Watch, Garmin, or custom chest strap:

```bash
python pipeline/edf_to_pipeline.py \
    --edf path/to/wearable_recording.edf \
    --out-dir pipeline/converted/ \
    --fs 125
```
This extracts the ECG channel, cleans it, splits it into 30-second segments, and saves it as `pipeline/converted/recording_ecg.csv`.

## Step 2: Sleep Filtering (Optional but Recommended)
Running an apnea model on daytime data will yield false positives (e.g., breath-holding during a workout or talking). We use actigraphy (or heart-rate variance heuristics if accelerometers aren't available) to filter out awake periods.

```bash
python pipeline/sleep_filter.py \
    --data pipeline/converted/ \
    --hr-threshold 80
```
This reads `recording_ecg.csv`, estimates sleep windows based on heart rate dipping, and writes `recording_ecg_sleep.csv`.

## Step 3: Run Inference
Now we feed the sleep-filtered data into the trained BiLSTM. The pipeline will detect that SpO2/ABP/GT_Resp are missing, set the modality flags appropriately, use EDR for respiration, and output predictions.

```bash
python pipeline/edf_test_loader.py \
    --data pipeline/converted/ \
    --mode infer \
    --model pipeline/apnea_model.keras \
    --scaler pipeline/apnea_scaler.pkl \
    --features pipeline/apnea_feature_cols.json
```

The script will output a per-segment probability sequence and a clinical summary of the night (e.g., AHI estimate).
