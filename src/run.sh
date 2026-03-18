#!/bin/bash

# Define the log file
LOG_FILE="running.txt"

# Clear previous logs
echo "--- Starting Pipeline at $(date) ---" > "$LOG_FILE"

# Activate your virtual environment
# Based on your previous 'ls' output, the path is src/venv.
source src/venv./bin/activate >> "$LOG_FILE" 2>&1

{
    echo "[1/7] Running Step 2 Validation (Initial)..."
    python3 step2_validation.py >> "$LOG_FILE" 2>&1

    echo "[2/7] Running Step 2 Validation (Multi-run)..."
    python3 step2_validation.py --runs 5 >> "$LOG_FILE" 2>&1

    echo "[3/7] Running Step 3 Validation..."
    python3 step3_validation.py >> "$LOG_FILE" 2>&1

    echo "[4/7] Running Calibration Scientist..."
    python3 calibration_scientist.py >> "$LOG_FILE" 2>&1

    echo "[5/7] Running Step 3 Calibration..."
    python3 step3_calibration.py >> "$LOG_FILE" 2>&1

    echo "[6/7] Running Step 4 VHLSS..."
    python3 step4_vhlss.py >> "$LOG_FILE" 2>&1

    echo "[7/7] Running Step 4 Personas IPFP..."
    python3 step4_Personas_IPFP.py >> "$LOG_FILE" 2>&1

    echo "[Final] Running Step 4 IPFP Feedback..."
    python3 step4_Personas_IPFP_feedback.py >> "$LOG_FILE" 2>&1

    echo "--- Pipeline Completed at $(date) ---"
} >> "$LOG_FILE" 2>&1
