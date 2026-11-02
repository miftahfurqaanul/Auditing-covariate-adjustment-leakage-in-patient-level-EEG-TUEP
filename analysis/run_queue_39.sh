#!/bin/bash
# Queue of round 39 (2026-10-22). Two chains at the same time; file-based completion markers.  Usage: ./run_queue_39.sh <cpu|gpu> <stage> [n_rep]
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore; mkdir -p hasil39
CH=$1; ST=$2; NR=${3:-20}
$PY round39_simulasi_kebocoran.py --run $ST $CH $NR > hasil39/${ST}_${CH}.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil39/${ST}_${CH}.log
touch hasil39/${ST}_${CH}_SELESAI
