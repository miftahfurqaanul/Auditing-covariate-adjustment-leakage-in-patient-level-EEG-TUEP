#!/bin/bash
# Queue of round 40. Usage: ./run_queue_40.sh <cpu|gpu> <stage> [n_rep] ; completion markers are file-based (not pgrep).
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore; mkdir -p hasil40
CH=$1; ST=$2; NR=${3:-20}
$PY round40_simulasi_perlipatan.py --run $ST $CH $NR > hasil40/${ST}_${CH}.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil40/${ST}_${CH}.log
touch hasil40/${ST}_${CH}_SELESAI
