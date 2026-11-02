#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
./run_queue_40.sh gpu GB 20
./run_queue_40.sh gpu GC 20
$PY round40_nm_perlipatan.py 50 > hasil40/NM_gpu.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil40/NM_gpu.log
touch hasil40/NM_gpu_SELESAI; touch hasil40/CHAIN_GPU_SELESAI
