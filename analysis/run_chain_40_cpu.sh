#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
./run_queue_40.sh cpu A 50
./run_queue_40.sh cpu C 100
$PY round40_rohira_perlipatan.py 30 > hasil40/ROH_cpu.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil40/ROH_cpu.log
touch hasil40/ROH_cpu_SELESAI; touch hasil40/CHAIN_CPU_SELESAI
