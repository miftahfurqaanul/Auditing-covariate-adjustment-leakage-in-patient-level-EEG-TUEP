#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
$PY round42_cpu.py --run LR 30 > hasil42/LR.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil42/LR.log; touch hasil42/LR_SELESAI
$PY round42_cpu.py --run RF 30 > hasil42/RF.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil42/RF.log; touch hasil42/RF_SELESAI
touch hasil42/CHAIN_CPU_SELESAI
