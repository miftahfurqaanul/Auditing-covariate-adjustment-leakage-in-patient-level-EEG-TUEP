#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
$PY round42_cpu.py --run XGB 20 > hasil42/XGB.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil42/XGB.log; touch hasil42/XGB_SELESAI
[ -f round42_gpu_nn.sh_ok ] && true
touch hasil42/CHAIN_XGB_SELESAI
