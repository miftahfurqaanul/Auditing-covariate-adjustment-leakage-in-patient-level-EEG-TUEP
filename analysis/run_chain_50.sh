#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
if [ "$1" = gpu ]; then
  $PY round50_emotiv.py --run XGB > hasil50/B_XGB.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil50/B_XGB.log
  $PY round50_A.py --run XGB > hasil50/A_XGB.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil50/A_XGB.log
  touch hasil50/CHAIN_GPU_SELESAI
else
  export NJOBS=16
  $PY round50_emotiv.py --run LR > hasil50/B_LR.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil50/B_LR.log
  $PY round50_emotiv.py --run RF > hasil50/B_RF.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil50/B_RF.log
  $PY round50_A.py --run RF > hasil50/A_RF.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil50/A_RF.log
  touch hasil50/CHAIN_CPU_SELESAI
fi
