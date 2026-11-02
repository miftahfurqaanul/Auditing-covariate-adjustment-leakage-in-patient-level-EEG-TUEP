#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
while ! grep -q "^OK" hasil36/diagnosis.log 2>/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES="" $PY round36b_log.py --null 10 > hasil36/diagnosis_log.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil36/diagnosis_log.log
