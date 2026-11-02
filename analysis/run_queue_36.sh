#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore; mkdir -p hasil36
CUDA_VISIBLE_DEVICES="" $PY round36_diagnosis_rohira.py --null 10 > hasil36/diagnosis.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil36/diagnosis.log
