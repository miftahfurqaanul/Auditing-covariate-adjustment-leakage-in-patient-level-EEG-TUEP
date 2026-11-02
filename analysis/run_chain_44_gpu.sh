#!/bin/bash
# NM without AS (D2) after the plasmode (Tang/NM) finishes; one GPU process at a time (file-based markers).
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
unset CUDA_VISIBLE_DEVICES; PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
until [ -f hasil42/CHAIN_GPU2_SELESAI ]; do sleep 30; done
$PY round44_nm.py --run > hasil44/E_nm.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil44/E_nm.log
touch hasil44/E_SELESAI
