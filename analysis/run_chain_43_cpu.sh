#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES="" NJOBS=8
$PY round43_fm_kompresi.py --run A > hasil43/A.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil43/A.log; touch hasil43/A_SELESAI
$PY round43_fm_kompresi.py --run C 20 > hasil43/C.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil43/C.log; touch hasil43/C_SELESAI
touch hasil43/CHAIN_SELESAI
