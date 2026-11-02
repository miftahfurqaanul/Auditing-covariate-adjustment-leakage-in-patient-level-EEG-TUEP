#!/bin/bash
# Queue of rounds 36c/37/38 (2026-10-22). GPU (one process): Tang L null, Tang Lp null, paired NM, then Tang L and Lp on real data. CPU: paired Rohira (null, then real data).
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore; mkdir -p hasil36 hasil37 hasil30
case "$1" in
  gpu)
    $PY round37_tang_log.py --null L 20 > hasil37/null_L.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil37/null_L.log
    $PY round37_tang_log.py --null Lp 20 > hasil37/null_Lp.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil37/null_Lp.log
    $PY round38_nm_paired.py --null 20 N0,V4 > hasil30/nm_paired.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil30/nm_paired.log
    $PY round37_tang_log.py --run L > hasil37/run_L.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil37/run_L.log
    $PY round37_tang_log.py --run Lp > hasil37/run_Lp.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil37/run_Lp.log
    touch hasil37/GPU_SELESAI ;;
  cpu)
    CUDA_VISIBLE_DEVICES="" $PY round36c_paired.py --null 30 > hasil36/paired_null.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil36/paired_null.log
    CUDA_VISIBLE_DEVICES="" $PY round36c_paired.py --run > hasil36/paired_run.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil36/paired_run.log
    touch hasil37/CPU_SELESAI ;;
esac
