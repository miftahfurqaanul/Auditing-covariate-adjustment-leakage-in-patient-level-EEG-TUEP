#!/bin/bash
# Queue of round 35 (2026-10-22). GPU: Tang V4t null, then real data. CPU: Rohira V4t null, then real data, then (after the GPU finishes) the Tang evaluation.
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore; mkdir -p hasil35
case "$1" in
  gpu)
    $PY round35_tang_v4t.py --null 20 > hasil35/tang_null.log 2>&1 && echo "NULL_OK $(date -Iseconds)" >> hasil35/tang_null.log
    $PY round35_tang_v4t.py --run > hasil35/tang_run.log 2>&1 && echo "RUN_OK $(date -Iseconds)" >> hasil35/tang_run.log
    touch hasil35/GPU_SELESAI ;;
  cpu)
    CUDA_VISIBLE_DEVICES="" $PY round35_rohira_v4t.py --null 20 > hasil35/rohira_null.log 2>&1 && echo "NULL_OK $(date -Iseconds)" >> hasil35/rohira_null.log
    CUDA_VISIBLE_DEVICES="" $PY round35_rohira_v4t.py --run > hasil35/rohira_run.log 2>&1 && echo "RUN_OK $(date -Iseconds)" >> hasil35/rohira_run.log
    while [ ! -f hasil35/GPU_SELESAI ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES="" $PY round35_eval_tang.py > hasil35/tang_eval.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil35/tang_eval.log
    touch hasil35/CPU_SELESAI ;;
esac
