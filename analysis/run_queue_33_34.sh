#!/bin/bash
# Queue of rounds 33-34 (run on the owner's instruction, 2026-10-21). Two chains:
#   GPU (satu proses): round34_tang_v4.py --run, lalu --null 20          -> hasil34/run.log, hasil34/null.log
#   CPU: wait for round32 (run by the owner) to finish, then round33, then (after the GPU chain finishes) round34_eval_tang.py
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
case "$1" in
  gpu)
    $PY round34_tang_v4.py --run > hasil34/run.log 2>&1 && echo "RUN_OK $(date -Iseconds)" >> hasil34/run.log
    $PY round34_tang_v4.py --null 20 > hasil34/null.log 2>&1 && echo "NULL_OK $(date -Iseconds)" >> hasil34/null.log
    touch hasil34/GPU_SELESAI ;;
  cpu)
    while pgrep -f "round32_anggaran_setara.py --run" > /dev/null; do sleep 20; done
    CUDA_VISIBLE_DEVICES="" $PY round33_pembanding_v4.py --run 20 10 > hasil33/run.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil33/run.log
    while [ ! -f hasil34/GPU_SELESAI ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES="" $PY round34_eval_tang.py > hasil34/eval.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil34/eval.log
    touch hasil34/CPU_SELESAI ;;
esac
