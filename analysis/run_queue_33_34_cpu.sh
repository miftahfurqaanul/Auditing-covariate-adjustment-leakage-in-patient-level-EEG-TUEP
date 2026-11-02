#!/bin/bash
# CPU chain of rounds 33-34 (version without waiting for round32: round32 had finished; a wait based on pgrep -f also matches the command line of other shells and never ends).
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
CUDA_VISIBLE_DEVICES="" $PY round33_pembanding_v4.py --run 20 10 > hasil33/run.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil33/run.log
while [ ! -f hasil34/GPU_SELESAI ]; do sleep 30; done
CUDA_VISIBLE_DEVICES="" $PY round34_eval_tang.py > hasil34/eval.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil34/eval.log
touch hasil34/CPU_SELESAI
