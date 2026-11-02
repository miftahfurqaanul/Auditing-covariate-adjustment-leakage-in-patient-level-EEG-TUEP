#!/bin/bash
# NM without AS (D2): 3 parallel GPU processes (owner's permission 2026-10-24; independent folds, fixed seeds), after the plasmode NM finishes.
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
unset CUDA_VISIBLE_DEVICES; PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
until [ -f hasil42/CHAIN_GPU2_SELESAI ]; do sleep 20; done
for i in 0 1 2; do $PY round44_nm.py --run --shard $i 3 > hasil44/E_nm_shard$i.log 2>&1 & done
wait
$PY round44_nm.py --assemble > hasil44/E_nm.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil44/E_nm.log
touch hasil44/E_SELESAI
