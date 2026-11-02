#!/bin/bash
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES="" NJOBS=8
for s in A B C D; do
  $PY round44_cpu.py --run $s > hasil44/$s.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil44/$s.log; touch hasil44/${s}_SELESAI
done
touch hasil44/CHAIN_CPU_SELESAI
