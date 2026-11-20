#!/bin/bash
# Putaran 52: lima proses GPU (NM real x3, NM plasmode x2) dan rantai CPU (real lalu null, NJOBS 18)
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python
mkdir -p hasil52
for i in 0 1 2; do nohup $PY round52_nm.py --run real --shard $i 3 > hasil52/nm_real_s$i.log 2>&1 & done
for i in 0 1; do nohup $PY round52_nm.py --run plasmode --shard $i 2 > hasil52/nm_plasmode_s$i.log 2>&1 & done
nohup bash -c "CUDA_VISIBLE_DEVICES='' NJOBS=18 $PY round52_cpu.py --run real > hasil52/cpu_real.log 2>&1; CUDA_VISIBLE_DEVICES='' NJOBS=18 $PY round52_cpu.py --run null > hasil52/cpu_null.log 2>&1; touch hasil52/CHAIN_CPU_SELESAI" > /dev/null 2>&1 &
