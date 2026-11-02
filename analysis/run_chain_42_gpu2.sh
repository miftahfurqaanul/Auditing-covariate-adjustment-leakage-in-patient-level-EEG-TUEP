#!/bin/bash
# Tang and NM (GPU) after XGB; one GPU process at a time (file-based markers).
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
unset CUDA_VISIBLE_DEVICES; PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; export PYTHONWARNINGS=ignore
until [ -f hasil42/XGB_SELESAI ]; do sleep 20; done
$PY round42_gpu_nn.py --check > hasil42/gpu_check.log 2>&1 || { echo "CHECK FAILED" >> hasil42/gpu_check.log; touch hasil42/GPU2_GAGAL; exit 1; }
$PY round42_gpu_nn.py --run Tang 20 > hasil42/Tang.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil42/Tang.log; touch hasil42/Tang_SELESAI
$PY round42_gpu_nn.py --run NM 20 > hasil42/NM.log 2>&1 && echo "OK $(date -Iseconds)" >> hasil42/NM.log; touch hasil42/NM_SELESAI
touch hasil42/CHAIN_GPU2_SELESAI
