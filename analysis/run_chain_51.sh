#!/bin/bash
# Putaran 51: tiga rantai (CPU plasmode, CPU label nyata, satu proses GPU). Resumable. Lihat PRAREGISTRASI_putaran51.md bagian 7.
cd "$(dirname "$0")"; PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; mkdir -p hasil51
ok() { echo "OK $(date -Iseconds)" >> "$1"; }
( NJOBS=12 $PY round51_plasmode.py --run emotiv RF > hasil51/emotiv_RF.log 2>&1 && ok hasil51/emotiv_RF.log
  NJOBS=12 $PY round51_plasmode.py --rho emotiv > hasil51/rho_emotiv.log 2>&1 && ok hasil51/rho_emotiv.log
  NJOBS=12 $PY round51_plasmode.py --run tuep RF > hasil51/tuep_RF.log 2>&1 && ok hasil51/tuep_RF.log
  NJOBS=12 $PY round51_plasmode.py --rho tuep > hasil51/rho_tuep.log 2>&1 && ok hasil51/rho_tuep.log; touch hasil51/CHAIN_CPU_SELESAI ) &
( CUDA_VISIBLE_DEVICES= NJOBS=4 $PY round51_nyata.py --run > hasil51/nyata.log 2>&1 && ok hasil51/nyata.log; touch hasil51/CHAIN_NYATA_SELESAI ) &
( $PY round51_plasmode.py --run emotiv XGB > hasil51/emotiv_XGB.log 2>&1 && ok hasil51/emotiv_XGB.log
  $PY round51_plasmode.py --run tuep XGB > hasil51/tuep_XGB.log 2>&1 && ok hasil51/tuep_XGB.log
  $PY round51_tang.py --run 20 > hasil51/tuep_Tang.log 2>&1 && ok hasil51/tuep_Tang.log; touch hasil51/CHAIN_GPU_SELESAI ) &
wait
