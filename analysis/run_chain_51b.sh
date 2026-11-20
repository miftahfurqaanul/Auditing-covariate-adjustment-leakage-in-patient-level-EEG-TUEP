#!/bin/bash
# Putaran 51, rencana eksekusi ulang (2026-11-03 ~21:35, atas permintaan pemilik "gunakan CPU dan GPU dengan maksimal"): koordinator rantai lama dihentikan;
# pekerjaan tersisa dibagi shard lewat round51_shard.py (kode analisis tidak berubah). Emotiv RF dan Emotiv XGB tetap berjalan dari rantai lama.
cd "$(dirname "$0")"; PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python; ok() { echo "OK $(date -Iseconds)" >> "$1"; }
for k in 0 1; do
  nohup $PY round51_shard.py plasmode tuep XGB 2 $k > hasil51/tuep_XGB_s$k.log 2>&1 && ok hasil51/tuep_XGB_s$k.log &
  nohup $PY round51_shard.py tang 2 $k 20 > hasil51/tuep_Tang_s$k.log 2>&1 && ok hasil51/tuep_Tang_s$k.log &
done
NJOBS=10 nohup $PY round51_shard.py plasmode tuep RF 1 0 > hasil51/tuep_RF_s0.log 2>&1 && ok hasil51/tuep_RF_s0.log &
( while pgrep -f "round51_plasmode.py --run emotiv RF" > /dev/null; do sleep 20; done
  NJOBS=12 $PY round51_plasmode.py --rho emotiv > hasil51/rho_emotiv.log 2>&1 && ok hasil51/rho_emotiv.log
  NJOBS=12 $PY round51_plasmode.py --rho tuep > hasil51/rho_tuep.log 2>&1 && ok hasil51/rho_tuep.log ) &
wait
