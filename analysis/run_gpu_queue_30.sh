#!/bin/bash
# GPU queue of round 30 (one process at a time): reproduction of one V0 fold -> NM-V4 50 folds -> reduced NM leakage test
cd /home/miftah_f/RTX-5090/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan
PY=/home/miftah_f/RTX-5090/EEG-TUEP-v2.0.1/.venv/bin/python
$PY round30_nm.py --repro > hasil30/nm_repro.log 2>&1
$PY round30_nm.py --run > hasil30/nm_run.log 2>&1
$PY round30_nm.py --null 20 > hasil30/nm_null.log 2>&1
echo QUEUE_DONE > hasil30/queue_done.flag
