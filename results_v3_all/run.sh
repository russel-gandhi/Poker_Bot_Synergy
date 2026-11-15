#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for b in v14_champion_submission submission_v23 vnew3 v5 v6; do
  python -m train.h2h submission_v3/bot.py opponents/$b/bot.py --stage 1 2 --seeds 81 82 83 84 85 --parallel 2 > results_v3_all/${b}_ours_first.txt 2>&1
  python -m train.h2h opponents/$b/bot.py submission_v3/bot.py --stage 1 2 --seeds 81 82 83 84 85 --parallel 2 > results_v3_all/${b}_theirs_first.txt 2>&1
  echo "done $b"
done
echo ALL_DONE
