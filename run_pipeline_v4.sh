#!/usr/bin/env bash
# v4: continue training from v3 (final_submission). v3 stays the champion unless v4 beats it.
set -e
cd "$(dirname "$0")"
python -m train.battle_royale --pool data/pool2.npz --name br4 --pop 24 --gens ${GENS:-300} --workers 12 --seed 4 \
    --init checkpoints/champion_v3.json \
    --fixed checkpoints/champion_v3.json checkpoints/champion_v1.json checkpoints/v2_robust.json checkpoints/v2_grand.json \
    --robust 0.3 --fixed-mult 3 --resume
python -m train.robust_select --run runs/br4 --pool data/pool2.npz --workers 12 --hof-last 12 \
    --extra checkpoints/champion_v3.json checkpoints/champion_v1.json checkpoints/v2_robust.json checkpoints/v2_grand.json
cp runs/br4/robust_champion.json checkpoints/candidate_v4.json
python -m train.export checkpoints/candidate_v4.json --out submission_v4
echo "=== H2H v4 (A) vs v3 (B) ==="
python -m train.h2h submission_v4/bot.py submission_v3/bot.py --stage 1 2 --seeds 71 72 73 74 75
echo "=== H2H v3 (A) vs v4 (B) ==="
python -m train.h2h submission_v3/bot.py submission_v4/bot.py --stage 1 2 --seeds 71 72 73 74 75
echo PIPELINE_V4_DONE
