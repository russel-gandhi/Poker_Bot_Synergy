#!/usr/bin/env bash
# v3: robust objective (worst case vs a fixed panel of opposite styles), friends' bots NOT used.
set -e
cd "$(dirname "$0")"
python -m train.battle_royale --pool data/pool2.npz --name br3 --pop 24 --gens ${GENS:-80} --workers 12 \
    --init checkpoints/champion_v1.json,checkpoints/v2_robust.json,checkpoints/bc_value.json \
    --fixed checkpoints/champion_v1.json checkpoints/v2_robust.json checkpoints/v2_grand.json \
    --robust 0.25 --fixed-mult 3 --resume
python -m train.robust_select --run runs/br3 --pool data/pool2.npz --workers 12 \
    --extra checkpoints/champion_v1.json checkpoints/v2_robust.json checkpoints/v2_grand.json
echo PIPELINE_V3_DONE
