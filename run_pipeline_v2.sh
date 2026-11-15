#!/usr/bin/env bash
# v2: range-aware features, 3 new sparring styles, old champion as permanent opponent,
# population seeded from old champion + solid value-style clone. Friends' bots are NOT used here.
set -e
cd "$(dirname "$0")"
NAME=${NAME:-br2}; GENS=${GENS:-120}; POP=${POP:-24}; WORKERS=${WORKERS:-12}
[ -f checkpoints/bc_value.json ] || python -m train.warmstart --pool data/pool2.npz --teacher value --matches 60 --out checkpoints/bc_value.json
python -m train.battle_royale --pool data/pool2.npz --name $NAME --pop $POP --gens $GENS --workers $WORKERS \
    --init checkpoints/champion_v1.json,checkpoints/bc_value.json --fixed checkpoints/champion_v1.json --resume
python -m train.grand_final --run runs/$NAME --pool data/pool2.npz --top 8 --sets 3 --workers $WORKERS \
    --extra checkpoints/champion_v1.json
echo PIPELINE_V2_DONE
