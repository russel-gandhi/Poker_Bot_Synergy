#!/usr/bin/env bash
# Full pipeline: deal pool -> BC warm start -> battle royale -> grand final -> export -> official tests
set -e
cd "$(dirname "$0")"
NAME=${NAME:-br1}; GENS=${GENS:-150}; POP=${POP:-24}; WORKERS=${WORKERS:-12}
[ -f data/pool.npz ] || python -m pokerbot.dealpool --deals 40000 --samples 256 --workers 8
[ -f checkpoints/bc.json ] || python -m train.warmstart --pool data/pool.npz --out checkpoints/bc.json --matches 60
python -m train.battle_royale --pool data/pool.npz --init checkpoints/bc.json --name $NAME \
    --pop $POP --gens $GENS --workers $WORKERS --resume
python -m train.grand_final --run runs/$NAME --pool data/pool.npz --top 8 --sets 3 --workers $WORKERS
python -m train.export runs/$NAME/final_champion.json --out submission
python -m train.test_bot --bot submission/bot.py --python "${BOT_PYTHON:-python}"
