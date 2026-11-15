"""Discard Hold'em bot: battle-royale champion policy + live Monte Carlo equity.
Python 3.11, numpy only. Never prints anything but protocol JSON to stdout."""
import json
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # single-threaded BLAS: no startup memory spikes; same decisions
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from pokerbot.agent import LiveOracle, PolicyAgent, genome_from_json  # noqa: E402

RANKS = "23456789TJQKA"


def log(*a):
    try:
        print(*a, file=sys.stderr, flush=True)
    except Exception:
        pass


def safe_action(req):
    """Emergency reply (low time bank or internal error). Always picks from legal_actions."""
    legal = req.get("legal_actions") or []
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal and req.get("to_call", 0) <= max(4, req.get("pot", 0) // 4):
        return {"type": "action", "action": "call"}
    if "fold" in legal:
        return {"type": "action", "action": "fold"}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    if "raise" in legal and "min_raise" in req:
        return {"type": "action", "action": "raise", "amount": int(req["min_raise"])}
    return {"type": "action", "action": "check"}  # unreachable for valid requests


def safe_discard(req):
    cards = req["hole_cards"]
    return {"type": "discard", "card": min(cards, key=lambda c: RANKS.index(c[0]))}


def main():
    with open(os.path.join(HERE, "model.json")) as f:
        genome = genome_from_json(json.load(f)["genome"])
    with open(os.path.join(HERE, "preflop.json")) as f:
        pre = json.load(f)
    oracle = LiveOracle(pre, n=256)
    agent = PolicyAgent(genome, oracle, np.random.default_rng())
    total_hands = 200
    hands_done = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        reply = None
        try:
            if t == "init":
                total_hands = int(msg.get("total_hands", 200))
            elif t == "new_hand":
                agent.on_new_hand(msg)
            elif t in ("action_request", "discard_request"):
                # time budget: keep a 40% safety margin over the remaining hands
                left = msg.get("time_left_ms", 30000)
                per_hand = left / max(total_hands - hands_done + 1, 1)
                oracle.n = 256 if per_hand > 60 else (128 if per_hand > 30 else (48 if per_hand > 12 else 16))
                if left < 1500:
                    reply = safe_action(msg) if t == "action_request" else safe_discard(msg)
                elif t == "action_request":
                    reply = agent.act(msg)
                else:
                    reply = agent.discard(msg)
            elif t == "discard_result":
                agent.on_discard_result(msg)
            elif t == "hand_result":
                hands_done += 1
                agent.on_hand_result(msg)
            elif t == "match_end":
                break
        except Exception as e:  # never crash: fall back to a safe legal reply
            log("error:", repr(e))
            if t == "action_request":
                reply = safe_action(msg)
            elif t == "discard_request":
                reply = safe_discard(msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
