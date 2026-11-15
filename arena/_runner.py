"""Protocol loop for running any in-repo agent class as a standalone bot (for official-engine tests)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from pokerbot.agent import LiveOracle
from pokerbot.baselines import BASELINES

def run(name):
    pre = json.load(open(os.path.join(ROOT, "data", "preflop.json")))
    agent = BASELINES[name](LiveOracle(pre, n=128), np.random.default_rng())
    for line in sys.stdin:
        m = json.loads(line); t = m.get("type"); r = None
        if t == "new_hand": agent.on_new_hand(m)
        elif t == "action_request": r = agent.act(m)
        elif t == "discard_request": r = agent.discard(m)
        elif t == "discard_result": agent.on_discard_result(m)
        elif t == "hand_result": agent.on_hand_result(m)
        elif t == "match_end": break
        if r is not None: print(json.dumps(r), flush=True)
