import random, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Starter_Kit"))
import numpy as np
from engine import best_hand
from pokerbot.core import evaluate, card_to_int, int_to_card, equity

rng = random.Random(1)
deck = [r + s for r in "23456789TJQKA" for s in "shdc"]
N = 20000
hands = [rng.sample(deck, 7) for _ in range(N)]
ref = [best_hand(h[:2], h[2:]) for h in hands]
mine = [int(x) for x in evaluate(np.array([[card_to_int(c) for c in h] for h in hands]))]
bad = 0
for i in range(N - 1):
    a = (ref[i] > ref[i + 1]) - (ref[i] < ref[i + 1])
    b = (mine[i] > mine[i + 1]) - (mine[i] < mine[i + 1])
    if a != b:
        bad += 1
        if bad < 5: print("MISMATCH", hands[i], ref[i], hands[i+1], ref[i+1], mine[i], mine[i+1])
    # category check
    if ref[i][0] != mine[i] >> 20:
        bad += 1
        if bad < 5: print("CAT", hands[i], ref[i], mine[i] >> 20)
# targeted: 5-card evaluation incl. wheel / steel wheel
for h in (["As","2s","3s","4s","5s"], ["Ah","2s","3d","4c","5s"], ["Ts","Js","Qs","Ks","As"]):
    print(h, evaluate(np.array([[card_to_int(c) for c in h]]))[0] >> 20)
print("mismatches:", bad)
X = np.array([[card_to_int(c) for c in h] for h in hands * 5])
t = time.time(); evaluate(X); dt = time.time() - t
print(f"throughput {len(X)/dt/1e6:.2f} M evals/s")
t = time.time()
e = equity([card_to_int("As"), card_to_int("Ad")], [], opp_n=2, n=20000)
print("AA vs random 2:", round(e, 3), "(expected ~0.85)", f"{time.time()-t:.3f}s")
e = equity([card_to_int(c) for c in ("7c","2d")], [], opp_n=2, n=20000); print("72o", round(e,3), "(~0.35)")
