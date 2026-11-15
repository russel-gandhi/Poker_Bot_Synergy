"""Robust model selection: full head-to-head matrix (simulator, Stage 2 length), then rank
candidates by their WORST matchup and by average, instead of total points vs a sibling-heavy field.

    python -m train.robust_select --run runs/br2 --pool data/pool2.npz --extra checkpoints/champion_v1.json
"""
import argparse
import glob
import os
import pickle
from multiprocessing import Pool

import numpy as np

import train.battle_royale as br
from pokerbot.genome_io import load_genome, save_genome
from pokerbot.baselines import BASELINES
from pokerbot.sim import DealPool, play_match


def _match(task):
    i, j, sa, sb, deals, seed = task
    rng = np.random.default_rng(seed)
    net, _, _ = play_match(br._make(sa, rng, False), br._make(sb, rng, False), br._POOL, deals)
    return i, j, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/br2")
    ap.add_argument("--pool", default="data/pool2.npz")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--hof-last", type=int, default=8)
    ap.add_argument("--sets", type=int, default=3)
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    cands = []
    st = pickle.load(open(os.path.join(a.run, "state.pkl"), "rb"))
    for m in st["pop"][:a.top]:
        cands.append((f"pop#{m['id']}", ("genome", m["g"])))
    for p in sorted(glob.glob(os.path.join(a.run, "hof_*.json")))[-a.hof_last:]:
        cands.append((os.path.basename(p)[:-5], ("genome", load_genome(p))))
    for p in a.extra:
        cands.append(("extra:" + os.path.basename(p)[:-5], ("genome", load_genome(p))))
    opps = cands + [(f"base:{b}", ("base", b)) for b in BASELINES]
    n_c, n_o = len(cands), len(opps)
    rng = np.random.default_rng(777)
    tasks = []
    for s in range(a.sets):
        deals = rng.choice(DealPool(a.pool).n, a.deals, replace=False)
        for i in range(n_c):
            for j in range(n_o):
                if j > i or j >= n_c:
                    tasks.append((i, j, cands[i][1], opps[j][1], deals, int(rng.integers(1 << 31))))
    M = np.zeros((n_c, n_o))
    with Pool(a.workers, initializer=br._init_worker, initargs=(a.pool,)) as wp:
        for i, j, net in wp.imap_unordered(_match, tasks, chunksize=1):
            per_hand = net / (2 * a.deals * a.sets)
            M[i, j] += per_hand
            if j < n_c:
                M[j, i] -= per_hand
    rows = []
    for i in range(n_c):
        others = [M[i, j] for j in range(n_o) if j != i]
        rows.append((min(others), float(np.mean(others)), i))
    rows.sort(reverse=True)
    names = [o[0] for o in opps]
    print(f"{'candidate':28s} {'worst':>7s} {'mean':>7s}  worst opponent")
    for worst, mean, i in rows:
        j = min((j for j in range(n_o) if j != i), key=lambda j: M[i, j])
        print(f"{cands[i][0]:28s} {worst:+7.2f} {mean:+7.2f}  {names[j]}")
    best = rows[0][2]
    out = os.path.join(a.run, "robust_champion.json")
    save_genome(cands[best][1][1], out, {"name": cands[best][0], "worst": rows[0][0], "mean": rows[0][1]})
    np.save(os.path.join(a.run, "robust_matrix.npy"), M)
    print("ROBUST CHAMPION:", cands[best][0], "->", out)


if __name__ == "__main__":
    main()
