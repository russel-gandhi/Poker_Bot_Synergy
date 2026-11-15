"""Grand Final: the last survivors + every Hall-of-Fame champion play a full round robin in the
Stage 2 format (500 duplicate deals per match, scored with match points then capped chips/hand,
C = 1000), on several fresh deal sets, plus every baseline. The winner becomes final_champion.json.

    python -m train.grand_final --run runs/br1 --pool data/pool.npz --top 8 --sets 3
"""
import argparse
import glob
import os
import pickle
from multiprocessing import Pool

import numpy as np

from pokerbot.agent import load_genome, save_genome
from pokerbot.baselines import BASELINES
from train.battle_royale import _init_worker, _make, CAP_PER_HAND  # noqa: F401
import train.battle_royale as br
from pokerbot.sim import DealPool, play_match


def _match(task):
    i, j, sa, sb, deals, seed = task
    rng = np.random.default_rng(seed)
    net, _, _ = play_match(_make(sa, rng, False), _make(sb, rng, False), br._POOL, deals)
    return i, j, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/br1")
    ap.add_argument("--pool", default="data/pool.npz")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--extra", nargs="*", default=[], help="extra candidate genomes (e.g. the old champion)")
    ap.add_argument("--sets", type=int, default=3)
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()

    cands = []
    st = pickle.load(open(os.path.join(a.run, "state.pkl"), "rb"))
    for m in st["pop"][:a.top]:          # pop is stored survivors-first, ranked
        cands.append((f"pop#{m['id']}", ("genome", m["g"])))
    for p in sorted(glob.glob(os.path.join(a.run, "hof_*.json"))):
        cands.append((os.path.basename(p)[:-5], ("genome", load_genome(p))))
    for p in a.extra:
        cands.append(("extra:" + os.path.basename(p)[:-5], ("genome", load_genome(p))))
    n_c = len(cands)
    field = cands + [(f"base:{b}", ("base", b)) for b in BASELINES]
    print(f"grand final: {n_c} candidates + {len(BASELINES)} baselines, {a.sets} deal sets x {a.deals} deals")

    pool_n = DealPool(a.pool).n
    rng = np.random.default_rng(2026)
    tasks = []
    for s in range(a.sets):
        deals = rng.choice(pool_n, a.deals, replace=False)
        for i in range(n_c):
            for j in range(i + 1, len(field)):
                tasks.append((i, j, field[i][1], field[j][1], deals, int(rng.integers(1 << 31))))
    pts = np.zeros(len(field)); cap = np.zeros(len(field)); played = np.zeros(len(field))
    C = 1000.0
    with Pool(a.workers, initializer=_init_worker, initargs=(a.pool,)) as wp:
        for k, (i, j, net) in enumerate(wp.imap_unordered(_match, tasks, chunksize=1)):
            p = 1.0 if net > 0 else (0.5 if net == 0 else 0.0)
            c = max(-C, min(C, net))
            pts[i] += p; pts[j] += 1 - p
            cap[i] += c; cap[j] -= c
            played[i] += 1; played[j] += 1
            if k % 50 == 0:
                print(f"  {k}/{len(tasks)} matches", flush=True)
    hands = played * 2 * a.deals
    order = sorted(range(n_c), key=lambda i: (pts[i] / played[i], cap[i] / hands[i]), reverse=True)
    lines = ["| rank | bot | match pts | win rate | capped chips/hand |", "|---|---|---|---|---|"]
    for r, i in enumerate(order):
        lines.append(f"| {r + 1} | {field[i][0]} | {pts[i]:.1f}/{played[i]:.0f} | "
                     f"{pts[i] / played[i]:.3f} | {cap[i] / hands[i]:+.3f} |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(a.run, "grand_final.md"), "w") as f:
        f.write(table + "\n")
    win = order[0]
    out = os.path.join(a.run, "final_champion.json")
    save_genome(field[win][1][1], out, {"name": field[win][0], "points": float(pts[win]),
                                        "played": float(played[win])})
    print(f"FINAL CHAMPION: {field[win][0]} -> {out}")


if __name__ == "__main__":
    main()
