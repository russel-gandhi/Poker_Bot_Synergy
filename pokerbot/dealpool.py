"""Build the preflop table and a pool of deals with precomputed equity features.

    python -m pokerbot.dealpool --deals 20000 --samples 256 --workers 14

Writes data/preflop.json and data/pool.npz. Uses the same estimator as LiveOracle so the
features the policy sees in training match what the live bot computes.
"""
import argparse
import json
import os
import time
from itertools import combinations
from multiprocessing import Pool

import numpy as np

from .core import equity_batch, iso_key, _PAIRS3

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _pre_chunk(args):
    keys, n, seed = args
    rng = np.random.default_rng(seed)
    my = np.array([[k // 4 * 4 + k % 4 for k in key] for key in keys])  # canonical cards are real cards
    out = []
    for i in range(0, len(my), 4):
        out.extend(equity_batch(my[i:i + 4], np.zeros((0,)), np.zeros((0,)), 3, n, rng).tolist())
    return list(zip([",".join(map(str, k)) for k in keys], out))


def build_preflop(n, workers):
    classes = sorted({iso_key(c) for c in combinations(range(52), 3)})
    print(f"preflop: {len(classes)} suit-isomorphic 3-card classes, {n} samples each")
    chunks = [(classes[i::workers * 4], n, i) for i in range(workers * 4)]
    table = {}
    with Pool(workers) as p:
        for res in p.imap_unordered(_pre_chunk, chunks):
            table.update(res)
    return table


def _deal_chunk(args):
    seed, count, n, pre_table = args
    rng = np.random.default_rng(seed)
    hole = np.zeros((count, 2, 3), dtype=np.int64)
    board = np.zeros((count, 5), dtype=np.int64)
    for d in range(count):
        deck = rng.permutation(52)
        hole[d, 0], hole[d, 1], board[d] = deck[0:3], deck[3:6], deck[6:11]
    pre = np.zeros((count, 2))
    for d in range(count):
        for s in range(2):
            pre[d, s] = pre_table[",".join(map(str, iso_key([int(c) for c in hole[d, s]])))]
    my3 = hole.reshape(count * 2, 3)
    flop = np.repeat(board[:, :3], 2, 0)
    full = np.repeat(board, 2, 0)
    # strength features carry 3 values: vs random, vs top-40% range, vs top-15% range
    f3 = np.stack(equity_batch(my3, flop, np.zeros(0), 3, n, rng, strong=True), -1).reshape(count, 2, 3)
    keep = np.zeros((count, 2, 3))
    turn = np.zeros((count, 2, 3, 3)); river = np.zeros((count, 2, 3, 3))
    for k, (i, j) in enumerate(_PAIRS3):
        pair = my3[:, [i, j]]
        disc = my3[:, [3 - i - j]]
        keep[:, :, k] = equity_batch(pair, flop, disc, 3, n, rng).reshape(count, 2)
        turn[:, :, k] = np.stack(equity_batch(pair, full[:, :4], disc, 2, n, rng, strong=True), -1).reshape(count, 2, 3)
        river[:, :, k] = np.stack(equity_batch(pair, full, disc, 2, n, rng, strong=True), -1).reshape(count, 2, 3)
    return hole, board, pre, f3, keep, turn, river


def build_pool(deals, n, workers, pre_table, seed=12345):
    chunk = 25
    jobs = [(seed + i, min(chunk, deals - i * chunk), n, pre_table)
            for i in range((deals + chunk - 1) // chunk)]
    parts = []
    t = time.time()
    with Pool(workers) as p:
        for k, r in enumerate(p.imap(_deal_chunk, jobs)):
            parts.append(r)
            if k % 40 == 0:
                print(f"  pool {len(parts) * chunk}/{deals} deals  {time.time() - t:.0f}s", flush=True)
    return [np.concatenate([p[i] for p in parts]) for i in range(7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=20000)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--pre-samples", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--out", default=os.path.join(DATA, "pool.npz"))
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    pre_path = os.path.join(DATA, "preflop.json")
    if os.path.exists(pre_path):
        pre = json.load(open(pre_path))
    else:
        t = time.time()
        pre = build_preflop(a.pre_samples, a.workers)
        json.dump(pre, open(pre_path, "w"))
        print(f"preflop table written ({time.time() - t:.0f}s)")
    arrs = build_pool(a.deals, a.samples, a.workers, pre, a.seed)
    np.savez_compressed(a.out, **dict(zip(("hole", "board", "pre", "f3", "keep", "turn", "river"), arrs)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
