"""Behaviour-cloning warm start: fit the policy network to the EquityBot heuristic so the
battle royale starts from a sane player instead of random noise (AlphaStar-style BC -> RL).

    python -m train.warmstart --pool data/pool.npz --out checkpoints/bc.json
"""
import argparse
import os

import numpy as np

from pokerbot.agent import (N_ACT, RAISE_FRAC, new_genome, policy_forward, save_genome, GRAD_KEYS)
from pokerbot.baselines import BASELINES
from pokerbot.sim import DealPool, play_match


def make_recorder(teacher_cls):
    class Recorder(teacher_cls):
        def act(self, req):
            reply = super().act(req)
            record(self, req, reply)
            return reply
    return Recorder


DATA = []


def record(self, req, reply):
    if True:
        legal = req["legal_actions"]
        mask = np.zeros(N_ACT, dtype=bool)
        mask[0] = "fold" in legal
        mask[1] = True
        mask[2:] = "raise" in legal
        if reply["action"] == "fold":
            a = 0
        elif reply["action"] in ("check", "call"):
            a = 1
        elif reply["amount"] >= req["max_raise"]:
            a = 5
        else:
            frac = (reply["amount"] - req["opp_bet"]) / max(req["pot"] + req["to_call"], 1)
            a = 2 + int(np.argmin([abs(frac - f) for f in RAISE_FRAC]))
        DATA.append((self.features(req), mask, a))


def batch_grad(g, X, M, A):
    H = np.tanh(X @ g["W1"] + g["b1"])
    L = H @ g["W2"] + X @ g["Wl"] + g["b2"]
    L = np.where(M, L, -1e9)
    L -= L.max(1, keepdims=True)
    P = np.exp(L) * M
    P /= P.sum(1, keepdims=True)
    loss = -np.log(P[np.arange(len(A)), A] + 1e-12).mean()
    D = P.copy()
    D[np.arange(len(A)), A] -= 1
    D /= len(A)
    dH = (D @ g["W2"].T) * (1 - H * H)
    return loss, {"W2": H.T @ D, "Wl": X.T @ D, "b2": D.sum(0), "W1": X.T @ dH, "b1": dH.sum(0)}


def fit(X, M, A, steps=3000, lr=3e-3, seed=0):
    g = new_genome(np.random.default_rng(seed), 0.05)
    m = {k: np.zeros_like(g[k]) for k in GRAD_KEYS}
    v = {k: np.zeros_like(g[k]) for k in GRAD_KEYS}
    rng = np.random.default_rng(seed)
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(A), 2048)
        loss, gr = batch_grad(g, X[idx], M[idx], A[idx])
        for k in GRAD_KEYS:
            m[k] = 0.9 * m[k] + 0.1 * gr[k]
            v[k] = 0.999 * v[k] + 0.001 * gr[k] ** 2
            g[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        if t % 500 == 0:
            full, _ = batch_grad(g, X, M, A)
            print(f"  bc step {t}: loss {full:.4f}")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool.npz")
    ap.add_argument("--out", default="checkpoints/bc.json")
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--teacher", default="equity", help="baseline style to clone")
    a = ap.parse_args()
    pool = DealPool(a.pool)
    rng = np.random.default_rng(0)
    Rec = make_recorder(BASELINES[a.teacher])
    opps = list(BASELINES)
    for i in range(a.matches):
        deals = rng.choice(pool.n, 200, replace=False)
        name = opps[i % len(opps)]
        play_match(lambda: Rec(None, rng), lambda: BASELINES[name](None, rng), pool, deals)
    D = DATA
    X = np.array([d[0] for d in D]); M = np.array([d[1] for d in D]); A = np.array([d[2] for d in D])
    print(f"BC dataset: {len(A)} decisions, action histogram {np.bincount(A, minlength=N_ACT)}")
    g = fit(X, M, A)
    # agreement
    agree = np.mean([np.argmax(policy_forward(g, X[i], M[i])[0]) == A[i] for i in range(0, len(A), 7)])
    print(f"BC argmax agreement with {a.teacher}: {agree:.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_genome(g, a.out, {"source": "bc_" + a.teacher})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
