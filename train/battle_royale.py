"""Battle Royale league training for Discard Hold'em.

Each generation is a tournament scored exactly like the real competition (duplicate matches,
match points win 1 / draw 0.5 / loss 0, tie-break = capped chips per hand). In it:

  1. Every population member plays every other member + the fixed gauntlet (random, station,
     maniac, nit, equity heuristic) + the Hall of Fame (past champions). All matches in a
     generation share one fresh set of deals (common random numbers -> fairer ranking).
  2. RL inner loop (REINFORCE, duplicate baseline): from every match each member gets a policy
     gradient where the advantage of a deal is its *duplicate* chip total, i.e. how much better it
     played those cards than the opponent did with the same cards. Luck largely cancels.
  3. Elimination: the bottom half of the leaderboard is knocked out.
  4. Survivors (with their RL-updated weights) breed: tournament selection, block crossover,
     self-adaptive Gaussian mutation of weights and of hyper-parameters (lr, sigma, temperature).
  5. Every few generations the champion enters the Hall of Fame, so later generations must keep
     beating old strategies (prevents rock-paper-scissors cycling).

    python -m train.battle_royale --pool data/pool.npz --init checkpoints/bc.json --gens 200
Resume with --resume (reads runs/<name>/state.pkl).
"""
import argparse
import copy
import csv
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

from pokerbot.agent import GRAD_KEYS, PolicyAgent, load_genome, new_genome, policy_grad_logp, save_genome
from pokerbot.baselines import BASELINES
from pokerbot.sim import DealPool, play_match

CAP_PER_HAND = 1.0          # Stage 1: C = 400 over 400 hands
_POOL = None


def _init_worker(path):
    global _POOL
    _POOL = DealPool(path)


def _make(spec, rng, record):
    kind, val = spec
    if kind == "base":
        return lambda: BASELINES[val](None, rng)
    return lambda: PolicyAgent(val, None, rng, record=record)


def _grad(genome, agents, per_deal_adv):
    """REINFORCE over both legs; advantage per deal = duplicate total from this agent's view."""
    acc = {k: np.zeros_like(genome[k]) for k in GRAD_KEYS}
    n = 0
    for ag in agents:
        for hand_id, x, h, p, a in ag.traj:
            adv = per_deal_adv[hand_id - 1]
            if adv == 0.0:
                continue
            gl = policy_grad_logp(genome, x, h, p, a)
            for k in GRAD_KEYS:
                acc[k] += adv * gl[k]
            n += 1
    scale = 1.0 / max(len(per_deal_adv), 1)
    return {k: v * scale for k, v in acc.items()}


def _run_match(task):
    i, j, spec_a, spec_b, deals, seed = task
    rng = np.random.default_rng(seed)
    want_a, want_b = spec_a[0] == "genome", spec_b[0] == "genome"
    net, per, legs = play_match(_make(spec_a, rng, want_a), _make(spec_b, rng, want_b), _POOL, deals)
    dup = (per[0] + per[1]) / 100.0          # A's duplicate total per deal, in 100-chip units
    ga = _grad(spec_a[1], [l[0] for l in legs], dup) if want_a else None
    gb = _grad(spec_b[1], [l[1] for l in legs], -dup) if want_b else None
    return i, j, net, ga, gb


class Member:
    _next = 0

    def __init__(self, genome, lr=0.02, sigma=0.02, parent=None, born=0):
        self.g = genome
        self.lr = lr
        self.sigma = sigma
        self.id = Member._next
        Member._next += 1
        self.parent = parent
        self.born = born
        self.points = 0.0
        self.capped = 0.0
        self.hands = 0
        self.played = 0
        self.grad = None
        self.vs = {}
        self.vsph = {}

    robust_w = 0.0              # >0: fitness adds robust_w * (worst chips/hand vs the panel)
    panel = ()

    def to_dict(self):
        return {"g": self.g, "lr": self.lr, "sigma": self.sigma, "id": self.id,
                "parent": self.parent, "born": self.born}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["g"], d["lr"], d["sigma"], d["parent"], d["born"])
        m.id = d["id"]
        return m

    def reset(self):
        self.points = self.capped = 0.0
        self.hands = self.played = 0
        self.grad = None
        self.vs = {}
        self.vsph = {}

    def worst(self):
        vals = [self.vsph[n] for n in Member.panel if n in self.vsph]
        return min(vals) if vals else 0.0

    def fitness(self):
        pts = self.points / max(self.played, 1)
        if Member.robust_w > 0:
            pts += Member.robust_w * self.worst()
        return (pts, self.capped / max(self.hands, 1))


def mutate(g, sigma, rng, temp_sigma=0.1):
    c = copy.deepcopy(g)
    for k in GRAD_KEYS + ("dw",):
        c[k] = c[k] + rng.normal(0, sigma, c[k].shape) * (1.0 if k != "dw" else 5.0)
    c["temp"] = np.clip(c["temp"] * np.exp(rng.normal(0, temp_sigma)), 0.3, 2.5)
    return c


def crossover(a, b, rng):
    c = copy.deepcopy(a)
    for k in GRAD_KEYS + ("dw", "temp"):
        if k in ("W1", "W2"):          # swap whole hidden units to keep them coherent
            for u in range(a["W1"].shape[1]):
                if rng.random() < 0.5:
                    c["W1"][:, u] = b["W1"][:, u]
                    c["b1"][u] = b["b1"][u]
                    c["W2"][u, :] = b["W2"][u, :]
        elif k != "b1" and rng.random() < 0.5:
            c[k] = b[k].copy()
    return c


def apply_grad(m, clip=5.0):
    if m.grad is None:
        return
    norm = np.sqrt(sum(float((v ** 2).sum()) for v in m.grad.values()))
    s = m.lr * (min(1.0, clip / norm) if norm > 0 else 0.0)
    for k in GRAD_KEYS:
        m.g[k] = m.g[k] + s * m.grad[k]


def score(a, b, net, hands):
    """Competition scoring for one match. a/b are Member or None (external)."""
    pa = 1.0 if net > 0 else (0.5 if net == 0 else 0.0)
    cap = max(-400.0, min(400.0, net))
    for m, p, c in ((a, pa, cap), (b, 1 - pa, -cap)):
        if m is not None:
            m.points += p
            m.capped += c
            m.hands += hands
            m.played += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool.npz")
    ap.add_argument("--init", default="checkpoints/bc.json", help="comma-separated warm-start genomes, spread across the population ('' = random)")
    ap.add_argument("--name", default="br1")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=200)
    ap.add_argument("--deals", type=int, default=200, help="deals per match (Stage 1 = 200)")
    ap.add_argument("--kill", type=float, default=0.5, help="fraction eliminated each generation")
    ap.add_argument("--elite", type=int, default=3, help="top members copied unmutated")
    ap.add_argument("--hof-every", type=int, default=5)
    ap.add_argument("--hof-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--robust", type=float, default=0.0,
                    help="weight of worst-case chips/hand vs the panel (fixed + strong baselines) in fitness")
    ap.add_argument("--panel-baselines", default="value,trapper,sticky,equity,nit")
    ap.add_argument("--fixed-mult", type=int, default=1, help="matches vs fixed opponents use this many x deals")
    ap.add_argument("--fixed", nargs="*", default=[],
                    help="genome files that are permanent opponents in every generation (e.g. old champion)")
    a = ap.parse_args()

    run_dir = os.path.join("runs", a.name)
    os.makedirs(run_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pool_n = DealPool(a.pool).n
    state_path = os.path.join(run_dir, "state.pkl")

    if a.resume and os.path.exists(state_path):
        st = pickle.load(open(state_path, "rb"))
        pop = [Member.from_dict(d) for d in st["pop"]]
        hof, gen0, Member._next = st["hof"], st["gen"] + 1, st["next_id"]
        rng = st["rng"]
        print(f"resumed {a.name} at generation {gen0}")
    else:
        bases = [load_genome(p) for p in a.init.split(",")] if a.init else [new_genome(rng)]
        pop = []
        for k in range(a.pop):
            base = bases[k % len(bases)]
            if k < len(bases):
                g = copy.deepcopy(base)
            elif k < a.pop * 3 // 4:
                g = mutate(base, 0.05 * (1 + k % 3), rng, 0.3)
            else:
                g = new_genome(rng, 0.3)   # random immigrants for diversity
            pop.append(Member(g, lr=float(np.exp(rng.uniform(np.log(0.005), np.log(0.08)))),
                              sigma=float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))))
        hof, gen0 = [], 0

    fixed = [load_genome(p) for p in a.fixed]
    Member.robust_w = a.robust
    Member.panel = tuple([f"fixed{k}" for k in range(len(fixed))] +
                         [b for b in a.panel_baselines.split(",") if b])
    log_path = os.path.join(run_dir, "log.csv")
    new_log = not os.path.exists(log_path) or not a.resume
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["gen", "champ_id", "champ_pts", "champ_chips_per_hand", "mean_pts", "hof_size",
                      "champ_vs_" + "|".join(BASELINES), "secs"])

    with Pool(a.workers, initializer=_init_worker, initargs=(a.pool,)) as wp:
        for gen in range(gen0, a.gens):
            t0 = time.time()
            for m in pop:
                m.reset()
            deals = rng.choice(pool_n, a.deals, replace=False)
            deals_big = rng.choice(pool_n, min(pool_n, a.deals * a.fixed_mult), replace=False)
            ext = [("base", n) for n in BASELINES] + [("genome", g) for g in fixed]                 + [("genome", h["g"]) for h in hof]
            ext_names = list(BASELINES) + [f"fixed{k}" for k in range(len(fixed))]                 + [f"hof{h['id']}" for h in hof]
            tasks = []
            for i in range(len(pop)):
                for j in range(i + 1, len(pop)):
                    tasks.append((i, j, ("genome", pop[i].g), ("genome", pop[j].g), deals,
                                  int(rng.integers(1 << 31))))
                for e, spec in enumerate(ext):
                    dl = deals_big if ext_names[e].startswith("fixed") else deals
                    tasks.append((i, -1 - e, ("genome", pop[i].g), spec, dl, int(rng.integers(1 << 31))))
            for i, j, net, ga, gb in wp.imap_unordered(_run_match, tasks, chunksize=2):
                A = pop[i]
                B = pop[j] if j >= 0 else None
                nd = a.deals * (a.fixed_mult if (j < 0 and ext_names[-1 - j].startswith("fixed")) else 1)
                score(A, B, net, 2 * nd)
                for m, gr in ((A, ga), (B, gb)):
                    if m is not None and gr is not None:
                        if m.grad is None:
                            m.grad = gr
                        else:
                            for k in GRAD_KEYS:
                                m.grad[k] += gr[k]
                if j < 0:
                    A.vs[ext_names[-1 - j]] = net
                    A.vsph[ext_names[-1 - j]] = net / (2.0 * nd)

            ranked = sorted(pop, key=lambda m: m.fitness(), reverse=True)
            champ = ranked[0]
            pts, cph = champ.fitness()
            vs = "|".join(f"{champ.vs.get(n, 0):+.0f}" for n in list(BASELINES) + [f"fixed{k}" for k in range(len(fixed))])
            secs = time.time() - t0
            log.writerow([gen, champ.id, f"{pts:.3f}", f"{cph:.3f}",
                          f"{np.mean([m.fitness()[0] for m in pop]):.3f}", len(hof), vs, f"{secs:.0f}"])
            logf.flush()
            print(f"gen {gen:4d} | champ #{champ.id} fit {pts:.3f} worst {champ.worst():+.2f} chips/hand {cph:+.3f} | "
                  f"vs {vs} | hof {len(hof)} | {secs:.0f}s", flush=True)

            save_genome(champ.g, os.path.join(run_dir, "champion.json"),
                        {"gen": gen, "id": champ.id, "points_rate": pts, "chips_per_hand": cph})
            if gen % a.hof_every == a.hof_every - 1:
                hof.append({"id": champ.id, "gen": gen, "g": copy.deepcopy(champ.g)})
                save_genome(champ.g, os.path.join(run_dir, f"hof_gen{gen:04d}_id{champ.id}.json"),
                            {"gen": gen, "id": champ.id})
                if len(hof) > a.hof_size:
                    hof.pop(0)

            # RL step for everyone (Lamarckian: survivors keep what they learned)
            for m in pop:
                apply_grad(m)

            # Battle royale: eliminate, then breed from survivors
            n_keep = max(2, int(round(len(pop) * (1 - a.kill))))
            survivors = ranked[:n_keep]
            children = []
            while len(survivors) + len(children) < a.pop:
                def pick():
                    c = rng.choice(len(survivors), 2, replace=False)
                    return survivors[min(c)]           # lower index = higher rank
                p1, p2 = pick(), pick()
                g = crossover(p1.g, p2.g, rng) if rng.random() < 0.5 else copy.deepcopy(p1.g)
                sigma = float(np.clip(p1.sigma * np.exp(rng.normal(0, 0.3)), 1e-3, 0.2))
                lr = float(np.clip(p1.lr * np.exp(rng.normal(0, 0.3)), 1e-3, 0.3))
                children.append(Member(mutate(g, sigma, rng), lr, sigma, parent=p1.id, born=gen + 1))
            # survivors outside the elite also get a small perturbation (exploration)
            for r, m in enumerate(survivors):
                if r >= a.elite:
                    m.g = mutate(m.g, m.sigma * 0.5, rng, 0.05)
            pop = survivors + children
            pickle.dump({"pop": [m.to_dict() for m in pop], "hof": hof, "gen": gen, "next_id": Member._next, "rng": rng},
                        open(state_path, "wb"))
    logf.close()
    print("done; champion at", os.path.join(run_dir, "champion.json"))


if __name__ == "__main__":
    main()
