"""Policy agent shared by the trainer (in-process sim) and the submitted bot.

The agent consumes protocol messages (dicts exactly as in the Protocol Specification)
and returns protocol replies. Hand strength comes from an *oracle*:
  - LiveOracle: Monte Carlo equity computed on the fly (used by bot.py)
  - the sim's precomputed oracle (same estimator, computed in bulk per deal)

Python 3.10+/3.11 compatible, numpy only.
"""
import json
import math

import numpy as np

from .core import card_to_int, equity, equity2, iso_key, _PAIRS3

N_FEAT = 30
N_FEAT_V1 = 24                           # first-generation models (padded with zeros on load)
N_HID = 16
N_ACT = 6                               # fold, check/call, raise .4/.8/1.5 pot, all-in
RAISE_FRAC = (0.4, 0.8, 1.5)
N_DISC_FEAT = 5


# ------------------------------------------------------------------ oracles
class LiveOracle:
    """Monte Carlo equities, cached per hand. `pre_table` maps iso-key str -> equity."""

    def __init__(self, pre_table, n=256, seed=None):
        self.pre_table = pre_table
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def new_hand(self):
        self.cache = {}

    def _get(self, key, fn):
        v = self.cache.get(key)
        if v is None:
            v = fn()
            self.cache[key] = v
        return v

    def pre(self, h3):
        k = ",".join(map(str, iso_key(h3)))
        v = self.pre_table.get(k)
        if v is None:
            v = equity(list(h3), [], opp_n=3, n=self.n, rng=self.rng)
        return v

    def flop3(self, h3, board):
        """(vs random, vs top-40% range, vs top-15% range)"""
        return self._get(("f3", tuple(sorted(h3))),
                         lambda: equity2(list(h3), list(board[:3]), opp_n=3, n=self.n, rng=self.rng))

    def keep(self, pair, board, discard):
        return self._get(("k", tuple(sorted(pair))),
                         lambda: equity(list(pair), list(board[:3]), dead=[discard], opp_n=3,
                                        n=self.n, rng=self.rng))

    def post(self, pair, board, discard):
        """(vs random, vs top-40% range, vs top-15% range)"""
        return self._get(("p", tuple(sorted(pair)), len(board)),
                         lambda: equity2(list(pair), list(board), dead=[discard], opp_n=2,
                                         n=self.n, rng=self.rng))


# ------------------------------------------------------------------ genome
def new_genome(rng, scale=0.1):
    return {
        "W1": rng.normal(0, scale, (N_FEAT, N_HID)),
        "b1": np.zeros(N_HID),
        "W2": rng.normal(0, scale, (N_HID, N_ACT)),
        "Wl": rng.normal(0, scale, (N_FEAT, N_ACT)),
        "b2": np.zeros(N_ACT),
        "dw": np.zeros(N_DISC_FEAT),
        "temp": np.array([1.0]),
    }


GRAD_KEYS = ("W1", "b1", "W2", "Wl", "b2")


def genome_to_json(g):
    return {k: np.asarray(v).tolist() for k, v in g.items()}


def genome_from_json(d):
    g = {k: np.asarray(v, dtype=float) for k, v in d.items()}
    return pad_genome(g)


def pad_genome(g):
    """Upgrade an older model to the current feature count. New input rows are zero, so the
    upgraded model plays exactly like the old one until training changes those weights."""
    f = g["W1"].shape[0]
    if f < N_FEAT:
        g["W1"] = np.vstack([g["W1"], np.zeros((N_FEAT - f, g["W1"].shape[1]))])
        g["Wl"] = np.vstack([g["Wl"], np.zeros((N_FEAT - f, g["Wl"].shape[1]))])
    return g


def policy_forward(g, x, mask):
    """x (F,), mask (A,) bool -> probs (A,), hidden (H,)"""
    h = np.tanh(x @ g["W1"] + g["b1"])
    logits = (h @ g["W2"] + x @ g["Wl"] + g["b2"]) / max(float(g["temp"][0]), 0.05)
    logits = np.where(mask, logits, -1e9)
    logits -= logits.max()
    p = np.exp(logits) * mask
    p /= p.sum()
    return p, h


def policy_grad_logp(g, x, h, p, a):
    """Gradient of log pi(a|x) w.r.t. GRAD_KEYS (temp treated as constant)."""
    t = max(float(g["temp"][0]), 0.05)
    d = -p.copy()
    d[a] += 1.0
    d /= t                                          # dlogp/dlogits_raw
    dh = (g["W2"] @ d) * (1 - h * h)
    return {"W2": np.outer(h, d), "Wl": np.outer(x, d), "b2": d,
            "W1": np.outer(x, dh), "b1": dh}


# ------------------------------------------------------------------ opponent model
class OppModel:
    """Per-leg opponent statistics with Bayesian-ish priors."""

    def __init__(self):
        self.hands = 0
        self.vpip = 0
        self.acts = 0
        self.raises = 0
        self.faced = 0          # times opp responded to our bet/raise
        self.folds = 0
        self.sd_seen = 0

    def update(self, history, my_seat_button):
        self.hands += 1
        vp = False
        last_me_raise = False
        for e in history:
            if e["actor"] == "opp":
                self.acts += 1
                if e["action"] == "raise":
                    self.raises += 1
                if e["street"] == "preflop" and e["action"] in ("call", "raise"):
                    vp = True
                if last_me_raise:
                    self.faced += 1
                    if e["action"] == "fold":
                        self.folds += 1
                last_me_raise = False
            else:
                last_me_raise = e["action"] == "raise"
        if vp:
            self.vpip += 1

    def feats(self):
        return ((self.folds + 1.0) / (self.faced + 3.0),
                (self.raises + 1.0) / (self.acts + 4.0),
                (self.vpip + 2.0) / (self.hands + 3.0))


# ------------------------------------------------------------------ agent
def _cards(lst):
    return [card_to_int(c) for c in lst]


class PolicyAgent:
    """Neural policy over abstract actions with equity + opponent-model features."""

    def __init__(self, genome, oracle, rng=None, record=False):
        self.g = genome
        self.oracle = oracle
        self.rng = rng if rng is not None else np.random.default_rng()
        self.record = record
        self.traj = []          # (hand_id, x, h, p, a) when recording
        self.opp = OppModel()
        self.seat_button = False
        self.hole = []
        self.discard_card = None
        self.opp_discard = None
        self.hand_id = 0

    # -------- events
    def on_new_hand(self, msg):
        self.hand_id = msg["hand_id"]
        self.seat_button = msg["seat"] == "button"
        self.hole = _cards(msg["hole_cards"])
        self.discard_card = None
        self.opp_discard = None
        if hasattr(self.oracle, "new_hand"):
            self.oracle.new_hand()

    def on_discard_result(self, msg):
        self.discard_card = card_to_int(msg["my_discard"])
        self.opp_discard = card_to_int(msg["opp_discard"])
        self.hole = _cards(msg["hole_cards"])

    def on_hand_result(self, msg):
        self.opp.update(msg.get("history") or [], self.seat_button)

    # -------- features
    def strength3(self, street, board):
        """(equity vs random, vs top-40% range, vs top-15% range)."""
        if street == "preflop":
            v = self.oracle.pre(self.hole)
            return v, v, v
        if len(self.hole) == 3:
            return self.oracle.flop3(self.hole, board)
        return self.oracle.post(self.hole, board, self.discard_card)

    def strength(self, street, board):
        return self.strength3(street, board)[0]

    def features(self, req):
        board = _cards(req["board"])
        street = req["street"]
        eq, eq40, eq15 = self.strength3(street, board)
        pot = max(req["pot"], 1)
        to_call = req["to_call"]
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0
        spr = req["my_stack"] / pot
        opp_r_st = me_r_st = opp_r_prev = 0
        last_opp_bet = 0
        for e in req["history"]:
            if e["action"] != "raise":
                continue
            if e["street"] == street:
                if e["actor"] == "opp":
                    opp_r_st += 1
                    last_opp_bet = e["amount"]
                else:
                    me_r_st += 1
            elif e["actor"] == "opp":
                opp_r_prev += 1
        f_fold, f_aggr, f_vpip = self.opp.feats()
        suits = [c & 3 for c in board]
        ranks = [c >> 2 for c in board]
        flushy = 1.0 if board and max(suits.count(s) for s in range(4)) >= 3 else 0.0
        paired = 1.0 if len(set(ranks)) < len(ranks) else 0.0
        od = (self.opp_discard >> 2) / 12.0 if self.opp_discard is not None else 0.0
        si = ("preflop", "flop", "turn", "river").index(street)
        x = np.zeros(N_FEAT)
        x[0] = 1.0
        x[1] = eq
        x[2] = eq * eq
        x[3] = pot_odds
        x[4] = eq - pot_odds
        x[5] = min(to_call / pot, 3.0)
        x[6] = math.log1p(spr) / 3.0
        x[7] = 1.0 if self.seat_button else 0.0
        x[8 + si] = 1.0
        x[12] = opp_r_st / 2.0
        x[13] = me_r_st / 2.0
        x[14] = opp_r_prev / 3.0
        x[15] = min(last_opp_bet / pot, 3.0)
        x[16] = f_fold
        x[17] = f_aggr
        x[18] = f_vpip
        x[19] = od
        x[20] = flushy
        x[21] = paired
        x[22] = (eq - 0.5) * (f_aggr - 0.25) * 4.0
        x[23] = (eq - 0.5) * x[5]
        # range-aware block (v2): how we fare against the hands that bet/raise
        opp_r_total = opp_r_st + opp_r_prev
        x[24] = eq40
        x[25] = eq15
        x[26] = eq40 - pot_odds
        x[27] = eq15 - pot_odds
        x[28] = min(req["pot"] / 400.0, 1.0)                 # share of all chips already in the pot
        x[29] = min(opp_r_total, 4) / 4.0 * (eq15 - 0.5)
        return x

    # -------- decisions
    def act(self, req):
        legal = req["legal_actions"]
        mask = np.zeros(N_ACT, dtype=bool)
        mask[0] = "fold" in legal
        mask[1] = ("check" in legal) or ("call" in legal)
        can_raise = "raise" in legal
        mask[2:] = can_raise
        x = self.features(req)
        p, h = policy_forward(self.g, x, mask)
        a = int(self.rng.choice(N_ACT, p=p))
        if self.record:
            self.traj.append((self.hand_id, x, h, p, a))
        return self.action_to_msg(a, req)

    @staticmethod
    def action_to_msg(a, req):
        legal = req["legal_actions"]
        if a == 0:
            return {"type": "action", "action": "fold"}
        if a == 1 or "raise" not in legal:
            return {"type": "action", "action": "check" if "check" in legal else "call"}
        if a == 5:
            amt = req["max_raise"]
        else:
            amt = req["opp_bet"] + int(round(RAISE_FRAC[a - 2] * (req["pot"] + req["to_call"])))
        amt = max(req["min_raise"], min(req["max_raise"], amt))
        return {"type": "action", "action": "raise", "amount": int(amt)}

    def discard(self, req):
        hole = _cards(req["hole_cards"])
        board = _cards(req["board"])
        best, best_s = 0, -1e9
        for k, (i, j) in enumerate(_PAIRS3):
            d = hole[3 - i - j]
            pair = (hole[i], hole[j])
            eq = self.oracle.keep(pair, board, d)
            r1, r2 = pair[0] >> 2, pair[1] >> 2
            f = (1.0 if r1 == r2 else 0.0, 1.0 if (pair[0] & 3) == (pair[1] & 3) else 0.0,
                 1.0 if abs(r1 - r2) in (1, 2) else 0.0, max(r1, r2) / 12.0, (d >> 2) / 12.0)
            s = eq + float(np.dot(self.g["dw"], f)) * 0.1
            if s > best_s:
                best, best_s = k, s
        i, j = _PAIRS3[best]
        return {"type": "discard", "card": req["hole_cards"][3 - i - j]}
