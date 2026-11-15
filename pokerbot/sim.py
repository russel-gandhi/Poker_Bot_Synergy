"""In-process Discard Hold'em engine for training. Mirrors Starter_Kit/engine.py rules exactly
(blinds, min-raise, all-in, 4-bet cap, simultaneous exposed discard, duplicate legs), but calls
agent objects directly instead of subprocesses. Parity is checked by tests/test_parity.py."""
import numpy as np

from .core import int_to_card, _PAIRS3

STARTING_STACK, SB, BB, MAX_RAISES = 200, 1, 2, 4
STREETS = ("preflop", "flop", "turn", "river")
BOARD_SEEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
SEAT_NAMES = ("button", "big_blind")


def default_discard(hole):
    return min(hole, key=lambda c: ("23456789TJQKA".index(c[0]), "shdc".index(c[1])))


class DealOracle:
    """Precomputed equities for one deal (both seats), looked up by card sets."""

    def __init__(self, table):
        self.t = table

    def new_hand(self):
        pass

    def pre(self, h3):
        return self.t[("pre", frozenset(h3))]

    def flop3(self, h3, board):
        return self.t[("f3", frozenset(h3))]

    def keep(self, pair, board, discard):
        return self.t[("k", frozenset(pair))]

    def post(self, pair, board, discard):
        return self.t[("p", frozenset(pair), len(board))]


class DealPool:
    """Arrays produced by pokerbot/dealpool.py."""

    def __init__(self, path):
        z = np.load(path)
        self.hole = z["hole"]        # (D,2,3) ints
        self.board = z["board"]      # (D,5)
        self.pre = z["pre"]          # (D,2)
        self.f3 = z["f3"]            # (D,2,3)   [vs random, vs top40%, vs top15%]
        self.keep = z["keep"]        # (D,2,3)
        self.turn = z["turn"]        # (D,2,keep,3)
        self.river = z["river"]      # (D,2,keep,3)
        self.n = len(self.board)
        self._oracles = {}

    def oracle(self, d):
        o = self._oracles.get(d)
        if o is None:
            t = {}
            for s in range(2):
                h = [int(c) for c in self.hole[d, s]]
                t[("pre", frozenset(h))] = float(self.pre[d, s])
                t[("f3", frozenset(h))] = tuple(float(v) for v in self.f3[d, s])
                for k, (i, j) in enumerate(_PAIRS3):
                    pr = frozenset((h[i], h[j]))
                    t[("k", pr)] = float(self.keep[d, s, k])
                    t[("p", pr, 4)] = tuple(float(v) for v in self.turn[d, s, k])
                    t[("p", pr, 5)] = tuple(float(v) for v in self.river[d, s, k])
            o = DealOracle(t)
            if len(self._oracles) < 4000:
                self._oracles[d] = o
        return o

    def cards(self, d):
        h0 = [int_to_card(int(c)) for c in self.hole[d, 0]]
        h1 = [int_to_card(int(c)) for c in self.hole[d, 1]]
        b = [int_to_card(int(c)) for c in self.board[d]]
        return h0, h1, b


def _legal_check(action, amount, req):
    if action not in req["legal_actions"]:
        return False
    if action == "raise":
        return isinstance(amount, int) and req["min_raise"] <= amount <= req["max_raise"]
    return True


def play_hand(agents, hand_id, seat_cards, board):
    """agents: (button_agent, bb_agent). Returns nets by seat."""
    hole = [list(seat_cards[0]), list(seat_cards[1])]
    stacks = [STARTING_STACK - SB, STARTING_STACK - BB]
    committed = [SB, BB]
    history = []
    discards = [None, None]

    def hist_for(seat):
        out = []
        for st, actor, action, amount in history:
            e = {"street": st, "actor": "me" if actor == seat else "opp", "action": action}
            if amount is not None:
                e["amount"] = amount
            out.append(e)
        return out

    for seat in (0, 1):
        agents[seat].on_new_hand({"type": "new_hand", "hand_id": hand_id, "seat": SEAT_NAMES[seat],
                                  "hole_cards": list(hole[seat]), "stack": 200, "opp_stack": 200})
    folder = None
    seen = 0
    for street in STREETS:
        seen = BOARD_SEEN[street]
        if street == "preflop":
            bets, last_raise, raises, actor = [SB, BB], BB, 1, 0
        else:
            bets, last_raise, raises, actor = [0, 0], 0, 0, 1
        if 0 not in stacks:
            acted = [False, False]
            while True:
                opp = 1 - actor
                to_call = bets[opp] - bets[actor]
                max_raise = min(bets[actor] + stacks[actor], bets[opp] + stacks[opp])
                legal = ["fold", "call"] if to_call > 0 else ["check"]
                can_raise = raises < MAX_RAISES and max_raise > bets[opp]
                if can_raise:
                    legal.append("raise")
                req = {"type": "action_request", "hand_id": hand_id, "street": street,
                       "board": board[:seen], "pot": sum(committed), "my_bet": bets[actor],
                       "opp_bet": bets[opp], "to_call": to_call, "my_stack": stacks[actor],
                       "opp_stack": stacks[opp], "legal_actions": legal}
                if can_raise:
                    req["min_raise"] = min(bets[opp] + max(last_raise, 2), max_raise)
                    req["max_raise"] = max_raise
                req["raises_left"] = MAX_RAISES - raises
                req["history"] = hist_for(actor)
                reply = agents[actor].act(req)
                action, amount = reply.get("action"), reply.get("amount")
                if not _legal_check(action, amount, req):
                    action = "check" if "check" in legal else "fold"
                    amount = None
                if action == "fold":
                    history.append((street, actor, "fold", None))
                    folder = actor
                    break
                if action == "check":
                    history.append((street, actor, "check", None))
                elif action == "call":
                    stacks[actor] -= to_call
                    committed[actor] += to_call
                    bets[actor] = bets[opp]
                    history.append((street, actor, "call", bets[actor]))
                else:
                    add = amount - bets[actor]
                    stacks[actor] -= add
                    committed[actor] += add
                    last_raise = amount - bets[opp]
                    bets[actor] = amount
                    raises += 1
                    acted[opp] = False
                    history.append((street, actor, "raise", amount))
                acted[actor] = True
                if acted[0] and acted[1] and bets[0] == bets[1]:
                    break
                actor = opp
            if folder is not None:
                break
        if street == "flop":
            for seat in (0, 1):
                rep = agents[seat].discard({"type": "discard_request", "hand_id": hand_id,
                                            "hole_cards": list(hole[seat]), "board": board[:3],
                                            "pot": sum(committed), "my_stack": stacks[seat],
                                            "opp_stack": stacks[1 - seat]})
                c = rep.get("card")
                discards[seat] = c if c in hole[seat] else default_discard(hole[seat])
            for seat in (0, 1):
                hole[seat].remove(discards[seat])
            for seat in (0, 1):
                agents[seat].on_discard_result({"type": "discard_result", "hand_id": hand_id,
                                                "my_discard": discards[seat],
                                                "opp_discard": discards[1 - seat],
                                                "hole_cards": list(hole[seat])})
    if folder is not None:
        result = "fold"
        nets = [0, 0]
        nets[folder], nets[1 - folder] = -committed[folder], committed[folder]
    else:
        result = "showdown"
        seen = 5
        from .core import evaluate, card_to_int
        sc = [int(evaluate(np.array([[card_to_int(c) for c in hole[s] + board]]))[0]) for s in (0, 1)]
        amt = committed[0]
        nets = [amt, -amt] if sc[0] > sc[1] else ([-amt, amt] if sc[1] > sc[0] else [0, 0])
    for seat in (0, 1):
        agents[seat].on_hand_result({"type": "hand_result", "hand_id": hand_id, "result": result,
                                     "net": nets[seat], "board": board[:seen],
                                     "my_hole_cards": list(hole[seat]),
                                     "opp_hole_cards": list(hole[1 - seat]) if result == "showdown" else None,
                                     "opp_discard": discards[1 - seat], "history": hist_for(seat)})
    return nets


def play_match(make_a, make_b, pool, deals):
    """Duplicate match. make_x(oracle_provider) -> fresh agent per leg.
    Agents get their oracle via `agent.oracle` being swapped per hand.
    Returns (net_A, per_deal_A (2, n) array of A's nets per leg, agents_by_leg)."""
    per = np.zeros((2, len(deals)))
    legs = []
    for leg in range(2):
        A, B = make_a(), make_b()
        bots = (A, B)
        for k, d in enumerate(deals):
            o = pool.oracle(int(d))
            A.oracle = o
            B.oracle = o
            h0, h1, b = pool.cards(int(d))
            button = (k + leg) % 2
            nets = play_hand((bots[button], bots[1 - button]), k + 1, (h0, h1), b)
            per[leg, k] = nets[0] if button == 0 else nets[1]
        legs.append((A, B))
    return float(per.sum()), per, legs
