"""Fixed-style opponents for the gauntlet. Same interface as PolicyAgent."""
import numpy as np

from .agent import PolicyAgent, _cards
from .core import _PAIRS3


class _Base(PolicyAgent):
    def __init__(self, oracle, rng=None, **kw):
        super().__init__({"dw": np.zeros(5)}, oracle, rng)
        self.kw = kw

    def _raise(self, req, frac):
        amt = req["opp_bet"] + int(round(frac * (req["pot"] + req["to_call"])))
        return {"type": "action", "action": "raise",
                "amount": int(max(req["min_raise"], min(req["max_raise"], amt)))}

    def _passive(self, req):
        return {"type": "action", "action": "check" if "check" in req["legal_actions"] else "call"}

    def _fold_or_check(self, req):
        return {"type": "action", "action": "check" if "check" in req["legal_actions"] else "fold"}


class RandomBot(_Base):
    """Mirror of the starter kit's random_bot.py."""

    def act(self, req):
        legal = req["legal_actions"]
        roll = self.rng.random()
        if "raise" in legal and roll < 0.2:
            size = (0.5, 0.75, 1.0, 2.0)[self.rng.integers(4)]
            return self._raise(req, size)
        if "check" in legal:
            return {"type": "action", "action": "check"}
        if roll < 0.8:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    def discard(self, req):
        return {"type": "discard", "card": req["hole_cards"][int(self.rng.integers(3))]}


class CallingStation(_Base):
    def act(self, req):
        return self._passive(req)


class Maniac(_Base):
    def act(self, req):
        if "raise" in req["legal_actions"] and self.rng.random() < 0.85:
            return self._raise(req, 1.0)
        return self._passive(req)


class Nit(_Base):
    def act(self, req):
        eq = self.strength(req["street"], _cards(req["board"]))
        if eq > 0.72 and "raise" in req["legal_actions"]:
            return self._raise(req, 0.7)
        if eq > 0.6:
            return self._passive(req)
        return self._fold_or_check(req)


class EquityBot(_Base):
    """Tight-aggressive pot-odds player: value-bets strength, calls with odds, small bluffs."""

    def act(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        eq = self.strength(street, _cards(req["board"]))
        pot = max(req["pot"], 1)
        to_call = req["to_call"]
        odds = to_call / (pot + to_call) if to_call else 0.0
        vb = {"preflop": 0.62, "flop": 0.68, "turn": 0.70, "river": 0.75}[street]
        if "raise" in legal and eq > vb:
            frac = 0.75 if eq < 0.85 else 1.0
            return self._raise(req, frac)
        if to_call == 0:
            if "raise" in legal and street != "preflop" and self.rng.random() < 0.12:
                return self._raise(req, 0.5)
            return {"type": "action", "action": "check"}
        if eq > odds + 0.05:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}


def _preflop_solid(bot, req, eq, open_t=0.60, call_floor=0.42):
    """Shared sound preflop: raise good hands, call with odds, fold junk."""
    legal = req["legal_actions"]
    to_call = req["to_call"]
    odds = to_call / (req["pot"] + to_call) if to_call else 0.0
    if "raise" in legal and eq > open_t + 0.06 * req["raises_left"] / 3 * (req["opp_bet"] > 2):
        return bot._raise(req, 0.9)
    if to_call == 0:
        return {"type": "action", "action": "check"}
    if eq > max(odds + 0.02, call_floor):
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold"}


class ValueHeavy(_Base):
    """Rarely folds to small bets, never bluffs, and only raises when strong against a strong
    range. Punishes small-ball barrelling and paying off big raises with one pair."""

    def act(self, req):
        street = req["street"]
        e, e40, e15 = self.strength3(street, _cards(req["board"]))
        if street == "preflop":
            return _preflop_solid(self, req, e)
        legal = req["legal_actions"]
        to_call = req["to_call"]
        odds = to_call / (req["pot"] + to_call) if to_call else 0.0
        if to_call:
            if "raise" in legal and e15 > 0.72:
                return self._raise(req, 0.9)
            if e40 > odds + 0.03:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}
        if "raise" in legal and e40 > 0.66:
            return self._raise(req, 0.7)
        return {"type": "action", "action": "check"}


class CheckRaiser(_Base):
    """Out of position it checks its strong hands and raises when bet into (a trap)."""

    def act(self, req):
        street = req["street"]
        e, e40, e15 = self.strength3(street, _cards(req["board"]))
        if street == "preflop":
            return _preflop_solid(self, req, e, 0.62, 0.40)
        legal = req["legal_actions"]
        to_call = req["to_call"]
        odds = to_call / (req["pot"] + to_call) if to_call else 0.0
        if to_call:
            if "raise" in legal and e15 > 0.62:
                return self._raise(req, 1.0)
            if e40 > odds + 0.02:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}
        if not self.seat_button and e15 > 0.62:
            return {"type": "action", "action": "check"}          # set the trap
        if "raise" in legal and (e40 > 0.6 or (e < 0.25 and self.rng.random() < 0.1)):
            return self._raise(req, 0.6)
        return {"type": "action", "action": "check"}


class StickyCaller(_Base):
    """Calls bets very lightly, raises only near-nut hands, and then raises big."""

    def act(self, req):
        street = req["street"]
        e, e40, e15 = self.strength3(street, _cards(req["board"]))
        legal = req["legal_actions"]
        to_call = req["to_call"]
        odds = to_call / (req["pot"] + to_call) if to_call else 0.0
        if street == "preflop":
            return _preflop_solid(self, req, e, 0.68, 0.34)
        if to_call:
            if "raise" in legal and e15 > 0.8:
                return self._raise(req, 1.5)
            if e > odds - 0.05:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}
        if "raise" in legal and e40 > 0.7:
            return self._raise(req, 0.5)
        return {"type": "action", "action": "check"}


BASELINES = {
    "value": ValueHeavy,
    "trapper": CheckRaiser,
    "sticky": StickyCaller,
    "random": RandomBot,
    "station": CallingStation,
    "maniac": Maniac,
    "nit": Nit,
    "equity": EquityBot,
}
