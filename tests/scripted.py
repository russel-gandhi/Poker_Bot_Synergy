"""Deterministic bot logic shared by the parity test (engine subprocess vs in-process sim)."""
import hashlib, json, sys

def h(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest(), 16)

class Scripted:
    def __init__(self, salt):
        self.salt = salt; self.hole = None
    def on_new_hand(self, m): self.hole = m["hole_cards"]
    def on_discard_result(self, m): pass
    def on_hand_result(self, m): pass
    def act(self, r):
        x = h(self.salt, r["hand_id"], r["street"], len(r["history"]), ",".join(self.hole), r["pot"])
        legal = r["legal_actions"]
        roll = x % 100
        if "raise" in legal and roll < 30:
            lo, hi = r["min_raise"], r["max_raise"]
            amt = lo + (x // 100) % (hi - lo + 1) if roll < 10 else min(hi, lo + (x // 7) % 20)
            return {"type": "action", "action": "raise", "amount": amt}
        if roll < 33:
            return {"type": "action", "action": "check"}   # sometimes illegal on purpose
        if "check" in legal: return {"type": "action", "action": "check"}
        if roll < 85: return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}
    def discard(self, r):
        x = h(self.salt, r["hand_id"], "d")
        if x % 10 == 0:
            return {"type": "discard", "card": "XX"}         # illegal on purpose
        return {"type": "discard", "card": r["hole_cards"][x % 3]}

def run(salt):
    b = Scripted(salt)
    for line in sys.stdin:
        m = json.loads(line); t = m["type"]
        if t == "new_hand": b.on_new_hand(m)
        elif t == "action_request": print(json.dumps(b.act(m)), flush=True)
        elif t == "discard_request": print(json.dumps(b.discard(m)), flush=True)
        elif t == "match_end": break
