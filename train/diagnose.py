"""Diagnose OUR bot's leaks from official-engine match logs (outcome data only).

    python -m train.diagnose logs/h2h_submission_vs_X:A logs/h2h_X_vs_submission:B

Each argument is <log dir>:<our label in that dir>. Reports where our chips are won and lost.
"""
import collections
import glob
import os
import re
import sys

from pokerbot.core import card_to_int, evaluate

LINE = re.compile(r"L\d #\d+\s+button (\w) \| (\w): (.+?)  (\w): (.+?) \| board ([^|]*)\|"
                  r"(?: discards (\w):(\S+) (\w):(\S+) \|)? (.*?) \| (fold|showdown).* (A|B) ([+-]\d+), (A|B) ([+-]\d+)$")
CATS = ("high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads", "str flush")


def hands(arg):
    d, ours = arg.rsplit(":", 1)
    for f in glob.glob(os.path.join(d, "*", "match.log")):
        for line in open(f):
            m = LINE.match(line.strip())
            if m:
                yield m, ours


def main():
    stats = collections.defaultdict(lambda: [0, 0])
    sd_cat = collections.defaultdict(lambda: [0, 0])
    total = [0, 0]

    def add(key, net):
        stats[key][0] += 1
        stats[key][1] += net

    for m, us in (h for a in sys.argv[1:] for h in hands(a)):
        them = "B" if us == "A" else "A"
        nets = {m.group(13): int(m.group(14)), m.group(15): int(m.group(16))}
        net = nets[us]
        total[0] += 1
        total[1] += net
        dealt = {m.group(2): m.group(3).split(), m.group(4): m.group(5).split()}
        board = m.group(6).split()
        disc = {}
        if m.group(7):
            disc = {m.group(7): m.group(8), m.group(9): m.group(10)}
        acts = m.group(11)
        result = m.group(12)
        streets, cur = {}, None
        for t in acts.split():
            if t.endswith(":"):
                cur = t[:-1]
                streets[cur] = []
            else:
                streets[cur].append(t)
        committed = abs(net) if net else 0
        flat = [t for st in streets.values() for t in st]
        our_raises = sum(t.startswith(us + "r") for t in flat)
        their_raises = sum(t.startswith(them + "r") for t in flat)
        # pot size buckets (by how much was at stake)
        size = "small pot (<20)" if committed < 20 else ("medium pot (20-99)" if committed < 100 else "BIG pot (100+)")
        add(f"{result:8s} | {size}", net)
        if their_raises >= 2:
            add("opponent raised 2+ times in the hand", net)
        if any(t.startswith(them + "r") for t in streets.get("river", [])):
            add("opponent bet/raised the river", net)
        for st in ("flop", "turn", "river"):
            toks = streets.get(st, [])
            if any(t.startswith(us + "r") for t in toks) and toks and toks[-1].startswith(them + "c"):
                add(f"we bet {st}, opponent CALLED", net)
            if toks and toks[-1] == them + "f":
                add(f"we bet {st}, opponent FOLDED", net)
        pf = streets.get("preflop", [])
        if sum(t[1] == "r" for t in pf) >= 3:
            add("preflop raising war (3+ raises)", net)
        if result == "fold" and flat and flat[-1] == us + "f" and committed >= 40:
            add("we FOLDED after investing 40+", net)
        # what we showed down in big pots
        if result == "showdown" and committed >= 100 and len(board) == 5:
            kept = [c for c in dealt[us] if c != disc.get(us)]
            if len(kept) == 2:
                sc = int(evaluate([[card_to_int(c) for c in kept + board]])[0]) >> 20
                sd_cat[CATS[sc]][0] += 1
                sd_cat[CATS[sc]][1] += net
    print(f"hands analysed: {total[0]}, our net {total[1]:+d} ({total[1] / max(total[0], 1):+.2f}/hand)\n")
    print(f"{'situation':44s} {'hands':>6s} {'our net':>9s} {'per hand':>9s}")
    for k in sorted(stats):
        n, v = stats[k]
        print(f"{k:44s} {n:6d} {v:+9d} {v / n:+9.1f}")
    print("\nour hand at showdown in BIG pots (100+ chips each):")
    for k in CATS:
        if k in sd_cat:
            n, v = sd_cat[k]
            print(f"  {k:12s} {n:5d} hands  our net {v:+7d}")


if __name__ == "__main__":
    main()
