"""Rulebook tally of v3 vs each friend bot, from OUR side, split by stage."""
import re, os
D = os.path.dirname(os.path.abspath(__file__))
OPP = ["v14_champion_submission", "submission_v23", "vnew3", "v5", "v6"]
def parse(path, ours_a):
    res = {1: [], 2: []}; st = None
    for line in open(path):
        m = re.search(r"=== Stage (\d)", line)
        if m: st = int(m.group(1)); continue
        m = re.match(r"\s*(\d+)\s+([+-]\d+)\s+([+-]\d+)\s", line)
        if m and st: res[st].append(int(m.group(2) if ours_a else m.group(3)))
        if "FORFEIT" in line and st: res[st].append(("F", line.strip()))
    return res
T = {1: [0, 0, 0], 2: [0, 0, 0]}
for b in OPP:
    a = parse(f"{D}/{b}_ours_first.txt", True); c = parse(f"{D}/{b}_theirs_first.txt", False)
    for st in (1, 2):
        nets = [n for n in a[st] + c[st] if isinstance(n, int)]
        H = 400 if st == 1 else 1000; C = H
        pts = sum(1 if n > 0 else 0.5 if n == 0 else 0 for n in nets)
        capped = sum(max(-C, min(C, n)) for n in nets)
        T[st][0] += pts; T[st][1] += len(nets); T[st][2] += capped
        print(f"S{st} vs {b:24s} {pts:4.1f}/{len(nets):<2d} pts  capped {capped:+6d}  {sum(nets)/(H*max(1,len(nets))):+.2f}/hand  {nets}")
for st in (1, 2):
    print(f"Stage {st} TOTAL: {T[st][0]:.1f}/{T[st][1]} match points, capped chips {T[st][2]:+d}")
