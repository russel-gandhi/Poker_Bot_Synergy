"""Tally rulebook match points from h2h result files, from OUR bot's point of view.
    python -m train.tally submission_v3 submission_v1
"""
import glob, re, sys
OPP = ["v14_champion_submission", "submission_v23", "vnew3", "v5", "v6"]
def rows(path, ours_is_a):
    out = []
    for line in open(path):
        m = re.match(r"\s+(\d+)\s+([+-]\d+)\s+([+-]\d+)\s", line)
        if m:
            out.append(int(m.group(2)) if ours_is_a else int(m.group(3)))
    return out
for me in sys.argv[1:]:
    print(f"\n#### {me}")
    tp = tm = 0; chips = 0
    for b in OPP:
        nets = rows(f"final_{me}_{b}_ours_first.txt", True) + rows(f"final_{me}_{b}_theirs_first.txt", False)
        pts = sum(1.0 if n > 0 else (0.5 if n == 0 else 0.0) for n in nets)
        tp += pts; tm += len(nets); chips += sum(nets)
        print(f"  vs {b:26s} {pts:.1f}/{len(nets)} match points   chips {sum(nets):+6d}  ({sum(nets)/(1000*max(len(nets),1)):+.2f}/hand)   {nets}")
    print(f"  TOTAL {tp:.1f}/{tm} match points, {chips/(1000*max(tm,1)):+.2f} chips/hand")
