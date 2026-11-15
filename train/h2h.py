"""Head-to-head series on the OFFICIAL engine, scored per the rulebook.
    python -m train.h2h BOT_A BOT_B --stage 2 --seeds 1 2 3 --python PY
"""
import argparse, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "Starter_Kit", "engine.py")

def one(args):
    a, b, stage, seed, py = args
    log = os.path.join(ROOT, "logs", "h2h_" + os.path.basename(os.path.dirname(a)) + "_vs_" + os.path.basename(os.path.dirname(b)), f"s{stage}_seed{seed}")
    out = subprocess.run([py, ENGINE, a, b, "--stage", str(stage), "--seed", str(seed),
                          "--log-dir", log, "--python", py], capture_output=True, text=True).stdout
    nets = [int(x) for x in re.findall(r"net\s+([+-]\d+)", out)]
    used = [float(x) for x in re.findall(r"time used\s+([\d.]+) s", out)]
    notices = [int(x) for x in re.findall(r"notices (\d+)", out)]
    res = re.search(r"Result: (.*)", out)
    return stage, seed, nets, used, notices, res.group(1) if res else "?", out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--stage", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--parallel", type=int, default=6)
    x = ap.parse_args()
    jobs = [(os.path.abspath(x.a), os.path.abspath(x.b), st, sd, x.python) for st in x.stage for sd in x.seeds]
    with ThreadPoolExecutor(x.parallel) as ex:
        res = list(ex.map(one, jobs))
    for stage in x.stage:
        C = 400 if stage == 1 else 1000; H = 400 if stage == 1 else 1000
        pts = [0.0, 0.0]; cap = [0.0, 0.0]
        print(f"\n=== Stage {stage} ({H} hands/match, C={C}) ===")
        print(f"{'seed':>6} {'A net':>8} {'B net':>8}  {'A time':>7} {'B time':>7}  notices  result")
        for st, sd, nets, used, notices, r, out in res:
            if st != stage: continue
            if len(nets) < 2 or "forfeit" in out:
                print(sd, "FORFEIT/ERROR:", r); print(out[-600:]); 
                if "A forfeited" in out: pts[1] += 1; cap[1] += C; cap[0] -= C
                elif "B forfeited" in out: pts[0] += 1; cap[0] += C; cap[1] -= C
                continue
            n = nets[0]
            pa = 1.0 if n > 0 else (0.5 if n == 0 else 0.0)
            pts[0] += pa; pts[1] += 1 - pa
            c = max(-C, min(C, n)); cap[0] += c; cap[1] -= c
            print(f"{sd:>6} {nets[0]:>+8d} {nets[1]:>+8d}  {used[0]:>7.1f} {used[1]:>7.1f}  {notices[0]}/{notices[1]}      {r}")
        m = sum(1 for r in res if r[0] == stage)
        print(f"match points  A {pts[0]:.1f}  B {pts[1]:.1f}   | capped chips/hand  A {cap[0]/(m*H):+.3f}  B {cap[1]/(m*H):+.3f}")

if __name__ == "__main__":
    main()
