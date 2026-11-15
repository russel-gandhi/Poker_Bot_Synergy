"""Testing phase: run the exported submission through the OFFICIAL Starter_Kit/engine.py
(real subprocesses, real clocks) against every arena bot, then a Stage 2 timing run.
Fails loudly on any forfeit, notice, or time-bank risk.

    python -m train.test_bot --bot submission/bot.py --python py311_path
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "Starter_Kit", "engine.py")
OPPS = ["Starter_Kit/example_bots/random_bot.py"] + [f"arena/{n}_bot.py" for n in
                                                     ("station", "maniac", "nit", "equity")]


def run(bot, opp, python, stage, seed, log_dir):
    cmd = [python, ENGINE, bot, opp, "--stage", str(stage), "--seed", str(seed),
           "--log-dir", log_dir, "--python", python]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(ROOT, "Starter_Kit")).stdout
    nets = [int(x) for x in re.findall(r"net\s+([+-]\d+)", out)]
    used = [float(x) for x in re.findall(r"time used\s+([\d.]+) s", out)]
    notices = [int(x) for x in re.findall(r"notices (\d+)", out)]
    forfeit = "forfeit" in out
    return nets, used, notices, forfeit, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default=os.path.join(ROOT, "submission", "bot.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    bot = os.path.abspath(a.bot)
    ok = True
    print(f"{'opponent':38s} {'stage':5s} {'seed':5s} {'net':>8s} {'time(s/2 legs)':>14s} notices")
    jobs = [(o, 1, s) for o in OPPS for s in range(a.seeds)] + [(OPPS[-1], 2, 99)]
    for opp, stage, seed in jobs:
        nets, used, notices, forfeit, out = run(bot, os.path.join(ROOT, opp), a.python, stage, seed,
                                                os.path.join(ROOT, "logs", "test"))
        if len(nets) < 2:
            print(out); ok = False; continue
        bank = 30 if stage == 1 else 75
        flag = ""
        if forfeit or notices[0] or used[0] / 2 > 0.6 * bank:
            ok = False
            flag = "  <-- PROBLEM"
        print(f"{opp:38s} {stage:5d} {seed:5d} {nets[0]:+8d} {used[0]:14.2f} {notices[0]:7d}{flag}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
