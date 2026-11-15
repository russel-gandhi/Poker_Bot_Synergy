import os, random, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from scripted import Scripted
from pokerbot.sim import play_hand
SEED, HANDS = "777", 400
out = subprocess.run([sys.executable, os.path.join(ROOT, "Starter_Kit", "engine.py"),
                      os.path.join(HERE, "parity_bot_A.py"), os.path.join(HERE, "parity_bot_B.py"),
                      "--seed", SEED, "--hands", str(HANDS), "--log-dir", "", "--no-time-limits"],
                     capture_output=True, text=True).stdout
eng = [int(x) for x in re.findall(r"net\s+([+-]\d+)", out)]
tot = [0, 0]
for leg in range(2):
    bots = (Scripted("A"), Scripted("B"))
    for deal in range(HANDS // 2):
        deck = [r + s for r in "23456789TJQKA" for s in "shdc"]
        random.Random(f"{SEED}:{deal}").shuffle(deck)
        button = (deal + leg) % 2
        nets = play_hand((bots[button], bots[1 - button]), deal + 1, (deck[0:3], deck[3:6]), deck[6:11])
        tot[button] += nets[0]; tot[1 - button] += nets[1]
print("engine:", eng, " sim:", tot)
assert eng == tot, "PARITY FAILURE"
print("PARITY OK")
