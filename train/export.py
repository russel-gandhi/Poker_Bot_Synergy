"""Package a genome as an uploadable submission (folder + zip with bot.py at the top level).

    python -m train.export runs/br1/final_champion.json --out submission
"""
import argparse
import json
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export(genome_path, out):
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "pokerbot"))
    shutil.copy(os.path.join(ROOT, "deploy", "bot.py"), os.path.join(out, "bot.py"))
    for f in ("__init__.py", "core.py", "agent.py"):
        shutil.copy(os.path.join(ROOT, "pokerbot", f), os.path.join(out, "pokerbot", f))
    shutil.copy(os.path.join(ROOT, "data", "preflop.json"), os.path.join(out, "preflop.json"))
    with open(genome_path) as f:
        model = json.load(f)
    with open(os.path.join(out, "model.json"), "w") as f:
        json.dump(model, f)
    zpath = out.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, files in os.walk(out):
            for fn in files:
                if fn.endswith(".pyc"):
                    continue
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, out))
    size = os.path.getsize(zpath)
    print(f"exported {genome_path} -> {out}/ and {zpath} ({size / 1024:.0f} KB, limit 5 MB)")
    return zpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("genome")
    ap.add_argument("--out", default=os.path.join(ROOT, "submission"))
    a = ap.parse_args()
    export(a.genome, a.out)


if __name__ == "__main__":
    main()
