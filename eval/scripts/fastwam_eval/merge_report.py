"""t10 + t40 -> the 50-task table, plus the seed-paired comparison against the A line.

    python3 merge_report.py                          # default tags
    python3 merge_report.py --tags jax-clean --baseline torch-clean

The pairing is what makes the comparison worth anything: both lines ran the SAME canonical
seeds, so a task-level average is not the only thing available -- the same episode can be
compared to itself. McNemar over those pairs asks "did the two models fail on the same
episodes", which a difference of aggregate rates cannot answer.
"""
import argparse
import json
import math
import os
from pathlib import Path
from collections import defaultdict

ROOT = os.environ.get("FW_ROOT", str(Path(__file__).resolve().parents[3] / "outputs/eval"))


def load(tag):
    """tag -> {(task, seed): success}. One json per finished episode; locks are ignored."""
    out = {}
    base = os.path.join(ROOT, "results", tag)
    if not os.path.isdir(base):
        return out
    for task in sorted(os.listdir(base)):
        d = os.path.join(base, task)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(".json"):
                continue
            try:
                r = json.load(open(os.path.join(d, name)))
            except Exception:
                continue
            out[(r["task"], int(r["seed"]))] = bool(r["success"])
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h) * 100, min(1.0, c + h) * 100)


ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--baseline", required=True, help="baseline tag evaluated under the same protocol")
# The denominator used to be the literal 1250 of the 50x25 table. Any other campaign --
# the fast tier is 10 tasks x 10 seeds = 100 -- then read as "100/1250 episodes", i.e. 8%
# complete, on a campaign that had finished. Ask the campaign how big it is instead.
ap.add_argument("--expect", type=int, default=0,
                help="episodes the campaign should contain (default: tasks x --seeds seen)")
ap.add_argument("--seeds", type=int, default=100, help="seeds per task, for --expect")
args = ap.parse_args()

runs = {t: load(t) for t in args.tags}
merged = {}
for t in args.tags:
    merged.update(runs[t])

by_task = defaultdict(list)
for (task, _), ok in merged.items():
    by_task[task].append(ok)

print(f"{'task':<28}{'done':>7}{'succ':>6}{'rate':>8}")
print("-" * 49)
for task in sorted(by_task):
    v = by_task[task]
    print(f"{task:<28}{len(v):>7}{sum(v):>6}{100*sum(v)/len(v):>7.1f}%")

n, k = len(merged), sum(merged.values())
lo, hi = wilson(k, n)
print("-" * 49)
print(f"{'TOTAL':<28}{n:>7}{k:>6}{(100*k/n if n else 0):>7.1f}%")
expect = args.expect or len(by_task) * args.seeds
print(f"\n{len(by_task)} tasks, {n}/{expect} episodes, SR {100*k/n if n else 0:.2f}% "
      f"(Wilson 95% {lo:.1f}-{hi:.1f})")
print("paper reference: 91.88% clean over 50 tasks x 100. Our seeds/episode count differ, "
      "so this is a comparison of the same stack, not of headline numbers.")
if len(by_task) < 50:
    print(f"NOTE: {len(by_task)} tasks, not 50. This is a SUBSET score and is not comparable "
          "to any 50-task number, ours or the paper's -- only to another run of the same subset.")

base = load(args.baseline)
if base:
    pairs = [(merged[key], base[key]) for key in merged if key in base]
    if pairs:
        b01 = sum(1 for a, b in pairs if (not a) and b)      # A line won
        b10 = sum(1 for a, b in pairs if a and (not b))      # we won
        both = sum(1 for a, b in pairs if a and b)
        neither = sum(1 for a, b in pairs if (not a) and (not b))
        print(f"\n--- seed-paired vs A line ({args.baseline}) over {len(pairs)} shared episodes ---")
        print(f"  both succeeded      {both}")
        print(f"  only ours           {b10}")
        print(f"  only A line         {b01}")
        print(f"  neither             {neither}")
        if b01 + b10:
            chi = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)     # McNemar, continuity-corrected
            print(f"  McNemar chi2(1) = {chi:.2f}  "
                  f"({'significant at 0.05' if chi > 3.84 else 'not significant at 0.05'})")
        else:
            print("  the two lines agreed on every shared episode; McNemar is undefined")
    else:
        print(f"\n(no shared (task,seed) pairs with {args.baseline} yet)")
else:
    print(f"\n(baseline tag {args.baseline!r} not found under {ROOT}/results)")
