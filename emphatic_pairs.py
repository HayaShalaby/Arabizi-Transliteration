"""Confusions inside the emphatic pairs, per configuration, pooled over seeds.
Usage: python emphatic_pairs.py      (run from the project folder)
Rate = times the reference letter was replaced by its partner / times the reference letter occurs.
"""
import glob, os, csv
from collections import Counter, defaultdict
from error_analysis import align, load

PAIRS = [("sad>seen", "ص", "س"), ("seen>sad", "س", "ص"),
         ("Ta>ta", "ط", "ت"), ("ta>Ta", "ت", "ط"),
         ("dad>dal", "ض", "د"), ("dal>dad", "د", "ض")]

groups = defaultdict(list)
for f in sorted(glob.glob("logs/Model_check_seed_*_ep_30_*")):
    if not (os.path.exists(f + "/ref.txt") and os.path.exists(f + "/pred.txt")):
        continue
    key = f.split("_attn_")[1].rsplit("_D", 1)[0] if "_attn_" in f else "global_dot"
    groups[key].append(f)

rows = []
for key, fs in sorted(groups.items()):
    refs, preds = load(fs)
    occ, conf = Counter(), Counter()
    for r, p in zip(refs, preds):
        for op, a, b in align(r, p):
            if op in ("match", "sub", "del"):
                occ[a] += 1
            if op == "sub":
                conf[(a, b)] += 1
    row = {"config": key, "runs": len(fs), "words": len(refs)}
    tot_c = tot_o = 0
    for name, a, b in PAIRS:
        row[name] = round(100.0 * conf[(a, b)] / occ[a], 2) if occ[a] else 0.0
        row[name + "_n"] = conf[(a, b)]
        tot_c += conf[(a, b)]
        tot_o += occ[a]
    row["pooled_rate"] = round(100.0 * tot_c / tot_o, 2)
    row["pooled_n"] = tot_c
    rows.append(row)

cols = list(rows[0].keys())
with open("emphatic_pairs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print("rate = % of reference letter occurrences replaced by its partner")
hdr = ["config", "runs"] + [p[0] for p in PAIRS] + ["pooled", "pooled_n"]
print("  ".join(f"{h:>12s}" for h in hdr))
for r in rows:
    vals = [r["config"], str(r["runs"])] + [f"{r[p[0]]:.2f}" for p in PAIRS] + [f"{r['pooled_rate']:.2f}", str(r["pooled_n"])]
    print("  ".join(f"{v:>12s}" for v in vals))
