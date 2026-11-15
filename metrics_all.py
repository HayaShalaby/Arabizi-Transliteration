import glob, re, sys, statistics as st
from collections import defaultdict
import sacrebleu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def cfg(name):
    m = re.search(r"_attn_(global|localp|local)_(dot|general)_D\d+$", name)
    if m:
        return f"{m.group(1)} + {m.group(2)}"
    return "global + dot (baseline)"

rows = defaultdict(list)
for f in sorted(glob.glob("logs/Model_check_seed_*")):
    try:
        R = [l.rstrip("\n") for l in open(f + "/ref.txt", encoding="utf-8")]
        P = [l.rstrip("\n") for l in open(f + "/pred.txt", encoding="utf-8")]
    except FileNotFoundError:
        continue
    seed = int(re.search(r"seed_(\d+)", f).group(1))
    r = [x.replace(" ", "") for x in R]
    p = [x.replace(" ", "") for x in P]
    acc = sum(a == b for a, b in zip(r, p)) / len(r)
    acc100 = sum(a == b for a, b in zip(r[:100], p[:100])) / 100
    cer = sum(ed(a, b) for a, b in zip(r, p)) / sum(len(a) for a in r)
    bleu = sacrebleu.corpus_bleu(P, [R], tokenize="none").score
    rows[cfg(f)].append((seed, acc, acc100, cer, bleu))

order = ["global + dot (baseline)", "global + general", "localp + dot", "localp + general"]
keys = [k for k in order if k in rows] + [k for k in rows if k not in order]
print(f"{'config':26s} n  acc_full        acc_first100   CER            BLEU")
out = ["config,seed,acc_full,acc_first100,cer,bleu"]
for k in keys:
    v = sorted(rows[k])
    def ms(i):
        xs = [x[i] for x in v]
        return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)
    a, a100, c, b = ms(1), ms(2), ms(3), ms(4)
    print(f"{k:26s} {len(v)}  {100*a[0]:.1f} +-{100*a[1]:.1f}   {100*a100[0]:.1f} +-{100*a100[1]:.1f}   {100*c[0]:.2f} +-{100*c[1]:.2f}   {b[0]:.2f} +-{b[1]:.2f}")
    for s, x, x100, cc, bb in v:
        out.append(f"{k},{s},{x:.4f},{x100:.4f},{cc:.4f},{bb:.2f}")
open("results_table.csv", "w").write("\n".join(out) + "\n")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for axi, idx, lab, sc in ((ax[0], 1, "Word accuracy, full test set (%)", 100), (ax[1], 3, "Character error rate (%)", 100)):
    for i, k in enumerate(keys):
        ys = [sc * x[idx] for x in sorted(rows[k])]
        m = st.mean(ys)
        sd = st.stdev(ys) if len(ys) > 1 else 0
        axi.bar(i, m, yerr=sd, capsize=4, color="#9bb7d4", edgecolor="k")
        axi.scatter([i] * len(ys), ys, color="k", s=14, zorder=3)
    axi.set_xticks(range(len(keys)))
    axi.set_xticklabels([k.replace(" (baseline)", "\n(baseline)") for k in keys], fontsize=8)
    axi.set_ylabel(lab)
    lo = min(sc * x[idx] for k in keys for x in rows[k])
    hi = max(sc * x[idx] for k in keys for x in rows[k])
    axi.set_ylim(lo - 3, hi + 3)
fig.suptitle("Attention type x scoring function (D=3); bars = mean, whiskers = std over seeds, dots = seeds")
fig.tight_layout()
fig.savefig("results_2x2.png", dpi=200)
print("wrote results_table.csv and results_2x2.png")
