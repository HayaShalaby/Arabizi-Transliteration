import glob, os, re
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from error_analysis import load

groups = defaultdict(list)
for f in sorted(glob.glob("logs/Model_check_seed_*_ep_30_*")):
    if not (os.path.exists(f + "/ref.txt") and os.path.exists(f + "/pred.txt")):
        continue
    key = f.split("_attn_")[1].rsplit("_D", 1)[0] if "_attn_" in f else "global_dot"
    groups[key].append(f)

fig, ax = plt.subplots(figsize=(7, 4))
for key, fs in sorted(groups.items()):
    refs, preds = load(fs)
    n, ok = Counter(), Counter()
    for r, p in zip(refs, preds):
        L = min(len(r), 8)
        n[L] += 1
        ok[L] += r == p
    Ls = [L for L in sorted(n) if n[L] >= 100]
    ax.plot(Ls, [100.0 * ok[L] / n[L] for L in Ls], marker="o", label=f"{key} ({len(fs)} runs)")
ax.set_xlabel("reference word length in letters (8 = 8 or more)")
ax.set_ylabel("word accuracy (%)")
ax.set_title("Word accuracy by word length")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("acc_by_length.png", dpi=200)

files = {"global_dot": "log_{s}.txt", "global_general": "log_global_general_{s}.txt",
         "localp_dot": "log_localp_dot_{s}.txt", "localp_general": "log_localp_general_{s}.txt",
         "multip_dot": "log_multip_dot_{s}.txt"}
pat = re.compile(r" - loss: ([\d.]+) - accuracy: [\d.]+ - val_loss: ([\d.]+)")
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for key, tpl in files.items():
    curves = []
    for s in (51, 53, 55):
        path = tpl.format(s=s)
        if not os.path.exists(path):
            continue
        m = pat.findall(open(path, errors="ignore").read())
        if len(m) == 30:
            curves.append([(float(a), float(b)) for a, b in m])
    if not curves:
        print("no complete log for", key)
        continue
    for i, name in enumerate(("train loss", "validation loss")):
        mean = [sum(c[e][i] for c in curves) / len(curves) for e in range(30)]
        axs[i].plot(range(1, 31), mean, label=f"{key} ({len(curves)} runs)")
        axs[i].set_title(name)
        axs[i].set_xlabel("epoch")
axs[0].set_ylabel("mean over seeds")
axs[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("loss_curves.png", dpi=200)
print("saved acc_by_length.png and loss_curves.png")
