import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = list(csv.DictReader(open("emphatic_pairs.csv")))
names = ["sad>seen", "seen>sad", "Ta>ta", "ta>Ta", "dad>dal", "dal>dad"]
colors = ["#7f7f7f", "#9ecae1", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
n = len(rows)
w = 0.8 / n
fig, ax = plt.subplots(figsize=(9, 4))
for i, r in enumerate(rows):
    xs = [j + (i - (n - 1) / 2) * w for j in range(len(names))]
    ax.bar(xs, [float(r[k]) for k in names], w, label=r["config"], color=colors[i])
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_ylabel("% of reference letters replaced by partner")
ax.set_title("Emphatic-pair confusions, 3 seeds pooled")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("emphatic_pairs.png", dpi=200)
