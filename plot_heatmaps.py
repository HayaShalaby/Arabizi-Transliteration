import glob, pickle, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(tag):
    f = glob.glob(f"viz/*_ep_30_*_attn_{tag}_dot_D3/attn_dump.pkl")[0]
    return pickle.load(open(f, "rb"))

models = [("global + dot", load("global")), ("local-p + dot", load("localp"))]
words = sys.argv[1:] or ["kefak", "7elo", "ertebat"]
jobs = (("w", "attention_heatmaps.png", "Attention weight"),
        ("occ", "occlusion_maps.png", "Occlusion: drop in probability of the predicted letter when an input letter is removed"),
        ("shap", "shapley_maps.png", "Exact Shapley value of each input letter for the predicted letter"))
for key, fname, title in jobs:
    fig, axs = plt.subplots(len(words), 2, figsize=(8, 3.0 * len(words)), squeeze=False)
    if key == "w":
        kw = dict(cmap="Blues", vmin=0, vmax=1)
    else:
        m = max(abs(d[wd][key][:len(d[wd]["pred"])]).max() for wd in words for _, d in models)
        kw = dict(cmap="RdBu_r", vmin=-m, vmax=m)
    for r, wd in enumerate(words):
        for c, (name, d) in enumerate(models):
            pred = d[wd]["pred"]
            M = d[wd][key][:len(pred)]
            ax = axs[r][c]
            ax.imshow(M, aspect="auto", **kw)
            ax.set_xticks(range(len(wd)))
            ax.set_xticklabels(list(wd))
            ax.set_yticks(range(len(pred)))
            ax.set_yticklabels(list(pred))
            ax.set_title(f"{name}: {wd}")
            ax.set_xlabel("Arabizi letter")
            ax.set_ylabel("output letter")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
print("saved attention_heatmaps.png and occlusion_maps.png")
