"""Error analysis for ATAR runs.

Usage:  python error_analysis.py logs/<run folder> [more run folders ...]
Each folder must hold ref.txt and pred.txt (characters separated by spaces, one word per line).
Several folders are pooled, e.g. the three baseline seeds.
"""
import sys
from collections import Counter

# letter groups that are written interchangeably in informal Arabic
GROUPS = {"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي", "ئ": "ء", "ؤ": "ء"}


def norm(w):
    return "".join(GROUPS.get(c, c) for c in w)


def align(a, b):
    """Levenshtein alignment. Returns list of (op, ref_char, pred_char)."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("match" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", a[i - 1], ""))      # reference letter missing from prediction
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1]))      # extra letter in prediction
            j -= 1
    return ops[::-1]


def load(folders):
    refs, preds = [], []
    for f in folders:
        r = [l.rstrip("\n").replace(" ", "") for l in open(f + "/ref.txt", encoding="utf-8")]
        p = [l.rstrip("\n").replace(" ", "") for l in open(f + "/pred.txt", encoding="utf-8")]
        assert len(r) == len(p), f
        refs += r
        preds += p
    return refs, preds


def pct(x, n):
    return f"{100.0 * x / n:.1f}%" if n else "n/a"


def main(folders):
    refs, preds = load(folders)
    n = len(refs)
    wrong = [(r, p) for r, p in zip(refs, preds) if r != p]
    print(f"words: {n} | wrong: {len(wrong)} | word accuracy: {pct(n - len(wrong), n)}")

    nacc = sum(norm(r) == norm(p) for r, p in zip(refs, preds))
    print(f"accuracy if interchangeable letters are merged (alef forms, ه/ة, ي/ى, ء forms): {pct(nacc, n)}")

    edits = Counter()
    sub, ins, dele = Counter(), Counter(), Counter()
    kinds = Counter()
    for r, p in wrong:
        ops = align(r, p)
        k = sum(o[0] != "match" for o in ops)
        edits["1 edit" if k == 1 else "2 edits" if k == 2 else "3+ edits"] += 1
        for o, a, b in ops:
            if o == "sub":
                sub[(a, b)] += 1
            elif o == "ins":
                ins[b] += 1
            elif o == "del":
                dele[a] += 1
        kinds["prediction shorter than reference"] += len(p) < len(r)
        kinds["prediction longer than reference"] += len(p) > len(r)
        kinds["same length"] += len(p) == len(r)
        doubled = any(p[i] == p[i + 1] for i in range(len(p) - 1)) and not any(r[i] == r[i + 1] for i in range(len(r) - 1))
        kinds["prediction repeats a letter the reference does not"] += doubled

    print("\nwrong words by number of character edits:")
    for k in ("1 edit", "2 edits", "3+ edits"):
        print(f"  {k:9s} {edits[k]:5d}  ({pct(edits[k], len(wrong))} of wrong words)")
    print("\nwrong words by length of prediction:")
    for k, v in kinds.items():
        print(f"  {k:55s} {v:5d}  ({pct(v, len(wrong))})")

    tot = sum(sub.values()) + sum(ins.values()) + sum(dele.values())
    print(f"\ncharacter edits in wrong words: {tot}  (substitutions {sum(sub.values())}, "
          f"extra letters {sum(ins.values())}, missing letters {sum(dele.values())})")
    print("\nmost common substitutions  reference -> prediction:")
    for (a, b), c in sub.most_common(15):
        merged = "  <- interchangeable" if norm(a) == norm(b) else ""
        print(f"  {a} -> {b}  {c:4d}{merged}")
    print("\nmost common missing letters:", ", ".join(f"{a}:{c}" for a, c in dele.most_common(8)))
    print("most common extra letters: ", ", ".join(f"{b}:{c}" for b, c in ins.most_common(8)))

    print("\naccuracy by reference word length:")
    by = Counter(), Counter()
    for r, p in zip(refs, preds):
        L = min(len(r), 8)
        by[0][L] += 1
        by[1][L] += r == p
    for L in sorted(by[0]):
        lab = f"{L}+" if L == 8 else str(L)
        print(f"  length {lab:2s}: {pct(by[1][L], by[0][L]):>6s}  (n={by[0][L]})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
