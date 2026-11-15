"""Slide figures from the locked test results (no re-scoring, no tuning).

Inputs : reports/test_results.md (point estimates + bootstrap 95% CIs), reports/test_predictions.parquet
Outputs: reports/figures/test_metrics.png, test_recall_subgroups.png, test_pr_curves.png, test_confusion.png
"""
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "reports"; F = R / "figures"
F.mkdir(parents=True, exist_ok=True)

# Reference palette (light mode) + neutral ink.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FINAL = "M2+silver"
NAMES = {"B0": "B0  always \"no\"", "B1": "B1  keyword rule", "B2": "B2  zero-shot", "M1": "M1  TF-IDF + LR",
         "M2": "M2  DistilRoBERTa", "M2+silver": "M2 + silver (final)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def r2(v):
    """Round half-up to 2 decimals, matching the report tables (0.565 -> 0.57)."""
    return str(Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def parse_tables(md):
    """{section: {model: {metric: (value, lo, hi)}}} from the markdown tables."""
    out, section = {}, None
    for line in md.splitlines():
        if line.startswith("## "):
            section = "all" if "All 200" in line else "agreed" if "agreed" in line else None
        m = re.match(r"\| (B0|B1|B2|M1|M2\+silver|M2) \| (.+) \|$", line)
        if section and m:
            cells = [re.findall(r"[\d.]+|nan", c) for c in m.group(2).split(" | ")]
            out.setdefault(section, {})[m.group(1)] = {
                k: tuple(float(x) for x in c) for k, c in zip(["precision", "recall", "f1", "pr_auc"], cells)}
    return out


def ci_panel(ax, data, metric, title, models, base=None):
    y = np.arange(len(models))[::-1]
    for yi, m in zip(y, models):
        v, lo, hi = data[m][metric]
        c = BLUE if m == FINAL else MUTED
        ax.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(v, yi, "o", ms=9, color=c, mec=SURFACE, mew=2, zorder=3)
        ax.text(hi + 0.02, yi, r2(v), va="center", color=INK if m == FINAL else INK2,
                fontweight="bold" if m == FINAL else "normal")
    if base is not None:
        ax.axvline(base, color=MUTED, ls=":", lw=1)
        ax.text(base, y[0] + 0.6, f"base rate {base:.2f}", color=INK2, fontsize=9, ha="center")
    ax.set_xlim(0, 1.12); ax.set_ylim(-0.6, len(models) - 0.2)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    return y


def fig_metrics(tables, y_test):
    models = ["B1", "B2", "M1", "M2", "M2+silver"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    y = ci_panel(axes[0], tables["all"], "recall", "Recall  (share of real needs found)", models)
    ci_panel(axes[1], tables["all"], "pr_auc", "PR-AUC  (ranking quality)", models, base=y_test.mean())
    axes[0].set_yticks(y, [NAMES[m] for m in models])
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == NAMES[FINAL]: lab.set_color(BLUE); lab.set_fontweight("bold")
    fig.suptitle("Locked test set: 200 posts, 93 real needs  ·  dot = score, line = 95% CI",
                 x=0.01, ha="left", color=INK2, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(F / "test_metrics.png", dpi=200); plt.close(fig)


def fig_subgroups(md):
    rows = {}
    for line in md.splitlines():
        m = re.match(r"\| (Bay Area|national|keyword hit|no keyword) \| (\d+) \| (\d+) \| (.+) \|$", line)
        if m:
            vals = [float(v) for v in m.group(4).split(" | ")]
            rows[m.group(1)] = dict(needs=int(m.group(3)), B1=vals[1], M2s=vals[5])
    order = [("keyword hit", "Posts with a need keyword"), ("no keyword", "Posts with NO need keyword"),
             ("national", "National subreddits"), ("Bay Area", "Bay Area subreddits")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(order))[::-1]
    for yi, (k, label) in zip(y, order):
        r = rows[k]
        ax.plot([r["B1"], r["M2s"]], [yi, yi], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.plot(r["B1"], yi, "o", ms=11, color=MUTED, mec=SURFACE, mew=2, zorder=3)
        ax.plot(r["M2s"], yi, "o", ms=11, color=BLUE, mec=SURFACE, mew=2, zorder=3)
        ax.text(r["B1"], yi + 0.3, f"{r['B1']:.0%}", ha="center", color=INK2, fontsize=10)
        ax.text(r["M2s"], yi + 0.3, f"{r['M2s']:.0%}", ha="center", color=INK, fontsize=10, fontweight="bold")
        ax.text(1.06, yi, f"{r['needs']} needs", va="center", color=INK2, fontsize=10)
    ax.set_yticks(y, [l for _, l in order]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.04, 1.16); ax.set_ylim(-0.6, len(order) - 0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.set_xticks([0, .25, .5, .75, 1])
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    ax.set_title("Recall by group: keyword rule vs. HelpRoute", loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.plot([], [], "o", color=MUTED, ms=9, label="B1 keyword rule"); ax.plot([], [], "o", color=BLUE, ms=9, label="M2 + silver (final)")
    ax.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(0, -0.28))
    fig.tight_layout(); fig.savefig(F / "test_recall_subgroups.png", dpi=200); plt.close(fig)


def fig_pr(pred):
    y = pred["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    series = [("M2+silver", BLUE, "M2 + silver (final)"), ("M2", ORANGE, "M2"), ("M1", AQUA, "M1 TF-IDF"), ("B2", YELLOW, "B2 zero-shot")]
    for m, c, name in series:
        pr, rc, _ = precision_recall_curve(y, pred[f"prob_{m}"])
        ap = average_precision_score(y, pred[f"prob_{m}"])
        ax.plot(rc, pr, color=c, lw=2.4 if m == FINAL else 2, zorder=3 if m == FINAL else 2, label=f"{name}  (PR-AUC {ap:.2f})")
    p1 = pred["pred_B1"].astype(bool).to_numpy()
    b1_p, b1_r = (p1 & (y == 1)).sum() / p1.sum(), (p1 & (y == 1)).sum() / y.sum()
    ax.plot(b1_r, b1_p, "s", ms=10, color=INK2, mec=SURFACE, mew=2, zorder=4, label="B1 keyword rule (one point)")
    ax.text(b1_r - 0.025, b1_p, "B1 keyword rule", color=INK, fontsize=9.5, ha="right", va="center")
    ax.axhline(y.mean(), color=MUTED, ls=":", lw=1); ax.text(0.01, y.mean() + 0.015, f"B0 / base rate {y.mean():.2f}", color=INK2, fontsize=9)
    ax.set(xlim=(0, 1.01), ylim=(0.3, 1.02), xlabel="Recall (share of real needs found)", ylabel="Precision (share of flags that are real needs)")
    ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    ax.set_title("Precision-recall curves, locked test set (200 posts)", loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(F / "test_pr_curves.png", dpi=200); plt.close(fig)


def fig_confusion(pred, models=("B1", "M2", "M2+silver")):
    """Confusion matrices on the locked test set. Cell color = share of the TRUE class (row), one hue light->dark."""
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#a9c9f0", BLUE, "#14427d"])
    titles = {"B1": "B1 keyword rule", "M2": "M2 DistilRoBERTa", "M2+silver": "M2 + silver (final)"}
    y = pred["y"].to_numpy()
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 5.0))
    for ax, m in zip(axes, models):
        cm = confusion_matrix(y, pred[f"pred_{m}"])
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        names = [["True negative", "False alarm"], ["Missed need", "Need found"]]
        for i in range(2):
            for j in range(2):
                dark = share[i, j] > 0.55
                ax.text(j, i - 0.12, f"{cm[i, j]}", ha="center", va="center", fontsize=22, fontweight="bold",
                        color="white" if dark else INK)
                ax.text(j, i + 0.18, f"{share[i, j]:.0%} of row\n{names[i][j]}", ha="center", va="center",
                        fontsize=9, color="white" if dark else INK2)
        ax.set_xticks([0, 1], ["Predicted:\nnot a need", "Predicted:\nneed"])
        ax.set_yticks([0, 1], ["Actual:\nnot a need", "Actual:\nneed"])
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color=SURFACE, lw=3); ax.tick_params(which="minor", length=0)
        ax.set_title(titles[m], loc="left", fontsize=12.5, fontweight="bold", color=BLUE if m == FINAL else INK)
    for ax in axes[1:]: ax.set_yticklabels([])
    fig.suptitle("Confusion matrices, locked test set (200 posts: 93 needs, 107 not)  ·  color = share of the actual class",
                 x=0.01, ha="left", color=INK2, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "test_confusion.png", dpi=200, bbox_inches="tight", pad_inches=0.25); plt.close(fig)


if __name__ == "__main__":
    md = (R / "test_results.md").read_text(encoding="utf-8")
    pred = pd.read_parquet(R / "test_predictions.parquet")
    tables = parse_tables(md)
    fig_metrics(tables, pred["y"].to_numpy())
    fig_subgroups(md)
    fig_pr(pred)
    print("wrote", *sorted(p.name for p in F.glob("test_*.png")))
    fig_confusion(pred)
