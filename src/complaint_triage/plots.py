"""Static charts for the README (light theme)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _short(label: str) -> str:
    return label.replace("_", " ").title().replace("Or ", "or ")


def confusion(y_true, y_pred, labels, title: str, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    ax.imshow(cm, cmap=BLUES, vmin=0, vmax=1)
    names = [_short(l) for l in labels]
    ax.set_xticks(range(len(labels)), names, rotation=30, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(labels)), names, fontsize=9, color=INK_2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > 0.55 else INK)
    ax.set_xlabel("Predicted queue", color=INK_2)
    ax.set_ylabel("Queue from CFPB label", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def per_class_f1(results: dict[str, dict], labels, path: Path,
                 title: str = "F1 by queue"):
    """Dot plot: one row per queue, one colored dot per model."""
    models = list(results)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(labels) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(labels))[::-1]
    for k, m in enumerate(models):
        f1 = [results[m]["per_class"].get(l, {}).get("f1-score", np.nan) for l in labels]
        ax.scatter(f1, y, s=64, color=SERIES[k % len(SERIES)], edgecolor=SURFACE,
                   linewidth=2, zorder=3, label=m)
    ax.set_yticks(y, [_short(l) for l in labels], color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score", color=INK_2)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models),
              fontsize=9, labelcolor=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def hybrid_curve(sweep, path: Path, title: str = "Accuracy vs. share of complaints sent to the LLM"):
    s = sweep.sort_values("share_to_llm")
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.plot(s["share_to_llm"] * 100, s["accuracy"] * 100, color=SERIES[0], linewidth=2)
    ax.scatter(s["share_to_llm"] * 100, s["accuracy"] * 100, s=40, color=SERIES[0],
               edgecolor=SURFACE, linewidth=2, zorder=3)
    lo, hi = s.iloc[0], s.iloc[-1]
    ax.annotate(f"Baseline only\n{lo['accuracy']:.1%}", (lo["share_to_llm"] * 100, lo["accuracy"] * 100),
                xytext=(8, -26), textcoords="offset points", fontsize=9, color=INK_2)
    ax.annotate(f"LLM for all\n{hi['accuracy']:.1%}", (hi["share_to_llm"] * 100, hi["accuracy"] * 100),
                xytext=(-70, 10), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_xlabel("Complaints sent to the LLM (%)", color=INK_2)
    ax.set_ylabel("Accuracy (%)", color=INK_2)
    ax.set_xlim(-2, 102)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
