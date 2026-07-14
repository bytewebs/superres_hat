#!/usr/bin/env python3
"""Render a clean SenHAT architecture diagram as PNG."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "senhat_architecture.png"

DEEP = "#163A5C"
ACCENT = "#006E96"
GREEN = "#5F9B6E"
ORANGE = "#D28246"
BOX = "#EEF4FC"


def draw_stack(ax, x, y, w, h, n, color, label, sublabel=""):
    dx, dy = 0.12, 0.09
    for i in range(n, 0, -1):
        rect = mpatches.Polygon(
            [(x + i*dx, y + i*dy), (x + w + i*dx, y + i*dy),
             (x + w + i*dx, y + h + i*dy), (x + i*dx, y + h + i*dy)],
            closed=True, facecolor=color, edgecolor=DEEP, alpha=0.15 + 0.08*i, linewidth=0.6)
        ax.add_patch(rect)
    front = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                           facecolor=color, edgecolor=DEEP, linewidth=1.4, alpha=0.55)
    ax.add_patch(front)
    ax.text(x + w/2, y + h + 0.35, label, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=DEEP)
    if sublabel:
        ax.text(x + w/2, y + h + 0.08, sublabel, ha="center", va="bottom",
                fontsize=9, color=DEEP, alpha=0.75)


def draw_box(ax, x, y, w, h, lines, fs=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                         facecolor=BOX, edgecolor=DEEP, linewidth=1.4)
    ax.add_patch(box)
    cy = y + h/2 + (len(lines)-1)*0.12
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        size = fs if i == 0 else fs - 1
        ax.text(x + w/2, cy - i*0.28, line, ha="center", va="center",
                fontsize=size, fontweight=weight, color=DEEP)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=DEEP, lw=2.0,
                                connectionstyle="arc3,rad=0.08"))


def main():
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=200)
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-0.8, 4.5)
    ax.axis("off")

    # Inputs
    draw_stack(ax, 0.2, 2.5, 2.0, 1.2, 4, GREEN, "10 m Input", "B2 B3 B4 B8")
    draw_stack(ax, 0.2, 0.3, 1.6, 0.9, 4, ORANGE, "20 m Input", "B4 B5 B6 B8a")

    # Fusion
    draw_box(ax, 3.3, 1.35, 2.6, 1.3, ["Multi-Res Fusion", "concat + attention"])

    # Shallow
    draw_stack(ax, 6.5, 1.1, 2.0, 1.2, 5, ACCENT, "Shallow Features")

    # HAT
    draw_box(ax, 9.2, 0.9, 3.4, 1.8, [
        "Hybrid Attention Transformer (HAT)",
        "RHAG  •  HAB  •  OCAB",
        "Overlapping Cross-Attention",
    ], fs=10.5)

    # Output
    draw_stack(ax, 13.4, 1.0, 2.2, 1.3, 5, ACCENT, "Super-Resolved", "10 m multispectral")

    # Constraints panel
    draw_box(ax, 16.0, 2.6, 2.2, 1.5, [
        "Optimization",
        "L1  •  NDVI  •  SCC",
    ], fs=9.5)

    # Arrows
    arrow(ax, 2.4, 3.0, 3.2, 2.2)
    arrow(ax, 2.0, 0.8, 3.2, 1.6)
    arrow(ax, 5.9, 1.9, 6.4, 1.9)
    arrow(ax, 8.7, 1.9, 9.1, 1.9)
    arrow(ax, 12.6, 1.9, 13.3, 1.9)

    plt.tight_layout(pad=0.2)
    plt.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
