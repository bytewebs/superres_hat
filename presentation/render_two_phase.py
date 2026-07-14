#!/usr/bin/env python3
"""Render a simple two-phase SenHAT training diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "senhat_two_phase.png"

DEEP = "#1A3D5C"
P1 = "#2E7D5A"      # green - fidelity
P2 = "#C45C26"      # orange - sharpening
BOX = "#F4F7FB"
ARROW = "#1A3D5C"


def box(ax, x, y, w, h, title, lines, color, title_fs=11):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor=DEEP, linewidth=1.5, alpha=0.25)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top",
            fontsize=title_fs, fontweight="bold", color=DEEP)
    cy = y + h/2 - 0.1
    for i, line in enumerate(lines):
        ax.text(x + w/2, cy - i*0.32, line, ha="center", va="center",
                fontsize=9.5, color=DEEP)


def arrow_h(ax, x1, x2, y, label=""):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2.2))
    if label:
        ax.text((x1+x2)/2, y + 0.22, label, ha="center", fontsize=8.5,
                color=DEEP, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=200)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    # ---- Shared model (top) ----
    box(ax, 4.5, 4.0, 7.0, 1.1, "SenHAT Model (same architecture both phases)", [
        "10 m + 20 m inputs  →  Transformer backbone  →  10 m multispectral output",
    ], BOX, title_fs=10.5)

    # ---- Phase 1 ----
    ax.text(0.3, 3.5, "PHASE 1", fontsize=13, fontweight="bold", color=P1)
    ax.text(0.3, 3.15, "Fidelity First", fontsize=10, color=P1)
    box(ax, 0.2, 0.5, 4.8, 2.4, "What we optimize", [
        "Pixel accuracy (L1)",
        "Spectral shape (VGG + NDI)",
        "Artifact cleanup (LDL)",
        "Goal: bands & boundaries correct",
    ], P1)

    # ---- Phase 2 ----
    ax.text(10.9, 3.5, "PHASE 2", fontsize=13, fontweight="bold", color=P2)
    ax.text(10.9, 3.15, "Sharpness Second", fontsize=10, color=P2)
    box(ax, 10.7, 0.5, 4.8, 2.4, "What we add", [
        "Adversarial texture sharpening",
        "MS-SSIM structure",
        "Same NDVI / spectral guardrails",
        "Goal: sharp edges, no fake detail",
    ], P2)

    # ---- Why two phases (center) ----
    box(ax, 5.5, 1.5, 5.0, 1.6, "Why two phases?", [
        "Phase 1 builds a trustworthy base map",
        "Phase 2 adds detail only after fidelity is stable",
        "Guardrails stop fake textures in both phases",
    ], "#D6E4F0")

    # Arrows: inputs → model → phases
    arrow_h(ax, 2.6, 5.0, 4.55, "")
    arrow_h(ax, 11.0, 5.0, 4.55, "")
    ax.annotate("", xy=(8.0, 3.9), xytext=(8.0, 4.0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2))
    ax.text(8.0, 3.65, "shared weights", ha="center", fontsize=8, color=DEEP, style="italic")

    # Phase flow arrow
    ax.annotate("", xy=(10.5, 2.5), xytext=(5.3, 2.5),
                arrowprops=dict(arrowstyle="-|>", color=DEEP, lw=2.5,
                                connectionstyle="arc3,rad=0"))
    ax.text(7.9, 2.75, "then enable", ha="center", fontsize=9, color=DEEP, fontweight="bold")

    plt.tight_layout(pad=0.3)
    plt.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
