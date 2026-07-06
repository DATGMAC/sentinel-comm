#!/usr/bin/env python3
"""
Render the social-preview banner (1280x640) -> docs/img/banner.png

Upload at: GitHub repo -> Settings -> Social preview.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#1a1a19"
INK     = "#ffffff"
INK2    = "#c3c2b7"
MUTED   = "#898781"
ACCENT  = "#3987e5"
GRAYBAR = "#4a4a47"


def main():
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 640)
    ax.axis("off")

    # ── Header ───────────────────────────────────────────────────────
    ax.text(80, 560, "sentinel-comm", fontsize=26, fontweight="bold",
            color=ACCENT, va="center")
    ax.text(420, 560, "—  persistent-kernel CPU→GPU command bus",
            fontsize=17, color=MUTED, va="center")

    # ── Slogan ───────────────────────────────────────────────────────
    ax.text(80, 448, "Loses the ping.", fontsize=54, fontweight="bold",
            color=MUTED, va="center")
    ax.text(80, 368, "Wins the flood.", fontsize=54, fontweight="bold",
            color=INK, va="center")

    # ── Illustration: one lonely ping vs a dense flood ───────────────
    #    label column | dots | number column
    y_ping, y_flood = 240, 155
    x_viz, dot_step, n_dots = 420, 24, 22

    ax.text(80, y_ping, "one-shot round-trip", fontsize=15, color=INK2,
            va="center")
    ax.scatter([x_viz], [y_ping], s=210, color=GRAYBAR, zorder=3)
    ax.text(1200, y_ping, "11.6 µs", fontsize=21, color=MUTED,
            va="center", ha="right", fontweight="bold")

    ax.text(80, y_flood, "pipelined stream", fontsize=15, color=INK2,
            va="center")
    xs = [x_viz + i * dot_step for i in range(n_dots)]
    ax.scatter(xs, [y_flood] * len(xs), s=210, color=ACCENT, zorder=3)
    ax.text(1200, y_flood, "0.63 µs/op", fontsize=21, color=ACCENT,
            va="center", ha="right", fontweight="bold")

    # ── Footer ───────────────────────────────────────────────────────
    ax.text(80, 62, "64-byte packets  ·  0.5 µs enqueue  ·  96 ns GPU dispatch"
                    "  ·  no cudaLaunchKernel  ·  C++/CUDA, 2 files  ·  MIT",
            fontsize=14.5, color=MUTED, va="center")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "img", "banner.png")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
