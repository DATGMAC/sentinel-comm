#!/usr/bin/env python3
"""
Render the README benchmark images (light + dark) from bench_dispatch numbers.

Update NUMBERS below from your own `./bench_dispatch` run, then:

    python3 benchmarks/plot_benchmarks.py     # writes docs/img/bench-{light,dark}.png

Design notes: two horizontal-bar panels — the win (pipelined burst) and the
honest loss (one-shot round-trip). sentinel-comm is the single emphasis hue;
baselines are neutral gray. Every bar is direct-labeled, so color never
carries identity alone.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# ── Measured numbers (RTX 5060 Ti, CUDA 12.8 — update from bench_dispatch) ──
BURST = [  # 1000 pipelined tiny ops, one final sync — µs per op
    ("sentinel-comm ring",  0.63, True),
    ("CUDA graph replays",  2.09, False),
    ("plain kernel launches", 2.53, False),
]
ROUNDTRIP = [  # submit one op, wait for completion — p50 µs
    ("CUDA graph + sync",   8.38, False),
    ("kernel launch + sync", 9.44, False),
    ("sentinel submit + fence", 11.63, True),
]
GPU_DISPATCH_NS = 96
ENQUEUE_US = 0.62

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", baseline="#c3c2b7", accent="#2a78d6",
                  gray_bar="#c9c8c1"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
                  muted="#898781", baseline="#383835", accent="#3987e5",
                  gray_bar="#4a4a47"),
}

BAR_H = 0.38          # thin marks
ROUND_PX = 4          # 4px rounded data-end
KAPPA = 0.5523        # cubic-bezier quarter-circle constant


def rounded_end_bar(ax, y0, y1, val, color, rpx=ROUND_PX):
    """Bar from x=0 to x=val: square at the baseline, 4px-rounded at the
    value end. Radius is computed in pixel space so corners stay circular."""
    o = ax.transData.transform((0, 0))
    inv = ax.transData.inverted()
    rx = inv.transform((o[0] + rpx, o[1]))[0]
    ry = inv.transform((o[0], o[1] + rpx))[1]
    rx = min(rx, val * 0.45)
    ry = min(ry, (y1 - y0) * 0.45)
    k = KAPPA
    verts = [
        (0, y0), (val - rx, y0),
        (val - rx + k * rx, y0), (val, y0 + ry - k * ry), (val, y0 + ry),
        (val, y1 - ry),
        (val, y1 - ry + k * ry), (val - rx + k * rx, y1), (val - rx, y1),
        (0, y1), (0, y0),
    ]
    codes = [Path.MOVETO, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), linewidth=0, facecolor=color))


def draw_bars(ax, rows, t, xmax):
    """Horizontal bars, rounded on the value end, square at the baseline."""
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.55, len(rows) - 1 + 0.85)

    for i, (name, val, emphasized) in enumerate(rows):
        y = len(rows) - 1 - i
        color = t["accent"] if emphasized else t["gray_bar"]
        rounded_end_bar(ax, y - BAR_H / 2, y + BAR_H / 2, val, color)
        # direct labels: name above-left, value at bar end (ink, never bar color)
        ax.text(0, y + BAR_H / 2 + 0.13, name, ha="left", va="bottom",
                fontsize=10.5, color=t["ink2"],
                fontweight="bold" if emphasized else "normal")
        ax.text(val + xmax * 0.015, y, f"{val:.2f} µs", ha="left", va="center",
                fontsize=10.5, color=t["ink"],
                fontweight="bold" if emphasized else "normal")

    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(0, color=t["baseline"], linewidth=1.2)


def render(mode, path):
    t = THEMES[mode]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    for ax, rows, title, sub, xpad in (
        (axes[0], BURST,
         "Sustained tiny-op cost",
         "1000 pipelined ops, one final sync — lower is better", 1.30),
        (axes[1], ROUNDTRIP,
         "One-shot round-trip (p50)",
         "submit one op, wait for it — the honest loss", 1.28),
    ):
        ax.set_facecolor(t["surface"])
        xmax = max(v for _, v, _ in rows) * xpad
        draw_bars(ax, rows, t, xmax)
        ax.text(0, len(rows) - 1 + 0.80, title, ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=t["ink"],
                transform=ax.transData)
        ax.text(0, len(rows) - 1 + 0.62, sub, ha="left", va="bottom",
                fontsize=9.5, color=t["muted"])

    fig.suptitle("")
    fig.text(0.012, 0.033,
             f"RTX 5060 Ti · CUDA 12.8 · benchmarks/bench_dispatch.cu    ·    "
             f"sc_submit enqueue: {ENQUEUE_US:.2f} µs    ·    "
             f"GPU-side dispatch: {GPU_DISPATCH_NS} ns",
             fontsize=8.5, color=t["muted"])

    fig.subplots_adjust(left=0.03, right=0.985, top=0.80, bottom=0.16,
                        wspace=0.14)
    fig.savefig(path, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "docs", "img")
    os.makedirs(out, exist_ok=True)
    render("light", os.path.join(out, "bench-light.png"))
    render("dark", os.path.join(out, "bench-dark.png"))
