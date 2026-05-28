"""
Patent drawings generator for AERIS invention patent application.
Generates 5 black-and-white line drawings in Chinese patent style.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial"],
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUT = Path(__file__).parent
BLACK = "#000000"
WHITE = "#FFFFFF"


def _box(ax, x, y, w, h, text, fontsize=9, bold=False):
    """Draw a rectangular box with centered text."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=WHITE, edgecolor=BLACK, linewidth=1.2,
    )
    ax.add_patch(rect)
    fw = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=fw)


def _diamond(ax, x, y, w, h, text, fontsize=8):
    """Draw a diamond (decision) shape."""
    pts = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y], [x, y+h/2]])
    ax.plot(pts[:, 0], pts[:, 1], color=BLACK, linewidth=1.2)
    ax.fill(pts[:, 0], pts[:, 1], facecolor=WHITE, edgecolor=BLACK, linewidth=1.2)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2, text="", fontsize=7):
    """Draw an arrow with optional label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0),
    )
    if text:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, text, fontsize=fontsize, ha="left", va="center")


# ============================================================
# Figure 1: Overall flowchart of AERIS routing method
# ============================================================
def fig1_overall_flow():
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Fig. 1", fontsize=11, fontweight="bold", pad=10)

    steps = [
        (3, 11.0, "S101: Start new round"),
        (3, 9.8,  "S102: CH election\n(multi-dimensional scoring)"),
        (3, 8.6,  "S103: Cluster association"),
        (3, 7.4,  "S104: CAS feature computation\n& mode selection"),
        (3, 6.2,  "S105: Intra-cluster data\ncollection & aggregation"),
        (3, 5.0,  "S106: Gateway candidate\nscoring & conditional activation"),
        (3, 3.8,  "S107: Skeleton backbone\nconstruction / update"),
        (3, 2.6,  "S108: Cascading uplink\ntransmission (4-level)"),
        (3, 1.4,  "S109: Metric logging\n& energy update"),
    ]

    for x, y, txt in steps:
        _box(ax, x, y, 4.0, 0.85, txt, fontsize=8.5)

    for i in range(len(steps) - 1):
        _arrow(ax, steps[i][0], steps[i][1] - 0.425,
               steps[i+1][0], steps[i+1][1] + 0.425)

    # Loop-back arrow from S109 to S101
    ax.annotate(
        "", xy=(1.0, 11.0), xytext=(1.0, 1.4),
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0,
                        connectionstyle="arc3,rad=0"),
    )
    ax.text(0.5, 6.2, "Next\nround", fontsize=7, ha="center", va="center")

    fig.savefig(OUT / "patent_fig1_overall_flow.pdf")
    fig.savefig(OUT / "patent_fig1_overall_flow.png")
    plt.close(fig)
    print("  Fig 1: overall flow")


# ============================================================
# Figure 2: CAS three-mode selection decision flow
# ============================================================
def fig2_cas_decision():
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Fig. 2", fontsize=11, fontweight="bold", pad=10)

    _box(ax, 3.5, 8.2, 4.5, 0.7, "S201: Compute 6-dim feature vector\n(q, e, d, r, rho, f)", fontsize=8.5)
    _box(ax, 3.5, 7.0, 4.5, 0.7, "S202: Normalize features to [0,1]\n(dynamic min-max per round)", fontsize=8.5)

    _box(ax, 1.5, 5.5, 2.2, 0.7, "S203: Score(direct)\nw_q=0.65, w_e=0.35\nw_d=-0.25 ...", fontsize=7)
    _box(ax, 3.5, 5.5, 2.2, 0.7, "S204: Score(chain)\nw_q=0.40, w_e=0.30\nw_d=0.20 ...", fontsize=7)
    _box(ax, 5.5, 5.5, 2.2, 0.7, "S205: Score(two-hop)\nw_q=0.25, w_e=0.20\nw_d=0.50 ...", fontsize=7)

    _box(ax, 3.5, 4.0, 4.5, 0.7, "S206: m* = arg max { Score(direct),\nScore(chain), Score(two-hop) }", fontsize=8.5)

    _box(ax, 1.5, 2.5, 1.8, 0.6, "Direct mode", fontsize=8.5, bold=True)
    _box(ax, 3.5, 2.5, 1.8, 0.6, "Chain mode", fontsize=8.5, bold=True)
    _box(ax, 5.5, 2.5, 1.8, 0.6, "Two-hop mode", fontsize=8.5, bold=True)

    _box(ax, 3.5, 1.2, 4.5, 0.6, "S207: Execute selected intra-cluster\ncommunication mode", fontsize=8.5)

    # Arrows
    _arrow(ax, 3.5, 7.85, 3.5, 7.35)
    _arrow(ax, 3.5, 6.65, 3.5, 6.15)

    # Fan out to three scores
    _arrow(ax, 2.5, 6.15, 1.5, 5.85)
    _arrow(ax, 3.5, 6.15, 3.5, 5.85)
    _arrow(ax, 4.5, 6.15, 5.5, 5.85)

    # Fan in from scores to argmax
    _arrow(ax, 1.5, 5.15, 2.5, 4.35)
    _arrow(ax, 3.5, 5.15, 3.5, 4.35)
    _arrow(ax, 5.5, 5.15, 4.5, 4.35)

    # From argmax to three modes
    _arrow(ax, 2.5, 3.65, 1.5, 2.85)
    _arrow(ax, 3.5, 3.65, 3.5, 2.85)
    _arrow(ax, 4.5, 3.65, 5.5, 2.85)

    # From modes to execute
    _arrow(ax, 1.5, 2.2, 2.5, 1.55)
    _arrow(ax, 3.5, 2.2, 3.5, 1.55)
    _arrow(ax, 5.5, 2.2, 4.5, 1.55)

    fig.savefig(OUT / "patent_fig2_cas_decision.pdf")
    fig.savefig(OUT / "patent_fig2_cas_decision.png")
    plt.close(fig)
    print("  Fig 2: CAS decision")


# ============================================================
# Figure 3: Cascading failure transfer logic
# ============================================================
def fig3_cascade():
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Fig. 3", fontsize=11, fontweight="bold", pad=10)

    _box(ax, 3, 11.2, 3.5, 0.6, "S301: CH has aggregated data", fontsize=8.5)
    _box(ax, 3, 10.0, 3.5, 0.6, "S302: Attempt direct route\n(CH -> BS)", fontsize=8.5)

    _diamond(ax, 3, 8.6, 2.8, 1.0, "S303:\nDirect\nsuccess?", fontsize=7.5)

    _diamond(ax, 3, 6.8, 2.8, 1.0, "S304:\nGateway\navailable?\n(I_g=1)", fontsize=7)

    _box(ax, 3, 5.4, 3.5, 0.6, "S305: Attempt gateway route\n(CH -> GW -> BS)", fontsize=8.5)

    _diamond(ax, 3, 4.0, 2.8, 1.0, "S306:\nGateway\nsuccess?", fontsize=7.5)

    _diamond(ax, 3, 2.2, 2.8, 1.0, "S307:\nSkeleton\navailable?\n(I_s=1)", fontsize=7)

    _box(ax, 3, 0.8, 3.5, 0.6, "S308: Attempt skeleton route\n(multi-hop backbone)", fontsize=8.5)

    # Right side: success exit
    _box(ax, 5.5, 8.6, 1.0, 0.5, "Done", fontsize=8, bold=True)

    # Arrows
    _arrow(ax, 3, 10.9, 3, 10.3)
    _arrow(ax, 3, 9.7, 3, 9.1)

    # Direct success? -> Yes -> Done
    ax.annotate("", xy=(5.0, 8.6), xytext=(4.4, 8.6),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.text(4.55, 8.8, "Yes", fontsize=7)

    # Direct success? -> No -> Gateway available?
    _arrow(ax, 3, 8.1, 3, 7.3, "No")

    # Gateway available? -> Yes -> Attempt gateway
    _arrow(ax, 3, 6.3, 3, 5.7, "Yes")

    # Gateway available? -> No -> Skeleton available?
    ax.annotate("", xy=(1.0, 6.8), xytext=(1.6, 6.8),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.annotate("", xy=(1.0, 2.2), xytext=(1.0, 6.8),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.annotate("", xy=(1.6, 2.2), xytext=(1.0, 2.2),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.text(1.1, 6.4, "No", fontsize=7)

    # Attempt gateway -> Gateway success?
    _arrow(ax, 3, 5.1, 3, 4.5)

    # Gateway success? -> Yes -> Done
    ax.annotate("", xy=(5.0, 4.0), xytext=(4.4, 4.0),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.text(4.55, 4.2, "Yes", fontsize=7)
    _box(ax, 5.5, 4.0, 1.0, 0.5, "Done", fontsize=8, bold=True)

    # Gateway success? -> No -> Skeleton available?
    _arrow(ax, 3, 3.5, 3, 2.7, "No")

    # Skeleton available? -> Yes -> Attempt skeleton
    _arrow(ax, 3, 1.7, 3, 1.1, "Yes")

    # Skeleton -> Done
    _box(ax, 5.5, 0.8, 1.0, 0.5, "Done", fontsize=8, bold=True)
    ax.annotate("", xy=(5.0, 0.8), xytext=(4.75, 0.8),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))

    # Skeleton not available -> fallback
    ax.text(1.1, 1.8, "No", fontsize=7)
    ax.annotate("", xy=(1.0, 2.2), xytext=(1.6, 2.2),
                arrowprops=dict(arrowstyle="<-", color=BLACK, lw=1.0))

    fig.savefig(OUT / "patent_fig3_cascade.pdf")
    fig.savefig(OUT / "patent_fig3_cascade.png")
    plt.close(fig)
    print("  Fig 3: cascade logic")


# ============================================================
# Figure 4: Gateway scoring and conditional activation
# ============================================================
def fig4_gateway():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Fig. 4", fontsize=11, fontweight="bold", pad=10)

    _box(ax, 3, 8.2, 4.0, 0.6, "S401: Measure CH->BS\ndirect route PDR", fontsize=8.5)

    _diamond(ax, 3, 6.8, 3.2, 1.0, "S402:\nPDR < adaptive\nthreshold?", fontsize=8)

    _box(ax, 5.2, 6.8, 1.2, 0.5, "Skip GW\n(not needed)", fontsize=7)

    _box(ax, 3, 5.2, 4.0, 0.7, "S403: Score each GW candidate:\nScore = 0.15*E + 0.20*C\n+ 0.35*L + (-0.60)*D", fontsize=7.5)

    _box(ax, 3, 3.8, 4.0, 0.6, "S404: Rank candidates\nby Score (descending)", fontsize=8.5)

    _box(ax, 3, 2.6, 4.0, 0.6, "S405: Select top-K candidates\nas gateway nodes (K=2~3)", fontsize=8.5)

    _box(ax, 3, 1.4, 4.0, 0.6, "S406: Activate gateway\nfor uplink relay", fontsize=8.5, bold=True)

    _arrow(ax, 3, 7.9, 3, 7.3)

    # Decision: No (PDR OK) -> skip
    ax.annotate("", xy=(4.6, 6.8), xytext=(4.2, 6.8),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.text(4.15, 7.05, "No", fontsize=7)

    # Decision: Yes -> score
    _arrow(ax, 3, 6.3, 3, 5.55, "Yes")
    _arrow(ax, 3, 4.85, 3, 4.1)
    _arrow(ax, 3, 3.5, 3, 2.9)
    _arrow(ax, 3, 2.3, 3, 1.7)

    fig.savefig(OUT / "patent_fig4_gateway.pdf")
    fig.savefig(OUT / "patent_fig4_gateway.png")
    plt.close(fig)
    print("  Fig 4: gateway activation")


# ============================================================
# Figure 5: System deployment topology
# ============================================================
def fig5_topology():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("Fig. 5", fontsize=11, fontweight="bold", pad=10)

    # Base station
    ax.plot(5, 6.5, marker="^", markersize=18, color=BLACK, markerfacecolor=WHITE, markeredgewidth=2)
    ax.text(5, 6.9, "BS (Base Station)", ha="center", fontsize=9, fontweight="bold")

    # Skeleton backbone lines (dashed)
    skeleton_chs = [(2.5, 4.5), (5, 4.5), (7.5, 4.5)]
    for sx, sy in skeleton_chs:
        ax.plot([sx, 5], [sy, 6.2], "--", color=BLACK, linewidth=1.0)

    # Three clusters
    clusters = [
        {"ch": (2.5, 4.5), "gw": (2.0, 5.3), "nodes": [(1.5, 3.8), (3.0, 3.5), (2.0, 3.2), (1.8, 4.2), (3.2, 4.2)], "label": "Cluster 1"},
        {"ch": (5.0, 4.5), "gw": (5.5, 5.3), "nodes": [(4.5, 3.8), (5.5, 3.5), (4.8, 3.2), (5.3, 4.2), (4.2, 4.2)], "label": "Cluster 2"},
        {"ch": (7.5, 4.5), "gw": (8.0, 5.3), "nodes": [(7.0, 3.8), (8.0, 3.5), (7.3, 3.2), (7.8, 4.2), (8.3, 4.2)], "label": "Cluster 3"},
    ]

    for cl in clusters:
        cx, cy = cl["ch"]
        # CH node (filled square)
        ax.plot(cx, cy, marker="s", markersize=12, color=BLACK, markerfacecolor=WHITE, markeredgewidth=2)
        ax.text(cx, cy - 0.35, "CH", ha="center", fontsize=7, fontweight="bold")

        # Gateway node (filled diamond)
        gx, gy = cl["gw"]
        ax.plot(gx, gy, marker="D", markersize=10, color=BLACK, markerfacecolor=WHITE, markeredgewidth=1.5)
        ax.text(gx + 0.3, gy, "GW", ha="left", fontsize=7)
        # GW -> BS (dotted line)
        ax.plot([gx, 5], [gy, 6.2], ":", color=BLACK, linewidth=0.8)
        # CH -> GW (solid)
        ax.plot([cx, gx], [cy, gy], "-", color=BLACK, linewidth=0.8)

        # Sensor nodes (small circles)
        for nx, ny in cl["nodes"]:
            ax.plot(nx, ny, "o", markersize=6, color=BLACK, markerfacecolor=WHITE, markeredgewidth=1.0)
            # Lines to CH
            ax.plot([nx, cx], [ny, cy], "-", color=BLACK, linewidth=0.4, alpha=0.5)

        # Cluster boundary (dashed ellipse)
        from matplotlib.patches import Ellipse
        ell = Ellipse((cx, cy - 0.3), 2.8, 2.6, fill=False, edgecolor=BLACK, linestyle="--", linewidth=0.8)
        ax.add_patch(ell)
        ax.text(cx, cy - 1.7, cl["label"], ha="center", fontsize=8)

    # Legend
    legend_y = 1.0
    ax.plot(1.0, legend_y, "^", markersize=10, color=BLACK, markerfacecolor=WHITE, markeredgewidth=1.5)
    ax.text(1.5, legend_y, "Base Station (BS)", fontsize=7.5, va="center")

    ax.plot(3.5, legend_y, "s", markersize=8, color=BLACK, markerfacecolor=WHITE, markeredgewidth=1.5)
    ax.text(4.0, legend_y, "Cluster Head (CH)", fontsize=7.5, va="center")

    ax.plot(6.0, legend_y, "D", markersize=8, color=BLACK, markerfacecolor=WHITE, markeredgewidth=1.5)
    ax.text(6.5, legend_y, "Gateway (GW)", fontsize=7.5, va="center")

    ax.plot(8.2, legend_y, "o", markersize=6, color=BLACK, markerfacecolor=WHITE, markeredgewidth=1.0)
    ax.text(8.6, legend_y, "Sensor Node", fontsize=7.5, va="center")

    legend_y2 = 0.4
    ax.plot([1.0, 1.8], [legend_y2, legend_y2], "-", color=BLACK, linewidth=1.0)
    ax.text(2.0, legend_y2, "Direct link", fontsize=7.5, va="center")
    ax.plot([3.5, 4.3], [legend_y2, legend_y2], "--", color=BLACK, linewidth=1.0)
    ax.text(4.5, legend_y2, "Skeleton backbone", fontsize=7.5, va="center")
    ax.plot([6.5, 7.3], [legend_y2, legend_y2], ":", color=BLACK, linewidth=1.0)
    ax.text(7.5, legend_y2, "Gateway relay", fontsize=7.5, va="center")

    fig.savefig(OUT / "patent_fig5_topology.pdf")
    fig.savefig(OUT / "patent_fig5_topology.png")
    plt.close(fig)
    print("  Fig 5: deployment topology")


if __name__ == "__main__":
    print("Generating patent drawings...")
    fig1_overall_flow()
    fig2_cas_decision()
    fig3_cascade()
    fig4_gateway()
    fig5_topology()
    print("All 5 patent figures generated in patent/")
