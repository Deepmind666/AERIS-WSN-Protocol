#!/usr/bin/env python3
"""Build LCN 2026 conference figures with a CV-inspired visual language."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_LCN26_AERIS" / "generated"

NS3_FILE = ROOT / "ns3_validation" / "results" / "lcn26_ns3_audit_20260420_012811" / "summary" / "ns3_focused_descriptive.csv"
STRICT_FILE = ROOT / "results" / "mega_experiments" / "scalability_4env_v50rigor_20260222_descriptive.csv"
ABLATION_FILE = ROOT / "results" / "mega_experiments" / "ablation_diag_multi_20260207_205448.json"
ENERGY_FILE = ROOT / "results" / "mega_experiments" / "energy_lifetime_stats.csv"
LATENCY_FILE = ROOT / "results" / "mega_experiments" / "latency_hop_v3_20260211_stats.csv"
MECH_FILE = ROOT / "results" / "lcn26_targeted_20260420" / "mechanism_grid_fat" / "mechanism_summary.csv"

ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_suburban", "outdoor_urban"]
ENV_LABELS = {
    "indoor_office": "Indoor Office",
    "indoor_factory": "Indoor Factory",
    "outdoor_suburban": "Outdoor Suburban",
    "outdoor_urban": "Outdoor Urban",
}
ENV_SHORT = {
    "indoor_office": "Office",
    "indoor_factory": "Factory",
    "outdoor_suburban": "Suburban",
    "outdoor_urban": "Urban",
}
NODE_ORDER_NS3 = [100, 500, 1000]
NODE_ORDER_STRICT = [100, 200, 300, 500, 800, 1000]
PROTO_ORDER = ["AERIS", "PEGASIS", "LEACH", "HEED", "TEEN"]
PROTO_DRAW_ORDER = ["LEACH", "HEED", "TEEN", "PEGASIS", "AERIS"]

COLORS = {
    "AERIS": "#2F5D7C",
    "PEGASIS": "#B07A8F",
    "LEACH": "#D58A5B",
    "HEED": "#7AA08D",
    "TEEN": "#C7A74D",
    "CAS": "#3E8E9B",
    "GW": "#D56C5B",
    "SK": "#5E6D8B",
    "SAFE": "#C9A24E",
    "grid": "#D9DEE5",
    "axis": "#556270",
    "text": "#24323F",
    "muted": "#7A8794",
    "benign_bg": "#FAF3F0",
    "harsh_bg": "#F2F7F5",
    "best": "#FFF2A8",
    "second": "#F6DFC2",
    "third": "#F4CED6",
}

MARKERS = {
    "AERIS": "o",
    "PEGASIS": "s",
    "LEACH": "^",
    "HEED": "D",
    "TEEN": "P",
}

LINESTYLES = {
    "AERIS": "-",
    "PEGASIS": "--",
    "LEACH": (0, (4, 2)),
    "HEED": (0, (2, 2)),
    "TEEN": (0, (1.5, 1.5)),
}


def apply_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.2,
            "axes.labelsize": 10.4,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 8.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "axes.linewidth": 0.9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.edgecolor": COLORS["axis"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "text.color": COLORS["text"],
        }
    )


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, stem: str) -> None:
    ensure_out_dir()
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png")
    plt.close(fig)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis="y")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D7DEE7", alpha=0.95),
    )


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def load_ns3_data() -> dict[tuple[str, int, str], tuple[float, float, int]]:
    data: dict[tuple[str, int, str], tuple[float, float, int]] = {}
    for row in load_csv_rows(NS3_FILE):
        proto = row["protocol"]
        if proto not in PROTO_ORDER:
            continue
        key = (row["environment"], int(row["num_nodes"]), proto)
        data[key] = (float(row["pdr_mean"]), float(row["pdr_std"]), int(row["n"]))
    return data


def load_strict_data() -> dict[tuple[str, int, str], tuple[float, float, int]]:
    data: dict[tuple[str, int, str], tuple[float, float, int]] = {}
    for row in load_csv_rows(STRICT_FILE):
        proto = row["protocol"]
        if proto not in PROTO_ORDER:
            continue
        key = (row["environment"], int(row["num_nodes"]), proto)
        data[key] = (float(row["pdr_mean"]), float(row["pdr_std"]), int(row["n"]))
    return data


def draw_scalability_panel(
    ax: plt.Axes,
    dataset: dict[tuple[str, int, str], tuple[float, float, int]],
    env: str,
    node_order: list[int],
    regime_tag: str,
) -> None:
    bg = COLORS["benign_bg"] if env == "indoor_office" else COLORS["harsh_bg"]
    ax.set_facecolor(bg)

    winner_proto = None
    winner_val = -1.0
    last_node = node_order[-1]
    for proto in PROTO_ORDER:
        mean, _, _ = dataset[(env, last_node, proto)]
        if mean > winner_val:
            winner_val = mean
            winner_proto = proto

    x = np.arange(len(node_order), dtype=float)

    for proto in PROTO_DRAW_ORDER:
        y = np.asarray([dataset[(env, n, proto)][0] for n in node_order], dtype=float)
        s = np.asarray([dataset[(env, n, proto)][1] for n in node_order], dtype=float)
        nrep = np.asarray([dataset[(env, n, proto)][2] for n in node_order], dtype=float)
        band = np.asarray([ci95(si, int(ni)) for si, ni in zip(s, nrep)], dtype=float)

        if proto in {"AERIS", "PEGASIS"}:
            alpha = 0.16 if proto == "AERIS" else 0.13
            ax.fill_between(x, y - band, y + band, color=COLORS[proto], alpha=alpha, linewidth=0)
            width = 2.5 if proto == "AERIS" else 2.1
            z = 4 if proto == "AERIS" else 3
            line_alpha = 1.0
        else:
            width = 1.4
            z = 2
            line_alpha = 0.7

        ax.plot(
            x,
            y,
            color=COLORS[proto],
            marker=MARKERS[proto],
            markersize=4.2 if proto in {"AERIS", "PEGASIS"} else 3.4,
            linewidth=width,
            linestyle=LINESTYLES[proto],
            alpha=line_alpha,
            zorder=z,
            label=proto,
        )

    style_axes(ax)
    ax.set_title(ENV_LABELS[env], pad=7)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.05, len(node_order) - 0.35)
    ax.set_xticks(x)
    labels = [str(n) for n in node_order]
    if node_order and node_order[0] == 50:
        ax.set_xticklabels(labels, rotation=24, ha="right")
    else:
        ax.set_xticklabels(labels)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax.text(
        0.98,
        0.95,
        f"{regime_tag} | {winner_proto} leads",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#D7DEE7", alpha=0.92),
    )

    for proto in ["AERIS", "PEGASIS"]:
        y_end = dataset[(env, last_node, proto)][0]
        ax.text(
            x[-1] + 0.06,
            y_end,
            f"{y_end:.2f}",
            va="center",
            ha="left",
            fontsize=8.2,
            color=COLORS[proto],
            fontweight="semibold",
        )


def build_ns3_canonical_figure() -> None:
    data = load_ns3_data()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.15), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        draw_scalability_panel(ax, data, env, NODE_ORDER_NS3, "Canonical")
        panel_label(ax, f"({chr(97 + idx)})")

    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")
    axes[2].set_xlabel("Number of nodes")
    axes[3].set_xlabel("Number of nodes")

    handles = [
        Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linestyle=LINESTYLES[p],
               linewidth=2.3 if p == "AERIS" else 2.0 if p == "PEGASIS" else 1.4, label=p)
        for p in PROTO_ORDER
    ]
    fig.legend(
        handles=handles,
        labels=PROTO_ORDER,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.subplots_adjust(top=0.83, left=0.08, right=0.99, bottom=0.11, wspace=0.12, hspace=0.24)
    save(fig, "fig_lcn26_ns3_canonical")


def build_strict_scalability_figure() -> None:
    data = load_strict_data()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.15), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        draw_scalability_panel(ax, data, env, NODE_ORDER_STRICT, "Strict")
        panel_label(ax, f"({chr(97 + idx)})")

    axes[0].set_ylabel("Mean PDR")
    axes[2].set_ylabel("Mean PDR")
    axes[2].set_xlabel("Number of nodes")
    axes[3].set_xlabel("Number of nodes")

    handles = [
        Line2D([0], [0], color=COLORS[p], marker=MARKERS[p], linestyle=LINESTYLES[p],
               linewidth=2.3 if p == "AERIS" else 2.0 if p == "PEGASIS" else 1.4, label=p)
        for p in PROTO_ORDER
    ]
    fig.legend(
        handles=handles,
        labels=PROTO_ORDER,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.subplots_adjust(top=0.83, left=0.08, right=0.99, bottom=0.11, wspace=0.12, hspace=0.24)
    save(fig, "fig_lcn26_strict_scalability")


def load_ablation_rows() -> list[dict]:
    raw = json.loads(ABLATION_FILE.read_text(encoding="utf-8"))["raw_results"]
    return [row for row in raw if row["protocol"] == "AERIS" and row["ablation_config"] in {"full", "no_gateway", "no_cas", "minimal"}]


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def build_ablation_figure() -> None:
    rows = load_ablation_rows()
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["ablation_config"])].append(float(row["pdr_expected"]))

    configs = ["full", "no_gateway", "no_cas", "minimal"]
    cfg_labels = {"full": "Full", "no_gateway": "-GW", "no_cas": "-CAS", "minimal": "Minimal"}
    module_rows = ["gateway", "cas", "skeleton", "safety"]
    module_labels = {"gateway": "Gateway", "cas": "CAS", "skeleton": "Skeleton", "safety": "Safety"}
    module_colors = {"gateway": COLORS["GW"], "cas": COLORS["CAS"], "skeleton": COLORS["SK"], "safety": COLORS["SAFE"]}

    flag_map = {}
    for cfg in configs:
        sample = next(row for row in rows if row["ablation_config"] == cfg)
        flag_map[cfg] = sample["diag_flags"]

    fig = plt.figure(figsize=(7.1, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 2.3], width_ratios=[2.1, 1.15], hspace=0.28, wspace=0.25)
    ax_setup = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[:, 1])

    # (a) setup matrix
    ax_setup.set_xlim(-0.5, len(configs) - 0.5)
    ax_setup.set_ylim(-0.5, len(module_rows) - 0.5)
    ax_setup.invert_yaxis()
    ax_setup.set_xticks(range(len(configs)))
    ax_setup.set_xticklabels([cfg_labels[c] for c in configs], fontsize=9)
    ax_setup.set_yticks(range(len(module_rows)))
    ax_setup.set_yticklabels([module_labels[m] for m in module_rows], fontsize=9)
    for j, cfg in enumerate(configs):
        for i, module in enumerate(module_rows):
            enabled = bool(flag_map[cfg][module])
            rect = plt.Rectangle(
                (j - 0.38, i - 0.30),
                0.76,
                0.60,
                facecolor=module_colors[module] if enabled else "#F8F8F8",
                edgecolor="#D4DBE3",
                linewidth=0.8,
                alpha=0.32 if enabled else 1.0,
            )
            ax_setup.add_patch(rect)
            ax_setup.scatter(
                [j],
                [i],
                s=34,
                marker="o",
                facecolors=module_colors[module] if enabled else "white",
                edgecolors=module_colors[module] if enabled else "#B8C3CF",
                linewidths=1.2,
                zorder=3,
            )
    for spine in ax_setup.spines.values():
        spine.set_visible(False)
    ax_setup.tick_params(length=0)
    ax_setup.set_title("Loss / module setup", fontsize=11.0, pad=2)
    panel_label(ax_setup, "(a)")

    # (b) absolute PDR bars
    width = 0.22
    x = np.arange(len(ENV_ORDER), dtype=float)
    plot_cfgs = ["full", "no_gateway", "no_cas"]
    bar_colors = [COLORS["AERIS"], COLORS["GW"], COLORS["CAS"]]
    for idx, cfg in enumerate(plot_cfgs):
        means = []
        errs = []
        for env in ENV_ORDER:
            m, s = mean_std(grouped[(env, cfg)])
            means.append(m)
            errs.append(ci95(s, len(grouped[(env, cfg)])))
        ax_bar.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            color=bar_colors[idx],
            alpha=0.82 if cfg == "full" else 0.72,
            edgecolor="white",
            linewidth=0.7,
            label={"full": "Full", "no_gateway": "-GW", "no_cas": "-CAS"}[cfg],
            zorder=3,
        )
        ax_bar.errorbar(
            x + (idx - 1) * width,
            means,
            yerr=errs,
            fmt="none",
            ecolor="#394552",
            elinewidth=0.9,
            capsize=2.8,
            zorder=4,
        )
        for xi, yi in zip(x + (idx - 1) * width, means):
            extra = 0.010 * idx
            dx = 0.0
            ha = "center"
            if yi > 0.90:
                extra = [0.006, 0.020, 0.034][idx]
                dx = [-0.018, 0.0, 0.018][idx]
                ha = ["right", "center", "left"][idx]
            ax_bar.text(
                xi + dx,
                yi + 0.012 + extra,
                f"{yi:.3f}",
                ha=ha,
                va="bottom",
                fontsize=6.9 if yi > 0.90 else 7.4,
                color=COLORS["text"],
            )
    style_axes(ax_bar)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([ENV_SHORT[e] for e in ENV_ORDER])
    ax_bar.set_ylabel("Mean PDR")
    ax_bar.set_ylim(0.25, 1.02)
    ax_bar.set_title("Absolute reliability", fontsize=11.0, pad=2)
    ax_bar.legend(ncol=3, loc="upper right", frameon=False, handlelength=1.4, columnspacing=0.9)
    panel_label(ax_bar, "(b)")

    # (c) gain vs full
    delta_envs = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
    y = np.arange(len(delta_envs), dtype=float)
    gw_delta = []
    cas_delta = []
    for env in delta_envs:
        full_m, _ = mean_std(grouped[(env, "full")])
        nogw_m, _ = mean_std(grouped[(env, "no_gateway")])
        nocas_m, _ = mean_std(grouped[(env, "no_cas")])
        gw_delta.append((nogw_m - full_m) * 100.0)
        cas_delta.append((nocas_m - full_m) * 100.0)

    ax_delta.axvline(0.0, color="#394552", linewidth=1.0, linestyle="--", zorder=1)
    ax_delta.hlines(y + 0.13, 0, gw_delta, color=COLORS["GW"], linewidth=1.9)
    ax_delta.hlines(y - 0.13, 0, cas_delta, color=COLORS["CAS"], linewidth=1.9)
    ax_delta.scatter(gw_delta, y + 0.13, color=COLORS["GW"], s=38, zorder=3)
    ax_delta.scatter(cas_delta, y - 0.13, color=COLORS["CAS"], s=38, marker="s", zorder=3)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels([ENV_SHORT[e] for e in delta_envs], fontsize=9)
    ax_delta.set_xlabel("Delta vs. Full (pts)")
    ax_delta.set_title("Gain vs. baseline", fontsize=11.0, pad=2)
    style_axes(ax_delta)
    ax_delta.grid(axis="x")
    lim = max(abs(min(gw_delta + cas_delta)), abs(max(gw_delta + cas_delta))) + 0.7
    ax_delta.set_xlim(-lim, lim)
    ax_delta.invert_yaxis()
    panel_label(ax_delta, "(c)")
    for ypos, val in zip(y + 0.13, gw_delta):
        ax_delta.text(val + (0.12 if val >= 0 else -0.12), ypos, f"{val:+.1f}", va="center",
                      ha="left" if val >= 0 else "right", fontsize=7.8, color=COLORS["GW"])
    for ypos, val in zip(y - 0.13, cas_delta):
        ax_delta.text(val + (0.12 if val >= 0 else -0.12), ypos, f"{val:+.1f}", va="center",
                      ha="left" if val >= 0 else "right", fontsize=7.8, color=COLORS["CAS"])

    handles = [
        Line2D([0], [0], color=COLORS["GW"], marker="o", linewidth=1.9, label="-GW vs Full"),
        Line2D([0], [0], color=COLORS["CAS"], marker="s", linewidth=1.9, label="-CAS vs Full"),
    ]
    ax_delta.legend(handles=handles, loc="lower right", frameon=False)

    fig.subplots_adjust(top=0.95, left=0.08, right=0.98, bottom=0.11)
    save(fig, "fig_lcn26_ablation_cv")


def build_tradeoff_figure() -> None:
    energy_rows = load_csv_rows(ENERGY_FILE)
    latency_rows = load_csv_rows(LATENCY_FILE)

    avg_metrics: dict[str, dict[str, float]] = {}
    for proto in PROTO_ORDER:
        subset = [r for r in energy_rows if r["protocol"] == proto]
        lat_subset = [r for r in latency_rows if r["protocol"] == proto]
        avg_metrics[proto] = {
            "pdr": float(np.mean([float(r["pdr_mean"]) for r in subset])),
            "energy": float(np.mean([float(r["energy_mean"]) for r in subset])),
            "life": float(np.mean([float(r["lifetime_mean"]) for r in subset])),
            "fnd": float(np.mean([float(r["fnd_mean"]) for r in subset])),
            "hops": float(np.mean([float(r["hops_mean"]) for r in lat_subset])),
        }

    metric_order = ["pdr", "energy", "life", "fnd", "hops"]
    metric_labels = ["PDR ↑", "Energy ↓", "Lifetime ↑", "FND ↑", "Hops ↓"]
    better = {"pdr": "high", "energy": "low", "life": "high", "fnd": "high", "hops": "low"}

    fig = plt.figure(figsize=(7.1, 5.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.44], hspace=0.28)
    ax_table = fig.add_subplot(gs[0])
    ax_plot = fig.add_subplot(gs[1])

    # top summary table
    ax_table.axis("off")
    table_data = []
    for proto in PROTO_ORDER:
        table_data.append(
            [
                proto,
                f"{avg_metrics[proto]['pdr']:.3f}",
                f"{avg_metrics[proto]['energy']:.1f}",
                f"{avg_metrics[proto]['life']:.1f}",
                f"{avg_metrics[proto]['fnd']:.1f}",
                f"{avg_metrics[proto]['hops']:.2f}",
            ]
        )
    col_labels = ["Methods"] + metric_labels
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.16, 1.0, 0.80],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D6DCE4")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#F4F7FA")
            cell.set_text_props(fontweight="semibold", color=COLORS["text"])
        elif c == 0:
            cell.set_text_props(fontweight="semibold", color=COLORS["text"])

    # highlight top-3 per metric
    rank_colors = [COLORS["best"], COLORS["second"], COLORS["third"]]
    for j, metric in enumerate(metric_order, start=1):
        vals = [(proto, avg_metrics[proto][metric]) for proto in PROTO_ORDER]
        vals = sorted(vals, key=lambda item: item[1], reverse=(better[metric] == "high"))
        top3 = [proto for proto, _ in vals[:3]]
        for rank_idx, proto in enumerate(top3):
            row_idx = PROTO_ORDER.index(proto) + 1
            cell = tbl[(row_idx, j)]
            cell.set_facecolor(rank_colors[rank_idx])
            if rank_idx == 0:
                cell.set_text_props(fontweight="bold")

    ax_table.text(
        0.0,
        1.01,
        "Protocol-level summary on the 100-node publication block",
        transform=ax_table.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="semibold",
    )
    ax_table.text(
        0.0,
        0.03,
        "Top-3 per metric are highlighted. Higher is better for PDR, lifetime, and first-node death (FND); lower is better for energy and hops.",
        transform=ax_table.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=COLORS["muted"],
    )

    # bottom environment-wise PDR vs FND for AERIS and PEGASIS
    env_positions = np.arange(len(ENV_ORDER), dtype=float)
    aeris_rows = {r["environment"]: r for r in energy_rows if r["protocol"] == "AERIS"}
    peg_rows = {r["environment"]: r for r in energy_rows if r["protocol"] == "PEGASIS"}

    aeris_pdr = np.asarray([float(aeris_rows[e]["pdr_mean"]) for e in ENV_ORDER], dtype=float)
    peg_pdr = np.asarray([float(peg_rows[e]["pdr_mean"]) for e in ENV_ORDER], dtype=float)
    aeris_fnd = np.asarray([float(aeris_rows[e]["fnd_mean"]) for e in ENV_ORDER], dtype=float)
    peg_fnd = np.asarray([float(peg_rows[e]["fnd_mean"]) for e in ENV_ORDER], dtype=float)

    width = 0.24
    ax_plot.bar(
        env_positions - width / 2,
        aeris_pdr,
        width=width,
        color=COLORS["AERIS"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax_plot.bar(
        env_positions + width / 2,
        peg_pdr,
        width=width,
        color=COLORS["PEGASIS"],
        alpha=0.60,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    style_axes(ax_plot)
    ax_plot.set_ylabel("PDR")
    ax_plot.set_ylim(0.0, 1.02)
    ax_plot.set_xticks(env_positions)
    ax_plot.set_xticklabels([ENV_SHORT[e] for e in ENV_ORDER])
    ax_plot.axvline(0.5, color="#B8C2CC", linestyle="--", linewidth=0.9)
    for x, y in zip(env_positions - width / 2, aeris_pdr):
        if y >= 0.82:
            ax_plot.text(x, y - 0.035, f"{y:.2f}", ha="center", va="top", fontsize=7.6, color="white", fontweight="semibold")
        else:
            ax_plot.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=7.8, color=COLORS["AERIS"])
    for x, y in zip(env_positions + width / 2, peg_pdr):
        if y >= 0.82:
            ax_plot.text(x, y - 0.070, f"{y:.2f}", ha="center", va="top", fontsize=7.5, color=COLORS["text"], fontweight="semibold")
        else:
            ax_plot.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=7.8, color=COLORS["PEGASIS"])

    ax2 = ax_plot.twinx()
    ax2.plot(env_positions, aeris_fnd, color=COLORS["AERIS"], marker="o", linewidth=2.0, zorder=4)
    ax2.plot(env_positions, peg_fnd, color=COLORS["PEGASIS"], marker="s", linewidth=1.8, linestyle="--", zorder=4)
    ax2.set_ylabel("First-node death (rounds)")
    ax2.set_ylim(0, max(float(np.max(peg_fnd)), float(np.max(aeris_fnd))) * 1.18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color(COLORS["axis"])
    ax2.tick_params(axis="y", colors=COLORS["axis"])
    for x, y in zip(env_positions, aeris_fnd):
        ax2.text(x - 0.03, y + 2.3, f"{y:.1f}", ha="right", va="bottom", fontsize=7.6, color=COLORS["AERIS"])
    x_last, y_last = env_positions[-1], peg_fnd[-1]
    ax2.text(x_last + 0.05, y_last + 2.3, f"{y_last:.1f}", ha="left", va="bottom", fontsize=7.6, color=COLORS["PEGASIS"])
    ax2.text(x_last - 0.20, aeris_fnd[-1] + 5.0, "AERIS FND", fontsize=7.6, color=COLORS["AERIS"], ha="right")
    ax2.text(x_last + 0.18, y_last + 6.0, "PEGASIS FND", fontsize=7.6, color=COLORS["PEGASIS"], ha="left")

    ax_plot.text(
        0.02,
        1.03,
        "Bars: mean PDR   |   Lines: first-node death",
        transform=ax_plot.transAxes,
        fontsize=8.2,
        color=COLORS["muted"],
        ha="left",
        va="bottom",
    )
    save(fig, "fig_lcn26_tradeoff_cv")


def main() -> None:
    apply_style()
    build_ns3_canonical_figure()
    build_strict_scalability_figure()
    build_ablation_figure()
    build_tradeoff_figure()
    print("[LCN26] Rebuilt CV-style conference figures in _LCN26_AERIS/generated")


if __name__ == "__main__":
    main()
