#!/usr/bin/env python3
"""Build the AERIS round-pipeline mechanism figure as a draw.io source.

The generated draw.io file keeps all text as separate text cells. All protocol
icons are emitted as text-free SVG assets and embedded into the draw.io source.
The companion preview HTML mirrors the same coordinates with HTML text boxes
over a text-free SVG graphics layer for fast visual QA.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "overleaf_upload_ready_20260501" / "generated"
ICON_DIR = OUT_DIR / "icons" / "aeris_round_pipeline"
LUCIDE_DIR = ROOT / "scripts" / "assets" / "lucide_static_isc"
FIGURE_STEM = "fig0_aeris_round_pipeline_lcn26"
DRAWIO_PATH = OUT_DIR / f"{FIGURE_STEM}.drawio"
GRAPHICS_SVG_PATH = OUT_DIR / f"{FIGURE_STEM}_graphics.svg"
STALE_PREVIEW_SVG_PATH = OUT_DIR / f"{FIGURE_STEM}_preview.svg"
PREVIEW_HTML_PATH = OUT_DIR / f"{FIGURE_STEM}_preview.html"
PDF_PATH = OUT_DIR / f"{FIGURE_STEM}.pdf"
MANIFEST_PATH = OUT_DIR / f"{FIGURE_STEM}_manifest.md"

W, H = 1440, 1080

COLORS = {
    # Academic, color-blind-aware palette:
    # - strong accents borrow from Okabe-Ito blue/sky-blue/green/orange;
    # - panel fills are low-chroma tints for labeled mechanism cells.
    "lavender": "#EEEAF7",
    "green": "#E4F1ED",
    "peach": "#F3E8C8",
    "blue": "#0072B2",
    "lightblue": "#56B4E9",
    "teal": "#009E73",
    "orange": "#E69F00",
    "yellow": "#F6EBC8",
    "bluegray": "#E6F0F7",
    "white": "#FFFFFF",
    "text": "#172033",
    "secondary": "#4B5868",
    "line": "#58667A",
    "border": "#B8C4D2",
    "purple": "#332288",
    "sand": "#DDCC77",
    "muted": "#EEF2F6",
}

DRAWIO_FONT = "Arial"
CSS_FONT = "'Arial','Helvetica Neue',Helvetica,sans-serif"


def svg_doc(width: int, height: int, body: str, viewbox: str | None = None) -> str:
    vb = viewbox or f"0 0 {width} {height}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{vb}" fill="none" stroke-linecap="round" stroke-linejoin="round">\n'
        f"{body}\n</svg>\n"
    )


def lucide_icon(name: str, color: str = COLORS["text"], stroke_width: float = 2.15) -> str:
    """Load a vendored Lucide icon and normalize it as a text-free SVG asset."""
    src = (LUCIDE_DIR / f"{name}.svg").read_text(encoding="utf-8")
    body_match = re.search(r"<svg\b[^>]*>\s*(.*?)\s*</svg>", src, flags=re.S)
    if not body_match:
        raise ValueError(f"Could not parse Lucide SVG: {name}")
    body = body_match.group(1)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()
    body = body.replace('stroke="currentColor"', "")
    body = re.sub(r'stroke-width="[^"]+"', "", body)
    body = f'<g stroke="{color}" stroke-width="{stroke_width}">\n{body}\n</g>'
    return svg_doc(64, 64, body, viewbox="0 0 24 24")


def regular_polygon(cx: float, cy: float, r: float, sides: int = 6) -> str:
    import math

    pts = []
    for idx in range(sides):
        angle = -math.pi / 2 + 2 * math.pi * idx / sides
        pts.append(f"{cx + r * math.cos(angle):.1f},{cy + r * math.sin(angle):.1f}")
    return " ".join(pts)


def icon_phase_cluster() -> str:
    c = COLORS["text"]
    blue = COLORS["blue"]
    body = [
        f'<circle cx="48" cy="48" r="10" fill="{blue}" stroke="{c}" stroke-width="3"/>',
    ]
    for x, y in [(16, 22), (82, 18), (96, 52), (72, 90), (25, 82), (12, 55)]:
        body.append(f'<line x1="48" y1="48" x2="{x}" y2="{y}" stroke="{c}" stroke-width="3"/>')
        body.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/>')
    return svg_doc(108, 108, "\n".join(body))


def icon_phase_intra() -> str:
    c = COLORS["text"]
    green = COLORS["teal"]
    body = [
        f'<circle cx="54" cy="24" r="8" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/>',
        f'<circle cx="20" cy="54" r="8" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/>',
        f'<circle cx="88" cy="54" r="8" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/>',
        f'<circle cx="54" cy="72" r="10" fill="{COLORS["green"]}" stroke="{c}" stroke-width="3"/>',
        f'<path d="M54 32v30M28 54h18M80 54H62" stroke="{c}" stroke-width="3"/>',
        f'<path d="M35 92c10-9 28-9 38 0M43 101c6-5 16-5 22 0" stroke="{c}" stroke-width="4"/>',
    ]
    return svg_doc(108, 108, "\n".join(body))


def tower(cx: float, cy: float, scale: float = 1.0, color: str = COLORS["blue"]) -> str:
    sw = max(2.0, 3.0 * scale)
    return "\n".join(
        [
            f'<circle cx="{cx}" cy="{cy - 34 * scale}" r="{5 * scale}" fill="{COLORS["white"]}" stroke="{color}" stroke-width="{sw}"/>',
            f'<path d="M{cx} {cy - 29 * scale} L{cx - 22 * scale} {cy + 34 * scale} H{cx + 22 * scale} Z" stroke="{color}" stroke-width="{sw}" fill="none"/>',
            f'<path d="M{cx - 13 * scale} {cy + 3 * scale} H{cx + 13 * scale} M{cx - 8 * scale} {cy - 12 * scale} H{cx + 8 * scale} M{cx - 17 * scale} {cy + 24 * scale} H{cx + 17 * scale}" stroke="{color}" stroke-width="{sw}"/>',
            f'<path d="M{cx - 22 * scale} {cy - 38 * scale} A{28 * scale} {28 * scale} 0 0 1 {cx + 22 * scale} {cy - 38 * scale}" stroke="{color}" stroke-width="{sw}" fill="none"/>',
            f'<path d="M{cx - 34 * scale} {cy - 46 * scale} A{44 * scale} {44 * scale} 0 0 1 {cx + 34 * scale} {cy - 46 * scale}" stroke="{color}" stroke-width="{sw}" fill="none"/>',
        ]
    )


def icon_phase_uplink() -> str:
    return svg_doc(108, 108, tower(54, 56, 1.0, COLORS["blue"]))


def icon_topology() -> str:
    line = COLORS["line"]
    blue = COLORS["blue"]
    text = COLORS["text"]
    body = [
        f'<ellipse cx="112" cy="78" rx="88" ry="58" fill="{COLORS["lavender"]}" fill-opacity=".38" stroke="{COLORS["border"]}" stroke-width="2" stroke-dasharray="9 7"/>',
        f'<ellipse cx="244" cy="92" rx="76" ry="54" fill="{COLORS["green"]}" fill-opacity=".32" stroke="{COLORS["border"]}" stroke-width="2" stroke-dasharray="9 7"/>',
    ]
    ch1 = (112, 78)
    ch2 = (232, 96)
    m1 = [(60, 30), (56, 78), (77, 125), (136, 26), (156, 62), (148, 126)]
    m2 = [(192, 52), (190, 126), (282, 58), (270, 132)]
    for x, y in m1:
        body.append(f'<path d="M{ch1[0]} {ch1[1]} L{x} {y}" stroke="{line}" stroke-width="2.3" stroke-dasharray="8 7"/>')
    for x, y in m2:
        body.append(f'<path d="M{ch2[0]} {ch2[1]} L{x} {y}" stroke="{line}" stroke-width="2.3" stroke-dasharray="8 7"/>')
    body.append(f'<path d="M{ch1[0]} {ch1[1]} L{ch2[0]} {ch2[1]}" stroke="{line}" stroke-width="2.7"/>')
    body.append(f'<path d="M{ch2[0]} {ch2[1]} L310 86" stroke="{blue}" stroke-width="4"/>')
    body.append(f'<path d="M328 86 L397 67" stroke="{blue}" stroke-width="4"/>')
    body.append(f'<path d="M270 132 C304 154 352 146 397 108" stroke="{blue}" stroke-width="2.8" stroke-dasharray="9 8"/>')
    for x, y in m1 + m2 + [(316, 144)]:
        body.append(f'<circle cx="{x}" cy="{y}" r="10" fill="{COLORS["white"]}" stroke="{text}" stroke-width="2.5"/>')
    body.append(f'<circle cx="{ch1[0]}" cy="{ch1[1]}" r="13" fill="{blue}" stroke="{text}" stroke-width="3"/>')
    body.append(f'<circle cx="{ch2[0]}" cy="{ch2[1]}" r="13" fill="{blue}" stroke="{text}" stroke-width="3"/>')
    body.append(f'<polygon points="{regular_polygon(320, 86, 15)}" fill="{COLORS["peach"]}" stroke="{text}" stroke-width="2.2"/>')
    body.append(tower(397, 102, 0.62, blue))
    return svg_doc(420, 166, "\n".join(body))


def icon_line_sample(kind: str) -> str:
    if kind == "intra":
        dash = ' stroke-dasharray="8 7"'
        color = COLORS["line"]
        width = 3
    elif kind == "skeleton":
        dash = ' stroke-dasharray="10 8"'
        color = COLORS["blue"]
        width = 3
    else:
        dash = ""
        color = COLORS["blue"]
        width = 4
    return svg_doc(72, 24, f'<path d="M8 12 H64" stroke="{color}" stroke-width="{width}"{dash}/>')


def icon_node(kind: str) -> str:
    if kind == "ch":
        body = f'<circle cx="24" cy="24" r="14" fill="{COLORS["blue"]}" stroke="{COLORS["text"]}" stroke-width="3"/>'
    elif kind == "gw":
        body = f'<polygon points="{regular_polygon(24, 24, 15)}" fill="{COLORS["peach"]}" stroke="{COLORS["text"]}" stroke-width="2.5"/>'
    elif kind == "bs":
        body = tower(24, 27, 0.34, COLORS["blue"])
    else:
        body = f'<circle cx="24" cy="24" r="12" fill="{COLORS["white"]}" stroke="{COLORS["text"]}" stroke-width="2.5"/>'
    return svg_doc(48, 48, body)


def simple_icon(name: str) -> str:
    c = COLORS["text"]
    blue = COLORS["blue"]
    line = COLORS["line"]
    if name == "clock":
        body = f'<circle cx="32" cy="32" r="22" stroke="{c}" stroke-width="4"/><path d="M32 18v16l12 7" stroke="{c}" stroke-width="4"/>'
    elif name == "battery":
        body = f'<rect x="18" y="12" width="27" height="40" rx="3" stroke="{c}" stroke-width="4"/><path d="M26 8h11" stroke="{c}" stroke-width="4"/><rect x="24" y="21" width="15" height="22" fill="{COLORS["lightblue"]}" fill-opacity=".28" stroke="none"/>'
    elif name == "neighborhood":
        body = f'<circle cx="32" cy="17" r="6" stroke="{c}" stroke-width="3"/><circle cx="17" cy="43" r="6" stroke="{c}" stroke-width="3"/><circle cx="47" cy="43" r="6" stroke="{c}" stroke-width="3"/><path d="M29 22L20 38M35 22l9 16M23 43h18" stroke="{c}" stroke-width="3"/>'
    elif name == "star":
        body = f'<path d="M32 10l7 14 15 2-11 11 3 16-14-8-14 8 3-16-11-11 15-2z" fill="{COLORS["white"]}" stroke="{c}" stroke-width="4"/>'
    elif name == "bars":
        body = f'<rect x="14" y="38" width="7" height="14" fill="{COLORS["lightblue"]}" fill-opacity=".28" stroke="{c}" stroke-width="3"/><rect x="29" y="28" width="7" height="24" fill="{COLORS["lightblue"]}" fill-opacity=".28" stroke="{c}" stroke-width="3"/><rect x="44" y="16" width="7" height="36" fill="{COLORS["lightblue"]}" fill-opacity=".28" stroke="{c}" stroke-width="3"/>'
    elif name == "join":
        body = f'<path d="M10 32h24" stroke="{c}" stroke-width="4"/><path d="M26 22l10 10-10 10" stroke="{c}" stroke-width="4"/><circle cx="48" cy="32" r="10" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/>'
    elif name == "balance":
        body = f'<path d="M32 12v40M18 18h28M18 18l-9 18h18zM46 18l-9 18h18z" stroke="{c}" stroke-width="3.5"/><path d="M22 54h20" stroke="{c}" stroke-width="4"/>'
    elif name == "table":
        body = f'<rect x="12" y="14" width="40" height="38" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3.5"/><path d="M12 27h40M12 40h40M25 14v38M39 14v38" stroke="{c}" stroke-width="3"/>'
    elif name == "dedup":
        body = f'<path d="M19 14h26v34H19zM14 20h26M9 26h26" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/><circle cx="44" cy="45" r="10" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/><path d="M39 45l4 4 7-9" stroke="{blue}" stroke-width="3"/>'
    elif name == "fuse":
        body = f'<path d="M14 12h36L36 29v13l-8 5V29z" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3.5"/><circle cx="20" cy="52" r="4" fill="{blue}"/><circle cx="32" cy="54" r="4" fill="{blue}"/><circle cx="44" cy="52" r="4" fill="{blue}"/>'
    elif name == "buffer":
        body = f'<ellipse cx="32" cy="16" rx="18" ry="7" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/><path d="M14 16v28c0 4 8 8 18 8s18-4 18-8V16" stroke="{c}" stroke-width="3" fill="none"/><path d="M14 30c0 4 8 8 18 8s18-4 18-8" stroke="{c}" stroke-width="3"/><path d="M14 44c0 4 8 8 18 8s18-4 18-8" stroke="{c}" stroke-width="3"/>'
    elif name == "shield":
        body = f'<path d="M32 8l22 9v15c0 16-10 25-22 30-12-5-22-14-22-30V17z" fill="{COLORS["bluegray"]}" stroke="{c}" stroke-width="3.5"/><path d="M22 33l7 7 14-17" stroke="{blue}" stroke-width="4"/>'
    else:
        body = f'<circle cx="32" cy="32" r="18" stroke="{line}" stroke-width="4"/>'
    return svg_doc(64, 64, body)


def mini_role_chip(kind: str) -> str:
    c = COLORS["text"]
    if kind == "member":
        body = f'<circle cx="24" cy="24" r="13" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/>'
    elif kind == "relay":
        body = f'<circle cx="24" cy="24" r="13" fill="{COLORS["bluegray"]}" stroke="{COLORS["blue"]}" stroke-width="3"/><path d="M17 24h14M25 18l6 6-6 6" stroke="{COLORS["blue"]}" stroke-width="2.6"/>'
    else:
        body = f'<circle cx="24" cy="24" r="14" fill="{COLORS["blue"]}" stroke="{c}" stroke-width="3"/>'
    return svg_doc(48, 48, body)


def route_atom(kind: str) -> str:
    c = COLORS["text"]
    blue = COLORS["blue"]
    if kind == "m":
        body = f'<circle cx="24" cy="24" r="12" fill="{COLORS["white"]}" stroke="{c}" stroke-width="3"/>'
    elif kind == "r":
        body = f'<circle cx="24" cy="24" r="12" fill="{COLORS["bluegray"]}" stroke="{blue}" stroke-width="3"/>'
    elif kind == "ch":
        body = f'<circle cx="24" cy="24" r="12" fill="{blue}" stroke="{c}" stroke-width="3"/>'
    elif kind == "gw":
        body = f'<polygon points="{regular_polygon(24, 24, 13)}" fill="{COLORS["peach"]}" stroke="{c}" stroke-width="2.5"/>'
    elif kind == "bs":
        body = tower(24, 26, 0.34, blue)
    else:
        raise ValueError(f"Unknown route atom: {kind}")
    return svg_doc(48, 48, body)


def route_arrow(color: str, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    return svg_doc(
        64,
        24,
        (
            f'<path d="M6 12H48" stroke="{color}" stroke-width="4"{dash}/>'
            f'<path d="M40 5L48 12L40 19" stroke="{color}" stroke-width="4" fill="none"/>'
        ),
    )


def atom_row_width(parts: list[tuple[str, float, float]]) -> float:
    return sum(w for _, w, _ in parts)


def place_atom_row(builder: FigureBuilder, x: float, y: float, parts: list[tuple[str, float, float]]) -> None:
    cursor = x
    for name, w, h in parts:
        builder.image(name, cursor, y, w, h)
        cursor += w


def route_icon(kind: str) -> str:
    c = COLORS["text"]
    line = COLORS["blue"]
    dash = ""
    fill_ch = COLORS["blue"]
    fill_m = COLORS["white"]
    if kind == "skeleton":
        dash = ' stroke-dasharray="9 7"'
    if kind == "fallback":
        dash = ' stroke-dasharray="8 7"'
        line = COLORS["line"]
    nodes: list[tuple[float, float, str, str]] = []
    links: list[tuple[tuple[float, float], tuple[float, float], str]] = []
    if kind == "direct_mode":
        nodes = [(36, 36, fill_m, c), (120, 36, fill_ch, c)]
        links = [((50, 36), (104, 36), "solid")]
    elif kind == "chain_mode":
        nodes = [(28, 36, fill_m, c), (92, 36, fill_m, c), (156, 36, fill_ch, c)]
        links = [((42, 36), (76, 36), "solid"), ((106, 36), (140, 36), "solid")]
    elif kind == "twohop_mode":
        nodes = [(28, 36, fill_m, c), (92, 36, fill_m, c), (156, 36, fill_ch, c)]
        links = [((42, 36), (76, 36), "solid"), ((106, 36), (140, 36), "solid")]
    elif kind == "direct_uplink":
        nodes = [(26, 38, fill_ch, c)]
        links = [((48, 38), (116, 38), "solid")]
    elif kind == "gateway":
        nodes = [(24, 42, fill_ch, c), (104, 42, COLORS["peach"], c)]
        links = [((42, 42), (82, 42), "solid"), ((126, 42), (170, 42), "solid")]
    elif kind == "skeleton":
        nodes = [(24, 42, fill_ch, c), (104, 42, fill_m, COLORS["blue"])]
        links = [((42, 42), (82, 42), "dash"), ((126, 42), (170, 42), "dash")]
    else:
        nodes = [(26, 42, fill_ch, c)]
        links = [((48, 42), (154, 42), "dash")]
    body = []
    for (x1, y1), (x2, y2), style in links:
        d = dash if style == "dash" else ""
        body.append(f'<path d="M{x1} {y1} H{x2}" stroke="{line}" stroke-width="4"{d}/>')
        body.append(f'<path d="M{x2 - 10} {y2 - 7} L{x2} {y2} L{x2 - 10} {y2 + 7}" stroke="{line}" stroke-width="4" fill="none"/>')
    for x, y, fill, stroke in nodes:
        if fill == COLORS["peach"]:
            body.append(f'<polygon points="{regular_polygon(x, y, 15)}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
        else:
            body.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    if kind in {"direct_uplink", "gateway", "skeleton", "fallback"}:
        body.append(tower(190, 55, 0.42, COLORS["blue"]))
    return svg_doc(220, 86, "\n".join(body))


def forwarding_icon() -> str:
    c = COLORS["text"]
    line = COLORS["line"]
    blue = COLORS["blue"]
    body = [
        f'<circle cx="150" cy="78" r="17" fill="{blue}" stroke="{c}" stroke-width="3"/>',
    ]
    for x, y in [(62, 28), (42, 78), (72, 125), (250, 28), (278, 78), (238, 125)]:
        body.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{COLORS["white"]}" stroke="{c}" stroke-width="2.5"/>')
        body.append(f'<path d="M{x + (13 if x < 150 else -13)} {y} L{150 + (-23 if x < 150 else 23)} {78}" stroke="{line}" stroke-width="3" stroke-dasharray="8 7"/>')
        endx = 150 + (-23 if x < 150 else 23)
        arrow = "M{} {} l-10 -5 M{} {} l-10 5".format(endx, 78, endx, 78) if x < 150 else "M{} {} l10 -5 M{} {} l10 5".format(endx, 78, endx, 78)
        body.append(f'<path d="{arrow}" stroke="{line}" stroke-width="3"/>')
    body.append(f'<path d="M137 55c8-7 20-7 28 0M130 44c14-13 34-13 48 0" stroke="{blue}" stroke-width="3"/>')
    return svg_doc(320, 156, "\n".join(body))


def make_icons() -> dict[str, str]:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for stale in ICON_DIR.glob("*.svg"):
        stale.unlink()
    icons = {
        "phase_cluster": icon_phase_cluster(),
        "phase_intra": icon_phase_intra(),
        "phase_uplink": icon_phase_uplink(),
        "topology": icon_topology(),
        "legend_ch": icon_node("ch"),
        "legend_m": icon_node("m"),
        "legend_gw": icon_node("gw"),
        "legend_bs": icon_node("bs"),
        "line_intra": icon_line_sample("intra"),
        "line_uplink": icon_line_sample("uplink"),
        "line_skeleton": icon_line_sample("skeleton"),
        "clock": lucide_icon("clock"),
        "battery": lucide_icon("battery-medium"),
        "neighborhood": lucide_icon("network"),
        "star": lucide_icon("star"),
        "bars": lucide_icon("chart-column-increasing"),
        "join": lucide_icon("log-in"),
        "balance": lucide_icon("scale"),
        "table": lucide_icon("table-2"),
        "dedup": lucide_icon("copy-check"),
        "fuse": lucide_icon("funnel"),
        "buffer": lucide_icon("database"),
        "route_m": route_atom("m"),
        "route_r": route_atom("r"),
        "route_ch": route_atom("ch"),
        "route_gw": route_atom("gw"),
        "route_bs": route_atom("bs"),
        "route_arrow_blue": route_arrow(COLORS["blue"]),
        "route_arrow_blue_dashed": route_arrow(COLORS["blue"], dashed=True),
        "route_arrow_line": route_arrow(COLORS["line"]),
        "route_arrow_line_dashed": route_arrow(COLORS["line"], dashed=True),
        "forwarding": forwarding_icon(),
        "role_m": mini_role_chip("member"),
        "role_r": mini_role_chip("relay"),
        "role_ch": mini_role_chip("ch"),
    }
    for name, svg in icons.items():
        (ICON_DIR / f"{name}.svg").write_text(svg, encoding="utf-8")
    return icons


@dataclass
class TextSpec:
    value: str
    x: float
    y: float
    w: float
    h: float
    size: int = 18
    color: str = COLORS["text"]
    bold: bool = False
    italic: bool = False
    align: str = "center"
    valign: str = "middle"


class FigureBuilder:
    def __init__(self, icons: dict[str, str]) -> None:
        self.icons = icons
        self.text_cells: list[str] = []
        self.image_cells: list[str] = []
        self.graphics: list[str] = []
        self.text_preview: list[str] = []
        self.image_preview: list[str] = []
        self.next_id = 10

    def _id(self, prefix: str) -> str:
        self.next_id += 1
        return f"{prefix}_{self.next_id}"

    def add_cell(self, cid: str, value: str, style: str, x: float, y: float, w: float, h: float) -> None:
        self.cells.append(
            f'        <mxCell id="{cid}" value="{escape(value)}" style="{escape(style)}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry" />\n'
            f"        </mxCell>"
        )

    def rect(self, x: float, y: float, w: float, h: float, fill: str = COLORS["white"], stroke: str = COLORS["border"], sw: float = 1.2, r: int = 8, dashed: bool = False, opacity: float = 1.0) -> None:
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        opacity_attr = f' fill-opacity="{opacity}"' if opacity < 1 else ""
        self.graphics.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"{opacity_attr} stroke="{stroke}" stroke-width="{sw}"{dash}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = COLORS["line"], sw: float = 2, dashed: bool = False, arrow: bool = False) -> None:
        cid = self._id("line")
        direction = "east" if abs(x2 - x1) >= abs(y2 - y1) else "south"
        style = f"shape=line;html=1;strokeColor={color};strokeWidth={sw};direction={direction};"
        if dashed:
            style += "dashed=1;dashPattern=8 6;"
        if arrow:
            style += "endArrow=blockThin;endFill=1;endSize=8;"
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        marker = f' marker-end="url(#arrow-{color[1:]})"' if arrow else ""
        self.graphics.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{dash}{marker}/>')

    def icon(self, name: str, x: float, y: float, w: float, h: float) -> None:
        b64 = base64.b64encode(self.icons[name].encode("utf-8")).decode("ascii")
        self.graphics.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/svg+xml;base64,{b64}"/>')

    def image(self, name: str, x: float, y: float, w: float, h: float) -> None:
        cid = self._id("img")
        b64 = base64.b64encode(self.icons[name].encode("utf-8")).decode("ascii")
        style = (
            "shape=image;html=1;imageAspect=0;aspect=fixed;"
            f"strokeColor=none;fillColor=none;image=data:image/svg+xml;base64,{b64};"
        )
        self.image_cells.append(
            f'        <mxCell id="{cid}" value="" style="{escape(style)}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry" />\n'
            f"        </mxCell>"
        )
        self.image_preview.append(
            f'<img alt="" src="data:image/svg+xml;base64,{b64}" '
            f'style="position:absolute;left:{x:.1f}px;top:{y:.1f}px;width:{w:.1f}px;height:{h:.1f}px;z-index:1;" />'
        )

    def text(self, spec: TextSpec) -> None:
        cid = self._id("txt")
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;fontFamily={DRAWIO_FONT};fontSize={spec.size};"
            f"fontColor={spec.color};align={spec.align};verticalAlign={spec.valign};whiteSpace=wrap;"
        )
        font_style = 0
        if spec.bold:
            font_style += 1
        if spec.italic:
            font_style += 2
        if font_style:
            style += f"fontStyle={font_style};"
        value = spec.value.replace("\n", "<br>")
        self.text_cells.append(
            f'        <mxCell id="{cid}" value="{escape(value)}" style="{escape(style)}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="{spec.x:.1f}" y="{spec.y:.1f}" width="{spec.w:.1f}" height="{spec.h:.1f}" as="geometry" />\n'
            f"        </mxCell>"
        )
        weight = "700" if spec.bold else "400"
        fstyle = "italic" if spec.italic else "normal"
        justify = {"left": "flex-start", "center": "center", "right": "flex-end"}.get(spec.align, "center")
        align_items = {"top": "flex-start", "middle": "center", "bottom": "flex-end"}.get(spec.valign, "center")
        safe_value = escape(spec.value).replace("\n", "<br>")
        css = (
            f"position:absolute;left:{spec.x:.1f}px;top:{spec.y:.1f}px;width:{spec.w:.1f}px;height:{spec.h:.1f}px;"
            f"display:flex;align-items:{align_items};justify-content:{justify};text-align:{spec.align};"
            f"font-family:{CSS_FONT};font-size:{spec.size}px;font-weight:{weight};font-style:{fstyle};"
            f"line-height:1.10;color:{spec.color};box-sizing:border-box;white-space:normal;letter-spacing:0;"
        )
        self.text_preview.append(f'<div class="label" style="{css}">{safe_value}</div>')

    def badge(self, x: float, y: float, w: float, h: float, label: str, fill: str, stroke: str, text_color: str = COLORS["text"]) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.0, r=16)
        self.text(TextSpec(label, x, y + 1, w, h, size=18, color=text_color, bold=True))

    def save_drawio(self) -> None:
        graphics_svg = self.graphics_svg()
        encoded = quote(graphics_svg, safe="")
        bg_style = (
            "shape=image;html=1;imageAspect=0;aspect=fixed;locked=1;"
            f"strokeColor=none;fillColor=none;image=data:image/svg+xml,{encoded};"
        )
        bg_cell = (
            f'        <mxCell id="graphics_layer" value="" style="{escape(bg_style)}" vertex="1" parent="1">\n'
            f'          <mxGeometry x="0" y="0" width="{W}" height="{H}" as="geometry" />\n'
            f"        </mxCell>"
        )
        xml = "\n".join(
            [
                '<mxfile host="app.diagrams.net" modified="2026-05-01T00:00:00.000Z" agent="Codex" version="29.5.6" type="device">',
                '  <diagram id="aeris-round-pipeline-v2" name="AERIS round pipeline">',
                f'    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" tooltips="1" connect="0" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" background="#FFFFFF" math="0" shadow="0">',
                "      <root>",
                '        <mxCell id="0" />',
                '        <mxCell id="1" parent="0" />',
                bg_cell,
                *self.image_cells,
                *self.text_cells,
                "      </root>",
                "    </mxGraphModel>",
                "  </diagram>",
                "</mxfile>",
                "",
            ]
        )
        DRAWIO_PATH.write_text(xml, encoding="utf-8")

    def graphics_svg(self) -> str:
        markers = []
        for color in {COLORS["line"], COLORS["blue"]}:
            key = color[1:]
            markers.append(
                f'<marker id="arrow-{key}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
                f'<path d="M0,0 L8,4 L0,8 Z" fill="{color}"/></marker>'
            )
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                "<defs>",
                *markers,
                "</defs>",
                f'<rect width="{W}" height="{H}" fill="{COLORS["white"]}"/>',
                *self.graphics,
                "</svg>",
                "",
            ]
        )

    def save_preview(self) -> None:
        graphics_svg = self.graphics_svg()
        GRAPHICS_SVG_PATH.write_text(graphics_svg, encoding="utf-8")
        if STALE_PREVIEW_SVG_PATH.exists():
            STALE_PREVIEW_SVG_PATH.unlink()
        PREVIEW_HTML_PATH.write_text(
            "\n".join(
                [
                    "<!doctype html>",
                    '<html><head><meta charset="utf-8">',
                    (
                        "<style>@page{size:15in 11.25in;margin:0}"
                        "html,body{margin:0;width:1440px;height:1080px;overflow:hidden;background:#fff}"
                        ".canvas{position:relative;width:1440px;height:1080px;overflow:hidden;background:#fff}"
                        ".canvas>svg{position:absolute;inset:0;display:block}.label{z-index:2}</style>"
                    ),
                    "</head><body>",
                    '<div class="canvas">',
                    graphics_svg,
                    *self.image_preview,
                    *self.text_preview,
                    "</div>",
                    "</body></html>",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def build_figure(icons: dict[str, str]) -> FigureBuilder:
    b = FigureBuilder(icons)

    # Title and vertical phase guides.
    b.text(TextSpec("AERIS round pipeline", 0, 18, W, 50, size=42, bold=True))
    b.line(468, 78, 468, 1038, color=COLORS["border"], sw=2, dashed=True)
    b.line(966, 78, 966, 1038, color=COLORS["border"], sw=2, dashed=True)

    # Phase headers.
    headers = [
        (28, 74, 420, "phase_cluster", "Phase 1", "Cluster Formation", COLORS["purple"]),
        (496, 74, 448, "phase_intra", "Phase 2", "Intra-Cluster Communication", COLORS["teal"]),
        (992, 74, 420, "phase_uplink", "Phase 3", "Uplink to BS", COLORS["blue"]),
    ]
    for x, y, w, icon, phase, title, rule in headers:
        b.icon(icon, x + 22, y - 4, 68, 68)
        title_size = 24 if title.startswith("Intra") else 26
        b.text(TextSpec(phase, x + 92, y - 2, w - 92, 34, size=31, bold=True))
        b.text(TextSpec(title, x + 92, y + 36, w - 92, 34, size=title_size + 1, bold=True))
        b.line(x, y + 84, x + w, y + 84, color=rule, sw=4)

    # Phase 1 topology and legend.
    b.rect(28, 176, 420, 190, fill=COLORS["white"], stroke=COLORS["border"], sw=1.5, r=8)
    b.icon("topology", 38, 187, 400, 166)
    b.rect(28, 384, 420, 92, fill=COLORS["white"], stroke=COLORS["border"], sw=1.2, r=8)
    legend_items = [
        ("legend_ch", "CH", 46, 400),
        ("legend_m", "M", 106, 400),
        ("legend_gw", "GW", 166, 400),
        ("legend_bs", "BS", 238, 400),
        ("line_intra", "intra", 296, 400),
        ("line_uplink", "uplink", 296, 429),
        ("line_skeleton", "skeleton", 296, 458),
    ]
    for icon, label, x, y in legend_items:
        if icon.startswith("line"):
            b.icon(icon, x, y - 4, 56, 18)
            b.text(TextSpec(label, x + 62, y - 9, 88, 28, size=20, align="left"))
        else:
            b.icon(icon, x, y, 34, 34)
            b.text(TextSpec(label, x - 8, y + 39, 50, 24, size=19, bold=True))

    # Phase 1 process cards.
    process = [
        (28, 498, COLORS["lavender"], "1", "Round state", [("clock", "time slot"), ("battery", "residual\nenergy"), ("neighborhood", "neighborhood")]),
        (28, 650, COLORS["peach"], "2", "CH scoring", [("battery", "energy"), ("star", "centrality"), ("bars", "link\nquality")]),
        (28, 802, COLORS["bluegray"], "3", "Cluster association", [("join", "join CH"), ("balance", "balance"), ("table", "update")]),
    ]
    for x, y, fill, num, title, items in process:
        b.rect(x, y, 420, 134, fill=fill, stroke=COLORS["border"], sw=1.3, r=8)
        b.rect(x + 12, y + 14, 44, 44, fill=COLORS["white"], stroke=COLORS["blue"], sw=1.6, r=22)
        b.text(TextSpec(num, x + 12, y + 14, 44, 44, size=25, bold=True))
        b.text(TextSpec(title, x + 72, y + 10, 270, 36, size=27, bold=True))
        if title == "CH scoring":
            b.rect(x + 322, y + 15, 90, 30, fill=COLORS["white"], stroke=COLORS["blue"], sw=1.2, r=15)
            b.text(TextSpec("Elect CH", x + 322, y + 16, 90, 28, size=18, bold=True))
        for idx, (icon, label) in enumerate(items):
            ix = x + 72 + idx * 122
            b.icon(icon, ix, y + 54, 46, 46)
            b.text(TextSpec(label, ix - 40, y + 96, 126, 36, size=19))
            if idx:
                b.line(x + 48 + idx * 122, y + 54, x + 48 + idx * 122, y + 116, color=COLORS["border"], sw=1.4, dashed=True)

    # Phase 2.
    b.rect(496, 182, 448, 52, fill=COLORS["white"], stroke=COLORS["border"], sw=1.1, r=10)
    b.text(TextSpec("Inputs:", 520, 191, 88, 32, size=24, bold=True, align="left"))
    b.text(TextSpec("energy \u00b7 link quality \u00b7 density \u00b7 buffer", 612, 191, 318, 32, size=19, align="left"))
    b.rect(496, 260, 448, 276, fill=COLORS["lavender"], stroke=COLORS["border"], sw=1.4, r=8)
    b.rect(558, 278, 46, 46, fill=COLORS["white"], stroke=COLORS["blue"], sw=1.6, r=23)
    b.text(TextSpec("1", 558, 278, 46, 46, size=25, bold=True))
    b.text(TextSpec("CAS mode selector", 620, 274, 250, 36, size=27, bold=True))
    b.text(TextSpec("select one mode", 632, 313, 206, 28, size=20))
    mode_cards = [
        (
            508,
            "Direct",
            "M\u00a0\u2192\u00a0CH",
            [("route_m", 18, 18), ("route_arrow_blue", 20, 12), ("route_ch", 18, 18)],
        ),
        (
            634,
            "Chain",
            "M\u00a0\u2192\u00a0M\u00a0\u2192\u00a0CH",
            [("route_m", 18, 18), ("route_arrow_blue", 20, 12), ("route_m", 18, 18), ("route_arrow_blue", 20, 12), ("route_ch", 18, 18)],
        ),
        (
            760,
            "Two-hop",
            "M\u00a0\u2192\u00a0R\u00a0\u2192\u00a0CH",
            [("route_m", 18, 18), ("route_arrow_blue", 20, 12), ("route_r", 18, 18), ("route_arrow_blue", 20, 12), ("route_ch", 18, 18)],
        ),
    ]
    for x, title, label, atoms in mode_cards:
        b.rect(x, 350, 116, 152, fill=COLORS["white"], stroke=COLORS["border"], sw=1.2, r=8)
        b.text(TextSpec(title, x + 8, 362, 100, 32, size=24, bold=True))
        b.text(TextSpec(label, x + 2, 402, 112, 28, size=18, bold=True))
        atom_x = x + (116 - atom_row_width(atoms)) / 2
        place_atom_row(b, atom_x, 440, atoms)
    for idx, (icon, label) in enumerate([("role_m", "M"), ("role_r", "R"), ("role_ch", "CH")]):
        cx = 625 + idx * 64
        b.icon(icon, cx, 502, 25, 25)
        b.text(TextSpec(label, cx + 27, 501, 32, 25, size=17, color=COLORS["secondary"], bold=True, align="left"))
    b.line(720, 536, 720, 564, color=COLORS["line"], sw=2.3, arrow=True)
    b.rect(496, 574, 448, 188, fill=COLORS["bluegray"], stroke=COLORS["border"], sw=1.3, r=8)
    b.rect(546, 590, 46, 46, fill=COLORS["white"], stroke=COLORS["blue"], sw=1.6, r=23)
    b.text(TextSpec("2", 546, 590, 46, 46, size=25, bold=True))
    b.text(TextSpec("Intra-cluster forwarding", 604, 588, 322, 36, size=25, bold=True))
    b.icon("forwarding", 560, 638, 326, 108)
    b.line(720, 762, 720, 790, color=COLORS["line"], sw=2.3, arrow=True)
    b.rect(496, 802, 448, 164, fill=COLORS["green"], stroke=COLORS["border"], sw=1.3, r=8)
    b.rect(580, 818, 46, 46, fill=COLORS["white"], stroke=COLORS["teal"], sw=1.6, r=23)
    b.text(TextSpec("3", 580, 818, 46, 46, size=25, bold=True))
    b.text(TextSpec("CH aggregation", 642, 816, 220, 36, size=27, bold=True))
    for idx, (icon, label) in enumerate([("dedup", "deduplicate"), ("fuse", "fuse"), ("buffer", "buffer")]):
        ix = 548 + idx * 135
        b.icon(icon, ix, 864, 58, 58)
        b.text(TextSpec(label, ix - 30, 923, 118, 28, size=20))
        if idx:
            b.line(ix - 38, 858, ix - 38, 946, color=COLORS["border"], sw=1.2, dashed=True)

    # Phase 3 decision selector.
    b.text(TextSpec("pre-transmission route selection", 1012, 168, 390, 34, size=24, italic=True))
    spine_x = 1014
    b.line(spine_x, 264, spine_x, 902, color=COLORS["line"], sw=2.3)
    q_specs = [
        (
            "Q1",
            "Direct\nacceptable?",
            238,
            262,
            1190,
            220,
            222,
            150,
            "Direct uplink",
            "CH \u2192 BS",
            [("route_ch", 22, 22), ("route_arrow_blue", 28, 12), ("route_bs", 22, 32)],
            COLORS["white"],
            COLORS["border"],
            "",
            "",
        ),
        (
            "Q2",
            "Gateway\nvalid?",
            438,
            458,
            1172,
            392,
            244,
            206,
            "Gateway-assisted\nuplink",
            "CH \u2192 GW \u2192 BS",
            [("route_ch", 22, 22), ("route_arrow_blue", 24, 12), ("route_gw", 22, 22), ("route_arrow_blue", 24, 12), ("route_bs", 22, 32)],
            COLORS["bluegray"],
            COLORS["blue"],
            "main gain",
            COLORS["blue"],
        ),
        (
            "Q3",
            "Skeleton\npath valid?",
            656,
            688,
            1190,
            626,
            222,
            178,
            "Skeleton reserve",
            "CH \u2192 CH \u2192 BS",
            [("route_ch", 22, 22), ("route_arrow_blue_dashed", 24, 12), ("route_ch", 22, 22), ("route_arrow_blue_dashed", 24, 12), ("route_bs", 22, 32)],
            COLORS["muted"],
            COLORS["border"],
            "reserve",
            COLORS["line"],
        ),
    ]
    for q, label, qy, arrow_y, card_x, card_y, card_w, card_h, title, route_label, atoms, fill, stroke, badge, badge_color in q_specs:
        b.rect(spine_x - 25, qy - 25, 50, 50, fill=COLORS["white"], stroke=COLORS["blue"], sw=1.6, r=25)
        b.text(TextSpec(q, spine_x - 25, qy - 25, 50, 50, size=21, bold=True))
        b.text(TextSpec(label, spine_x + 30, qy - 28, 100, 60, size=19, align="left"))
        b.text(TextSpec("Yes", 1154, arrow_y - 35, 40, 26, size=18, color=COLORS["secondary"]))
        b.line(card_x - 48, arrow_y, card_x - 10, arrow_y, color=COLORS["line"], sw=2, arrow=True)
        ch = card_h
        sw = 3.0 if title.startswith("Gateway") else 1.3
        b.rect(card_x, card_y, card_w, ch, fill=fill, stroke=stroke, sw=sw, r=8)
        title_h = 46 if "\n" in title else 30
        b.text(TextSpec(title, card_x + 10, card_y + 8, card_w - 20, title_h + 4, size=21 if title.startswith("Gateway") else 24, bold=True))
        atom_w = atom_row_width(atoms)
        atom_x = card_x + (card_w - atom_w) / 2
        place_atom_row(b, atom_x, card_y + 58, atoms)
        label_y = card_y + (110 if badge == "reserve" else 118)
        b.text(TextSpec(route_label, card_x + 10, label_y, card_w - 20, 34, size=23, color=COLORS["text"], bold=True))
        if badge:
            if badge == "reserve":
                b.badge(card_x + (card_w - 100) / 2, card_y + ch - 32, 100, 28, badge, COLORS["line"], COLORS["line"], text_color=COLORS["white"])
            else:
                b.badge(card_x + (card_w - 118) / 2, card_y + ch - 42, 118, 30, badge, badge_color, badge_color, text_color=COLORS["white"])
    b.text(TextSpec("No", spine_x + 20, 341, 48, 26, size=18, color=COLORS["secondary"]))
    b.line(spine_x, 288, spine_x, 414, color=COLORS["line"], sw=2, arrow=True)
    b.text(TextSpec("No", spine_x + 20, 543, 48, 26, size=18, color=COLORS["secondary"]))
    b.line(spine_x, 488, spine_x, 632, color=COLORS["line"], sw=2, arrow=True)
    b.text(TextSpec("No", spine_x + 20, 761, 48, 26, size=18, color=COLORS["secondary"]))
    b.line(spine_x, 682, spine_x, 916, color=COLORS["line"], sw=2)
    b.line(spine_x, 916, 1190, 916, color=COLORS["line"], sw=2, arrow=True)
    b.rect(1190, 840, 222, 170, fill=COLORS["yellow"], stroke=COLORS["border"], sw=1.3, r=8)
    b.text(TextSpec("One-shot fallback", 1200, 852, 202, 34, size=24, bold=True))
    b.image("route_ch", 1267, 900, 22, 22)
    b.image("route_arrow_line_dashed", 1289, 904, 24, 12)
    b.image("route_bs", 1313, 894, 22, 32)
    b.text(TextSpec("CH \u2192 BS", 1200, 938, 202, 34, size=23, color=COLORS["text"], bold=True))
    b.badge(1232, 978, 138, 30, "single attempt", COLORS["yellow"], COLORS["border"])

    return b


def write_manifest(icon_names: list[str]) -> None:
    lines = [
        "# AERIS Round Pipeline Figure v3",
        "",
        "Generated by `scripts/build_aeris_round_pipeline_drawio.py`.",
        "The default output directory is the active LCN Overleaf package: `overleaf_upload_ready_20260501/generated/`.",
        "Common module icons are sourced from vendored `lucide-static` SVG assets under `scripts/assets/lucide_static_isc/` (ISC license).",
        "",
        "Artifacts:",
        f"- Draw.io source: `{DRAWIO_PATH.relative_to(ROOT).as_posix()}`",
        f"- Text-free graphics layer SVG: `{GRAPHICS_SVG_PATH.relative_to(ROOT).as_posix()}`",
        f"- Visual QA preview HTML: `{PREVIEW_HTML_PATH.relative_to(ROOT).as_posix()}`",
        f"- PDF export target: `{PDF_PATH.relative_to(ROOT).as_posix()}`",
        f"- Text-free SVG icon directory: `{ICON_DIR.relative_to(ROOT).as_posix()}`",
        "",
        "Rules applied:",
        "- Every non-text graphical element is produced in a text-free SVG graphics layer.",
        "- Every icon asset is also saved as a standalone text-free SVG.",
        "- Every visible label is a separate draw.io text cell.",
        "- The QA preview uses HTML text boxes over the text-free SVG graphics layer; no full-figure SVG with text is emitted.",
        "- Figure uses the requested 1440 x 1080 canvas.",
        "- Palette v4 uses a restrained academic scheme: Okabe-Ito blue/sky-blue/green/orange accents and Paul-Tol-style low-chroma tints for labeled cells.",
        "- Text uses Arial in the draw.io source and an Arial/Helvetica fallback stack in the PDF preview export.",
        "- Common process icons use a consistent Lucide stroke family; protocol-specific route mini-diagrams are decomposed into atomic SVG image cells so individual nodes and arrows remain editable in draw.io.",
        "- The previous bottom Strict-mode note bar has been removed.",
        "- Gateway-assisted uplink is emphasized; Skeleton and fallback remain secondary.",
        "- No Freepik asset is embedded in this revision; the icon set is generated locally to avoid attribution ambiguity in the submission PDF.",
        "",
        "Generated icon assets:",
        *[f"- `{name}.svg`" for name in sorted(icon_names)],
        "",
    ]
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    icons = make_icons()
    builder = build_figure(icons)
    builder.save_drawio()
    builder.save_preview()
    write_manifest(list(icons))
    print(f"[saved] {DRAWIO_PATH}")
    print(f"[saved] {PREVIEW_HTML_PATH}")
    print(f"[saved] {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
