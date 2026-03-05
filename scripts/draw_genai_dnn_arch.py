"""
GenAI & DNN Models across Communication Network Layers
Publication-quality architecture diagram with clear model icons.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np
import os

# ============================================================
# Canvas
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(26, 15), dpi=200)
ax.set_xlim(0, 26)
ax.set_ylim(0, 15)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#fafafa')

# ============================================================
# Color palettes
# ============================================================
BLU_DARK = '#0d1b4a'
BLU_MID  = '#1a237e'
BLU_CARD = '#283593'
BLU_ICON = '#90caf9'
BLU_ARROW = '#1565c0'

GRN_DARK = '#0d3311'
GRN_MID  = '#1b5e20'
GRN_CARD = '#2e7d32'
GRN_ICON = '#a5d6a7'
GRN_ARROW = '#2e7d32'

PUR = {'bg': '#ede7f6', 'bd': '#7e57c2', 'tx': '#4527a0',
       'card': '#d1c4e9', 'card2': '#b39ddb', 'hdr': '#7c4dff'}
ORA = {'bg': '#fff3e0', 'bd': '#ef6c00', 'tx': '#e65100',
       'card': '#ffe0b2', 'card2': '#ffcc80', 'hdr': '#ff6d00'}
CYA = {'bg': '#e0f7fa', 'bd': '#00838f', 'tx': '#006064',
       'card': '#b2ebf2', 'card2': '#80deea', 'hdr': '#00b8d4'}

WHITE = '#ffffff'
DARK_TEXT = '#212121'
GREY_TEXT = '#424242'


# ============================================================
# Helper functions
# ============================================================
def rounded_rect(x, y, w, h, color, ec='none', lw=0, alpha=1.0,
                 radius=0.12, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=ec, linewidth=lw,
                         alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box


def text_center(x, y, s, **kwargs):
    defaults = dict(ha='center', va='center', fontsize=9,
                    color=DARK_TEXT, zorder=10)
    defaults.update(kwargs)
    return ax.text(x, y, s, **defaults)


def draw_arrow_line(x1, y1, x2, y2, color, lw=2.0, zorder=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_width=0.15,head_length=0.1',
                                color=color, lw=lw),
                zorder=zorder)


# ============================================================
# Icon drawing functions — GenAI models
# ============================================================
def draw_icon_llm(cx, cy, s, ic):
    """LLM: multi-layer fully connected network (3-5-5-2)."""
    layers = [3, 5, 5, 2]
    xs = np.linspace(cx - s * 0.45, cx + s * 0.45, len(layers))
    for i, (lx, n) in enumerate(zip(xs, layers)):
        ys = np.linspace(cy - s * 0.42, cy + s * 0.42, n)
        for y in ys:
            ax.plot(lx, y, 'o', color=ic, ms=5.5, zorder=12)
            if i < len(layers) - 1:
                nys = np.linspace(cy - s * 0.42, cy + s * 0.42, layers[i+1])
                for ny in nys:
                    ax.plot([lx, xs[i+1]], [y, ny], '-', color=ic,
                            lw=0.6, alpha=0.45, zorder=11)


def draw_icon_diffusion(cx, cy, s, ic):
    """Diffusion: noisy dots -> arrow -> clean circle."""
    rng = np.random.RandomState(42)
    for _ in range(15):
        dx = rng.uniform(-0.4, -0.05) * s
        dy = rng.uniform(-0.35, 0.35) * s
        ax.plot(cx + dx, cy + dy, '.', color=ic, ms=4.0, alpha=0.6, zorder=12)
    ax.annotate('', xy=(cx + s*0.3, cy), xytext=(cx + s*0.02, cy),
                arrowprops=dict(arrowstyle='->', color=ic, lw=2.0), zorder=12)
    circ = Circle((cx + s*0.42, cy), s*0.15, fc=ic, ec='white',
                  lw=1.2, zorder=12)
    ax.add_patch(circ)


def draw_icon_gan(cx, cy, s, ic):
    """GAN: G <-> D adversarial."""
    off = s * 0.28
    rounded_rect(cx-off-s*0.2, cy-s*0.2, s*0.36, s*0.4,
                 ic, ec='white', lw=1.0, alpha=0.35, radius=0.05, zorder=11)
    text_center(cx-off, cy, 'G', fontsize=13, fontweight='bold', color=ic, zorder=12)
    rounded_rect(cx+off-s*0.16, cy-s*0.2, s*0.36, s*0.4,
                 ic, ec='white', lw=1.0, alpha=0.35, radius=0.05, zorder=11)
    text_center(cx+off+0.02, cy, 'D', fontsize=13, fontweight='bold', color=ic, zorder=12)
    ax.annotate('', xy=(cx+off-s*0.18, cy+s*0.08),
                xytext=(cx-off+s*0.18, cy+s*0.08),
                arrowprops=dict(arrowstyle='->', color=ic, lw=1.5), zorder=12)
    ax.annotate('', xy=(cx-off+s*0.18, cy-s*0.08),
                xytext=(cx+off-s*0.18, cy-s*0.08),
                arrowprops=dict(arrowstyle='->', color=ic, lw=1.5), zorder=12)


def draw_icon_vae(cx, cy, s, ic):
    """VAE: encoder triangle -> z -> decoder triangle."""
    ex = cx - s * 0.38
    tri_h, tri_w = s * 0.4, s * 0.26
    ax.fill([ex, ex, ex+tri_w], [cy-tri_h, cy+tri_h, cy],
            color=ic, alpha=0.5, zorder=12)
    ax.plot([ex, ex, ex+tri_w, ex], [cy-tri_h, cy+tri_h, cy, cy-tri_h],
            color=ic, lw=1.2, zorder=12)
    circ = Circle((cx, cy), s*0.13, fc=ic, ec='white', lw=1.2, alpha=0.7, zorder=12)
    ax.add_patch(circ)
    text_center(cx, cy, 'z', fontsize=10, fontweight='bold', color='white', zorder=13)
    dx = cx + s * 0.38
    ax.fill([dx, dx, dx-tri_w], [cy-tri_h, cy+tri_h, cy],
            color=ic, alpha=0.5, zorder=12)
    ax.plot([dx, dx, dx-tri_w, dx], [cy-tri_h, cy+tri_h, cy, cy-tri_h],
            color=ic, lw=1.2, zorder=12)
    ax.plot([ex+tri_w, cx-s*0.13], [cy, cy], '-', color=ic, lw=1.2, zorder=11)
    ax.plot([cx+s*0.13, dx-tri_w], [cy, cy], '-', color=ic, lw=1.2, zorder=11)


def draw_icon_transformer(cx, cy, s, ic):
    """Transformer: stacked attention layers + curved arrow."""
    for i, dy in enumerate([-0.28, 0.0, 0.28]):
        y0 = cy + dy * s
        w, h = s * 0.65, s * 0.2
        rounded_rect(cx-w/2, y0-h/2, w, h, ic, ec='white', lw=1.0,
                     alpha=0.3+0.15*i, radius=0.04, zorder=11)
    ax.annotate('', xy=(cx+s*0.4, cy+s*0.3),
                xytext=(cx+s*0.4, cy-s*0.3),
                arrowprops=dict(arrowstyle='->', color=ic, lw=2.0,
                                connectionstyle='arc3,rad=-0.4'),
                zorder=12)


# ============================================================
# Icon drawing functions — DNN models
# ============================================================
def draw_icon_cnn(cx, cy, s, ic):
    """CNN: 3 stacked offset rectangles (conv filters)."""
    for i in range(3):
        dx = -s*0.18 + i*s*0.18
        dy = s*0.15 - i*s*0.15
        w, h = s*0.48, s*0.58
        rounded_rect(cx+dx-w/2, cy+dy-h/2, w, h, ic, ec='white', lw=1.2,
                     alpha=0.28+0.18*i, radius=0.04, zorder=11+i)


def draw_icon_rnn(cx, cy, s, ic):
    """RNN: 3 nodes with recurrent loop arrow."""
    xs = np.linspace(cx-s*0.38, cx+s*0.38, 3)
    for x in xs:
        c = Circle((x, cy), s*0.13, fc=ic, ec='white', lw=1.2, alpha=0.7, zorder=12)
        ax.add_patch(c)
    for i in range(2):
        ax.plot([xs[i]+s*0.13, xs[i+1]-s*0.13], [cy, cy],
                '-', color=ic, lw=1.5, zorder=11)
    ax.annotate('', xy=(xs[0]+s*0.06, cy+s*0.2),
                xytext=(xs[2]-s*0.06, cy+s*0.2),
                arrowprops=dict(arrowstyle='->', color=ic, lw=1.8,
                                connectionstyle='arc3,rad=-0.5'),
                zorder=12)


def draw_icon_gnn(cx, cy, s, ic):
    """GNN: 4 nodes + graph edges."""
    pts = [(cx-s*0.28, cy+s*0.22), (cx+s*0.28, cy+s*0.22),
           (cx-s*0.28, cy-s*0.22), (cx+s*0.28, cy-s*0.22)]
    for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                '-', color=ic, lw=1.2, alpha=0.5, zorder=11)
    for px, py in pts:
        c = Circle((px, py), s*0.12, fc=ic, ec='white', lw=1.2, alpha=0.8, zorder=12)
        ax.add_patch(c)


def draw_icon_rl(cx, cy, s, ic):
    """RL: Agent <-> Environment bidirectional arrows."""
    off = s * 0.28
    rounded_rect(cx-off-s*0.2, cy-s*0.2, s*0.36, s*0.4,
                 ic, ec='white', lw=1.0, alpha=0.35, radius=0.05, zorder=11)
    text_center(cx-off, cy, 'A', fontsize=13, fontweight='bold', color=ic, zorder=12)
    rounded_rect(cx+off-s*0.16, cy-s*0.2, s*0.36, s*0.4,
                 ic, ec='white', lw=1.0, alpha=0.35, radius=0.05, zorder=11)
    text_center(cx+off+0.02, cy, 'E', fontsize=13, fontweight='bold', color=ic, zorder=12)
    ax.annotate('', xy=(cx+off-s*0.18, cy+s*0.08),
                xytext=(cx-off+s*0.18, cy+s*0.08),
                arrowprops=dict(arrowstyle='->', color=ic, lw=1.5), zorder=12)
    ax.annotate('', xy=(cx-off+s*0.18, cy-s*0.08),
                xytext=(cx+off-s*0.18, cy-s*0.08),
                arrowprops=dict(arrowstyle='->', color=ic, lw=1.5), zorder=12)


def draw_icon_fl(cx, cy, s, ic):
    """FL: central server + 3 client nodes + lines."""
    c0 = Circle((cx, cy), s*0.16, fc=ic, ec='white', lw=1.5, alpha=0.9, zorder=12)
    ax.add_patch(c0)
    text_center(cx, cy, 'S', fontsize=9, fontweight='bold', color='white', zorder=13)
    for ang in [90, 210, 330]:
        rad = np.radians(ang)
        px, py = cx + s*0.38*np.cos(rad), cy + s*0.38*np.sin(rad)
        c2 = Circle((px, py), s*0.1, fc=ic, ec='white', lw=1.0, alpha=0.6, zorder=12)
        ax.add_patch(c2)
        ax.plot([cx, px], [cy, py], '-', color=ic, lw=1.2, alpha=0.6, zorder=11)


ICON_FUNCS = {
    'LLM': draw_icon_llm, 'Diffusion': draw_icon_diffusion,
    'GAN': draw_icon_gan, 'VAE': draw_icon_vae,
    'Transformer': draw_icon_transformer,
    'CNN': draw_icon_cnn, 'RNN': draw_icon_rnn,
    'GNN': draw_icon_gnn, 'RL': draw_icon_rl, 'FL': draw_icon_fl,
}

# ============================================================
# Layout constants
# ============================================================
TITLE_Y = 14.5
LEFT_X = 0.3          # left panel x
RIGHT_X = 22.5        # right panel x
PANEL_W = 3.2         # side panel width
LAYER_H = 3.8         # each swim-lane height
LAYER_GAP = 0.25      # gap between layers
MID_X = LEFT_X + PANEL_W + 0.6   # middle area left edge
MID_W = RIGHT_X - MID_X - 0.6    # middle area width

# Vertical positions of the three layers (bottom of each)
# Available: y=1.0 to y=13.8 (below title)
BOT_MARGIN = 1.0
LAYER_BOT = [BOT_MARGIN + i * (LAYER_H + LAYER_GAP) for i in range(3)]
# LAYER_BOT[0]=Physical, [1]=Network, [2]=Application
LAYER_MID = [b + LAYER_H / 2 for b in LAYER_BOT]

# Side panel vertical span
SIDE_BOT = LAYER_BOT[0]
SIDE_TOP = LAYER_BOT[2] + LAYER_H
SIDE_H = SIDE_TOP - SIDE_BOT

# ============================================================
# Title
# ============================================================
text_center(13, TITLE_Y,
            'The Role of GenAI and DNN Models across Communication Network Layers',
            fontsize=16, fontweight='bold', color=DARK_TEXT)

# ============================================================
# Left panel — GenAI Models
# ============================================================
rounded_rect(LEFT_X, SIDE_BOT, PANEL_W, SIDE_H, BLU_DARK,
             ec=BLU_MID, lw=2, radius=0.15, zorder=1)
text_center(LEFT_X + PANEL_W/2, SIDE_TOP - 0.35, 'GenAI Models',
            fontsize=13, fontweight='bold', color=WHITE)

genai_models = ['LLM', 'Diffusion', 'GAN', 'VAE', 'Transformer']
card_h_side = (SIDE_H - 1.0) / 5 - 0.12
for i, name in enumerate(genai_models):
    cy = SIDE_TOP - 0.75 - i * (card_h_side + 0.12) - card_h_side / 2
    cx = LEFT_X + PANEL_W / 2
    cw, ch = PANEL_W - 0.4, card_h_side
    rounded_rect(cx - cw/2, cy - ch/2, cw, ch, BLU_CARD,
                 ec=BLU_ICON, lw=0.8, alpha=0.85, radius=0.08, zorder=3)
    text_center(cx, cy - ch/2 + 0.22, name,
                fontsize=9.5, fontweight='bold', color=WHITE, zorder=10)
    icon_cy = cy + 0.1
    ICON_FUNCS[name](cx, icon_cy, ch * 0.7, BLU_ICON)

# ============================================================
# Right panel — DNN Models
# ============================================================
rounded_rect(RIGHT_X, SIDE_BOT, PANEL_W, SIDE_H, GRN_DARK,
             ec=GRN_MID, lw=2, radius=0.15, zorder=1)
text_center(RIGHT_X + PANEL_W/2, SIDE_TOP - 0.35, 'DNN Models',
            fontsize=13, fontweight='bold', color=WHITE)

dnn_models = ['CNN', 'RNN', 'GNN', 'RL', 'FL']
for i, name in enumerate(dnn_models):
    cy = SIDE_TOP - 0.75 - i * (card_h_side + 0.12) - card_h_side / 2
    cx = RIGHT_X + PANEL_W / 2
    cw, ch = PANEL_W - 0.4, card_h_side
    rounded_rect(cx - cw/2, cy - ch/2, cw, ch, GRN_CARD,
                 ec=GRN_ICON, lw=0.8, alpha=0.85, radius=0.08, zorder=3)
    text_center(cx, cy - ch/2 + 0.22, name,
                fontsize=9.5, fontweight='bold', color=WHITE, zorder=10)
    icon_cy = cy + 0.1
    ICON_FUNCS[name](cx, icon_cy, ch * 0.7, GRN_ICON)

# ============================================================
# Middle swim-lanes — three layers
# ============================================================
layer_colors = [CYA, ORA, PUR]  # Physical, Network, Application
layer_names = ['Physical Layer', 'Network Layer', 'Application Layer']
layer_cards = [
    [('Channel Modeling\n& Estimation',
      ['\u2022 Path loss prediction',
       '\u2022 Fading characterization',
       '\u2022 MIMO channel estimation',
       '\u2022 Beamforming optimization']),
     ('RF Sensing &\nSignal Processing',
      ['\u2022 Spectrum sensing',
       '\u2022 Modulation recognition',
       '\u2022 Interference cancellation',
       '\u2022 Signal reconstruction'])],
    [('Traffic Management\n& Optimization',
      ['\u2022 Routing optimization',
       '\u2022 Load balancing',
       '\u2022 QoS management',
       '\u2022 Resource allocation']),
     ('Anomaly Detection\n& Security',
      ['\u2022 Intrusion detection',
       '\u2022 Traffic classification',
       '\u2022 Threat identification',
       '\u2022 Privacy preservation'])],
    [('Object Detection\n& Recognition',
      ['\u2022 Image classification',
       '\u2022 Semantic segmentation',
       '\u2022 Video analytics',
       '\u2022 Pattern recognition']),
     ('Digital Twins\n& Simulation',
      ['\u2022 Network modeling',
       '\u2022 Scenario generation',
       '\u2022 Predictive maintenance',
       '\u2022 Performance forecasting'])],
]

for li, (lc, lname, cards) in enumerate(zip(layer_colors, layer_names, layer_cards)):
    bot = LAYER_BOT[li]
    # Layer background
    rounded_rect(MID_X, bot, MID_W, LAYER_H, lc['bg'],
                 ec=lc['bd'], lw=2, radius=0.15, zorder=1)
    # Layer header bar
    rounded_rect(MID_X + 0.15, bot + LAYER_H - 0.55, MID_W - 0.3, 0.45,
                 lc['bd'], alpha=0.85, radius=0.08, zorder=3)
    text_center(MID_X + MID_W/2, bot + LAYER_H - 0.33, lname,
                fontsize=11, fontweight='bold', color=WHITE, zorder=10)

    # Two cards per layer
    card_w = (MID_W - 0.8) / 2
    card_h = LAYER_H - 0.85
    for ci, (ctitle, bullets) in enumerate(cards):
        cx = MID_X + 0.25 + ci * (card_w + 0.3)
        cy = bot + 0.15
        rounded_rect(cx, cy, card_w, card_h, lc['card'],
                     ec=lc['bd'], lw=1.0, alpha=0.7, radius=0.1, zorder=3)
        # Card title
        text_center(cx + card_w/2, cy + card_h - 0.32, ctitle,
                    fontsize=8.5, fontweight='bold', color=lc['tx'],
                    zorder=10, linespacing=1.1)
        # Bullet points
        for bi, bt in enumerate(bullets):
            ax.text(cx + 0.2, cy + card_h - 0.72 - bi * 0.42, bt,
                    fontsize=6.8, color=GREY_TEXT, va='top', ha='left',
                    zorder=10)

# ============================================================
# Inter-layer arrows (Empower / Underpin)
# ============================================================
for li in range(2):
    y_from = LAYER_BOT[li] + LAYER_H
    y_to = LAYER_BOT[li + 1]
    mid_x_arrow = MID_X + MID_W / 2
    # Upward arrow
    ax.annotate('', xy=(mid_x_arrow, y_to),
                xytext=(mid_x_arrow, y_from),
                arrowprops=dict(arrowstyle='->,head_width=0.2,head_length=0.1',
                                color='#616161', lw=2.5),
                zorder=5)
    label = 'Underpin' if li == 0 else 'Empower'
    text_center(mid_x_arrow + 0.8, (y_from + y_to) / 2, label,
                fontsize=8, fontweight='bold', color='#616161',
                fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.15', fc='#ffffffcc', ec='none'))

# ============================================================
# Side arrows: GenAI -> layers (blue), layers -> DNN (green)
# ============================================================
left_edge = LEFT_X + PANEL_W
right_edge = RIGHT_X
for li in range(3):
    ymid = LAYER_MID[li]
    # Blue arrow from left panel to layer
    ax.annotate('', xy=(MID_X, ymid),
                xytext=(left_edge + 0.05, ymid),
                arrowprops=dict(arrowstyle='->,head_width=0.18,head_length=0.1',
                                color=BLU_ARROW, lw=2.2),
                zorder=5)
    # Green arrow from layer to right panel
    ax.annotate('', xy=(right_edge - 0.05, ymid),
                xytext=(MID_X + MID_W, ymid),
                arrowprops=dict(arrowstyle='->,head_width=0.18,head_length=0.1',
                                color=GRN_ARROW, lw=2.2),
                zorder=5)

# ============================================================
# Bottom legend
# ============================================================
legend_y = 0.45
legend_items = [
    (BLU_DARK, 'GenAI Models'),
    (GRN_DARK, 'DNN Models'),
    (PUR['bd'], 'Application Layer'),
    (ORA['bd'], 'Network Layer'),
    (CYA['bd'], 'Physical Layer'),
]
total_w = len(legend_items) * 3.8
start_x = 13 - total_w / 2
for i, (c, lab) in enumerate(legend_items):
    lx = start_x + i * 3.8
    rounded_rect(lx, legend_y - 0.12, 0.5, 0.24, c,
                 radius=0.04, zorder=10)
    ax.text(lx + 0.65, legend_y, lab, fontsize=7.5,
            va='center', ha='left', color=DARK_TEXT, zorder=10)

# ============================================================
# Save outputs
# ============================================================
out_dir = r'c:/AERIS-WSN-Protocol/for_submission/figures'
os.makedirs(out_dir, exist_ok=True)
base = os.path.join(out_dir, 'genai_dnn_architecture')

plt.tight_layout(pad=0.3)
fig.savefig(base + '.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
fig.savefig(base + '.svg', bbox_inches='tight',
            facecolor=fig.get_facecolor())
fig.savefig(base + '.pdf', bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Saved: {base}.png")
print(f"Saved: {base}.svg")
print(f"Saved: {base}.pdf")
print("Done.")
