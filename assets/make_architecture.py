"""Render assets/architecture.png.

Pillow, drawn at 2x and downsampled. Dark card with light text so it reads on
both the GitHub light and dark themes.

Run:  python assets/make_architecture.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 960 * S, 600 * S
OUT = Path(__file__).with_name("architecture.png")

BG, FG, MUTED, LINE = (13, 17, 23), (201, 209, 217), (139, 148, 158), (110, 118, 129)
ACCENT, GREEN, RED = (188, 140, 255), (63, 185, 80), (248, 113, 113)
FONTS = r"C:\Windows\Fonts"


def font(n, s):
    return ImageFont.truetype(f"{FONTS}\\{n}", s * S)


f_title, f_head = font("seguisb.ttf", 15), font("seguisb.ttf", 12)
f_small, f_lbl = font("segoeui.ttf", 10), font("segoeuii.ttf", 9)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def box(x, y, w, h, c=LINE, width=2):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=6 * S, outline=c, width=int(width * S))


def text(x, y, s, f=f_small, fill=MUTED, anchor="mm"):
    d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)


def _head(p0, p1, c, size=6):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** .5, 1e-6)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    s = size * S
    d.polygon([(x1, y1),
               (x1 - ux * s + px * s * .5, y1 - uy * s + py * s * .5),
               (x1 - ux * s - px * s * .5, y1 - uy * s - py * s * .5)], fill=c)


def _dashed(p0, p1, c, w, on=6, off=4):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** .5, 1e-6)
    ux, uy = dx / dist, dy / dist
    pos = 0.
    while pos < dist:
        seg = min(on * S, dist - pos)
        d.line([(x0 + ux * pos, y0 + uy * pos),
                (x0 + ux * (pos + seg), y0 + uy * (pos + seg))], fill=c, width=int(w * S))
        pos += (on + off) * S


def arrow(pts, c=LINE, w=1.5, dash=False):
    pts = [(x * S, y * S) for x, y in pts]
    for i in range(len(pts) - 1):
        (_dashed if dash else lambda a, b, cc, ww: d.line([a, b], fill=cc, width=int(ww * S)))(
            pts[i], pts[i + 1], c, w)
    _head(pts[-2], pts[-1], c)


text(480, 26, "Stock-Price-Forecaster — walk-forward evaluation against the baselines that matter",
     f_title, FG)

# ── data ────────────────────────────────────────────────────────────────────
box(30, 54, 250, 62)
text(155, 76, "Daily bars — 10 tickers", f_head, FG)
text(155, 95, "2021–2025 · ~5,000 out-of-sample days")

# ── split ───────────────────────────────────────────────────────────────────
arrow([(280, 85), (316, 85)])
box(318, 54, 290, 62, ACCENT)
text(463, 76, "Walk-forward split — 5 expanding folds", f_head, FG)
text(463, 95, "scaler fit on the train slice only · no peeking")
text(622, 78, "a regression test fails", f_lbl, ACCENT, anchor="lm")
text(622, 94, "if the leak returns", f_lbl, ACCENT, anchor="lm")

# ── target ──────────────────────────────────────────────────────────────────
arrow([(463, 116), (463, 146)])
box(318, 148, 290, 56)
text(463, 168, "Target: next-day returns", f_head, FG)
text(463, 187, "not price — price flatters every model")

# ── models ──────────────────────────────────────────────────────────────────
arrow([(463, 204), (463, 232)])
box(30, 234, 900, 92)
text(52, 254, "Forecasters, all scored on the same folds", f_head, FG, anchor="lm")
names = ["persistence", "momentum", "ARIMA", "XGBoost", "LSTM", "PatchTST"]
subs = ["r̂ = 0 (floor)", "r̂ = r(t−1)", "champion", "18 causal feats", "on returns", "transformer"]
x0, bw, gap = 52, 136, 12
for i, (n, s) in enumerate(zip(names, subs)):
    x = x0 + i * (bw + gap)
    box(x, 270, bw, 46)
    text(x + bw / 2, 286, n, f_head, FG)
    text(x + bw / 2, 305, s)

# ── backtest ────────────────────────────────────────────────────────────────
arrow([(480, 326), (480, 354)])
box(180, 356, 600, 62, ACCENT)
text(480, 378, "Cost-aware long/flat backtest — 5 bps per side", f_head, FG)
text(480, 397, "always reported beside buy-and-hold · zero-cost runs refused at the schema level")

# ── verdict ─────────────────────────────────────────────────────────────────
arrow([(480, 418), (480, 446)])
box(180, 448, 600, 54, RED)
text(480, 468, "Result: no forecaster beats buy-and-hold", f_head, FG)
text(480, 487, "best net Sharpe 1.34 (ARIMA) vs 1.83 buy-and-hold · reported, not buried")

# ── serving ─────────────────────────────────────────────────────────────────
box(30, 448, 130, 54)
text(95, 468, "Flask :5000", f_head, FG)
text(95, 487, "UI")
box(800, 448, 130, 54)
text(865, 468, "FastAPI :8000", f_head, FG)
text(865, 487, "/predict /backtest")

text(30, 534, "Conformal prediction intervals replace a hand-written confidence number · MLflow logs every run",
     f_lbl, MUTED, anchor="lm")
text(30, 554, "Docker · Redis · Streamlit ops dashboard · 90 offline tests",
     f_lbl, MUTED, anchor="lm")

img.resize((W // S, H // S), Image.LANCZOS).save(OUT, "PNG", optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
