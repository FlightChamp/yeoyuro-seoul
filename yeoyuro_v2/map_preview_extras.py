"""
yeoyuro_v2/map_preview_extras.py
================================
실시간 미리보기 보조 그림.
- stamp     : 미리보기 왼쪽 위에 번호·저장 시각을 새김 → 화면의 그림이 방금 저장한 결과인지 눈으로 확인
- zoom_changes : 바뀐 곳 주변만 크게 그림. 빈 원 = 이전 위치, 빨간 점 = 새 위치
"""
from __future__ import annotations

from pathlib import Path

LINE_COLORS = {"1": "#0052A4", "2": "#009D3E", "3": "#EF7C1C", "4": "#00A5DE", "5": "#996CAC",
               "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C", "9": "#BDB092"}


def stamp(png: Path, text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        f = ImageFont.load_default()
    d.rectangle([8, 8, 8 + 22 * len(text), 58], fill=(255, 243, 196))
    d.text((18, 14), text, fill=(120, 60, 0), font=f)
    im.save(png)


def _font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "AppleGothic"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def zoom_changes(frames: dict, old_final: dict, new_final: dict, out: Path, pad: int = 260) -> bool:
    """바뀐 점·역명·클릭이 있으면 주변 확대 그림을 out 에 저장하고 True."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    moved = []
    for k, b in new_final.items():
        a = old_final.get(k)
        if a is None or b is None:
            continue
        try:
            if abs(float(a[0]) - float(b[0])) > 0.4 or abs(float(a[1]) - float(b[1])) > 0.4:
                moved.append((k, (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
        except (TypeError, ValueError):
            continue
    if not moved:
        return False
    _font()
    vn = frames["Station_Visual_Nodes"].copy()
    xy = vn.set_index("visual_node_id")[["x_px", "y_px"]]
    seq = frames["Line_Sequences"]
    xs = [p[0] for _, a, b in moved for p in (a, b)]
    ys = [p[1] for _, a, b in moved for p in (a, b)]
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    w = x1 - x0
    fig, ax = plt.subplots(figsize=(9, 9 * (y1 - y0) / max(w, 1)), dpi=100)
    for (line, br), g in seq.groupby(["line_id", "branch_code"]):
        g = g.sort_values("path_order")
        pts = xy.reindex(g.visual_node_id).dropna()
        if str(line) == "2" and br == "main" and len(pts):
            pts = pd.concat([pts, pts.iloc[[0]]])
        ax.plot(pts.x_px, pts.y_px, color=LINE_COLORS.get(str(line), "#888"), lw=5, zorder=1, solid_capstyle="round")
    ax.scatter(vn.x_px, vn.y_px, s=40, c="white", edgecolors="#374151", zorder=2)
    lb = frames["Station_Labels"]
    for r in lb.itertuples():
        if x0 <= r.label_x_px <= x1 and y0 <= r.label_y_px <= y1:
            ax.text(r.label_x_px, r.label_y_px + 22, r.station_key, ha="center", va="top", fontsize=10, zorder=3)
    for k, a, b in moved:
        ax.scatter([a[0]], [a[1]], s=420, facecolors="none", edgecolors="#DC2626", linewidths=2, linestyle="--", zorder=4)
        ax.scatter([b[0]], [b[1]], s=140, c="#DC2626", zorder=5)
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5), zorder=5)
        ax.text(b[0] + 14, b[1] - 18, f"{k}\n({a[0]:.0f}, {a[1]:.0f}) → ({b[0]:.0f}, {b[1]:.0f})", color="#991B1B",
                fontsize=9, zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.grid(color="#E5E7EB", lw=0.6)
    ax.set_title("바뀐 곳 확대 — 빈 원 = 이전 위치, 빨간 점 = 새 위치", fontsize=12)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True
