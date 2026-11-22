"""
yeoyuro_v2/map_canvas.py
========================
좌표 편집용 미리보기: 5120×2880 캔버스를 **1:1** 로 그린다 (그림 1픽셀 = 노선도 좌표 1). (D-090)

- 100px 격자, 500px 마다 숫자 → 그림에서 바로 좌표를 읽을 수 있음
- 1~9호선 + 예정 노선(phase 3-A·3-B). 예정 노선은 같은 색 + 가운데 흰 점선으로 구분
- 예정 노선의 좌표 빈칸 점은 그리지 않고 선을 끊는다 (오른쪽 아래에 개수)
- 앱 노선도와는 별개 (앱은 예정 행을 빼고 그림)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from yeoyuro_v2.map_planned import LINE_COLORS, LINE_INFO, PHASE_COL, PLANNED, is_planned

W, H = 5120, 2880


def drawable(m: dict) -> dict:
    """좌표가 있는 점만 남긴 사본 (빈칸 점을 지나는 선은 끊긴다)."""
    out = dict(m)
    vn = m["Station_Visual_Nodes"]
    out["Station_Visual_Nodes"] = vn.dropna(subset=["x_px", "y_px"]).reset_index(drop=True)
    lb = m["Station_Labels"]
    out["Station_Labels"] = lb.dropna(subset=["label_x_px", "label_y_px"]).reset_index(drop=True)
    ca = m["Station_Click_Areas"]
    out["Station_Click_Areas"] = ca.dropna(subset=["click_x_px", "click_y_px"]).reset_index(drop=True)
    return out


def planned_status(m: dict) -> dict:
    """예정 노선 입력 현황 (include_planned=True 로 계산한 표에서)."""
    vn = m["Station_Visual_Nodes"]
    if PHASE_COL not in vn.columns:
        return {"n": 0, "filled": 0, "blank": [], "mismatch": []}
    p = vn[vn[PHASE_COL].map(is_planned)]
    blank = p[p.x_px.isna() | p.y_px.isna()].visual_node_id.tolist()
    # 같은 역인데 좌표가 다른 점 (예정 점 ↔ 같은 역의 다른 점) — 의도한 것이 아니면 맞추기
    mismatch = []
    have = vn.dropna(subset=["x_px", "y_px"])
    for k, g in have[have.station_key.isin(set(p.station_key))].groupby("station_key"):
        pts = set(zip(g.x_px.round(), g.y_px.round()))
        if len(pts) > 1:
            mismatch.append(f"{k} " + " / ".join(f"{r.visual_node_id}({r.x_px:.0f},{r.y_px:.0f})" for r in g.itertuples()))
    return {"n": len(p), "filled": len(p) - len(blank), "blank": blank, "mismatch": mismatch}


def _font(plt):
    from matplotlib import font_manager
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "AppleGothic"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _paths(seq: pd.DataFrame, xy: pd.DataFrame):
    """(line_id, branch, planned, [연속 좌표 묶음…]) — 좌표 빈칸에서 끊는다. 2호선 순환·6호선 응암순환 규칙은 가져오기(14)와 같다."""
    has_phase = PHASE_COL in seq.columns
    for (line, br), g in seq.groupby(["line_id", "branch_code"], sort=False):
        planned_rows = g[g[PHASE_COL].map(is_planned)] if has_phase else g.iloc[0:0]
        for planned, gg in ((False, g.drop(planned_rows.index)), (True, planned_rows)):
            if gg.empty:
                continue
            ids = gg.sort_values("path_order").visual_node_id.tolist()
            if str(line) == "6" and "구산_L6" in ids and "새절_L6" in ids:
                i = ids.index("구산_L6")
                if i + 1 < len(ids) and ids[i + 1] == "새절_L6":
                    ids = ids[:i + 1] + ["응암_L6"] + ids[i + 1:]
            if str(line) == "2" and br == "main" and not planned:
                ids = ids + ids[:1]
            runs, cur = [], []
            for nid in ids:
                if nid in xy.index:
                    cur.append((float(xy.at[nid, "x_px"]), float(xy.at[nid, "y_px"])))
                else:
                    if len(cur) > 1:
                        runs.append(cur)
                    cur = []
            if len(cur) > 1:
                runs.append(cur)
            yield str(line), br, planned, runs


def render_canvas(m: dict, out: Path) -> Path:
    """m = materialize(..., include_planned=True) 결과. out 에 5120×2880 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font(plt)
    d = drawable(m)
    vn, lb, seq = d["Station_Visual_Nodes"], d["Station_Labels"], d["Line_Sequences"]
    tl = d.get("Transfer_Links", pd.DataFrame())
    xy = vn.drop_duplicates("visual_node_id").set_index("visual_node_id")[["x_px", "y_px"]]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), W, H, color="white", zorder=0))
    for g in range(0, W + 1, 100):
        ax.axvline(g, color="#C7CBD1" if g % 500 == 0 else "#EEF0F3", lw=1.0 if g % 500 == 0 else 0.6, zorder=0.1)
    for g in range(0, H + 1, 100):
        ax.axhline(g, color="#C7CBD1" if g % 500 == 0 else "#EEF0F3", lw=1.0 if g % 500 == 0 else 0.6, zorder=0.1)
    for gx in range(0, W + 1, 500):
        for gy in range(0, H + 1, 500):
            ax.text(gx + 6, gy + 18, f"{gx},{gy}", fontsize=10, color="#9CA3AF", zorder=0.2)
    for line, br, planned, runs in _paths(seq, xy):
        c = LINE_COLORS.get(line, "#888888")
        for run in runs:
            xs, ys = [p[0] for p in run], [p[1] for p in run]
            ax.plot(xs, ys, color=c, lw=9 if planned else 10, solid_capstyle="round", solid_joinstyle="round",
                    zorder=2 if planned else 1)
            if planned:
                ax.plot(xs, ys, color="white", lw=2.2, ls=(0, (4, 4)), zorder=2.1)
    if len(tl):
        for r in tl.itertuples():
            a, b = r.from_visual_node_id, r.to_visual_node_id
            if bool(r.connector_visible) and a in xy.index and b in xy.index:
                ax.plot([xy.at[a, "x_px"], xy.at[b, "x_px"]], [xy.at[a, "y_px"], xy.at[b, "y_px"]],
                        color="#9CA3AF", lw=4, zorder=2.5)
    is_p = vn[PHASE_COL].map(is_planned) if PHASE_COL in vn.columns else pd.Series(False, index=vn.index)
    tr = vn["is_transfer_station"].astype(str).str.lower().isin(["true", "1"])
    # 예정 점은 기존 점 아래에 (기존 역에 붙는 예정 점은 기존 표시가 보이고, 새 역만 초록 원으로 보임)
    for mask, s, ec, lw, z in ((is_p, 150, "#15803D", 3.2, 2.8), (~tr & ~is_p, 70, "#4B5563", 2.4, 3),
                               (tr & ~is_p, 230, "#111827", 3.6, 4)):
        g = vn[mask]
        ax.scatter(g.x_px, g.y_px, s=s, c="white", edgecolors=ec, linewidths=lw, zorder=z)
    planned_only = set(vn.station_key) - set(vn[~is_p].station_key)
    for r in lb.itertuples():
        new = r.station_key in planned_only
        ax.text(r.label_x_px, r.label_y_px + 30, r.station_key, ha="center", va="top", fontsize=17,
                fontweight="bold" if new or str(getattr(r, "label_weight", "")) == "bold" else "normal",
                color="#166534" if new else "#111827", zorder=5)
    # 범례
    st = planned_status(m)
    lines = [k for k in LINE_INFO if k in set(seq.line_id.astype(str))]
    x0, y0 = W - 700, H - 70 - 48 * (len(lines) + 2)
    ax.add_patch(plt.Rectangle((x0 - 30, y0 - 60), 720, 48 * (len(lines) + 2) + 110, fc="white", ec="#D1D5DB",
                               lw=1.5, zorder=6))
    ax.text(x0, y0 - 20, f"예정 노선 점 {st['filled']}/{st['n']} 입력 · 흰 점선 = 예정 · 초록 원 = 예정 점", fontsize=15,
            zorder=7)
    for i, k in enumerate(lines):
        y = y0 + 30 + i * 48
        ax.plot([x0, x0 + 90], [y, y], color=LINE_COLORS[k], lw=10, zorder=7)
        if k in {p[0] for p in PLANNED} and k not in [str(n) for n in range(1, 10)]:
            ax.plot([x0, x0 + 90], [y, y], color="white", lw=2.2, ls=(0, (4, 4)), zorder=7.1)
        ax.text(x0 + 110, y, LINE_INFO[k][0] + (" (예정)" if k in ("GJ", "SB", "AR") else
                                               " (+코레일 예정)" if k == "1" and is_p.any() else ""),
                va="center", fontsize=15, zorder=7)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_tmp.png")
    fig.savefig(tmp, dpi=100)
    plt.close(fig)
    tmp.replace(out)
    return out
