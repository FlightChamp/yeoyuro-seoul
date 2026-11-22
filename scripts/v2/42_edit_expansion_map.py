"""
scripts/v2/42_edit_expansion_map.py
===================================
Phase 3 확장 노선(코레일 1호선·경의중앙선·수인분당선·공항철도) 역 좌표를 **직접 찍는** 도구.
범위 밖(D-089): 경의중앙선 서울역–신촌–가좌 지선, 수인분당선 왕십리–청량리.

기본 노선도 파일(yeoyuro_seoul_vector_map_coordinate_workbook.xlsx)은 **건드리지 않는다.**
확장 역 좌표는 별도 입력 파일에 적고, 다 찍은 뒤 3A-2 에서 기본 파일로 합친다 (9호선 때와 같은 방식).

하는 일
-------
1. 입력 파일이 없으면 새로 만든다 (있으면 절대 덮어쓰지 않음)
       data/master/draft/yeoyuro_map_expansion_input.xlsx
   - '역좌표_입력' 시트: 역마다 한 줄. 노란 칸(x_px, y_px)만 채우면 된다
     · 이미 노선도에 있는 역(서울역·노량진·공덕 등)은 기존 좌표가 회색으로 미리 채워져 있음 (같은 역 = 같은 좌표)
     · 3A-1 배치 초안이 있으면 '참고_x/참고_y' 에 적어 둠 (그대로 쓸 필요 없음)
   - '노선순서' 시트: 노선·지선별 역 순서 (선을 그리는 순서, 참고용)
2. 입력 파일을 엑셀로 열고, 브라우저 미리보기를 띄운다 (http://localhost:8790~)
   - 미리보기 = 5120×2880 캔버스 그대로 (1픽셀 = 노선도 좌표 1). 100px 격자, 500px 마다 숫자
   - 그림 위에 마우스를 올리면 좌표 표시, **클릭하면 'x<탭>y' 가 복사됨 → 엑셀 x_px 칸에 붙여넣기(Ctrl+V)하면 x·y 두 칸이 한 번에 채워짐**
   - 엑셀에서 저장(Ctrl+S)할 때마다 그림·검사 결과가 자동으로 바뀜 (확대·스크롤 위치 유지)
3. 검사: 빈칸, 다른 역과 같은/너무 가까운 좌표, 기존 역 좌표를 바꾼 경우, 캔버스 밖

실행
----
    python scripts/v2/42_edit_expansion_map.py            # 입력 파일 열기 + 실시간 미리보기 (끝내려면 Ctrl+C)
    python scripts/v2/42_edit_expansion_map.py --once     # 한 번만 검사·그림 저장하고 끝 (reports/figures/map_live/expansion_preview.png)
    python scripts/v2/42_edit_expansion_map.py --no-excel # 엑셀은 열지 않고 미리보기만

역 순서는 노선 구조 기준(경의중앙·수인분당·공항철도는 시각표 확보 전, 3-B 에서 재확인).
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
INPUT = ROOT / "data" / "master" / "draft" / "yeoyuro_map_expansion_input.xlsx"
DRAFT_3A = ROOT / "data" / "marts" / "v2" / "korail_line1" / "map_draft.csv"
LIVE = ROOT / "reports" / "figures" / "map_live" / "expansion"      # git 무시 폴더
ONCE_PNG = ROOT / "reports" / "figures" / "map_live" / "expansion_preview.png"
W, H = 5120, 2880
SHEET = "역좌표_입력"
NEAR_PX = 24                     # 다른 역과 이보다 가까우면 경고 (같은 좌표는 오류)

BASE_COLORS = {"1": "#0052A4", "2": "#009D3E", "3": "#EF7C1C", "4": "#00A5DE", "5": "#996CAC",
               "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C", "9": "#BDB092"}

# 확장 노선: 키 → (표시 이름, 색, 단계, {지선: [역 순서]})
EXPANSION = {
    "K1": ("1호선(코레일)", "#0052A4", "3-A", {
        "경부": ["서울역", "남영", "용산", "노량진", "대방", "신길", "영등포", "신도림", "구로",
               "가산디지털단지", "독산", "금천구청"],
        "경인": ["구로", "구일", "개봉", "오류동", "온수"],
        "경원": ["청량리", "회기", "외대앞", "신이문", "석계", "광운대", "월계", "녹천", "창동",
               "방학", "도봉", "도봉산"],
    }),
    "GJ": ("경의중앙선", "#77C4A3", "3-B", {
        "중앙": ["용산", "이촌", "서빙고", "한남", "옥수", "응봉", "왕십리", "청량리", "회기", "중랑",
               "상봉", "망우", "양원"],
        "경의": ["용산", "효창공원앞", "공덕", "서강대", "홍대입구", "가좌", "디지털미디어시티", "수색"],
    }),
    "SB": ("수인분당선", "#F5A200", "3-B", {
        "본선": ["왕십리", "서울숲", "압구정로데오", "강남구청", "선정릉", "선릉", "한티", "도곡",
               "구룡", "개포동", "대모산입구", "수서", "복정"],
    }),
    "AR": ("공항철도", "#0090D2", "3-B", {
        "본선": ["서울역", "공덕", "홍대입구", "디지털미디어시티", "마곡나루", "김포공항"],
    }),
}
# 범위에서 뺀 구간 (D-089): 배차가 길고 일부 열차만 들어가 시각표 기대 대기로는 추천이 불안정
#   - 경의중앙선 서울역–신촌(경의선)–가좌 지선
#   - 수인분당선 왕십리–청량리 (청량리 연장 운행분)
EXCLUDED = {"신촌(경의선)": "경의중앙선 서울역 지선 제외 (D-089)"}
COLS = ["순번", "단계", "역", "확장 노선", "상태", "x_px", "y_px", "참고_x", "참고_y", "참고_출처", "메모"]


# ---------------------------------------------------------------------- 읽기
def _copy_read(path: Path, **kw) -> dict:
    """엑셀이 파일을 열고 있어도 읽을 수 있게 임시 사본에서 읽는다."""
    tmp = Path(tempfile.mkdtemp()) / "wb.xlsx"
    shutil.copy2(path, tmp)
    try:
        return pd.read_excel(tmp, sheet_name=None, **kw)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def read_base() -> dict:
    """기본 노선도: 노드(역 이름·좌표), 노선 순서, 역명 위치. 읽기만 한다."""
    from yeoyuro_v2.map_workbook import materialize
    m = materialize(BASE)
    vn = m["Station_Visual_Nodes"].copy()
    vn["x_px"] = pd.to_numeric(vn.x_px, errors="coerce")
    vn["y_px"] = pd.to_numeric(vn.y_px, errors="coerce")
    return {"nodes": vn, "seq": m["Line_Sequences"], "labels": m["Station_Labels"]}


def base_coords(base: dict) -> dict[str, tuple[float, float]]:
    vn = base["nodes"].dropna(subset=["x_px", "y_px"])
    return {r.station_name: (float(r.x_px), float(r.y_px)) for r in vn.itertuples()}


def station_list() -> list[tuple[str, str, list[str]]]:
    """(역, 단계, 지나는 확장 노선) — 노선·지선 순서대로, 중복 없이."""
    seen: dict[str, list] = {}
    order = []
    for key, (name, _, phase, branches) in EXPANSION.items():
        for br, stations in branches.items():
            for s in stations:
                if s not in seen:
                    seen[s] = [phase, []]
                    order.append(s)
                if name not in seen[s][1]:
                    seen[s][1].append(name)
                if phase == "3-A":
                    seen[s][0] = "3-A"
    return [(s, seen[s][0], seen[s][1]) for s in order]


# ---------------------------------------------------------------------- 입력 파일 만들기
def build_input(base: dict, path: Path = INPUT) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    if path.exists():
        raise FileExistsError(path)
    bc = base_coords(base)
    draft = {}
    if DRAFT_3A.exists():
        d = pd.read_csv(DRAFT_3A)
        draft = {r.station: (r.x_px, r.y_px) for r in d.itertuples()}
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "안내"
    guide = [
        "확장 노선 역 좌표 입력 파일 (Phase 3) — 기본 노선도 파일과 별개",
        "",
        "1. '역좌표_입력' 시트의 노란 칸(x_px, y_px)만 채웁니다. 캔버스 5120 × 2880, y 는 아래로 증가.",
        "2. 회색 칸 = 이미 노선도에 있는 역의 기존 좌표입니다. 같은 역은 같은 좌표를 씁니다 (바꾸면 경고).",
        "3. 브라우저 미리보기에서 원하는 위치를 클릭하면 'x<탭>y' 가 복사됩니다 → x_px 칸을 선택하고 Ctrl+V.",
        "4. 저장(Ctrl+S)할 때마다 미리보기가 바뀝니다. 위쪽 막대에 빈칸·겹침 검사가 나옵니다.",
        "5. 참고_x/참고_y = 3A-1 자동 배치 초안 (써도 되고 무시해도 됩니다).",
        "6. 역 순서는 '노선순서' 시트. 선은 이 순서대로 이어 그립니다.",
        "",
        "다 찍으면 3A-2 에서 이 파일을 기본 노선도 파일로 합칩니다 (지금은 앱 노선도에 영향 없음).",
    ]
    for i, t in enumerate(guide, 1):
        ws0.cell(i, 1, t).font = Font(bold=(i == 1), size=12 if i == 1 else 11)
    ws0.column_dimensions["A"].width = 110

    ws = wb.create_sheet(SHEET)
    head = PatternFill("solid", fgColor="1F3864")
    yellow = PatternFill("solid", fgColor="FFF2CC")
    gray = PatternFill("solid", fgColor="E7E6E6")
    for j, c in enumerate(COLS, 1):
        cell = ws.cell(1, j, c)
        cell.fill, cell.font = head, Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center")
    for i, (s, phase, lines) in enumerate(station_list(), 2):
        exist = s in bc
        ref = bc.get(s) or draft.get(s)
        src = "기존 노선도" if exist else ("3A-1 배치 초안" if s in draft else "")
        vals = [i - 1, phase, s, ", ".join(lines), "기존 좌표" if exist else "새 역",
                bc[s][0] if exist else None, bc[s][1] if exist else None,
                ref[0] if ref else None, ref[1] if ref else None, src, None]
        for j, v in enumerate(vals, 1):
            ws.cell(i, j, v)
        for j in (6, 7):
            ws.cell(i, j).fill = gray if exist else yellow
    for j, wdt in enumerate([6, 6, 16, 26, 10, 9, 9, 9, 9, 14, 30], 1):
        ws.column_dimensions[get_column_letter(j)].width = wdt
    ws.freeze_panes = "C2"

    ws2 = wb.create_sheet("노선순서")
    for j, c in enumerate(["노선", "지선", "순서", "역", "단계"], 1):
        cell = ws2.cell(1, j, c)
        cell.fill, cell.font = head, Font(bold=True, color="FFFFFF")
    r = 2
    for key, (name, _, phase, branches) in EXPANSION.items():
        for br, stations in branches.items():
            for k, s in enumerate(stations, 1):
                for j, v in enumerate([name, br, k, s, phase], 1):
                    ws2.cell(r, j, v)
                r += 1
    for j, wdt in enumerate([14, 12, 6, 16, 6], 1):
        ws2.column_dimensions[get_column_letter(j)].width = wdt
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


def read_input(path: Path = INPUT) -> pd.DataFrame:
    """범위 밖 역(예전 입력 파일에 남은 줄)은 무시한다."""
    df = _copy_read(path)[SHEET]
    df = df[df["역"].notna()].copy()
    keep = {s for s, _, _ in station_list()}
    df.attrs["ignored"] = sorted(set(df["역"]) - keep)
    df = df[df["역"].isin(keep)].copy()
    for c in ("x_px", "y_px"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------- 검사
def check(inp: pd.DataFrame, base: dict) -> dict:
    bc = base_coords(base)
    have = inp.dropna(subset=["x_px", "y_px"])
    blanks = list(inp[inp.x_px.isna() | inp.y_px.isna()]["역"])
    out = [r["역"] for _, r in have.iterrows() if not (0 <= r.x_px <= W and 0 <= r.y_px <= H)]
    moved = [f"{r['역']} 기존 ({bc[r['역']][0]:.0f}, {bc[r['역']][1]:.0f}) → ({r.x_px:.0f}, {r.y_px:.0f})"
             for _, r in have.iterrows() if r["역"] in bc and
             (abs(bc[r["역"]][0] - r.x_px) > 0.5 or abs(bc[r["역"]][1] - r.y_px) > 0.5)]
    pts = {r["역"]: (r.x_px, r.y_px) for _, r in have.iterrows()}
    others = {k: v for k, v in bc.items() if k not in pts}           # 기존 노선도의 다른 역
    same, near = [], []
    items = list(pts.items())
    for i, (a, (x, y)) in enumerate(items):
        for b, (u, v) in items[i + 1:] + list(others.items()):
            d = ((x - u) ** 2 + (y - v) ** 2) ** 0.5
            if d < 0.5:
                same.append(f"{a} = {b} ({x:.0f}, {y:.0f})")
            elif d < NEAR_PX:
                near.append(f"{a}–{b} {d:.0f}px")
    n = len(inp)
    return {"n": n, "filled": n - len(blanks), "blanks": blanks, "out": out, "moved": moved,
            "same": same, "near": near, "ok": not (same or out), "ignored": inp.attrs.get("ignored", [])}


# ---------------------------------------------------------------------- 그림 (캔버스 1:1)
def render(inp: pd.DataFrame, base: dict, out: Path, res: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "AppleGothic"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), W, H, color="white", zorder=0))
    for g in range(0, W + 1, 100):                                   # 격자
        ax.axvline(g, color="#E5E7EB" if g % 500 else "#C7CBD1", lw=0.6 if g % 500 else 1.0, zorder=0.1)
    for g in range(0, H + 1, 100):
        ax.axhline(g, color="#E5E7EB" if g % 500 else "#C7CBD1", lw=0.6 if g % 500 else 1.0, zorder=0.1)
    for g in range(0, W + 1, 500):
        for y in range(250, H, 500):
            ax.text(g + 6, y, str(g), fontsize=9, color="#9CA3AF", zorder=0.2)
    for g in range(0, H + 1, 500):
        for x in range(250, W, 500):
            ax.text(x, g - 6, str(g), fontsize=9, color="#9CA3AF", zorder=0.2)
    # 기존 노선도 (흐리게)
    vn = base["nodes"].dropna(subset=["x_px", "y_px"])
    xy = vn.set_index("visual_node_id")[["x_px", "y_px"]]
    for (line, br), g in base["seq"].groupby(["line_id", "branch_code"]):
        g = g.sort_values("path_order")
        p = xy.reindex(g.visual_node_id).dropna()
        if str(line) == "2" and br == "main" and len(p):
            p = pd.concat([p, p.iloc[[0]]])
        ax.plot(p.x_px, p.y_px, color=BASE_COLORS.get(str(line), "#888"), lw=5, alpha=0.30,
                solid_capstyle="round", zorder=1)
    ax.scatter(vn.x_px, vn.y_px, s=40, c="white", edgecolors="#9CA3AF", linewidths=1.2, zorder=2)
    for r in vn.drop_duplicates("station_name").itertuples():
        ax.text(r.x_px, r.y_px + 16, r.station_name, ha="center", va="top", fontsize=8, color="#9CA3AF", zorder=2)
    # 확장 노선
    pts = {r["역"]: (r.x_px, r.y_px) for _, r in inp.dropna(subset=["x_px", "y_px"]).iterrows()}
    status = dict(zip(inp["역"], inp["상태"]))
    for key, (name, color, phase, branches) in EXPANSION.items():
        for br, stations in branches.items():
            run = []
            for s in stations + [None]:                              # 빈칸에서 선을 끊는다
                if s is not None and s in pts:
                    run.append(pts[s])
                    continue
                if len(run) > 1:
                    ax.plot([p[0] for p in run], [p[1] for p in run], color=color, lw=9, alpha=0.9,
                            solid_capstyle="round", solid_joinstyle="round", zorder=3)
                run = []
    for s, (x, y) in pts.items():
        new = status.get(s) == "새 역"
        ax.scatter([x], [y], s=150 if new else 220, c="white", edgecolors="#111827" if new else "#DC2626",
                   linewidths=2.4, zorder=4)
        ax.text(x + 18, y - 14, s, fontsize=15, fontweight="bold", color="#111827", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    leg = [f"{name} ({phase})" for _, (name, _, phase, _) in EXPANSION.items()]
    for i, ((_, (name, color, phase, _)), t) in enumerate(zip(EXPANSION.items(), leg)):
        ax.plot([W - 520, W - 440], [H - 220 + i * 44] * 2, color=color, lw=9)
        ax.text(W - 425, H - 220 + i * 44, t, va="center", fontsize=15)
    ax.text(W - 520, H - 270, f"확장 역 {res['filled']}/{res['n']} 입력 · 빨간 테두리 = 기존 역", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=100)
    plt.close(fig)
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------- 실시간 미리보기
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>확장 노선 좌표 찍기</title>
<style>
body{font-family:'Malgun Gothic',sans-serif;margin:0;background:#f3f4f6}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:8px 12px;font-size:15px;z-index:2}
#msg{font-size:13px;color:#555;margin-top:4px;max-height:110px;overflow:auto}
#pos{font-family:Consolas,monospace;background:#111827;color:#fff;padding:2px 8px;border-radius:4px}
#copied{color:#047857;font-weight:bold}
button{font-size:13px;margin-left:8px}
#wrap{padding:8px;overflow:auto}
img{display:block;background:#fff;border:1px solid #ddd;cursor:crosshair}
img.fit{width:100%;height:auto}
img.full{width:auto;max-width:none}
.flash{animation:f 1.2s}@keyframes f{from{background:#fff3c4}to{background:#fff}}
</style></head><body>
<div id="bar"><b id="st">검사 대기 중…</b> · <span id="when"></span>
 · 좌표 <span id="pos">x -, y -</span> <span id="copied"></span>
 <button id="zb">원본 크기로 (1:1)</button>
 <div style="font-size:13px;margin-top:4px">그림을 <b>클릭</b>하면 'x 탭 y' 가 복사됩니다 → 엑셀 x_px 칸 선택 후 Ctrl+V · 엑셀에서 저장(Ctrl+S)하면 그림이 바뀝니다</div>
 <div id="msg"></div></div>
<div id="wrap"><img id="img" class="fit" alt="미리보기"></div>
<script>
let ver=-1;
const img=document.getElementById('img'), zb=document.getElementById('zb');
zb.onclick=()=>{img.className=img.className==='fit'?'full':'fit';zb.textContent=img.className==='fit'?'원본 크기로 (1:1)':'화면에 맞추기';};
function xy(e){const r=img.getBoundingClientRect();
  return [Math.round((e.clientX-r.left)*img.naturalWidth/r.width), Math.round((e.clientY-r.top)*img.naturalHeight/r.height)];}
img.onmousemove=e=>{const [x,y]=xy(e);document.getElementById('pos').textContent='x '+x+', y '+y;};
img.onclick=async e=>{const [x,y]=xy(e);const t=x+'\\t'+y;
  try{await navigator.clipboard.writeText(t);document.getElementById('copied').textContent='복사됨: '+x+', '+y;}
  catch(err){window.prompt('복사가 막혔습니다. 직접 복사하세요 (x 탭 y)',t);}};
async function poll(){
  try{
    const r=await fetch('status.json?'+Date.now(),{cache:'no-store'});
    const s=await r.json();
    if(s.version!==ver){
      ver=s.version;
      if(s.image){const n=new Image();n.onload=()=>{img.src=n.src};n.src=s.image+'?'+s.version;}
      document.getElementById('st').textContent=s.status;
      document.getElementById('when').textContent='마지막 검사 '+s.when+' (저장 '+s.saved+')';
      document.getElementById('msg').innerHTML=s.detail;
      const b=document.getElementById('bar');b.classList.remove('flash');void b.offsetWidth;b.classList.add('flash');
    }
  }catch(e){document.getElementById('st').textContent='감시 스크립트가 꺼져 있습니다 (터미널에서 다시 실행)';}
  setTimeout(poll,1000);
}
poll();
</script></body></html>"""


class Quiet(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def serve() -> int:
    for port in range(8790, 8800):
        try:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", port), partial(Quiet, directory=str(LIVE)))
            srv.daemon_threads = True
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            return port
        except OSError:
            continue
    raise RuntimeError("8790~8799 포트를 모두 쓸 수 없습니다")


def summary(res: dict) -> tuple[str, str]:
    parts = []
    if res["same"]:
        parts.append("<b>다른 역과 같은 좌표(고쳐야 함)</b>: " + " / ".join(res["same"]))
    if res["out"]:
        parts.append("<b>캔버스 밖</b>: " + ", ".join(res["out"]))
    if res["moved"]:
        parts.append("기존 역 좌표가 바뀜 (기존 역은 기본 파일에서 옮기세요): " + " / ".join(res["moved"]))
    if res["near"]:
        parts.append(f"{NEAR_PX}px 안으로 가까움(확인): " + ", ".join(res["near"][:20]))
    if res["blanks"]:
        parts.append(f"빈칸 {len(res['blanks'])}개: " + ", ".join(res["blanks"]))
    if res.get("ignored"):
        parts.append("범위에서 빠져 무시한 줄 (지워도 됨): " + ", ".join(res["ignored"]))
    head = f"{'✅' if res['ok'] and not res['blanks'] else ('⚠️' if res['ok'] else '❌')} 확장 역 {res['filled']}/{res['n']} 입력"
    return head, "<br>".join(parts)


def publish(version: int, saved: str, png: Path | None, status: str, detail: str) -> None:
    image = ""
    if png is not None:
        image = f"preview_{version}.png"
        tmp = LIVE / (image + ".tmp")
        shutil.copy(png, tmp)
        try:
            from yeoyuro_v2.map_preview_extras import stamp
            stamp(tmp, f"#{version}  saved {saved}")
        except Exception:                                            # noqa: BLE001
            pass
        tmp.replace(LIVE / image)
        for old in LIVE.glob("preview_*.png"):
            if old.name != image:
                old.unlink(missing_ok=True)
    st = {"version": version, "status": status, "detail": detail, "image": image,
          "when": datetime.now().strftime("%H:%M:%S"), "saved": saved}
    tmp = LIVE / "status.json.tmp"
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    tmp.replace(LIVE / "status.json")


def run_once(base: dict, out: Path) -> dict:
    inp = read_input()
    res = check(inp, base)
    render(inp, base, out, res)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="한 번만 검사·그림 저장")
    ap.add_argument("--no-excel", action="store_true", help="엑셀을 열지 않음")
    a = ap.parse_args(argv)
    base = read_base()
    if not INPUT.exists():
        build_input(base)
        print(f"[새로 만듦] {INPUT}")
    else:
        print(f"[기존 파일 사용 — 덮어쓰지 않음] {INPUT}")
    if a.once:
        res = run_once(base, ONCE_PNG)
        head, _ = summary(res)
        print(f"{head} · 빈칸 {len(res['blanks'])} · 같은 좌표 {len(res['same'])} · 기존 역 이동 {len(res['moved'])} "
              f"· 가까움 {len(res['near'])}\n그림: {ONCE_PNG}")
        return 0 if res["ok"] else 1
    if not a.no_excel and hasattr(os, "startfile"):
        os.startfile(str(INPUT))                                     # noqa: S606 (Windows 전용)
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "index.html").write_text(PAGE, encoding="utf-8")
    publish(0, "-", None, "검사 중…", "")
    port = serve()
    url = f"http://localhost:{port}/index.html"
    webbrowser.open(url)
    print(f"입력 파일: {INPUT}\n미리보기: {url}  (브라우저가 안 열리면 주소창에 붙여 넣기)\n"
          "엑셀에서 저장할 때마다 그림이 바뀝니다. 끝내려면 Ctrl+C")
    png = LIVE / "_render.png"
    last, version = None, 0
    while True:
        try:
            m = INPUT.stat().st_mtime
            if m != last:
                time.sleep(0.8)                                      # 엑셀이 저장을 끝낼 시간
                last = INPUT.stat().st_mtime
                saved = datetime.fromtimestamp(last).strftime("%H:%M:%S")
                version += 1
                t0 = time.time()
                try:
                    res = run_once(base, png)
                except Exception as e:                               # noqa: BLE001
                    publish(version, saved, None, "⚠️ 읽기 실패 — 다시 저장하면 재시도합니다", f"{type(e).__name__}: {e}")
                    print(f"[{datetime.now():%H:%M:%S}] 읽기 실패 ({type(e).__name__}) — 다시 저장하면 재시도")
                    continue
                head, detail = summary(res)
                publish(version, saved, png, head, detail)
                print(f"[{datetime.now():%H:%M:%S}] 저장 감지 ({saved}) → {head} · 같은 좌표 {len(res['same'])} · "
                      f"가까움 {len(res['near'])} · 기존 역 이동 {len(res['moved'])} · #{version} ({time.time() - t0:.1f}초)")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("감시 종료")
            return 0


if __name__ == "__main__":
    sys.exit(main())
