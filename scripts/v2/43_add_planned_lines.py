"""
scripts/v2/43_add_planned_lines.py
==================================
노선도 기본 파일에 Phase 3 예정 노선(코레일 1호선·경의중앙선·수인분당선·공항철도)을 **한 번** 추가한다. (D-090)
실행 전 자동 백업 (data/master/archive/). 이미 추가된 파일이면 아무것도 하지 않는다.

바뀌는 것
---------
- Station_Visual_Nodes · Line_Sequences · Transfer_Links : 맨 오른쪽에 `phase` 열 추가 (기존 행은 빈칸 = 운영 중)
- Station_Visual_Nodes : 예정 노선별 점 추가 (phase '3-A' / '3-B')
    · 이미 노선도에 있는 역(노량진·공덕·왕십리 등)은 그 역의 기존 좌표를 미리 채움 (같은 역 = 같은 좌표)
    · 새 역은 x_px·y_px 를 비워 둠 (노란 칸) — 3A-1 배치 초안이 있으면 alias_or_note 에 참고값
- Line_Sequences : 예정 노선 그리는 순서 / Transfer_Links : 예정 점과 같은 역 기존 점을 잇는 회색 연결선
- Station_Display_Master · Station_Click_Areas · Station_Labels : 새 역 행 (역명·클릭 위치는 기존과 같은 수식)
- Lines · 안내 시트 : 예정 노선 설명
- 다른 역과 좌표가 같으면 주황 표시하는 조건부 서식 범위를 새 행까지 넓힘

바뀌지 않는 것
--------------
- 기존 1~9호선 행의 값 (좌표·이동량·순서 모두 그대로)
- 앱 노선도: 가져오기(14)·앱은 phase 가 채워진 행을 빼고 읽는다 (yeoyuro_v2.map_workbook)

실행
----
    python scripts/v2/43_add_planned_lines.py          (엑셀을 닫고 실행)
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.formatting.rule import FormulaRule
from openpyxl.formatting.formatting import ConditionalFormattingList
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from copy import copy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.map_planned import LINES_ROWS, PHASE_COL, PLANNED, node_id, planned_nodes   # noqa: E402

BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
DRAFT_3A = ROOT / "data" / "marts" / "v2" / "korail_line1" / "map_draft.csv"
YELLOW = PatternFill("solid", fgColor="FFF2CC")
GREY = PatternFill("solid", fgColor="EDEDED")
ORANGE = PatternFill("solid", fgColor="FCE4D6")
PLAN = PatternFill("solid", fgColor="E2EFDA")          # 예정 행 표시 (phase 칸)
L = get_column_letter


def _head(ws) -> dict:
    return {c.value: c.column for c in ws[1] if c.value}


def _last_row(ws, key_col: int) -> int:
    r = ws.max_row
    while r > 1 and ws.cell(r, key_col).value in (None, ""):
        r -= 1
    return r


def _copy_style(ws, src_row: int, dst_row: int, ncol: int):
    for c in range(1, ncol + 1):
        s, d = ws.cell(src_row, c), ws.cell(dst_row, c)
        if s.has_style:
            d.font, d.border, d.alignment, d.number_format = copy(s.font), copy(s.border), copy(s.alignment), s.number_format
            d.fill = copy(s.fill)


def _add_phase_col(ws) -> int:
    h = _head(ws)
    if PHASE_COL in h:
        return h[PHASE_COL]
    c = ws.max_column + 1
    hc = ws.cell(1, c, PHASE_COL)
    src = ws.cell(1, c - 1)
    hc.font, hc.fill, hc.border, hc.alignment = copy(src.font), copy(src.fill), copy(src.border), copy(src.alignment)
    ws.column_dimensions[L(c)].width = 9
    return c


def migrate(path: Path = BASE) -> int:
    if (path.parent / ("~$" + path.name)).exists():
        print("[중단] 엑셀이 기본 파일을 열고 있습니다. 엑셀을 닫고 다시 실행하세요.")
        return 3
    wb = openpyxl.load_workbook(path)
    vn = wb["Station_Visual_Nodes"]
    hv = _head(vn)
    if PHASE_COL in hv:
        print("이미 예정 노선이 추가된 파일입니다. 아무것도 바꾸지 않았습니다.")
        return 0
    bak = path.parent / "archive" / f"{path.stem}_before_planned_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    bak.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, bak)

    # ---- 기존 점 (역 → 좌표, 역 → 점 id 목록)
    last_vn = _last_row(vn, hv["visual_node_id"])
    exist_ids, by_station = set(), {}
    for r in range(2, last_vn + 1):
        nid, k = vn.cell(r, hv["visual_node_id"]).value, vn.cell(r, hv["station_key"]).value
        exist_ids.add(nid)
        by_station.setdefault(k, []).append((nid, vn.cell(r, hv["x_px"]).value, vn.cell(r, hv["y_px"]).value,
                                             vn.cell(r, hv["line_id"]).value))
    draft = {}
    if DRAFT_3A.exists():
        d = pd.read_csv(DRAFT_3A)
        draft = {r.station: (int(round(r.x_px)), int(round(r.y_px))) for r in d.itertuples()}
    new_nodes = [n for n in planned_nodes() if n[0] not in exist_ids]
    lines_of = {}
    for nid, s, line, br, phase in planned_nodes():
        lines_of.setdefault(s, set()).add(str(line))
    for k, lst in by_station.items():
        lines_of.setdefault(k, set()).update(str(x[3]) for x in lst)

    # ---- Station_Visual_Nodes
    pc = _add_phase_col(vn)
    hv = _head(vn)
    X, Y, K = L(hv["x_px"]), L(hv["y_px"]), L(hv["station_key"])
    no = max(int(vn.cell(r, hv["visual_node_no"]).value or 0) for r in range(2, last_vn + 1))
    added_first = set()
    r = last_vn
    for nid, s, line, br, phase in new_nodes:
        r += 1
        no += 1
        _copy_style(vn, last_vn, r, vn.max_column)
        old = by_station.get(s)
        x, y = (old[0][1], old[0][2]) if old else (None, None)
        note = None
        if not old and s in draft:
            note = f"3A-1 배치 초안 ({draft[s][0]}, {draft[s][1]})"
        first = (not old) and s not in added_first
        added_first.add(s)
        vals = {"mark_status": "기존 좌표" if old else "입력 필요", "visual_node_no": no, "visual_node_id": nid,
                "station_key": s, "station_name": s, "display_name": s if s.endswith("역") else s + "역",
                "line_id": int(line) if line.isdigit() else line, "branch_codes": br, "x_px": x, "y_px": y,
                "x_pct": f"={X}{r}/5120*100", "y_pct": f"={Y}{r}/2880*100", "marker_radius_px": 12,
                "is_transfer_station": len(lines_of.get(s, ())) > 1, "is_branch_node": s in ("구로", "용산"),
                "label_owner": first, "alias_or_note": note, "memo": f"예정 {phase}"}
        for col, v in vals.items():
            if col in hv:
                vn.cell(r, hv[col], v)
        vn.cell(r, pc, phase).fill = PLAN
        for c in (hv["x_px"], hv["y_px"]):
            vn.cell(r, c).fill = YELLOW
        for c in (hv["x_pct"], hv["y_pct"]):
            vn.cell(r, c).fill = GREY
    last_new = r
    vn.conditional_formatting = ConditionalFormattingList()
    vn.conditional_formatting.add(
        f"A2:{L(vn.max_column)}{last_new}",
        FormulaRule(formula=[f'AND(${X}2<>"",COUNTIFS(${X}$2:${X}${last_new},${X}2,${Y}$2:${Y}${last_new},${Y}2,'
                             f'${K}$2:${K}${last_new},"<>"&${K}2)>0)'], fill=ORANGE, stopIfTrue=False))
    vn.auto_filter.ref = f"A1:{L(vn.max_column)}{last_new}"

    # ---- Line_Sequences
    ls = wb["Line_Sequences"]
    hs = _head(ls)
    lp = _add_phase_col(ls)
    hs = _head(ls)
    r = _last_row(ls, hs["visual_node_id"])
    src = r
    for line, br, phase, _, stations in PLANNED:
        for i, s in enumerate(stations, 1):
            r += 1
            _copy_style(ls, src, r, ls.max_column)
            vals = {"line_id": int(line) if line.isdigit() else line, "branch_code": br, "path_order": i,
                    "visual_node_id": node_id(s, line), "station_key": s, "station_name": s,
                    "display_name": s if s.endswith("역") else s + "역",
                    "is_branch_endpoint": i in (1, len(stations)), "alias_or_note": None}
            for col, v in vals.items():
                ls.cell(r, hs[col], v)
            ls.cell(r, lp, phase).fill = PLAN
    ls.auto_filter.ref = f"A1:{L(ls.max_column)}{r}"

    # ---- Transfer_Links: 예정 점 ↔ 같은 역의 기존 점(없으면 먼저 나온 예정 점)
    tl = wb["Transfer_Links"]
    ht = _head(tl)
    tp = _add_phase_col(tl)
    ht = _head(tl)
    r = _last_row(tl, ht["station_key"])
    src = r
    seen_new = {}
    for nid, s, line, br, phase in new_nodes:
        anchor = by_station[s][0] if s in by_station else seen_new.get(s)
        if anchor is None:
            seen_new[s] = (nid, None, None, line)
            continue
        r += 1
        _copy_style(tl, src, r, tl.max_column)
        a_line = anchor[3]
        vals = {"station_key": s, "station_name": s,
                "from_line_id": int(a_line) if str(a_line).isdigit() else a_line,
                "to_line_id": int(line) if line.isdigit() else line,
                "from_visual_node_id": anchor[0], "to_visual_node_id": nid, "transfer_style": "connector",
                "connector_visible": True, "connector_color": "#9CA3AF", "memo": f"예정 {phase} 노선도 연결선"}
        for col, v in vals.items():
            tl.cell(r, ht[col], v)
        tl.cell(r, tp, phase).fill = PLAN
    tl.auto_filter.ref = f"A1:{L(tl.max_column)}{r}"

    # ---- 새 역: Display_Master · Click_Areas · Labels
    new_stations = []
    for nid, s, line, br, phase in new_nodes:
        if s not in by_station and s not in new_stations:
            new_stations.append(s)
    branches = {}
    for line, br, phase, _, stations in PLANNED:
        for s in stations:
            branches.setdefault(s, []).append(f"{line}:{br}")
    phase_of = {s: ph for _, s, _, _, ph in new_nodes}
    dm = wb["Station_Display_Master"]
    hd = _head(dm)
    r = _last_row(dm, hd["station_key"])
    src = r
    sno = max(int(dm.cell(i, hd["station_no"]).value or 0) for i in range(2, r + 1))
    for s in new_stations:
        r += 1
        sno += 1
        _copy_style(dm, src, r, dm.max_column)
        vals = {"station_no": sno, "station_key": s, "station_name": s, "display_name": s + "역",
                "available_lines": ",".join(sorted(lines_of[s])), "candidate_station_uids": None,
                "project_branches": ";".join(dict.fromkeys(branches[s])),
                "is_transfer_station": len(lines_of[s]) > 1, "is_event_station": False,
                "alias_or_note": f"예정 {phase_of[s]}", "memo": "앱·경로 계산 미반영 (phase 예정)"}
        for col, v in vals.items():
            if col in hd:
                dm.cell(r, hd[col], v)
    dm.auto_filter.ref = f"A1:{L(dm.max_column)}{r}"

    rng = lambda col: f"Station_Visual_Nodes!${col}$2:${col}$2000"        # noqa: E731
    for ws, pre in ((wb["Station_Click_Areas"], "click"), (wb["Station_Labels"], "label")):
        h = _head(ws)
        r = _last_row(ws, h["station_key"])
        src = r
        tmpl = {c: ws.cell(src, h[c]).value for c in h}
        SK, DX, DY = L(h["station_key"]), L(h[f"{pre}_dx"]), L(h[f"{pre}_dy"])
        for s in new_stations:
            r += 1
            _copy_style(ws, src, r, ws.max_column)
            for c, v in tmpl.items():
                if c in ("mark_status", "station_key", "station_name", "display_name", "available_lines",
                         f"{pre}_x_px", f"{pre}_y_px", f"{pre}_x_pct", f"{pre}_y_pct", f"{pre}_dx", f"{pre}_dy",
                         "manual_override_note", "memo"):
                    continue
                ws.cell(r, h[c], v)
            ws.cell(r, h["mark_status"], "예정")
            ws.cell(r, h["station_key"], s)
            ws.cell(r, h["station_name"], s)
            ws.cell(r, h["display_name"], s + "역")
            ws.cell(r, h["available_lines"], ",".join(sorted(lines_of[s])))
            ws.cell(r, h[f"{pre}_dx"], 0).fill = YELLOW
            ws.cell(r, h[f"{pre}_dy"], 0).fill = YELLOW
            ws.cell(r, h[f"{pre}_x_px"], f"=ROUND(AVERAGEIF({rng(K)},${SK}{r},{rng(X)})+{DX}{r},0)").fill = GREY
            ws.cell(r, h[f"{pre}_y_px"], f"=ROUND(AVERAGEIF({rng(K)},${SK}{r},{rng(Y)})+{DY}{r},0)").fill = GREY
            ws.cell(r, h[f"{pre}_x_pct"], f"={L(h[pre + '_x_px'])}{r}/5120*100").fill = GREY
            ws.cell(r, h[f"{pre}_y_pct"], f"={L(h[pre + '_y_px'])}{r}/2880*100").fill = GREY
        ws.auto_filter.ref = f"A1:{L(ws.max_column)}{r}"

    # ---- Lines · 안내
    ln = wb["Lines"]
    hl = _head(ln)
    r = _last_row(ln, hl["line_id"])
    src = r
    for row in LINES_ROWS:
        r += 1
        _copy_style(ln, src, r, ln.max_column)
        for c, v in zip(("line_id", "line_name", "color", "project_range", "branch_codes", "note"), row):
            ln.cell(r, hl[c], int(v) if c == "line_id" and str(v).isdigit() else v)
    ln.auto_filter.ref = f"A1:{L(ln.max_column)}{r}"
    if "안내" in wb.sheetnames:
        g = wb["안내"]
        g.append(["예정 노선 (Phase 3, D-090)",
                  "Station_Visual_Nodes·Line_Sequences·Transfer_Links 의 phase 열이 '3-A'·'3-B' 인 행은 예정 노선이다. "
                  "좌표 편집 미리보기(29_watch_map)에는 같이 그려지지만, 앱 노선도·경로 계산에는 들어가지 않는다. "
                  "'입력 필요' 행의 노란 칸(x_px, y_px)을 채운다. 미리보기 그림을 클릭하면 'x 탭 y' 가 복사되어 x_px 칸에 Ctrl+V 로 붙여 넣을 수 있다. "
                  "범위: 3-A 1호선 경부·경인·경원(코레일), 3-B 경의중앙선 용산~양원·용산~수색, 수인분당선 왕십리~복정, 공항철도 서울역~김포공항."])
    wb.save(path)
    n_blank = sum(1 for n in new_nodes if n[1] not in by_station)
    print(f"추가 완료: {path}\n백업: {bak}\n예정 점 {len(new_nodes)}개 (기존 좌표 채움 {len(new_nodes) - n_blank} · "
          f"입력 필요 {n_blank}) · 새 역 {len(new_stations)}개 · 순서 {sum(len(p[4]) for p in PLANNED)}행")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(BASE))
    a = ap.parse_args(argv)
    return migrate(Path(a.xlsx))


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
