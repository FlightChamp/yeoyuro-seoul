"""
scripts/v2/30_build_map_workbook.py
===================================
1~9호선 노선도 좌표를 한 파일로 정리한 **새 기본 워크북**을 만든다 (v2.6).

입력 : 사용자가 9호선 좌표를 채운 작업본 (--src, 기본: data/master/yeoyuro_seoul_vector_map_coordinate_workbook_v26.xlsx)
출력 : data/master/yeoyuro_seoul_vector_map_coordinate_workbook.xlsx   ← 새 기본 파일 (source of truth)
       (기존 v1 파일은 data/master/archive/ 로 옮겨 둔다 — 실행 명령에서 처리)

정리 규칙
---------
- 모든 좌표·퍼센트는 수식이 아닌 숫자 값 (엑셀 저장 여부와 무관)
- Station_Visual_Nodes / Line_Sequences 는 노선(1→9)·지선·순서대로 정렬, visual_node_no 재부여
- 같은 역의 다른 노선 노드는 같은 좌표 (기존 관례). **다른 역끼리 같은 좌표는 금지** → 앞뒤 역 중간점으로
  임시 보정하고 mark_status='확인 필요' + 주황색 표시 (사용자가 최종 결정)
- v1 가져오기 스크립트(14)의 필수 6개 시트·컬럼 그대로 유지
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
W, H = 5120, 2880
LINE_COLORS = {"1": "#0052A4", "2": "#009D3E", "3": "#EF7C1C", "4": "#00A5DE", "5": "#996CAC",
               "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C", "9": "#BDB092"}
HEAD = PatternFill("solid", fgColor="1F3864")
INPUT = PatternFill("solid", fgColor="FFF2CC")
CHECK = PatternFill("solid", fgColor="F8CBAD")
THIN = Border(*(Side(style="thin", color="D9D9D9"),) * 4)


def write_sheet(wb, name, df, widths=None, input_cols=(), freeze="C2", flag_rows=()):
    ws = wb.create_sheet(name)
    for j, c in enumerate(df.columns, start=1):
        cell = ws.cell(1, j, c)
        cell.font, cell.fill = Font(bold=True, color="FFFFFF"), HEAD
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for i, row in enumerate(df.itertuples(index=False), start=2):
        for j, v in enumerate(row, start=1):
            v = None if (v is None or (isinstance(v, float) and pd.isna(v))) else v
            if hasattr(v, "item"):
                v = v.item()
            cell = ws.cell(i, j, v)
            cell.border = THIN
            if df.columns[j - 1] in input_cols:
                cell.fill = INPUT
        if (i - 2) in flag_rows:
            for j in range(1, len(df.columns) + 1):
                ws.cell(i, j).fill = CHECK
    ws.row_dimensions[1].height = 30
    for j, c in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(c, max(10, min(28, len(str(c)) + 4)))
    if freeze:
        ws.freeze_panes = freeze
    ws.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
    return ws


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook_v26.xlsx"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    src = Path(args.src)
    xl = pd.ExcelFile(src)
    d = {s: xl.parse(s) for s in xl.sheet_names}

    # ---------------- 노드: 정렬 + 다른 역 겹침 보정
    vn = d["Station_Visual_Nodes"].copy()
    ls = d["Line_Sequences"].copy()
    ls["line_id"] = ls.line_id.astype(int)
    border = {"main": 0, "common": 0, "seongsu_branch": 1, "sinjeong_branch": 2, "hanam_branch": 1, "macheon_branch": 2}
    ls["_b"] = ls.branch_code.map(border).fillna(9)
    ls = ls.sort_values(["line_id", "_b", "path_order"]).drop(columns="_b").reset_index(drop=True)
    order = {vid: k for k, vid in enumerate(dict.fromkeys(ls.visual_node_id))}
    vn["_o"] = vn.visual_node_id.map(order)
    vn = vn.sort_values(["line_id", "_o"]).drop(columns="_o").reset_index(drop=True)
    vn["visual_node_no"] = range(1, len(vn) + 1)
    vn["x_px"], vn["y_px"] = vn.x_px.astype(int), vn.y_px.astype(int)

    fixes = []
    for (x, y), g in vn.groupby(["x_px", "y_px"]):
        stations = g.visual_node_id.str.rsplit("_L", n=1).str[0]
        if stations.nunique() <= 1:
            continue
        # 같은 좌표에 다른 역 → 9호선 쪽에서 원래 그 자리의 주인이 아닌 역을 앞뒤 역 중간점으로
        for vid in g.visual_node_id:
            st, ln = vid.rsplit("_L", 1)
            others = [s_ for s_ in stations if s_ != st]
            owner_exists = any(f"{st}_L{l}" in set(vn.visual_node_id) and l != ln
                               for l in [str(i) for i in range(1, 10)])
            if owner_exists:
                continue                                  # 환승역 쪽은 그대로 둔다
            seq = ls[(ls.line_id == int(ln))].reset_index(drop=True)
            k = seq.index[seq.visual_node_id == vid][0]
            prv = vn.set_index("visual_node_id").loc[seq.visual_node_id[k - 1], ["x_px", "y_px"]]
            nxt = vn.set_index("visual_node_id").loc[seq.visual_node_id[k + 1], ["x_px", "y_px"]]
            nx, ny = int(round((prv.x_px + x) / 2)) if (prv.x_px, prv.y_px) != (x, y) else int(round((nxt.x_px + x) / 2)), \
                int(round((prv.y_px + y) / 2)) if (prv.x_px, prv.y_px) != (x, y) else int(round((nxt.y_px + y) / 2))
            i = vn.index[vn.visual_node_id == vid][0]
            fixes.append((vid, (x, y), (nx, ny), others))
            vn.loc[i, ["x_px", "y_px"]] = [nx, ny]
            vn.loc[i, "mark_status"] = "확인 필요"
            vn.loc[i, "memo"] = f"{', '.join(others)} 와 같은 좌표({x}, {y})였음 → 앞 역과의 중간점으로 임시 보정. 위치 확인 후 '완료'로"
    vn.loc[vn.mark_status != "확인 필요", "mark_status"] = "완료"
    vn.loc[vn.line_id == 9, "memo"] = vn.loc[vn.line_id == 9, "memo"].where(vn.mark_status == "확인 필요", "")
    vn["x_pct"], vn["y_pct"] = (vn.x_px / W * 100).round(4), (vn.y_px / H * 100).round(4)
    flag_nodes = set(vn.index[vn.mark_status == "확인 필요"])
    fixed_xy = {v.rsplit("_L", 1)[0]: n for v, _, n, _ in fixes}

    # ---------------- 역 단위 시트 (클릭·라벨): 값, 보정된 역은 새 좌표 따라감
    dm = d["Station_Display_Master"].copy()
    ca, lb = d["Station_Click_Areas"].copy(), d["Station_Labels"].copy()
    for t, xc, yc in ((ca, "click_x_px", "click_y_px"), (lb, "label_x_px", "label_y_px")):
        for st, (nx, ny) in fixed_xy.items():
            t.loc[t.station_key == st, [xc, yc]] = [nx, ny]
        t[xc], t[yc] = t[xc].astype(int), t[yc].astype(int)
        t[xc.replace("_px", "_pct")] = (t[xc] / W * 100).round(4)
        t[yc.replace("_px", "_pct")] = (t[yc] / H * 100).round(4)
        t["mark_status"] = "완료"
    ca["source_method"] = ca.source_method.replace({"auto_from_visual_nodes": "auto_from_visual_node"})
    lb["memo"] = lb.memo.where(~lb.memo.astype(str).str.contains("라벨 이동|덮어써도"), "")
    keyorder = {k: i for i, k in enumerate(dict.fromkeys(vn.station_key))}
    for t in (dm, ca, lb):
        t["_o"] = t.station_key.map(keyorder)
        t.sort_values("_o", inplace=True)
        t.drop(columns="_o", inplace=True)
        t.reset_index(drop=True, inplace=True)
    dm["station_no"] = range(1, len(dm) + 1)
    dm["memo"] = dm.memo.fillna("").replace({"v2.6 9호선": ""})

    tl = d["Transfer_Links"].copy()
    tl = tl.sort_values(["from_line_id", "to_line_id", "station_key"]).reset_index(drop=True)
    tl["memo"] = "환승 connector"

    # ---------------- 설정·안내·범위·출처
    cfg = d["Config"].copy()
    cfg = cfg[~cfg.parameter.astype(str).str.startswith("line")]
    cfg = pd.concat([cfg, pd.DataFrame([{"parameter": f"line{k}_color", "value": v, "note": f"{k}호선 노선 색"}
                                        for k, v in LINE_COLORS.items()])], ignore_index=True)
    scope = d["Project_Scope"].copy()
    scope["line_id"] = scope.line_id.astype(int)
    scope = scope.drop_duplicates(subset=["line_id", "branch_code"], keep="last").sort_values(["line_id"]).reset_index(drop=True)
    src_tbl = d["Sources"].copy()
    src_tbl = pd.concat([src_tbl, pd.DataFrame([
        {"source_name": "서울메트로9호선_역사정보_20260531", "url": "철도데이터포털 (서울시메트로9호선㈜)",
         "usage_note": "9호선 38역 목록·순서·위경도 참고 (노선도 좌표는 사용자 수작업)"},
        {"source_name": "서울교통공사_서울 도시철도 열차운행시각표_20260901", "url": "https://www.data.go.kr/data/15098251/fileData.do",
         "usage_note": "9호선 급행 정차역 16개 확인"}])], ignore_index=True).drop_duplicates(subset=["source_name"])

    guide = pd.DataFrame({"항목": [
        "파일", "버전", "좌표계", "작업 단위", "환승역 규칙", "금지", "편집 방법", "검증", "앱 반영", "주의 (v2.6 생성 시)"],
        "내용": [
        "여유로 서울 벡터 노선도 좌표 기본 파일 (source of truth). 1~9호선 270역 / 노드 315개",
        "v2.6 — 1~8호선(v1 좌표, 5호선 개화산·방화 y 조정) + 9호선(사용자 입력) 통합. 수식 없음, 모든 값 숫자",
        "캔버스 5120 × 2880 px, 왼쪽 위 (0,0), y 는 아래로 증가. *_pct 는 px 에서 계산된 참고값",
        "Station_Visual_Nodes = 노선별 역 점(노란 칸 x_px, y_px). Station_Click_Areas / Station_Labels = 역 단위 1행",
        "같은 역의 다른 노선 노드는 같은 좌표에 둔다 (예: 당산_L2 = 당산_L9)",
        "서로 다른 역을 같은 좌표에 두지 않는다. 겹치면 검증 FAIL",
        "좌표를 바꾸면 Station_Click_Areas·Station_Labels 의 같은 역 좌표도 함께 바꾼다 (라벨은 겹칠 때만 따로 이동)",
        "python scripts\\v2\\30b_check_map_workbook.py  → 검증표 + reports/figures/vector_map_preview.png",
        "9호선 경로 검색 연결 후 python scripts\\14_import_map_workbook.py --root . --xlsx data\\master\\yeoyuro_seoul_vector_map_coordinate_workbook.xlsx",
        ("주황색 행 " + ", ".join(f"{v.rsplit('_L', 1)[0]}({o[0]},{o[1]}→{n[0]},{n[1]})" for v, o, n, _ in fixes)
         + " — 다른 역과 좌표가 같아 임시 보정함. 위치를 정한 뒤 mark_status 를 '완료'로") if fixes else "없음"]})
    layer = d["Map_Layer_Order"].copy()
    handoff = pd.DataFrame({"AI 인수인계": [
        "이 파일은 여유로 서울(Yeoyuro Seoul) 벡터 클릭 노선도의 좌표 기본 파일이다. 자동 레이아웃으로 대체하지 않는다.",
        "필수 시트: Station_Display_Master, Station_Visual_Nodes, Station_Click_Areas, Station_Labels, Line_Sequences, Transfer_Links",
        "visual_node_id = '{역}_L{노선}'. 9호선은 노선도에서 한 줄(L9)이고, 일반/급행(9L/9X) 구분은 경로 계산에서만 쓴다.",
        "같은 역의 노선별 노드는 같은 좌표, 다른 역끼리 같은 좌표 금지. 좌표는 숫자 값만 (수식 금지).",
        "2호선 main 은 시청→…→충정로 후 시청으로 닫힌다. 6호선 응암순환은 가져오기에서 렌더링 보정한다."]})

    wb = Workbook()
    wb.remove(wb.active)
    write_sheet(wb, "Guide", guide, {"항목": 18, "내용": 140}, freeze=None)
    write_sheet(wb, "Config", cfg, {"parameter": 28, "value": 14, "note": 60}, freeze=None)
    write_sheet(wb, "Station_Display_Master", dm, {"station_key": 14, "candidate_station_uids": 34, "project_branches": 30})
    ws = write_sheet(wb, "Station_Visual_Nodes", vn, {"visual_node_id": 18, "station_key": 14, "memo": 60},
                     input_cols=("x_px", "y_px"), flag_rows=flag_nodes)
    hh = {c: j for j, c in enumerate(vn.columns, start=1)}
    for col, hi in (("x_px", W), ("y_px", H)):
        L = get_column_letter(hh[col])
        dv = DataValidation(type="whole", operator="between", formula1="0", formula2=str(hi), showErrorMessage=True,
                            errorTitle=col, error=f"0 ~ {hi} 사이 정수")
        dv.add(f"{L}2:{L}{len(vn) + 1}")
        ws.add_data_validation(dv)
    write_sheet(wb, "Station_Click_Areas", ca, {"station_key": 14}, input_cols=("click_x_px", "click_y_px"))
    write_sheet(wb, "Station_Labels", lb, {"station_key": 14, "memo": 40}, input_cols=("label_x_px", "label_y_px"))
    write_sheet(wb, "Line_Sequences", ls, {"visual_node_id": 18})
    write_sheet(wb, "Transfer_Links", tl, {"from_visual_node_id": 18, "to_visual_node_id": 18})
    write_sheet(wb, "Map_Layer_Order", layer, freeze=None)
    write_sheet(wb, "Project_Scope", scope, {"project_range": 34, "note": 60}, freeze=None)
    write_sheet(wb, "Sources", src_tbl, {"source_name": 44, "url": 50, "usage_note": 60}, freeze=None)
    write_sheet(wb, "AI_Handoff_Prompt", handoff, {"AI 인수인계": 140}, freeze=None)
    wb.active = 3
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f"저장: {args.out}")
    print(f"- 역 {len(dm)} / 노드 {len(vn)} (노선별 {vn.line_id.value_counts().sort_index().to_dict()}) / 순서 {len(ls)} / 환승선 {len(tl)}")
    for v, o, n, others in fixes:
        print(f"- 임시 보정: {v} {o} → {n} (원래 {', '.join(others)} 와 같은 좌표) → '확인 필요'")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
