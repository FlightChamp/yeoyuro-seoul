"""
tests/test_map_planned.py
=========================
D-090 — 노선도 기본 파일 안의 예정 노선(Phase 3) 처리.

| 검사 | 내용 |
|---|---|
| 앱 불변 | 기본값(예정 제외)으로 읽은 6개 시트 = 1~9호선 315점·270역, 앱 노선도 csv 가 커밋본과 바이트 단위로 같음 |
| 예정 포함 | 예정 점·순서가 map_planned 정의와 같음, 기존 역에 붙는 예정 점은 그 역 좌표로 채워짐 |
| 추가 스크립트 | 백업 원본에 다시 실행하면 커밋된 파일과 같은 표, 두 번째 실행은 아무것도 바꾸지 않음 |
| 미리보기 | 1:1 캔버스 5120×2880 |
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.map_planned import PHASE_COL, PLANNED, node_id, planned_nodes   # noqa: E402
from yeoyuro_v2.map_workbook import materialize                                 # noqa: E402

BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
SIX = ["Station_Display_Master", "Station_Visual_Nodes", "Station_Click_Areas", "Station_Labels",
       "Line_Sequences", "Transfer_Links"]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m_ops():
    return materialize(BASE)


@pytest.fixture(scope="module")
def m_all():
    return materialize(BASE, include_planned=True)


def test_default_reads_only_lines_1_to_9(m_ops):
    vn = m_ops["Station_Visual_Nodes"]
    assert len(vn) == 315 and vn.x_px.notna().all()
    assert sorted(vn.line_id.unique().tolist()) == list(range(1, 10)) and vn.line_id.dtype == "int64"
    for s in SIX:
        assert PHASE_COL not in m_ops[s].columns
    assert len(m_ops["Station_Display_Master"]) == len(m_ops["Station_Labels"]) == 270
    assert not m_ops["_missing"]


def test_app_map_unchanged(tmp_path):
    """예정 노선을 넣은 기본 파일로 앱 노선도를 다시 가져와도 커밋된 앱 노선도 csv 와 바이트 단위로 같다."""
    app = ROOT / "data" / "app_v26"
    if not (app / "data" / "marts" / "route_edge_mart.parquet").exists():
        pytest.skip("앱 데이터 폴더 없음")
    dst = tmp_path / "app"
    shutil.copytree(app, dst)
    b30 = _load("b30_planned", "scripts/v2/30_build_app_data.py")
    ok, msg = b30.import_map(dst, BASE)
    assert ok, msg
    for f in ("map_visual_nodes", "map_click_areas", "map_labels", "map_line_segments", "map_transfer_links",
              "station_display_master"):
        assert (dst / "data" / "master" / f"{f}.csv").read_bytes() == (app / "data" / "master" / f"{f}.csv").read_bytes(), f


def test_planned_rows_match_definition(m_all):
    vn, seq = m_all["Station_Visual_Nodes"], m_all["Line_Sequences"]
    p = vn[vn[PHASE_COL] != ""]
    ops = vn[vn[PHASE_COL] == ""]
    assert set(p.visual_node_id) == {n[0] for n in planned_nodes()} - set(ops.visual_node_id)
    for line, br, phase, _, stations in PLANNED:
        g = seq[(seq.line_id.astype(str) == line) & (seq.branch_code == br)].sort_values("path_order")
        assert g.visual_node_id.tolist() == [node_id(s, line) for s in stations], (line, br)
        assert (g[PHASE_COL] == phase).all()
    # 경의중앙선 서울역 지선·수인분당선 청량리 제외 (D-089)
    assert "신촌(경의선)" not in set(vn.station_key) and "청량리_LSB" not in set(vn.visual_node_id)
    # 기존 역에 붙는 예정 점은 좌표가 채워져 있어야 함 (값은 사용자가 바꿀 수 있으므로 비교하지 않음)
    ops_keys = set(ops.station_key)
    assert p[p.station_key.isin(ops_keys)][["x_px", "y_px"]].notna().all().all()
    # 수인분당선 야탑 연장 (D-091): 복정 → 가천대 → 태평 → 모란(8호선 환승) → 야탑
    sb = seq[(seq.line_id.astype(str) == "SB")].sort_values("path_order").station_key.tolist()
    assert sb[-5:] == ["복정", "가천대", "태평", "모란", "야탑"]
    tl = m_all["Transfer_Links"]
    assert ((tl.from_visual_node_id == "모란_L8") & (tl.to_visual_node_id == "모란_LSB")).any()


def _structure(m: dict) -> dict:
    """좌표 값(사용자가 계속 고침)을 뺀 뼈대: 점·순서·연결선·역 목록."""
    vn, ls, tl = m["Station_Visual_Nodes"], m["Line_Sequences"], m["Transfer_Links"]
    return {"nodes": sorted(zip(vn.visual_node_id, vn.station_key, vn.line_id.astype(str), vn[PHASE_COL])),
            "seq": sorted(zip(ls.line_id.astype(str), ls.branch_code, ls.path_order.astype(int), ls.visual_node_id,
                              ls[PHASE_COL])),
            "links": sorted(zip(tl.from_visual_node_id, tl.to_visual_node_id, tl[PHASE_COL])),
            "stations": sorted(m["Station_Display_Master"].station_key)}


@pytest.mark.parametrize("kind", ["before_planned_2", "before_planned_update"])
def test_migration_reproducible_and_idempotent(tmp_path, kind):
    """추가 전(처음)·갱신 전(야탑 연장 전) 백업에 스크립트를 다시 돌리면 커밋된 파일과 뼈대가 같고, 한 번 더 돌리면 변경 없음."""
    mig = _load("mig43", "scripts/v2/43_add_planned_lines.py")
    baks = sorted((ROOT / "data" / "master" / "archive").glob(f"*_{kind}*.xlsx"))
    if not baks:
        pytest.skip(f"{kind} 백업 없음")
    p = tmp_path / "wb.xlsx"
    shutil.copy(baks[-1], p)
    assert mig.migrate(p) == 0
    assert _structure(materialize(p, include_planned=True)) == _structure(materialize(BASE, include_planned=True))
    before = p.read_bytes()
    assert mig.migrate(p) == 0 and p.read_bytes() == before                     # 두 번째: 변경 없음


def _blank_points(path, node_ids):
    """사본에서 지정한 점의 x_px·y_px 를 비운다 (사용자가 옮긴 좌표와 무관하게 자동 배치를 검사하려고)."""
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb["Station_Visual_Nodes"]
    hdr = [c.value for c in ws[1]]
    ii, xi, yi = hdr.index("visual_node_id"), hdr.index("x_px"), hdr.index("y_px")
    hit = 0
    for row in ws.iter_rows(min_row=2):
        if row[ii].value in node_ids:
            row[xi].value = None
            row[yi].value = None
            hit += 1
    wb.save(path)
    return hit


def test_fill_blanks_only_touches_blank_planned_points(tmp_path):
    """자동 배치: 빈 예정 점만 채우고(앞뒤 사이 등간격·끝은 연장), 좌표가 있던 점은 그대로.

    기본 파일의 가천대·태평·야탑은 사용자가 옮길 수 있으므로, 사본에서 세 점을 비운 뒤 자동 배치를 검사한다."""
    mig = _load("mig43f", "scripts/v2/43_add_planned_lines.py")
    # (1) 기본 파일 그대로: 빈 점이 없으면 아무 좌표도 바꾸지 않음
    p0 = tmp_path / "wb0.xlsx"
    shutil.copy(BASE, p0)
    b0 = materialize(p0, include_planned=True)["Station_Visual_Nodes"].set_index("visual_node_id")
    mig.fill_blanks(p0)
    a0 = materialize(p0, include_planned=True)["Station_Visual_Nodes"].set_index("visual_node_id")
    had0 = b0.x_px.notna()
    assert (a0.loc[had0[had0].index, ["x_px", "y_px"]] == b0.loc[had0[had0].index, ["x_px", "y_px"]]).all().all()
    # (2) 세 점을 비운 사본: 비운 점만 채워지고 복정~야탑이 같은 간격
    p = tmp_path / "wb.xlsx"
    shutil.copy(BASE, p)
    new_pts = ("가천대_LSB", "태평_LSB", "야탑_LSB")
    assert _blank_points(p, new_pts) == 3
    before = materialize(p, include_planned=True)["Station_Visual_Nodes"].set_index("visual_node_id")
    mig.fill_blanks(p)
    after = materialize(p, include_planned=True)["Station_Visual_Nodes"].set_index("visual_node_id")
    had = before.x_px.notna()
    assert (after.loc[had[had].index, ["x_px", "y_px"]] == before.loc[had[had].index, ["x_px", "y_px"]]).all().all()
    if after.loc[["복정_LSB", "모란_LSB"], "x_px"].notna().all():
        sb = ["복정_LSB", "가천대_LSB", "태평_LSB", "모란_LSB", "야탑_LSB"]
        pts = after.loc[sb, ["x_px", "y_px"]]
        assert pts.notna().all().all()
        step = pts.diff().dropna()
        assert (step.sub(step.iloc[0]).abs() <= 1).all().all()                # 복정~야탑 같은 간격(반올림 1px)


def test_canvas_preview(tmp_path, m_all):
    pytest.importorskip("matplotlib")
    from PIL import Image
    from yeoyuro_v2.map_canvas import planned_status, render_canvas
    out = render_canvas(m_all, tmp_path / "p.png")
    assert Image.open(out).size == (5120, 2880)
    st = planned_status(m_all)
    assert st["n"] == len(st["blank"]) + st["filled"] and st["n"] > 0
