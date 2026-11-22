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
    # 기존 역에 붙는 예정 점 = 그 역 기존 좌표, 새 역 = 빈칸
    first = ops.groupby("station_key")[["x_px", "y_px"]].first()
    for r in p.itertuples():
        if r.station_key in first.index:
            assert (r.x_px, r.y_px) == tuple(first.loc[r.station_key]), r.visual_node_id
    assert m_all["_clash"] == []


def test_migration_reproducible_and_idempotent(tmp_path):
    mig = _load("mig43", "scripts/v2/43_add_planned_lines.py")
    baks = sorted((ROOT / "data" / "master" / "archive").glob("*_before_planned_*.xlsx"))
    if not baks:
        pytest.skip("추가 전 백업 없음")
    p = tmp_path / "wb.xlsx"
    shutil.copy(baks[-1], p)
    assert mig.migrate(p) == 0
    a, b = materialize(p, include_planned=True), materialize(BASE, include_planned=True)
    for s in SIX:
        pd.testing.assert_frame_equal(a[s].drop(columns=[c for c in a[s] if c.startswith("_")]),
                                      b[s].drop(columns=[c for c in b[s] if c.startswith("_")]), check_dtype=False)
    before = p.read_bytes()
    assert mig.migrate(p) == 0 and p.read_bytes() == before                     # 두 번째: 변경 없음


def test_canvas_preview(tmp_path, m_all):
    pytest.importorskip("matplotlib")
    from PIL import Image
    from yeoyuro_v2.map_canvas import planned_status, render_canvas
    out = render_canvas(m_all, tmp_path / "p.png")
    assert Image.open(out).size == (5120, 2880)
    st = planned_status(m_all)
    assert st["n"] == len(st["blank"]) + st["filled"] and st["n"] > 0
