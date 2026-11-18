"""
tests/test_project_page.py — '프로젝트 & 검증' 화면의 v2 핵심 결과 (Phase 2-4, D-084).

1. yeoyuro_v2.headline 이 커밋된 mart 에서 리포트와 같은 수치를 계산하는가
2. 화면이 오류 없이 뜨고, 그 수치·비교 금지 단서·근거 문서를 보여 주는가
"""
from pathlib import Path

import pandas as pd
import pytest

import app_harness as H
from yeoyuro_v2 import headline as HL

ROOT = Path(__file__).resolve().parents[1]
MARTS = ROOT / "data" / "marts" / "v2"
L9 = MARTS / "line9_shiftability_mart.parquet"


def test_v23_summary_matches_report():
    s = HL.v23_summary(pd.read_parquet(MARTS / "od_shiftability_mart.parquet"))
    assert (s["n_units"], s["n_od"], s["n_hot"], s["n_structural"]) == (3720, 620, 1690, 48)
    assert s["shares"] == {"time_shiftable": 73.5, "dual": 23.1, "structural": 2.8, "route_shiftable": 0.5}
    assert s["time_alternative"] == 96.6


@pytest.mark.skipif(not L9.exists(), reason="9호선 평가 mart 없음")
def test_v26_summary_matches_report():
    s = HL.v26_summary(pd.read_parquet(L9))
    assert s["A"]["shares"]["structural"] == 7.6 and s["B"]["shares"]["structural"] == 5.1
    assert s["A"]["n_od"] == 300 and s["B"]["n_od"] == 300


def test_summaries_handle_missing_mart():
    assert HL.v23_summary(None) is None and HL.v26_summary(None) is None
    t = HL.comparison_table(HL.v23_summary(pd.read_parquet(MARTS / "od_shiftability_mart.parquet")), None)
    assert list(t.columns) == ["혼잡 이동 중 유형", "1~8호선 (v2.3)"]


def test_project_page_shows_v2_results():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(H.APP), default_timeout=300)
    at.session_state["nav_page"] = "프로젝트 & 검증"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    met = {m.label: m.value for m in at.metric}
    assert met["출발 시각을 바꿀 대안 있음"] == "96.6%"
    assert met["경로만 바꿔서 해결"] == "0.5%"
    assert met["회피 어려움"] == "2.8%"
    tables = [d.value for d in at.dataframe if "혼잡 이동 중 유형" in d.value.columns]
    assert tables and tables[0]["1~8호선 (v2.3)"].tolist() == ["73.5%", "23.1%", "2.8%", "0.5%"]
    if L9.exists():
        assert tables[0]["9호선 포함 · B군 (전체 균등)"].tolist()[2] == "5.1%"
    assert any("직접 비교하지 않습니다" in i.value for i in at.info)
    assert any("docs/v2/v2_results.md" in c.value for c in at.caption)
