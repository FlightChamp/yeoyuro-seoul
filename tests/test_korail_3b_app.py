"""
tests/test_korail_3b_app.py — Phase 3B-3 앱 통합 (D-105 5절 A1~A4, D-106 통합 토글).

- 하나의 토글 '광역철도 확장 (시범)' 로 3-A·3-B 를 같이 켠다. 3-B 는 정차표가 있을 때만 (없으면 3-A 만 + 안내)
- 끔: 확장 구간(3-A·3-B)은 회색 점선, 확장 전용 역 선택 불가 / 켬: 노선 색 실선, 확장 역 선택 가능
- 공항철도 구간 경로는 열차 연결을 확인해도 '참고 경로' (확정 추천 아님, B13)
정차표가 필요한 검사는 없으면 건너뛴다 (P4: 사용자 PC 에서 52 실행 뒤).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
from yeoyuro_v2 import korail_app as KA          # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"
WORKBOOK = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
HAS_APP = (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists()
HAS_3B = KA.k3b_ready(ROOT)
need_app = pytest.mark.skipif(not HAS_APP, reason="app_v26 데이터 없음")
need_3b = pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (사용자 PC 에서 52 실행 뒤, P4)")
ONLY_3B = ["수색", "가좌", "서강대", "서빙고", "한남", "응봉", "중랑", "망우", "양원", "서울숲", "압구정로데오", "한티",
           "구룡", "개포동", "대모산입구", "가천대", "태평", "야탑"]


@pytest.fixture(scope="module")
def disp():
    return pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")


# ---------------------------------------------------------------------- 역 목록 (A1)
@need_app
def test_station_lists(disp):
    base = KA.korail_only_stations(disp)
    assert len(base) == 18 and not set(base) & set(ONLY_3B)                 # 기본(include_3b=False) = 3-A 그대로
    both = KA.korail_only_stations(disp, include_3b=True)
    assert both[:18] == base and sorted(both[18:]) == sorted(ONLY_3B)
    x = KA.extended_display(disp, include_3b=True)
    assert len(x) == len(disp) + 36 and len(KA.extended_display(disp)) == len(disp) + 18
    row = x.set_index("station_key")
    assert set(row.at["공덕", "available_lines"].split(",")) == {"5", "6", "GJ", "AR"}
    assert set(row.at["망우", "candidate_station_uids"].split(";")) == {"GJ_망우", "GC_망우"}
    assert "GC_상봉" in row.at["상봉", "candidate_station_uids"] and "7_상봉" in row.at["상봉", "candidate_station_uids"]
    assert set(row.at["김포공항", "available_lines"].split(",")) == {"5", "9", "AR"}
    assert row.at["회기", "available_lines"] == "1,GJ"                      # 3-A 새 역에 3-B 배지
    assert disp.equals(pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv"))   # 원본 불변


def test_k3b_ready_is_file_check(tmp_path):
    assert KA.k3b_ready(tmp_path) is False                                   # 정차표 없는 환경 = 3-B 끔
    (tmp_path / "data" / "marts" / "v2" / "korail_3b").mkdir(parents=True)
    (tmp_path / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet").write_bytes(b"")
    assert KA.k3b_ready(tmp_path) is True


# ---------------------------------------------------------------------- 노선도 재료 (A3)
@pytest.mark.skipif(not WORKBOOK.exists(), reason="노선도 작업 엑셀 없음")
def test_map_overlay_has_lines_and_new_stations():
    ov = KA.map_overlay(WORKBOOK)
    assert set(ov["b_segs"].line_id) == {"GJ", "SB", "AR"} and set(ov["a_segs"].line_id) == {"1"}
    assert sorted(ov["b_new_stations"]) == sorted(ONLY_3B)
    assert len(ov["b_click"]) == 18 and len(ov["b_labels"]) == 18 and len(ov["b_links"]) > 0


# ---------------------------------------------------------------------- 계산기 (A2·B13)
@pytest.fixture(scope="module")
def planner3b(disp):
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    return KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", query_date="2026-10-14", repo_root=ROOT,
                               include_3b=True)


@need_3b
def test_planner_3b_routes(planner3b):
    sd = KA.service_day("2026-10-14", "08:00")
    p = planner3b.plan("양원", "야탑", sd)
    assert p.status_3b == "on" and p.status == "ok" and p.best.status == KA.CONNECT_OK
    assert set(p.best.lines_3b) == {"GJ", "SB"} and all(lg.korail for lg in p.best.legs)
    assert p.best.timetable_valid_from == {"GJ": "2026-09-01", "SB": "2026-08-22"}
    assert p.baseline_min is None                                            # 3-B 전용 역: 1~9호선 비교 없음
    q = planner3b.plan("가좌", "강남", sd)
    assert q.best is not None and q.best.status == KA.CONNECT_OK and "GJ" in q.best.lines_3b


@need_3b
def test_airport_is_reference_not_confirmed(planner3b):
    """B13: 공항철도 구간 경로는 확정 추천(best)이 되지 않고 '참고 경로' 로만."""
    sd = KA.service_day("2026-10-14", "08:00")
    p = planner3b.plan("김포공항", "서울역", sd)
    assert p.reference is not None and p.reference.is_reference
    assert p.reference.status == KA.CONNECT_OK and "AR" in p.reference.lines_3b
    assert p.reference.freshness_status and p.reference.timetable_valid_from["AR"] == "2025-12-29"
    assert p.best is None or not p.best.is_reference
    for o, d in (("마곡나루", "홍대입구"), ("서울역", "김포공항"), ("디지털미디어시티", "공덕")):
        x = planner3b.plan(o, d, sd)
        assert x.best is None or "AR" not in x.best.lines_3b
        assert x.status in ("ok", "no_benefit", "none", "reference")


@need_3b
def test_saturday_uses_holiday_timetable(disp):
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    sd = KA.service_day("2026-10-17", "14:00")                               # 토요일
    assert sd.day_type == "saturday"
    p = KA.ExtensionPlanner(DATA9, mod, sr, disp, sd.timetable_day_type, query_date=str(sd.date), repo_root=ROOT,
                            include_3b=True).plan("왕십리", "야탑", sd)
    assert p.best is not None and p.best.status == KA.CONNECT_OK
    st = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet")
    hol = set(st[(st.day_type == "sunday") & (st.group == "SB")].train)
    assert all(r.train in hol for r in p.best.rides if r.train.startswith("K"))


# ---------------------------------------------------------------------- 화면 (AppTest)
def _route(at):
    at.session_state["nav_page"] = "경로 추천"
    at.session_state["view_route"] = "경로"


def _search(at, o, d, date, hhmm):
    at.selectbox(key="sel_origin").set_value(o)
    at.selectbox(key="sel_dest").set_value(d)
    at.date_input[0].set_value(date)
    at.time_input[0].set_value(dt.time(*map(int, hhmm.split(":"))))
    at.run()
    [b for b in at.button if b.label == "경로 검색"][0].click()
    at.run()


@need_app
def test_app_off_dotted_extension_and_no_extension_stations():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    _route(at)
    at.run()
    assert not at.exception and at.toggle(key="ext_on").value is False
    assert at.toggle(key="ext_on").label == KA.TOGGLE_LABEL == "광역철도 확장 (시범)"
    opts = " ".join(at.selectbox(key="sel_origin").options)
    assert "야탑" not in opts and "월계" not in opts                          # 끔: 확장 전용 역 선택 불가
    spec = at.get("plotly_chart")[0].proto.spec
    assert '"dash":"dot"' in spec.replace(" ", "")                          # 끔: 확장 구간 회색 점선


@need_3b
def test_app_on_3b_route_and_reference():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=900)
    _route(at)
    at.run()
    at.toggle(key="ext_on").set_value(True)
    at.run()
    opts = " ".join(at.selectbox(key="sel_origin").options)
    assert "야탑" in opts and "양원" in opts                                  # 켬: 3-B 전용 역
    spec = at.get("plotly_chart")[0].proto.spec.replace(" ", "")
    assert '"dash":"dot"' not in spec and "#77C4A3".lower() in spec.lower()  # 켬 + 3-B 자료: 노선 색 실선
    _search(at, "양원", "야탑", dt.date(2026, 10, 14), "08:00")
    assert not at.exception
    caps = " ".join(c.value for c in at.caption)
    mds = " ".join(m.value for m in at.markdown)
    assert any(KA.NOT_IN_BASE in i.value for i in at.info)
    assert "시각표 기준일" in caps and KA.CONNECT_OK in mds and KA.BADGE_UNKNOWN in mds
    _search(at, "김포공항", "서울역", dt.date(2026, 10, 14), "08:00")
    assert not at.exception
    caps = " ".join(c.value for c in at.caption)
    mds = " ".join(m.value for m in at.markdown)
    assert KA.REFERENCE_TITLE in mds and KA.AR_REFERENCE in caps            # 공항철도 = 참고 경로 문구
    _search(at, "왕십리", "야탑", dt.date(2026, 10, 17), "14:00")            # 토요일
    assert KA.SATURDAY_3B in " ".join(c.value for c in at.caption)
