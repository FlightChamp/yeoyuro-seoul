"""
tests/test_korail_app.py — 광역철도 확장 앱 통합 (D-101, 사전 등록 9절)

전체 검증(W1 구현 전 앱과 화면 비교, W2 K 표본 412쌍 × 41개 시각, W8 시간)은 scripts/v2/49_check_korail_app.py 가
계산해 reports/v2/korail_app_3a2d.{md,json} 에 남긴다. 여기서는 빠른 대표 검사만 한다.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA9 = ROOT / "data" / "app_v26"
WORKBOOK = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet").exists()
    or not (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists(), reason="mart 없음")

from yeoyuro_v2 import korail_app as KA          # noqa: E402

ONLY = ["남영", "용산", "대방", "영등포", "구로", "독산", "금천구청", "구일", "개봉", "오류동",
        "회기", "외대앞", "신이문", "광운대", "월계", "녹천", "방학", "도봉"]
COMMON = ["노량진", "신길", "신도림", "가산디지털단지", "온수", "석계", "창동", "도봉산"]


@pytest.fixture(scope="module")
def disp():
    return pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")


# ---------------------------------------------------------------------- B (W3 일부)
def test_station_lists(disp):
    """B2·B3: 전용 18 · 공통 8 · 켬 목록 = 끔 + 18, 이름 중복 0, 원본 불변."""
    before = disp.copy()
    assert KA.korail_only_stations(disp) == ONLY
    assert KA.common_stations(disp) == COMMON
    x = KA.extended_display(disp)
    assert len(x) == len(disp) + 18 and x.station_key.is_unique
    assert set(x.station_key) - set(disp.station_key) == set(ONLY)
    for n in COMMON:
        assert "1" in x.loc[x.station_key == n, "available_lines"].iloc[0].split(",")
    assert x.loc[x.station_key == "구로", "candidate_station_uids"].iloc[0] == "1_구로@경부;1_구로@경인"
    pd.testing.assert_frame_equal(disp, before)                 # 원본은 건드리지 않음
    assert KA.is_korail_only("월계", disp) and not KA.is_korail_only("석계", disp) and not KA.is_korail_only(None, disp)


# ---------------------------------------------------------------------- D (W5 일부)
def test_service_day():
    """D1 (엔진 parse_hhmm 경계): 04:29 까지 전날 운행일, 요일 시각표도 전날 날짜."""
    s = KA.service_day("2026-11-28", "00:30")                   # 토 00:30 → 금요일 밤
    assert s.date == dt.date(2026, 11, 27) and s.day_type == "weekday" and s.minute == 1470 and s.late_night
    assert "11월 27일(금) 심야" in s.label
    s = KA.service_day("2026-11-30", "04:29")                   # 월 04:29 → 일요일 밤 (휴일 시각표)
    assert s.date == dt.date(2026, 11, 29) and s.day_type == "sunday" and s.minute == 28 * 60 + 29
    s = KA.service_day("2026-11-30", "04:30")                   # 월 04:30 → 같은 날 이른 아침
    assert s.date == dt.date(2026, 11, 30) and s.day_type == "weekday" and not s.late_night
    assert KA.service_day("2026-11-29", "08:00").day_type == "sunday"


# ---------------------------------------------------------------------- B5 (W4 일부)
def test_map_overlay():
    if not WORKBOOK.exists():
        pytest.skip("노선도 엑셀 없음")
    ov = KA.map_overlay(WORKBOOK)
    assert len(ov["a_nodes"]) == 26 and len(ov["a_segs"]) > 0 and len(ov["b_segs"]) > 0
    assert sorted(ov["a_new_stations"]) == sorted(ONLY)          # 클릭·역명은 3-A 새 역만
    assert set(ov["a_click"].station_key) == set(ONLY) and set(ov["a_labels"].station_key) == set(ONLY)


# ---------------------------------------------------------------------- C (W2 일부)
@pytest.fixture(scope="module")
def planner(disp):
    sys.path[:0] = [str(ROOT / "app" / "streamlit")]
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    return KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", repo_root=ROOT)


def test_late_night_not_recommended(planner):
    """D-099 S3 예외: 금천구청→월계 23:00 은 K704(석계 00:20 도착 뒤 열차 없음) 경로를 확정 추천하지 않는다."""
    p = planner.plan("금천구청", "월계", KA.service_day("2026-11-24", "23:00"))
    assert p.best is not None and p.best.status == KA.CONNECT_OK
    assert "K704" not in [r.train for r in p.best.rides]
    # 같은 선로 경로라도 시각표 확인은 석계 대신 구로에서 갈아타는 열차 조합(K7038→K224)으로 완주된다 (A7: 두 값은 다를 수 있음)
    assert [r.train for r in p.best.rides] == ["K7038", "K224"]
    assert p.excluded_train.get("막차 종료", 0) >= 1
    assert p.best.train_arrive is not None and p.best.train_total_min is not None


def test_none_after_last_train(planner):
    """C4: 막차 뒤(금요일 밤 00:30 = 토 00:30 입력)에는 확정 추천 없음 + 사유."""
    p = planner.plan("금천구청", "월계", KA.service_day("2026-11-28", "00:30"))
    assert p.status == "none" and p.best is None and p.main_reason


def test_best_is_always_connected(planner):
    """C3: 확정 추천은 언제나 '연결 확인'. 빠른 연결 불가 후보가 있어도 그 앞에 오지 않는다."""
    for o, d in (("가산디지털단지", "월계"), ("신도림", "석계"), ("독산", "개봉"), ("회기", "구로"), ("상봉", "건대입구")):
        for hh in ("05:40", "08:00", "23:30", "00:10"):
            p = planner.plan(o, d, KA.service_day("2026-11-25", hh))
            if p.best is not None:
                assert p.best.status == KA.CONNECT_OK
                ok = [c for c in p.candidates if c.status == KA.CONNECT_OK]
                assert p.best.expected_min == min(c.expected_min for c in ok)
            assert p.status in ("ok", "no_benefit", "none")


def test_baseline_and_benefit(planner):
    """A4: 1~9호선 공통 역끼리면 같은 엔진의 1~9호선만 최단과 비교, 코레일 전용 역이면 비교 없음."""
    p = planner.plan("신도림", "석계", KA.service_day("2026-11-25", "08:00"))
    assert p.baseline_min is not None and p.status == "ok" and p.best.korail and p.saving_min > 0
    assert any(lg.korail for lg in p.best.legs)
    q = planner.plan("상봉", "건대입구", KA.service_day("2026-11-25", "08:00"))
    assert q.status == "no_benefit" and not q.best.korail
    r = planner.plan("가산디지털단지", "월계", KA.service_day("2026-11-25", "08:00"))
    assert r.baseline_min is None


# ---------------------------------------------------------------------- 화면 (W1·W3·W6 일부)
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"


def _route(at):
    at.session_state["nav_page"] = "경로 추천"
    at.session_state["view_route"] = "경로"


def test_app_off_default_and_toggle_clear():
    """A1·B1·B4: 기본 끔(목록 그대로), 켜고 월계를 고른 뒤 끄면 비우고 안내."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    _route(at)
    at.run()
    assert not at.exception
    assert at.toggle(key="ext_on").value is False
    n_off = len(at.selectbox(key="sel_origin").options)
    at.toggle(key="ext_on").set_value(True)
    at.run()
    assert len(at.selectbox(key="sel_origin").options) == n_off + 18
    at.selectbox(key="sel_origin").set_value("월계")
    at.selectbox(key="sel_dest").set_value("석계")
    at.run()
    at.toggle(key="ext_on").set_value(False)
    at.run()
    assert not at.exception
    assert at.session_state["origin_station_key"] is None
    assert at.session_state["destination_station_key"] == "석계"
    assert any(KA.OFF_CLEARED in w.value for w in at.warning)
    assert len(at.selectbox(key="sel_origin").options) == n_off


def test_app_on_card_texts():
    """A5·A7·W6: 코레일 전용 역 → 기존 결과 대신 안내 + 확장 카드(두 기준 문구·혼잡도 미확인·운행일)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    _route(at)
    at.run()
    at.toggle(key="ext_on").set_value(True)
    at.run()
    at.selectbox(key="sel_origin").set_value("가산디지털단지")
    at.selectbox(key="sel_dest").set_value("월계")
    at.date_input[0].set_value(dt.date(2026, 11, 24))
    at.time_input[0].set_value(dt.time(8, 0))
    at.run()
    [b for b in at.button if b.label == "경로 검색"][0].click()
    at.run()
    assert not at.exception
    caps = " ".join(c.value for c in at.caption)
    mds = " ".join(m.value for m in at.markdown)
    assert any(KA.NOT_IN_BASE in i.value for i in at.info)
    for t in (KA.BASIS_EXPECTED, KA.BASIS_TIMETABLE, KA.BASIS_STATIC, "운행일 기준: 11월 24일(화)", KA.NO_COMFORT):
        assert t in caps, t
    assert KA.BADGE_UNKNOWN in mds and KA.CONNECT_OK in mds
    assert [m.label for m in at.metric][:3] == ["기대 소요", "시각표 확인 도착", "1~9호선만 이용 시"]
