"""
tests/test_holidays.py — 공휴일 시각표 요일 (D-104, 사전 등록 11-4절 H1~H6·H8)

확장 카드 전·후 비교(H9)와 앱 화면(H7) 전체는 scripts/v2/51_check_holidays.py → reports/v2/holidays_3a2f.{md,json}.
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import holidays as H              # noqa: E402
from yeoyuro_v2 import service_day as SD          # noqa: E402

D = dt.date.fromisoformat

# 사전 등록 11-2절 표 (날짜만, 같은 날 여러 사유는 한 번)
EXPECTED = {
    2025: "01-01 01-27 01-28 01-29 01-30 03-01 03-03 05-05 05-06 06-03 06-06 08-15 10-03 10-05 10-06 10-07 10-08 10-09 12-25",
    2026: "01-01 02-16 02-17 02-18 03-01 03-02 05-01 05-05 05-24 05-25 06-03 06-06 07-17 08-15 08-17 09-24 09-25 09-26 "
          "10-03 10-05 10-09 12-25",
    2027: "01-01 02-06 02-07 02-08 02-09 03-01 05-01 05-03 05-05 05-13 06-06 07-17 07-19 08-15 08-16 09-14 09-15 09-16 "
          "10-03 10-04 10-09 10-11 12-25 12-27",
}


# ---------------------------------------------------------------------- H1 자료
def test_table_matches_prereg():
    got = {y: sorted(d.strftime("%m-%d") for d in H._table() if d.year == y) for y in (2025, 2026, 2027)}
    for y, s in EXPECTED.items():
        assert got[y] == sorted(s.split()), y
    assert all(H.supported(d) for d in H._table())


def test_rows_have_source_and_kind():
    with open(H.FILE, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r["name"] and r["source"] and r["checked_at"] for r in rows)
    assert {r["kind"] for r in rows} <= H.KINDS
    dup = Counter((r["date"], r["name"]) for r in rows)
    assert max(dup.values()) == 1                                   # 같은 날·같은 사유 중복 없음
    assert sorted(H.holiday_names(D("2025-05-05"))) == ["부처님오신날", "어린이날"]   # 같은 날 두 사유
    for r in rows:                                                   # 대체공휴일은 평일
        if r["kind"] == "대체":
            assert D(r["date"]).weekday() < 5, r


# ---------------------------------------------------------------------- H2·H3
def test_year_differences():
    assert not H.is_holiday(D("2025-05-01")) and not H.is_holiday(D("2025-07-17"))
    assert H.is_holiday(D("2026-05-01")) and H.is_holiday(D("2026-07-17"))
    assert H.is_holiday(D("2027-05-03")) and H.is_holiday(D("2027-07-19"))


def test_temporary_and_election():
    for d in ("2025-01-27", "2025-06-03", "2026-06-03"):
        assert H.is_holiday(D(d)), d
    assert not H.is_holiday(D("2025-10-10"))                         # 지정되지 않음


# ---------------------------------------------------------------------- H4 시각표 요일
@pytest.mark.parametrize("date,stats,tt", [
    ("2026-11-25", "weekday", "weekday"),      # 평일
    ("2026-11-28", "saturday", "saturday"),    # 토
    ("2026-11-29", "sunday", "sunday"),        # 일
    ("2026-05-01", "weekday", "sunday"),       # 평일 공휴일 (노동절, 금)
    ("2027-05-03", "weekday", "sunday"),       # 대체공휴일 (월)
    ("2027-07-19", "weekday", "sunday"),
    ("2026-06-06", "saturday", "sunday"),      # 토요일 공휴일 → END (설계 가정 P7)
])
def test_timetable_day_type(date, stats, tt):
    c = SD.service_clock(date, "08:00")
    assert (c.day_type, c.timetable_day_type) == (stats, tt)
    assert bool(c.holiday) == H.is_holiday(D(date))


# ---------------------------------------------------------------------- H5 심야
@pytest.mark.parametrize("date,hhmm,sdate,tt", [
    ("2026-04-30", "23:50", "2026-04-30", "weekday"),
    ("2026-05-01", "00:30", "2026-04-30", "weekday"),
    ("2026-05-02", "00:30", "2026-05-01", "sunday"),
    ("2026-07-18", "00:30", "2026-07-17", "sunday"),
    ("2025-06-04", "00:30", "2025-06-03", "sunday"),
    ("2027-05-04", "00:30", "2027-05-03", "sunday"),
    ("2026-05-02", "04:29", "2026-05-01", "sunday"),
    ("2026-05-02", "04:30", "2026-05-02", "saturday"),
])
def test_late_night_holiday(date, hhmm, sdate, tt):
    c = SD.service_clock(date, hhmm)
    assert (str(c.service_date), c.timetable_day_type) == (sdate, tt)
    assert c.day_type == SD.day_type_of(c.service_date)                # 통계 요일은 그대로 (P2)


# ---------------------------------------------------------------------- H6 비공휴일 불변
def test_non_holiday_unchanged():
    d = D("2025-01-01")
    while d <= D("2027-12-31"):
        for hm in ("00:30", "08:00"):
            c = SD.service_clock(d, hm)
            if not H.is_holiday(c.service_date):
                assert c.timetable_day_type == c.day_type and c.holiday == () and "공휴일" not in c.label
        d += dt.timedelta(days=1)


# ---------------------------------------------------------------------- H8 범위 밖
def test_out_of_range():
    for d in ("2024-12-25", "2028-01-01"):
        c = SD.service_clock(d, "08:00")
        assert not c.holiday_supported and c.holiday == () and c.timetable_day_type == c.day_type
    assert SD.service_clock("2025-01-01", "00:30").holiday_supported is False   # 운행일 2024-12-31
    assert SD.service_clock("2026-11-25", "08:00").holiday_supported


def test_extension_service_day_carries_holiday():
    from yeoyuro_v2 import korail_app as KA
    s = KA.service_day("2026-05-02", "00:30")
    assert s.timetable_day_type == "sunday" and s.day_type == "weekday" and s.holiday == ("노동절",)
    assert "공휴일(노동절)" in s.label


# ---------------------------------------------------------------------- H7 화면
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"


def _search(on, o, d, date, hm):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.session_state["nav_page"] = "경로 추천"
    at.session_state["view_route"] = "경로"
    at.run()
    if on:
        at.toggle(key="ext_on").set_value(True)
        at.run()
    at.selectbox(key="sel_origin").set_value(o)
    at.selectbox(key="sel_dest").set_value(d)
    at.date_input[0].set_value(D(date))
    h, m = map(int, hm.split(":"))
    at.time_input[0].set_value(dt.time(h, m))
    at.run()
    [b for b in at.button if b.label == "경로 검색"][0].click()
    at.run()
    assert not at.exception
    return " ".join(c.value for c in at.caption)


@pytest.mark.skipif(not (ROOT / "data" / "app_v26" / "data" / "marts" / "route_edge_mart.parquet").exists(),
                    reason="앱 데이터 없음")
def test_app_holiday_notices():
    from yeoyuro_v2 import korail_app as KA
    caps = _search(True, "가산디지털단지", "월계", "2026-05-01", "08:00")          # 평일 공휴일, 확장 카드만
    assert KA.HOLIDAY_EXT in caps and "공휴일(노동절)" in caps
    caps = _search(False, "신림", "강남", "2026-05-01", "08:00")                  # 기존 카드: 안내만
    assert KA.HOLIDAY_BASE in caps and KA.HOLIDAY_EXT not in caps
    caps = _search(False, "신림", "강남", "2026-05-04", "08:00")                  # 비공휴일: 안내 없음
    assert KA.HOLIDAY_BASE not in caps and KA.OUT_OF_RANGE not in caps
    caps = _search(False, "신림", "강남", "2028-01-04", "08:00")                  # 범위 밖
    assert KA.OUT_OF_RANGE in caps


# ---------------------------------------------------------------------- D-104b (사후): 시각표 요일 ≠ 통계 요일
@pytest.mark.skipif(not (ROOT / "data" / "app_v26" / "data" / "marts" / "route_edge_mart.parquet").exists(),
                    reason="app_v26 데이터 없음")
def test_holiday_planner_uses_timetable_day_for_korail():
    """평일 공휴일 계산기: 1~9호선 통계는 평일(조회 날짜 요일), 코레일 열차 조회는 END.
    D-104 첫 구현은 엔진 평가기가 통계 요일(평일)로 END 색인을 찾아 모든 코레일 후보를 '불가능한 연결' 로 뺐다."""
    import pandas as pd
    sys.path.insert(0, str(ROOT / "app" / "streamlit"))
    import station_routing as sr
    from yeoyuro_v2 import korail_app as KA
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    data9 = ROOT / "data" / "app_v26"
    disp = pd.read_csv(data9 / "data" / "master" / "station_display_master.csv")
    sd = KA.service_day("2026-05-01", "08:00")
    assert (sd.day_type, sd.timetable_day_type) == ("weekday", "sunday")
    p = KA.ExtensionPlanner(data9, mod, sr, disp, sd.timetable_day_type, query_date=str(sd.date), repo_root=ROOT)
    r = p.plan("혜화", "용산", sd)
    assert p.on.scorer(0).day_type == "weekday"                       # 통계 요일 그대로 (P3)
    assert p.on.evaluator(0).tt_day == "sunday"                       # 열차 조회는 휴일 시각표
    assert "불가능한 연결" not in r.excluded_engine
    assert r.best is not None and r.best.status == KA.CONNECT_OK
