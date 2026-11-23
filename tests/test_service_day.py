"""
tests/test_service_day.py — 심야 운행일 일관화 (D-103, 사전 등록 10절 T1·T2·T3·T5·T6)

앱 전·후 화면 비교(T7)와 심야 전후 경로 비교(T8)는 scripts/v2/50_check_late_night.py 가 계산해
reports/v2/late_night_3a2e.{md,json} 에 남긴다.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import service_day as SD           # noqa: E402
from yeoyuro_v2 import time_dependent as td        # noqa: E402

STOP_TIMES = ROOT / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet"


# ---------------------------------------------------------------------- T1·T6 경계
@pytest.mark.parametrize("date,hhmm,sdate,dtype,minute,late", [
    ("2026-11-28", "00:30", "2026-11-27", "weekday", 1470, True),     # 토 00:30 → 금 DAY
    ("2026-11-29", "00:30", "2026-11-28", "saturday", 1470, True),    # 일 00:30 → 토 SAT
    ("2026-11-30", "00:30", "2026-11-29", "sunday", 1470, True),      # 월 00:30 → 일 END
    ("2026-11-30", "04:29", "2026-11-29", "sunday", 1709, True),
    ("2026-11-30", "04:30", "2026-11-30", "weekday", 270, False),
    ("2026-11-30", "00:00", "2026-11-29", "sunday", 1440, True),
    ("2026-11-30", "00:01", "2026-11-29", "sunday", 1441, True),
    ("2026-11-29", "23:59", "2026-11-29", "sunday", 1439, False),
    ("2026-11-30", "01:14", "2026-11-29", "sunday", 1514, True),
    ("2026-11-30", "01:15", "2026-11-29", "sunday", 1515, True),
    ("2026-11-30", "05:00", "2026-11-30", "weekday", 300, False),
    ("2027-01-01", "00:30", "2026-12-31", "weekday", 1470, True),     # 해·달 넘김
])
def test_boundary(date, hhmm, sdate, dtype, minute, late):
    c = SD.service_clock(date, hhmm)
    assert (str(c.service_date), c.day_type, c.minute, c.late_night) == (sdate, dtype, minute, late)
    assert c.hhmm == "%02d:%02d" % (minute // 60, minute % 60) and c.input_hhmm == hhmm
    assert td.parse_hhmm(c.hhmm) == td.parse_hhmm(hhmm) == minute      # 엔진 해석과 같음


def test_accepts_time_objects():
    c = SD.service_clock(dt.date(2026, 11, 28), dt.time(0, 30))
    assert c.service_date == dt.date(2026, 11, 27) and c.hhmm == "24:30"
    assert "11월 27일(금) 심야" in c.label


# ---------------------------------------------------------------------- T3 비심야 인자 불변
def test_non_late_args_unchanged():
    """04:30~23:59 (30분 간격) × 7요일: 운행일 인자 = 지금 앱 인자 (요일·시각·날짜)."""
    base = dt.date(2026, 11, 23)                                       # 월
    for k in range(7):
        d = base + dt.timedelta(days=k)
        w = d.weekday()
        old_dt = "saturday" if w == 5 else ("sunday" if w == 6 else "weekday")
        for m in range(270, 1440, 30):
            hm = "%02d:%02d" % (m // 60, m % 60)
            c = SD.service_clock(d, hm)
            assert (c.day_type, c.hhmm, c.service_date, c.late_night) == (old_dt, hm, d, False)


# ---------------------------------------------------------------------- R2 프로필 경계 (자료)
@pytest.mark.skipif(not STOP_TIMES.exists(), reason="시각표 mart 없음")
def test_profile_boundary_holds_on_timetable():
    """서울교통공사 프로필: 모든 출발·도착이 (경계, 경계+24시간] 안 — 경계 안쪽(01:15~04:59)에 열차 0건 (F2)."""
    s = pd.read_parquet(STOP_TIMES)
    b = SD.SEOUL_METRO.late_night_until_min
    for col in ("arr_min", "dep_min"):
        x = s[col].dropna()
        assert x.min() >= b and x.max() < b + 1440, col
    assert s.dep_min.min() >= 300 and s.dep_min.max() < 1515           # 첫 05:00, 마지막 25:14


@pytest.mark.skipif(not STOP_TIMES.exists(), reason="시각표 mart 없음")
def test_midnight_trains_one_record():
    """T2: 자정을 넘는 열차는 한 운행일·한 열차번호의 기록이고 정차 순서대로 시각이 줄지 않는다."""
    s = pd.read_parquet(STOP_TIMES)
    cross = s[s.dep_min.fillna(s.arr_min) >= 1440][["day_type", "group", "train"]].drop_duplicates()
    assert len(cross) > 0
    m = s.merge(cross, on=["day_type", "group", "train"])
    for _, g in m.groupby(["day_type", "group", "train", "direction"]):
        g = g.sort_values("stop_seq")
        t = g.dep_min.fillna(g.arr_min).to_numpy()
        assert (t[1:] >= t[:-1] - 1e-9).all()


# ---------------------------------------------------------------------- T5 두 카드 같은 운행일
def test_extension_uses_same_clock():
    from yeoyuro_v2 import korail_app as KA
    for date in ("2026-11-27", "2026-11-28", "2026-11-29", "2026-11-30"):
        for hm in ("00:00", "00:30", "01:14", "04:29", "04:30", "08:00", "23:50"):
            a, b = KA.service_day(date, hm), SD.service_clock(date, hm)
            assert (a.date, a.day_type, a.minute, a.late_night, a.label) == \
                (b.service_date, b.day_type, float(b.minute), b.late_night, b.label)


# ---------------------------------------------------------------------- 앱이 운행일 인자를 엔진에 넘기는지
def test_app_passes_service_args():
    src = (ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py").read_text(encoding="utf-8")
    assert "day_of_week" not in src                     # 달력 요일 직접 계산은 남아 있지 않음
    assert src.count("_service_clock(") >= 3
    # 확장 카드 계산기도 운행일 날짜로 만든다 (RouteScorer 는 조회 날짜의 요일로 통계 요일을 다시 정하므로,
    # 심야에 입력 날짜를 넘기면 1~9호선 구간 통계만 다음 날 요일이 된다)
    assert "get_ext_planner(sd.timetable_day_type, str(sd.date))" in src     # D-104: 시각표 요일
