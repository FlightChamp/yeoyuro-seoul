"""
yeoyuro_v2/service_day.py — 입력 날짜·시각 → 운행일 (D-103, 사전 등록 10절)
=====================================================================

열차 운행시각표는 **운행일** 기준이다 (서울교통공사 DAY/SAT/END, 자정 뒤는 24:xx·25:xx). 혼잡도 통계도
같은 운행일 기준으로 판단했다 (F3·F4). 사용자가 넣는 것은 **달력 날짜·시각**이므로, 자정 뒤 입력은
전날 운행일의 연속 시각으로 바꿔야 시각표·혼잡도와 맞는다.

- 경계는 **시각표 프로필마다** 둔다 (R1). 지금은 서울교통공사 시각표(1~9호선 + 코레일 1호선) 하나.
  3-B 노선(경의중앙·수인분당·공항철도)은 3B-0 감사 뒤 자기 프로필로 추가한다.
- 서울교통공사 프로필 경계 = 엔진 `time_dependent.parse_hhmm` 과 같은 04:29 (F2: 01:15~04:59 열차 0건, 테스트로 고정)
- 04:30 이후 입력은 운행일 = 입력 날짜, 시각 = 입력 시각 그대로 (R5, 비심야 불변)
- 요일 유형은 운행일 날짜의 요일로 정한다 (통계 요일 `day_type`, 지금과 같음)
- D-104: **시각표 요일** `timetable_day_type` — 운행일이 공휴일(2025~2027 날짜표)이면 END(`sunday`).
  열차 단위 시각표를 쓰는 계산(확장 카드)만 이 값을 쓴다. 혼잡·배차 통계 요일은 바꾸지 않는다 (P2·P3)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from yeoyuro_v2 import time_dependent as td

DAY_KO = "월화수목금토일"


@dataclass(frozen=True)
class TimetableProfile:
    name: str
    late_night_until_min: int      # 이 분(자정부터) 미만의 입력은 전날 운행일 (서울교통공사: 270 = 04:30)
    note: str = ""


SEOUL_METRO = TimetableProfile(
    "seoul_metro_20260901", td.BIN_START_MIN - 60,
    "서울교통공사 열차운행시각표 20260901 (1~9호선 + 코레일 1호선). 엔진 parse_hhmm 과 같은 경계")
PROFILES = {"seoul_metro": SEOUL_METRO}


@dataclass(frozen=True)
class ServiceClock:
    input_date: dt.date
    input_hhmm: str               # 사용자가 넣은 시각 (화면 표시용)
    service_date: dt.date         # 운행일
    day_type: str                 # weekday / saturday / sunday
    minute: int                   # 운행일 기준 연속 분 (심야면 1440 이상)
    hhmm: str                     # 운행일 기준 연속 시각 ("24:30") — 엔진에 넘기는 값
    late_night: bool
    timetable_day_type: str = ""  # D-104: 공휴일이면 sunday(END), 아니면 day_type
    holiday: tuple = ()           # 운행일의 공휴일 명칭
    holiday_supported: bool = True  # 운행일이 공휴일 날짜표 범위(2025~2027) 안인지

    @property
    def label(self) -> str:
        d = self.service_date
        s = "운행일 기준: %d월 %d일(%s)%s" % (d.month, d.day, DAY_KO[d.weekday()], " 심야" if self.late_night else "")
        if self.holiday:
            s += " · 공휴일(%s)" % ", ".join(self.holiday)
        return s


def day_type_of(date: dt.date) -> str:
    """운행일 날짜 → 요일 유형 (공휴일 미구분, R10)."""
    w = date.weekday()
    return "saturday" if w == 5 else ("sunday" if w == 6 else "weekday")


def _as_date(date) -> dt.date:
    if isinstance(date, dt.datetime):
        return date.date()
    if isinstance(date, dt.date):
        return date
    return dt.date.fromisoformat(str(date)[:10])


def _as_hhmm(t) -> str:
    if isinstance(t, (dt.time, dt.datetime)):
        return "%02d:%02d" % (t.hour, t.minute)
    hh, mm = (int(x) for x in str(t).split(":")[:2])
    return "%02d:%02d" % (hh, mm)


def service_clock(date, time, profile: str = "seoul_metro") -> ServiceClock:
    """달력 날짜·시각 → 운행일·요일 유형·연속 시각."""
    p = PROFILES[profile]
    d = _as_date(date)
    hm = _as_hhmm(time)
    hh, mm = (int(x) for x in hm.split(":"))
    m = hh * 60 + mm
    late = m < p.late_night_until_min and hh < 24
    sd = d - dt.timedelta(days=1) if late else d
    minute = m + 1440 if late else m
    from yeoyuro_v2 import holidays as H
    names = tuple(H.holiday_names(sd))
    dtype = day_type_of(sd)
    return ServiceClock(d, hm, sd, dtype, minute, "%02d:%02d" % (minute // 60, minute % 60), late,
                        "sunday" if names else dtype, names, H.supported(sd))
