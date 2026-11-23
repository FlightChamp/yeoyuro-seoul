"""
yeoyuro_v2/holidays.py — 대한민국 공휴일 날짜표 조회 (D-104, 사전 등록 11절)
=====================================================================

- 자료: `data/master/kr_public_holidays_2025_2027.csv` (날짜·명칭·분류·근거·확인일). 같은 날 여러 사유는 행을 나눈다
- 범위: 2025-01-01 ~ 2027-12-31. 범위 밖은 `supported()` 가 False — 호출 측이 '지원 범위 밖' 을 알린다 (P6)
- 앱 실행 중 외부 API 를 부르지 않는다. 새 임시공휴일이 지정되면 자료 파일만 고친다
"""
from __future__ import annotations

import csv
import datetime as dt
from functools import lru_cache
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "data" / "master" / "kr_public_holidays_2025_2027.csv"
FIRST, LAST = dt.date(2025, 1, 1), dt.date(2027, 12, 31)
KINDS = {"법정", "대체", "임시", "선거"}


@lru_cache(maxsize=1)
def _table() -> dict:
    out: dict = {}
    with open(FILE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(dt.date.fromisoformat(r["date"]), []).append(r)
    return out


def supported(date: dt.date) -> bool:
    return FIRST <= date <= LAST


def holiday_names(date: dt.date) -> list[str]:
    return [r["name"] for r in _table().get(date, [])]


def is_holiday(date: dt.date) -> bool:
    return date in _table()


def rows() -> list[dict]:
    return [r for d in sorted(_table()) for r in _table()[d]]
