"""P1 보조분석(36) 계산 로직 — 합성 데이터만 쓴다."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tc", ROOT / "scripts" / "v2" / "36_transfer_car_shift.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)
PC = [f"{i}번째 칸 혼잡도(%)" for i in range(1, 11)]


def _rows(train, start, a, b, vals, n=6):
    t0 = pd.Timestamp(start)
    return [{"편성번호": train, "t": t0 + pd.Timedelta(seconds=10 * k), "a": a, "b": b, **dict(zip(PC, vals))}
            for k in range(n)]


def _day(train="404", date="2026-06-17", drop_end=True):
    before = [80.0] * 10
    after = [80.0] * 10
    if drop_end:
        after[8], after[9] = 60.0, 60.0          # 9·10번째 칸에서 하차 집중
    rows = _rows(train, f"{date} 08:00:00", "동대문", "동대문역사문화공원", before)
    rows += _rows(train, f"{date} 08:01:30", "동대문역사문화공원", "충무로", after)
    return rows


def test_segments_and_passages():
    d = pd.DataFrame(_day())
    seg = tc.segments(d, PC)
    assert len(seg) == 2
    ps = tc.station_passages(seg, PC)
    assert len(ps) == 1 and ps.iloc[0].station == "동대문역사문화공원" and ps.iloc[0].next == "충무로"
    assert ps.iloc[0].d10 == -20 and ps.iloc[0].d1 == 0


def test_gap_breaks_link():
    rows = _rows("404", "2026-06-17 08:00:00", "동대문", "동대문역사문화공원", [80] * 10)
    rows += _rows("404", "2026-06-17 08:30:00", "동대문역사문화공원", "충무로", [80] * 10)   # 30분 뒤 → 다른 통과
    ps = tc.station_passages(tc.segments(pd.DataFrame(rows), PC), PC)
    assert ps.empty


def test_effect_sign_and_mirror():
    d = pd.DataFrame(_day())
    ps = tc.station_passages(tc.segments(d, PC), PC)
    tg = {("동대문역사문화공원", "충무로"): {9, 10}}
    e = tc.effects(ps, tg)
    assert e.iloc[0].e < 0                       # 하차 칸에서 더 많이 줄면 음수
    em = tc.effects(ps, tg, mirrored=True)       # 반전 기준 → 1·2번째 칸을 보게 됨
    assert em.iloc[0].e > 0
    centered_total = e.iloc[0].e
    assert np.isclose(centered_total, -20 - (-40 / 10) - (0 - (-40 / 10)))


def test_target_cars_from_tip():
    tip = pd.DataFrame([
        {"station_name": "동대문역사문화공원", "from_line": "4", "to_line": "2", "arrive_toward": "충무로 방면",
         "alight_car": 10, "depart_toward": "", "board_car": 10},
        {"station_name": "동대문역사문화공원", "from_line": "4", "to_line": "5", "arrive_toward": "충무로 방면",
         "alight_car": 2, "depart_toward": "", "board_car": 1},
        {"station_name": "총신대입구", "from_line": "4", "to_line": "7", "arrive_toward": "동작 방면",
         "alight_car": 1, "depart_toward": "", "board_car": 1},
    ])
    t = tc.target_cars(tip, "alight")
    assert t[("동대문역사문화공원", "충무로")] == {1, 2, 3, 9, 10}      # 두 노선 ±1 합집합
    assert t[("이수", "동작")] == {1, 2}                                # 별칭 · 끝 칸 자르기


def test_hier_boot_and_summary():
    rows = []
    for k, day in enumerate(["2026-06-17", "2026-06-18", "2026-06-19"]):
        for tr in ("404", "406"):
            rows += _day(train=tr, date=day)
    ps = tc.station_passages(tc.segments(pd.DataFrame(rows), PC), PC)
    e = tc.effects(ps, {("동대문역사문화공원", "충무로"): {9, 10}})
    lo, hi = tc.hier_boot(e, b=200)
    assert hi < 0 and lo <= hi
    s = tc.summarize(e)
    assert s["n_passages"] == 6 and s["combos"] == 1 and s["combos_negative"] == 1 and s["pass"] is True
