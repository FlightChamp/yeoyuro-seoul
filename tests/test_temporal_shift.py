"""
tests/test_temporal_shift.py
============================
v2.2 Time Shift 판정 regression test.

1. Top 은 성공 후보만, 규칙 순서(|이동폭| → 노출 감소 → 늦게 우선)로 정렬된다.
2. Calm 이면 Top 이 비어 있다.
3. 후보 시각은 ±30/60/90 안이고 운행 창 밖은 제외된다.
4. 같은 경로 후보의 수치는 엔진으로 직접 평가한 값과 같다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import load_v1  # noqa: E402
from yeoyuro_v2.temporal_shift import evaluate_time_shift  # noqa: E402
from yeoyuro_v2.time_dependent import TimeDependentEvaluator, parse_hhmm  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    if not (ROOT / "data" / "marts" / "congestion_edge_lookup.parquet").exists():
        pytest.skip("runtime mart 없음")
    mod, sr, disp = load_v1(ROOT)
    return mod, sr, disp, sr.load_headway(ROOT)


def _run(ctx, o, d, dep):
    mod, sr, disp, hw = ctx
    rs = mod.RouteScorer(ROOT, "weekday", dep)
    res = sr.find_route_by_station(rs, disp, o, d, "fast")
    ev = TimeDependentEvaluator(rs, hw, "step")
    path = res["fastest"]["path"]
    return ev, path, evaluate_time_shift(ev, path, parse_hhmm(dep))


@pytest.mark.parametrize("o,d,dep", [("노원", "양천구청", "17:40"), ("사당", "교대", "08:10"),
                                     ("응암", "고속터미널", "08:00")])
def test_top_rules(ctx, o, d, dep):
    ev, path, ts = _run(ctx, o, d, dep)
    assert ts["type"] == "time_shiftable"
    keys = [(abs(x.shift_min), -x.exposure_100_drop_min, -x.shift_min) for x in ts["top"]]
    assert keys == sorted(keys)
    assert all(x.success for x in ts["top"]) and len(ts["top"]) <= 3
    assert all(abs(x.shift_min) in (30, 60, 90) for x in ts["rows"])
    for x in ts["rows"]:
        if x.same_route:
            r = ev.evaluate(path, x.depart_min)
            assert x.exposure_100_min == pytest.approx(round(r.exposure[100], 2))


def test_calm_has_no_top(ctx):
    _, _, ts = _run(ctx, "신촌", "잠실", "08:30")
    assert ts["type"] == "calm" and ts["top"] == []


def test_window_edge_skips(ctx):
    _, _, ts = _run(ctx, "신촌", "홍대입구", "06:00")
    assert -60 in ts["skipped_shifts"] and -90 in ts["skipped_shifts"]
    assert all(x.depart_min >= parse_hhmm("05:30") for x in ts["rows"])
