"""
tests/test_time_dependent.py
============================
v2.0 시간 진행형 평가 엔진 regression test.

1. fixed 정책은 v1 find_route_by_station 결과를 재현해야 한다 (v1 재현성 보장).
2. 이동 중 bin 을 하나만 통과하는 짧은 경로는 step == fixed 여야 한다.
3. step 은 bin 경계를 넘는 edge 를 분할해야 한다 (노출 시간 합 = 차내 시간 이하).
4. 환승 대기는 출발 bin 이 아니라 환승역 도착 bin 으로 조회해야 한다.

실행
----
    pytest tests/test_time_dependent.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

td = pytest.importorskip("yeoyuro_v2.time_dependent")
from yeoyuro_v2 import load_v1  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    if not (ROOT / "data" / "marts" / "congestion_edge_lookup.parquet").exists():
        pytest.skip("runtime mart 없음")
    mod, sr, disp = load_v1(ROOT)
    return mod, sr, disp, sr.load_headway(ROOT)


@pytest.mark.parametrize("o,d,dep", [
    ("신촌", "잠실", "08:30"), ("서울역", "강남", "08:20"),
    ("응암", "고속터미널", "08:00"), ("까치산", "성수", "17:50"), ("방화", "마천", "07:40"),
])
def test_fixed_reproduces_v1(ctx, o, d, dep):
    mod, sr, disp, hw = ctx
    rs = mod.RouteScorer(ROOT, "weekday", dep)
    r1 = sr.find_route_by_station(rs, disp, o, d, "calm")
    assert r1["ok"]
    ev = td.TimeDependentEvaluator(rs, hw, "fixed")
    for c in r1["candidates"]:
        r = ev.evaluate(c["path"], td.parse_hhmm(dep))
        assert abs(r.actual_time_min - c["actual_time_min"]) <= 0.11
        assert abs(r.perceived_time_min - c["perceived_time_min"]) <= 0.11
        assert round(r.max_congestion, 1) == pytest.approx(c["max_congestion"], abs=0.05)
        assert r.transfer_count == c["transfer_count"]


def test_single_bin_trip_equals_fixed(ctx):
    mod, sr, disp, hw = ctx
    rs = mod.RouteScorer(ROOT, "weekday", "08:31")
    r1 = sr.find_route_by_station(rs, disp, "신촌", "홍대입구", "calm")
    path = r1["fastest"]["path"]
    a = td.TimeDependentEvaluator(rs, hw, "fixed").evaluate(path, td.parse_hhmm("08:31"))
    b = td.TimeDependentEvaluator(rs, hw, "step").evaluate(path, td.parse_hhmm("08:31"))
    assert b.arrive_min < td.parse_hhmm("09:00")          # 전제: bin 하나 안에서 끝남
    assert a.max_congestion == pytest.approx(b.max_congestion)
    assert a.exposure == pytest.approx(b.exposure)


def test_step_splits_at_boundary_and_bounds_exposure(ctx):
    mod, sr, disp, hw = ctx
    rs = mod.RouteScorer(ROOT, "weekday", "07:40")
    r1 = sr.find_route_by_station(rs, disp, "방화", "마천", "calm")
    r = td.TimeDependentEvaluator(rs, hw, "step").evaluate(
        r1["fastest"]["path"], td.parse_hhmm("07:40"), keep_trace=True)
    assert len(r.bins_used) >= 3                          # 80분대 이동은 여러 bin 을 지난다
    for x in td.EXPOSURE_THRESHOLDS:
        assert 0 <= r.exposure[x] <= r.in_vehicle_min + 1e-6
    assert r.exposure[80] >= r.exposure[100] >= r.exposure[130]
    # trace 시각은 단조 증가해야 한다
    starts = [t.t_start for t in r.trace]
    assert starts == sorted(starts)


def test_transfer_wait_uses_arrival_bin(ctx):
    mod, sr, disp, hw = ctx
    rs = mod.RouteScorer(ROOT, "weekday", "08:20")
    r1 = sr.find_route_by_station(rs, disp, "서울역", "강남", "calm")
    path = r1["fastest"]["path"]
    ev = td.TimeDependentEvaluator(rs, hw, "step")
    r = ev.evaluate(path, td.parse_hhmm("08:20"), keep_trace=True)
    waits = [t for t in r.trace if t.kind == "transfer_wait"]
    assert waits, "환승 경로여야 한다"
    w = waits[0]
    nxt = path[path.index(w.u) + 1]
    m = ev.edge_meta[(w.u, nxt)]
    expected = sr.expected_wait(hw, m["line"], m["from_station"], m["direction"], "weekday",
                                td.bin_label(td.bin_index_of(w.t_start)))
    assert w.t_end - w.t_start == pytest.approx(expected or 0.0)


def test_bin_helpers():
    assert td.bin_index_of(td.parse_hhmm("05:30")) == 0
    assert td.bin_index_of(td.parse_hhmm("08:29")) == 5
    assert td.bin_index_of(td.parse_hhmm("00:30")) == 38
    assert td.bin_label(38) == "00:30~01:00"
