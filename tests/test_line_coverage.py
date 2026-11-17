"""
tests/test_line_coverage.py
===========================
Phase 1-4 (D-077) — 1~9호선 대표 경로 통합 테스트.

앱과 같은 계산 경로(get_scorer → find_route_by_station → v2 재계산 + provenance)를 Streamlit 없이 실행해,
노선마다 대표 출발·도착 쌍이 끝까지 계산되는지 한 번에 검사한다. 기존 test_app_ia 의 기준값 테스트가
"숫자가 바뀌지 않았는가"를 본다면, 이 테스트는 "모든 노선에서 경로가 성립하는가"를 본다.

검사 항목 (사례마다)
 1. 경로 탐색 성공 (ok)
 2. 계산 출처 = v2_time_dependent (조용한 v1 대체가 없음)
 3. 경로 연속성: 인접한 두 노드가 모두 실제 간선(주행 route_edge_mart ∪ 환승 transfer_edge_mart)
 4. 기대 노선 사용: 각 노선 대표 OD 는 그 노선을, 9호선은 급행(9X)·일반(9L)을 올바르게 탄다
 5. 값 범위: 소요시간 > 0, v2/v1 소요시간 비율 0.8~1.5, 100%+ 노출 ≤ 소요시간, 최대 혼잡도 0~250%

시간대: 평일 출근(08:00), 토요일 낮(14:00), 일요일 밤(22:30) — 배차·혼잡 조회가 시간대마다 다르기 때문.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "app_v26"
pytestmark = pytest.mark.skipif(not (DATA / "data" / "marts" / "route_edge_mart.parquet").exists(),
                                reason="앱 데이터 폴더(data/app_v26) 없음")

# (출발, 도착, 반드시 쓰는 노선 집합, 설명)
CASES = [
    ("서울역", "청량리", {"1"}, "1호선 단독"),
    ("강남", "잠실", {"2"}, "2호선 단독"),
    ("교대", "옥수", {"3"}, "3호선 단독"),
    ("노원", "수유", {"4"}, "4호선 단독"),
    ("광화문", "여의도", {"5"}, "5호선 단독"),
    ("합정", "이태원", {"6"}, "6호선 단독"),
    ("상봉", "건대입구", {"7"}, "7호선 단독"),
    ("석촌", "잠실", {"8"}, "8호선 단독 (1정거장)"),
    ("김포공항", "고속터미널", {"9X"}, "9호선 급행역 ↔ 급행역 = 급행"),
    ("등촌", "신목동", {"9L"}, "9호선 일반역만 = 일반"),
    ("서울역", "잠실", {"2", "4"}, "환승 1회"),
]
TIMES = [("weekday", "08:00", "2026-10-14"), ("saturday", "14:00", "2026-10-17"),
         ("sunday", "22:30", "2026-10-18")]


@lru_cache(maxsize=None)
def _ctx(day_type: str, hhmm: str, date: str):
    """시간대별 스코어러·평가기는 만들기 비싸므로 한 번만 만든다."""
    sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    from yeoyuro_v2.time_dependent import TimeDependentEvaluator
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    rs = mod.RouteScorer(DATA, day_type, hhmm, date)
    return sr, rs, sr.load_display_master(DATA), TimeDependentEvaluator(rs, sr.load_headway(DATA), "step")


@lru_cache(maxsize=1)
def _edges() -> frozenset:
    import pandas as pd
    marts = DATA / "data" / "marts"
    ride = pd.read_parquet(marts / "route_edge_mart.parquet", columns=["from_node", "to_node"])
    xfer = pd.read_parquet(marts / "transfer_edge_mart.parquet", columns=["from_node", "to_node"])
    return frozenset(zip(ride.from_node, ride.to_node)) | frozenset(zip(xfer.from_node, xfer.to_node))


def _route(o, d, day_type, hhmm, date):
    from yeoyuro_v2 import provenance as pv
    from yeoyuro_v2.app_bridge import retime_result
    from yeoyuro_v2.time_dependent import parse_hhmm
    sr, rs, disp, ev = _ctx(day_type, hhmm, date)
    res = sr.find_route_by_station(rs, disp, o, d, "calm")
    if not res.get("ok"):
        return res
    return pv.retime_with_provenance(res, lambda: ev, retime_result, depart_min=parse_hhmm(hhmm), mode="calm")


def _line(node: str) -> str:
    return node.split("_", 1)[0]


@pytest.mark.parametrize("when", TIMES, ids=lambda t: "%s-%s" % (t[0], t[1]))
@pytest.mark.parametrize("o,d,lines,desc", CASES, ids=[c[3] for c in CASES])
def test_representative_route(o, d, lines, desc, when):
    res = _route(o, d, *when)
    assert res.get("ok"), "%s→%s 경로 탐색 실패" % (o, d)
    assert res["provenance"]["source"] == "v2_time_dependent", res["provenance"]

    rec = res["recommended"]
    path = list(rec["path"])
    gaps = [(a, b) for a, b in zip(path, path[1:]) if (a, b) not in _edges()]
    assert not gaps, "실제 간선이 아닌 연결: %s" % gaps[:3]

    used = {_line(n) for n in path}
    assert lines <= used, "기대 노선 %s, 실제 %s" % (sorted(lines), sorted(used))

    act, v1 = rec["actual_time_min"], rec["v1"]["actual_time_min"]
    assert act > 0
    assert 0.8 <= act / v1 <= 1.5, "v2 %.1f분 / v1 %.1f분" % (act, v1)
    assert 0 <= (rec.get("exposure_100_min") or 0) <= act + 1e-6
    assert 0 <= rec["max_congestion"] <= 250


def test_line9_local_to_express_switch():
    """일반역(등촌) → 먼 급행역(고속터미널): 일반으로 출발해 급행으로 갈아타거나 일반으로 끝까지 가는 것 모두 허용,
    단 출발 구간은 반드시 9L (등촌은 급행이 서지 않음)."""
    res = _route("등촌", "고속터미널", *TIMES[0])
    assert res.get("ok") and res["provenance"]["source"] == "v2_time_dependent"
    path = res["recommended"]["path"]
    rides = [n for n in path if _line(n) in ("9L", "9X")]
    assert rides and _line(rides[0]) == "9L", path[:4]
    assert {_line(n) for n in path} <= {"9L", "9X"}
