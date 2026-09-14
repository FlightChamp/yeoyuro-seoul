"""
tests/test_through_junction.py
==============================
직결 분기역(5호선 강동) 통과 regression test.

5호선은 방화~하남검단산 / 방화~마천 이 **하나의 계통으로 직결 운행**한다.
근거는 서울교통공사 `열차운행현황` 의 구간 표기다.

    5호선 : 방화~하남검단산/마천          영업거리 59.8   시격 2.5/6.5   운행횟수 428
    2호선 : 성수~성수[성수지선/신정지선]     영업거리 48.8[5.4/6.0]      운행횟수 528[226/220]

2호선 지선은 대괄호로 분리돼 별도 시격·별도 운행횟수를 갖는 셔틀이지만,
5호선은 마천지선을 포함해 단일 계통으로 집계된다.

그래프는 강동을 본선 노드와 지선 노드로 나누고 사이에 환승 엣지를 두었다.
이 엣지는 경로에 따라 의미가 다르다.

    천호(본선) <-> 둔촌동(마천)   같은 열차로 통과. 환승 아님
    길동(하남) <-> 둔촌동(마천)   강동에서 갈아타야 함. 환승 맞음

엣지 하나로는 구분되지 않으므로 경로의 앞뒤 노드를 보고 판정한다.
**과잉 수정 방지가 이 파일의 핵심이다.** 통과를 허용하면서도
하남↔마천 환승은 반드시 남아야 한다.

실행
----
    pytest tests/test_through_junction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app" / "streamlit"))
sys.path.insert(0, str(ROOT / "scripts"))

station_routing = pytest.importorskip("station_routing")


@pytest.fixture(scope="module")
def ctx():
    if not (ROOT / "data" / "master" / "station_display_master.csv").exists():
        pytest.skip("station_display_master.csv 없음. 12 스크립트를 먼저 실행하세요.")
    mod = station_routing.load_scorer_class(ROOT)
    rs = mod.RouteScorer(ROOT, "weekday", "08:30", None)
    disp = station_routing.load_display_master(ROOT)
    return rs, disp, mod


def rec(ctx, a, b):
    rs, disp, _ = ctx
    r = station_routing.find_route_by_station(rs, disp, a, b, "calm")
    assert r["ok"], "%s → %s 경로 실패: %s" % (a, b, r.get("reason"))
    return r["recommended"]


# ---------------------------------------------------------------- 판정 함수
def test_through_junction_is_registered(ctx):
    _, _, mod = ctx
    pair = frozenset(("5_강동", "5_강동@macheon_branch"))
    assert pair in mod.THROUGH_JUNCTIONS
    assert mod.THROUGH_JUNCTIONS[pair] == "5_천호"


def test_seongsu_and_sindorim_are_not_through(ctx):
    """2호선 지선은 별도 셔틀이다. 통과 대상에 들어가면 안 된다."""
    _, _, mod = ctx
    for pair in (("2_성수", "2_성수@seongsu_branch"),
                 ("2_신도림", "2_신도림@sinjeong_branch")):
        assert mod.through_junction_trunk(*pair) is None


# ---------------------------------------------------------------- 통과 (환승 아님)
@pytest.mark.parametrize("a,b", [
    ("광나루", "오금"),        # 본선 동쪽 -> 마천지선
    ("답십리", "올림픽공원"),   # 본선 -> 마천지선
    ("방화", "마천"),          # 전 구간 직결
    ("여의도", "거여"),
])
def test_trunk_to_macheon_is_not_a_transfer(ctx, a, b):
    """본선에서 마천 방면으로 가는 것은 같은 열차 통과다."""
    ev = rec(ctx, a, b)
    assert ev["transfer_count"] == 0, (
        "%s → %s 는 환승 0회여야 한다 (실제 %d회)" % (a, b, ev["transfer_count"]))
    lines = {sg["line"] for sg in ev["segments"] if sg["kind"] == "ride"}
    assert lines == {"5"}, "5호선 한 개 노선으로만 가야 한다 (실제 %s)" % lines


def test_through_pass_keeps_one_ride_segment(ctx):
    """통과하면 승차 구간이 끊기지 않는다."""
    ev = rec(ctx, "광나루", "오금")
    rides = [sg for sg in ev["segments"] if sg["kind"] == "ride"]
    transfers = [sg for sg in ev["segments"] if sg["kind"] == "transfer"]
    assert len(rides) == 1
    assert len(transfers) == 0


def test_junction_counts_as_intermediate_stop(ctx):
    """통과하는 분기역은 중간 정차역으로 세어진다.

    광나루-천호-강동-둔촌동-올림픽공원-방이-오금 = 엣지 6개, 중간역 5개.
    강동이 빠지면 4개가 되므로 여기서 잡힌다.
    """
    ev = rec(ctx, "광나루", "오금")
    assert ev["dwell_stop_count"] == 5
    assert "강동" in ev["segments"][0]["stations"]


# ---------------------------------------------------------------- 환승 유지 (과잉 수정 방지)
@pytest.mark.parametrize("a,b,least", [
    ("길동", "둔촌동", 1),
    ("마천", "하남검단산", 1),
    ("올림픽공원", "명일", 1),
    ("거여", "고덕", 1),
])
def test_branch_to_branch_still_transfers(ctx, a, b, least):
    """하남 방면 <-> 마천 방면은 강동에서 실제로 갈아타야 한다.

    이 테스트가 깨지면 과잉 수정이다. 통과 예외가 너무 넓게 적용된 것이다.
    """
    ev = rec(ctx, a, b)
    assert ev["transfer_count"] >= least, (
        "%s → %s 는 환승이 필요하다 (실제 %d회)" % (a, b, ev["transfer_count"]))


def test_branch_transfer_happens_at_gangdong(ctx):
    ev = rec(ctx, "길동", "둔촌동")
    ats = [sg["at"] for sg in ev["segments"] if sg["kind"] == "transfer"]
    assert "강동" in ats, "강동에서 환승해야 한다 (실제 %s)" % ats


def test_seongsu_branch_still_transfers(ctx):
    """2호선 성수지선은 여전히 환승이 필요하다."""
    ev = rec(ctx, "건대입구", "신설동")
    assert ev["transfer_count"] >= 1


# ---------------------------------------------------------------- 계산 일치
@pytest.mark.parametrize("a,b", [
    ("광나루", "오금"),
    ("방화", "마천"),
    ("길동", "둔촌동"),
])
def test_segment_sum_matches_total(ctx, a, b):
    """통과 처리 후에도 표시값 합 = 카드 총계.

    evaluate() 만 고치고 describe_path() 를 빠뜨리면 여기서 잡힌다.
    실제로 1단계 패치 직후 방화→마천이 90.6 vs 92.7 로 어긋났다.
    """
    ev = rec(ctx, a, b)
    seg_total = sum(sg.get("minutes") or 0 for sg in ev["segments"])
    wait = ev.get("transfer_wait_min") or 0
    assert seg_total + wait == pytest.approx(ev["actual_time_min"], abs=0.2)
