"""
tests/test_korail_engine.py — Phase 3A-2b 코레일 엔진 확장 (D-095 6-1절 T1·T1a·T1b·T2·T3·T5, V1 일부)

전체 V1(3,720건)·T1(24개 방향쌍 × 6개 출발)은 scripts/v2/46_check_korail_engine.py 가 계산하고
reports/v2/korail_engine_3a2b.json 에 남긴다. 여기서는 빠른 대표 검사 + 그 결과 파일을 확인한다.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA9 = ROOT / "data" / "app_v26"
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet").exists()
    or not (ROOT / "data" / "marts" / "congestion_edge_lookup.parquet").exists(), reason="mart 없음")

from yeoyuro_v2 import korail_engine as KE        # noqa: E402
from yeoyuro_v2 import korail_line1 as K          # noqa: E402
from yeoyuro_v2 import time_dependent as td       # noqa: E402
from yeoyuro_v2 import train_timetable as T       # noqa: E402
from yeoyuro_v2 import load_v1                    # noqa: E402


@pytest.fixture(scope="module")
def v1():
    return load_v1(ROOT)


@pytest.fixture(scope="module")
def on9():
    if not (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists():
        pytest.skip("앱 데이터 없음")
    sys.path[:0] = [str(ROOT / "app" / "streamlit")]
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    r = KE.KorailTDRouter(DATA9, mod, sr, sr.load_display_master(DATA9), "weekday", include_korail=True)
    return r, sr


def test_off_is_plain_tdrouter(v1):
    """V1(일부)·T5: 끈 라우터 = 기존 TDRouter, 켰다 끈 뒤에도 원본 그래프·결과 불변."""
    mod, sr, disp = v1
    th0, disp0 = dict(mod.THROUGH_JUNCTIONS), disp.copy()
    od = [("서울역", "강남"), ("응암", "고속터미널"), ("방화", "마천"), ("까치산", "성수")]
    plain = td.TDRouter(ROOT, mod, sr, disp, "weekday")
    base_adj = {u: [dict(e) for e in l] for u, l in plain.scorer(td.bin_index_of(480)).adj.items()}
    want = [(tuple(x.path), x.actual_time_min) for o, d in od for x in plain.evaluate_od(o, d, 480)[:1]]
    on = KE.KorailTDRouter(ROOT, mod, sr, disp, "weekday", include_korail=True)
    on.evaluate_od("신도림", "구일", 480)
    off = KE.KorailTDRouter(ROOT, mod, sr, disp, "weekday", include_korail=False)
    got = [(tuple(x.path), x.actual_time_min) for o, d in od for x in off.evaluate_od(o, d, 480)[:1]]
    assert got == want
    assert plain.scorer(td.bin_index_of(480)).adj == base_adj
    assert on._base_scorers[td.bin_index_of(480)].adj.keys() == base_adj.keys()
    assert "1_남영" not in on._base_scorers[td.bin_index_of(480)].nodes
    assert dict(mod.THROUGH_JUNCTIONS) == th0 and disp.equals(disp0)


def test_guro_transfer_between_branches(on9):
    """T1: 구일→가산디지털단지·독산→개봉 = 구로 갈아타기 1번·보행 1분·다른 열차."""
    r, sr = on9
    for o, d in [("구일", "가산디지털단지"), ("독산", "개봉")]:
        q = r.evaluate_od(o, d, 480)[0]
        pairs = [{a, b} for a, b in zip(q.path, q.path[1:])]
        assert pairs.count(set(KE.GURO_NODES)) == 1
        assert q.transfer_count == 1 and math.isclose(q.transfer_walk_min, 1.0)
        ev = r.evaluator(td.bin_index_of(480))
        c = T.verify_connection(r.kidx, T.legs_from_path(ev, q.path), 480, "weekday")
        assert c.status == T.CONNECTED and c.rides[0].train != c.rides[1].train
        assert c.rides[1].dep_min >= c.rides[0].arr_min + 1.0 - 1e-9


def test_guro_revisit_detour_blocked_twice(on9):
    """T1a: 신도림에서 방향을 바꿔 구로를 다시 지나는 우회 = 재방문 규칙 + 열차 단위 검증 두 겹."""
    r, sr = on9
    detour = ["1_구일", KE.GURO_GI, "1_신도림", KE.GURO_GB, "1_가산디지털단지"]
    assert sr.has_station_revisit(detour)
    assert all(p != detour for p in r.candidates("구일", "가산디지털단지", 480))
    ev = r.evaluator(td.bin_index_of(480))
    assert T.verify_connection(r.kidx, T.legs_from_path(ev, detour), 480, "weekday").status == T.IMPOSSIBLE


def test_guro_as_origin_and_destination(on9):
    """T1b: 구로 두 노드가 모두 후보, 각 방면 무환승."""
    r, sr = on9
    assert set(KE.GURO_NODES) <= set(sr.candidates_of(r.display, "구로"))
    for o, d, node in [("구로", "구일", KE.GURO_GI), ("구로", "가산디지털단지", KE.GURO_GB),
                       ("구일", "구로", KE.GURO_GI), ("가산디지털단지", "구로", KE.GURO_GB)]:
        q = r.evaluate_od(o, d, 480)[0]
        assert q.transfer_count == 0 and node in q.path


def test_through_trains_have_no_transfer(on9):
    """T2: 직통 열차 OD 는 갈아타기·보행 0 (서울역·청량리·구로 통과)."""
    r, _ = on9
    for o, d in [("신도림", "구일"), ("시청", "남영"), ("종로3가", "회기"), ("온수", "서울역")]:
        q = r.evaluate_od(o, d, 480)[0]
        assert q.transfer_count == 0 and q.transfer_walk_min == 0, (o, d, q.path)


def test_E_depends_on_alight_station(on9):
    """T3: 같은 승차역·시각, 다른 하차역 → 그 하차역에 서는 열차만으로 E."""
    r, _ = on9
    ev = r.evaluator(td.bin_index_of(480))
    e_kwangun = ev.korail_wait({"codes": [1018, 1019], "nodes": ["a", "b"]}, 480.0).E
    e_wolgye = ev.korail_wait({"codes": [1018, 1019, 1020], "nodes": ["a", "b"]}, 480.0).E
    n_k = sum(480 <= x < 510 for x in r.kidx.pair("weekday", "1", 1018, 1019, (1018, 1019))[0])
    n_w = sum(480 <= x < 510 for x in r.kidx.pair("weekday", "1", 1018, 1020, (1018, 1019, 1020))[0])
    assert n_w < n_k and e_wolgye > e_kwangun            # 광운대 종착 열차는 월계 E 에서 빠짐


def test_korail_congestion_stays_missing(on9):
    """E8: 코레일 간선 혼잡 = 결측 (v1 채우기 값 지움), 판정 = 혼잡 판정 제외."""
    r, _ = on9
    ev = r.evaluator(td.bin_index_of(480))
    assert all(math.isnan(ev.cong_med[(u, v, 5)]) for (u, v) in list(ev.korail_edges)[:20])
    q = r.evaluate_od("회기", "가산디지털단지", 480)[0]
    assert q.korail and q.congestion_judgement == KE.JUDGE_EXCLUDED and q.korail_ride_min > 0
    assert not math.isnan(q.perceived_time_min)


def test_cost_search_excludes_korail(on9):
    """E8: 혼잡 기반 탐색(Yen cost)은 코레일 뺀 원래 그래프 — 코레일 후보는 소요시간 탐색에서만."""
    r, _ = on9
    s = r.scorer(td.bin_index_of(480))
    assert "1_남영" in s.nodes and "1_남영" not in s._korail_base.nodes


def test_guro_terminating_split_only_in_connection(on9):
    """E3a: 구로 종착 → 구로 시발 이어 타기는 연결 검증에서만, 같은 방면 안에서만."""
    r, _ = on9
    leg = T.Leg("1", 1007, 1813, "1_신도림", "1_구일", 0.0, (1007, 1701, 1813))
    for t in range(300, 1500, 7):
        rs, why = T._ride_leg(r.kidx, "weekday", leg, float(t), 60)
        if why == "guro_split":
            assert rs[0].alight_code == 1701 and rs[1].board_code == 1701
            assert rs[1].dep_min >= rs[0].arr_min + 1.0 - 1e-9
    cross = T.Leg("1", 1813, 1702, "1_구일", "1_가산디지털단지", 0.0, (1813, 1701, 1702))
    assert T._ride_leg(r.kidx, "weekday", cross, 480.0, 60)[1] == T.IMPOSSIBLE


def test_report_numbers():
    p = ROOT / "reports" / "v2" / "korail_engine_3a2b.json"
    if not p.exists():
        pytest.skip("46 결과 없음")
    res = json.loads(p.read_text(encoding="utf-8"))
    assert res["V1"]["mismatch"] == 0 and res["V1"]["cases"] == 3720
    assert all(res["T5"].values())
    t1 = res["T"]["T1"]
    assert t1["impossible"] == 0 and t1["revisit"] == 0 and t1["direct_guro_ok"] == t1["direct_guro"] > 0
    assert res["T"]["T1a"]["train_check"] == T.IMPOSSIBLE and not res["T"]["T1a"]["in_any_candidate"]


def test_meaningless_guro_transfer_is_not_a_candidate(on9):
    """구현 중 발견(D-096): 지선 → 구로 → 다른 방면 노드 → 신도림 은 같은 열차 재탑승(중복 환승) → 후보 아님."""
    assert KE.valid_guro_transfer(["1_구일", KE.GURO_GI, KE.GURO_GB, "1_가산디지털단지"])
    assert not KE.valid_guro_transfer(["1_구일", KE.GURO_GI, KE.GURO_GB, "1_신도림"])
    assert not KE.valid_guro_transfer(["1_신도림", KE.GURO_GB, KE.GURO_GI, "1_구일"])     # 종착→시발은 연결 검증에서만
    r, _ = on9
    for o, d in [("구일", "독산"), ("개봉", "금천구청"), ("오류동", "신도림")]:
        for p in r.candidates(o, d, 480):
            assert KE.valid_guro_transfer(p), p
