"""
tests/test_train_timetable.py — Phase 3A-2a 열차 단위 시각표·시각표 연결 검증 (D-093 2-1절, D-094)

| 검사 | 내용 |
|---|---|
| 정제 | 묶음 1·2~8·9L·9X, 1호선 급행 없음, 음수 정차 없음, 열차 안 시각 역행 없음 |
| 3A-1 대조 | 1호선 코레일 서울 구간 + 서울역~청량리 행이 3A-1 mart 와 같음 |
| 방향 | 2호선 신촌→잠실은 시청 쪽으로 가는 열차만 (반대 방향으로 한 바퀴 도는 열차 배제) |
| 순환 경계 | 2호선 성수·6호선 응암에서 기록이 끊겨도 이어 탐 (보행 0) |
| 사유 | 첫차 전·막차 뒤·60분 초과 구분 (R3) |
| 불가능한 연결 | 경인↔경부 남쪽을 갈아타기 없이 가는 구간 = 불가능 (R6 오류 기준) |
| 엔진 후보 | 앱 데이터 후보 경로를 따라가면 불가능한 연결 0 |
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2 import korail_line1 as K          # noqa: E402
from yeoyuro_v2 import train_timetable as T       # noqa: E402

MART = ROOT / "data" / "marts" / "v2" / "train_timetable"
pytestmark = pytest.mark.skipif(not (MART / "stop_times.parquet").exists(), reason="열차 단위 시각표 mart 없음")


@pytest.fixture(scope="module")
def st():
    return pd.read_parquet(MART / "stop_times.parquet")


@pytest.fixture(scope="module")
def idx(st):
    return T.TrainIndex(st)


def test_clean_mart(st):
    assert set(st.group) == {"1", "2", "3", "4", "5", "6", "7", "8", "9L", "9X"}
    assert set(st.day_type) == {"weekday", "saturday", "sunday"}
    assert ((st.dep_min - st.arr_min).dropna() >= 0).all()
    s = st.sort_values(["day_type", "group", "train", "direction", "stop_seq"])
    nxt = s.groupby(["day_type", "group", "train", "direction"]).arr_min.shift(-1)
    assert not (nxt < s.dep_min - 1e-9).any()
    a = json.loads((MART / "audit.json").read_text(encoding="utf-8"))
    assert a["clean"]["rows_kept"] == len(st)
    assert len(a["clean"]["negative_dwell_rows"]) <= T.NEG_DWELL_MAX


def test_line1_matches_3a1_mart(st):
    km = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "korail_line1" / "stop_times.parquet")
    s1 = st[(st.group == "1") & st.code.isin(list(K.SCOPE))]
    key = ["day_type", "train", "direction", "code"]
    m = s1[key + ["arr_min", "dep_min"]].merge(km[key + ["arr_min", "dep_min"]], on=key, how="outer",
                                                indicator=True, suffixes=("", "_k"))
    assert (m._merge == "both").all()
    assert np.allclose(m.arr_min.fillna(-1), m.arr_min_k.fillna(-1))
    assert np.allclose(m.dep_min.fillna(-1), m.dep_min_k.fillna(-1))


def _l2(*codes):
    return tuple(codes)


def test_direction_is_respected_on_circle_line(idx):
    # 신촌(240) → 이대(241) … 시청(201) … 잠실(216): 시청 쪽 순서 전체를 via 로 줌
    via = (240, 241, 242, 243, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216)
    rs, why = T._ride_leg(idx, "weekday", T.Leg("2", 240, 216, "2_신촌", "2_잠실", 0.0, via), 480, 60)
    assert why is None and rs
    assert rs[-1].arr_min - rs[0].dep_min < 50           # 반대 방향 한 바퀴(약 60분+)가 아님
    assert all(r.direction == rs[0].direction for r in rs)


def test_loop_record_boundaries(idx):
    # 2호선 동대문역사문화공원(205) → 잠실(216): 성수(211)에서 기록이 끊김 → 두 기록 이어 탐
    via = tuple(range(205, 217))
    rs, why = T._ride_leg(idx, "weekday", T.Leg("2", 205, 216, "a", "b", 0.0, via), 480, 60)
    assert why is None and len(rs) in (1, 2)
    if len(rs) == 2:
        assert rs[1].board_code == 211 and rs[1].dep_min >= rs[0].arr_min - 1e-9
    # 6호선 새절(2617) → 응암(2611) → 역촌(2612) → 불광(2613)
    rs, why = T._ride_leg(idx, "weekday", T.Leg("6", 2617, 2613, "a", "b", 0.0, (2617, 2611, 2612, 2613)), 480, 60)
    assert why is None and rs and rs[-1].alight_code == 2613


def test_no_service_reasons(idx):
    via = (150, 151)
    r, why = idx.next_ride("weekday", "1", 150, 151, 100.0, via=via)          # 01:40 — 첫차 전
    assert r is None and why == K.NO_SERVICE_BEFORE_FIRST
    r, why = idx.next_ride("weekday", "1", 150, 151, 1700.0, via=via)         # 28:20 — 막차 뒤
    assert r is None and why == K.NO_SERVICE_AFTER_LAST
    r, why = idx.next_ride("weekday", "1", 150, 151, 480.0, via=via)
    assert r is not None and 0 <= r.wait_min <= 15 and r.arr_min > r.dep_min


def test_impossible_connection_detected(idx):
    # 구일(1813) → 구로(1701) → 가산디지털단지(1702) 를 한 열차로: 경인↔경부 직통 없음 → 불가능
    leg = T.Leg("1", 1813, 1702, "1_구일", "1_가산디지털단지", 0.0, (1813, 1701, 1702))
    assert T.impossible_legs(idx, [leg], "weekday") == [leg]
    c = T.verify_connection(idx, [leg], 480, "weekday")
    assert c.status == T.IMPOSSIBLE
    # 구로에서 1분 걸어 갈아타면 (D-093) 연결됨
    legs = [T.Leg("1", 1813, 1701, "1_구일", "1_구로", 0.0, (1813, 1701)),
            T.Leg("1", 1701, 1702, "1_구로", "1_가산디지털단지", 1.0, (1701, 1702))]
    c = T.verify_connection(idx, legs, 480, "weekday")
    assert c.status == T.CONNECTED and c.rides[1].dep_min >= c.rides[0].arr_min + 1.0 - 1e-9
    assert c.rides[0].train != c.rides[1].train


def test_engine_candidates_have_no_impossible_connection(st, idx):
    data = ROOT / "data" / "app_v26"
    if not (data / "data" / "marts" / "route_edge_mart.parquet").exists():
        pytest.skip("앱 데이터 없음")
    sys.path[:0] = [str(ROOT / "app" / "streamlit")]
    import station_routing as sr
    from yeoyuro_v2 import time_dependent as td
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = sr.load_display_master(data)
    r = td.TDRouter(data, mod, sr, disp, day_type="weekday")
    ev = r.evaluator(td.bin_index_of(480))
    names = sorted(disp.station_name.unique())
    rng = random.Random(7)
    n = 0
    for _ in range(40):
        o, d = rng.sample(names, 2)
        for p in r.candidates(o, d, 480):
            legs = T.legs_from_path(ev, p)
            assert T.impossible_legs(idx, legs, "weekday") == [], (o, d, p)
            c = T.verify_connection(idx, legs, 480, "weekday")
            assert c.status == T.CONNECTED, (o, d, c.reason)
            n += 1
    assert n > 100
    # 신정지선 까치산: 그래프 코드 260 → 시각표 200
    res = r.evaluate_od("까치산", "신도림", 480)
    legs = T.legs_from_path(ev, res[0].path)
    assert legs[0].board_code == 200


def test_report_matches_audit():
    a = json.loads((MART / "audit.json").read_text(encoding="utf-8"))
    rep = (ROOT / "reports" / "v2" / "train_timetable_3a2a.md").read_text(encoding="utf-8")
    assert f"{a['clean']['rows_kept']:,}" in rep
    for e in a.get("engine", []):
        assert e["impossible"] == 0
