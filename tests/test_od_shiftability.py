"""
tests/test_od_shiftability.py
=============================
v2.3 분류기(classify) 단위 테스트. 엔진 없이 합성 evidence 로 규칙만 검사한다.
사전 등록(docs/v2/preregistration_v23.md §2·§3)의 규칙이 코드와 일치하는지 확인하는 용도.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2.od_shiftability import classify  # noqa: E402
from yeoyuro_v2.shift_rules import T0, T1, reduction_ok  # noqa: E402


def m(actual=30.0, perceived=35.0, transfers=1, exp=10.0, exp90=12.0, mx=120.0, **kw):
    return {"actual": actual, "perceived": perceived, "transfers": transfers, "exp100": exp,
            "exp100_p90": exp90, "exp130": 0.0, "max": mx, "arrive": 540.0, **kw}


def ev(base=None, route=(), time=()):
    return {"base": base or m(), "route": list(route), "time": list(time)}


def test_t1_values_match_preregistration():
    assert T1.set_id == "T1_prereg_v23"
    assert (T1.min_exposure_drop_min, T1.min_exposure_drop_rel, T1.use_max_cong_drop) == (5.0, 0.5, False)
    assert T1.time_shift_primary_window == 60 and T1.time_shift_max_time_increase_min == 5.0
    assert (T1.max_time_loss_min, T1.max_perceived_excess_min, T1.max_extra_transfer,
            T1.max_edge_jaccard, T1.calm_gate_exposure_min) == (15.0, 5.0, 1, 0.65, 3.0)


def test_reduction_rules():
    assert reduction_ok(10, 5, 120, 120, T1)          # 절대 5분
    assert reduction_ok(4, 2, 120, 120, T1)           # 상대 50%
    assert not reduction_ok(10, 6, 140, 100, T1)      # 최대혼잡 감소는 T1 에서 무시
    assert reduction_ok(10, 6, 140, 100, T0)          # T0 에서는 인정


def test_calm():
    assert classify(ev(m(exp=0.0, mx=90.0)), T1)["type"] == "calm"


def test_route_only():
    r = m(actual=40.0, perceived=38.0, exp=2.0, mx=95.0, jac=0.3)
    out = classify(ev(route=[r]), T1)
    assert out["type"] == "route_shiftable" and out["route_ok"] and not out["time_ok"]


def test_time_window_primary_only():
    t90 = m(actual=30.0, exp=0.0, mx=80.0, shift=90, same=True, jac=1.0)
    assert classify(ev(time=[t90]), T1)["type"] == "structural"           # ±90 은 주 창 밖
    assert classify(ev(time=[t90]), replace(T1, time_shift_primary_window=90))["type"] == "time_shiftable"


def test_dual_and_bindings():
    r = m(actual=40.0, perceived=38.0, exp=2.0, mx=95.0, jac=0.3)
    t = m(actual=31.0, exp=1.0, mx=90.0, shift=30, same=True, jac=1.0)
    assert classify(ev(route=[r], time=[t]), T1)["type"] == "dual"
    r_bad = m(actual=50.0, perceived=38.0, exp=0.0, mx=80.0, jac=0.3)    # 시간손실 20분
    t_bad = m(actual=40.0, exp=0.0, mx=80.0, shift=30, same=True, jac=1.0)  # 소요 +10분
    out = classify(ev(route=[r_bad], time=[t_bad]), T1)
    assert out["type"] == "structural"
    assert out["route_bind"] == "time_loss" and out["time_bind"] == "time_increase"


def test_similar_route_rule():
    t = m(actual=30.0, exp=0.0, mx=80.0, shift=30, same=False, jac=0.5)
    out = classify(ev(time=[t]), T1)
    assert out["type"] == "structural" and out["time_bind"] == "route_changed"


def test_out_of_window_base_is_invalid():
    # 기준 경로가 01:00 이후 도착 → 자료 범위 밖 → 분석 제외 (D-036)
    assert classify(ev(m(arrive=1510.0)), T1)["type"] == "invalid"


def test_out_of_window_candidate_is_ignored():
    good_but_oow = m(actual=31.0, exp=0.0, mx=80.0, shift=60, same=True, jac=1.0, arrive=1505.0)
    out = classify(ev(time=[good_but_oow]), T1)
    assert out["type"] == "structural" and out["time_bind"] == "out_of_window"
