"""
tests/test_event_scenario.py
============================
v2.5 이벤트 시나리오 mart 불변식 + 엔진의 λ 배율 동작.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = ROOT / "data" / "marts" / "v2" / "event_scenario_mart.parquet"


@pytest.fixture(scope="module")
def mart():
    if not P.exists():
        pytest.skip("event_scenario_mart 없음")
    return pd.read_parquet(P)


def test_untreated_cannot_gain_exposure(mart):
    # 간접 단위의 기준 경로는 배수 구간을 지나지 않으므로, 같은 경로라면 노출이 늘 수 없다
    u = mart[~mart.treated]
    assert (u.event_exposure_100_min_l03 <= u.baseline_exposure_100_min + 1e-6).all()


def test_lambda_monotone_exposure(mart):
    t = mart[mart.treated]
    # 같은 단위에서 λ 가 커지면 기준 경로 최대 기대혼잡은 줄지 않는다 (경로가 바뀌는 경우 제외 허용 오차)
    share = (t.event_max_congestion_l06 + 1e-6 >= t.event_max_congestion_l015).mean()
    assert share >= 0.95


def test_columns_and_flags(mart):
    for c in ("baseline_type", "event_shiftability_type", "type_transition", "event_sensitive_flag",
              "threshold_set_id"):
        assert c in mart.columns
    assert set(mart.threshold_set_id) == {"T1_prereg_v23"}
    assert not (mart.event_sensitive_flag & ~mart.treated).any()


def test_event_scale_engine():
    from yeoyuro_v2 import load_v1
    from yeoyuro_v2.time_dependent import TimeDependentEvaluator, parse_hhmm
    mod, sr, disp = load_v1(ROOT)
    rs = mod.RouteScorer(ROOT, "saturday", "18:00", "2025-09-27")      # 불꽃축제
    hw = sr.load_headway(ROOT)
    res = sr.find_route_by_station(rs, disp, "공덕", "여의나루", "fast")
    path = res["fastest"]["path"]
    t = parse_hhmm("18:00")
    m = [TimeDependentEvaluator(rs, hw, "step", event_scale=s).evaluate(path, t).max_congestion
         for s in (0.0, 1.0, 2.0)]
    assert m[0] < m[1] < m[2]
    base = mod.RouteScorer(ROOT, "saturday", "18:00", None)
    m0 = TimeDependentEvaluator(base, hw, "step").evaluate(path, t).max_congestion
    assert m0 == pytest.approx(m[0])
