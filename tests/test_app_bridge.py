"""
tests/test_app_bridge.py
========================
v2.1 앱 브리지 regression test. Streamlit 없이 검사한다.

1. retime 후 카드 값(소요시간)은 엔진 결과와 같아야 한다.
2. 소요시간 분해(승차+정차+도보+대기) 합계 = 카드 소요시간 (화면 타임라인과 카드 일치).
3. 대안이 있으면 v1 채택 조건을 v2 값으로 만족해야 한다.
4. 시간대 대안은 ±120분 안이고 최대혼잡 개선이 10%p 이상이어야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import load_v1  # noqa: E402
from yeoyuro_v2.app_bridge import retime_result  # noqa: E402
from yeoyuro_v2.time_dependent import TimeDependentEvaluator, parse_hhmm  # noqa: E402

CASES = [("노원", "양천구청", "17:40"), ("서울역", "강남", "08:20"),
         ("신촌", "잠실", "08:30"), ("응암", "고속터미널", "08:00")]


@pytest.fixture(scope="module")
def ctx():
    if not (ROOT / "data" / "marts" / "congestion_edge_lookup.parquet").exists():
        pytest.skip("runtime mart 없음")
    mod, sr, disp = load_v1(ROOT)
    return mod, sr, disp, sr.load_headway(ROOT)


@pytest.mark.parametrize("o,d,dep", CASES)
def test_retime_consistency(ctx, o, d, dep):
    mod, sr, disp, hw = ctx
    rs = mod.RouteScorer(ROOT, "weekday", dep)
    res = sr.find_route_by_station(rs, disp, o, d, "calm")
    ev = TimeDependentEvaluator(rs, hw, "step")
    t = parse_hhmm(dep)
    out = retime_result(res, lambda b: ev, t, "calm")
    assert out["engine"] == "v2_time_dependent"
    for c in out["candidates"]:
        r = ev.evaluate(c["path"], t)
        assert c["actual_time_min"] == pytest.approx(round(r.actual_time_min, 1))
        assert c["exposure_100_min"] == pytest.approx(round(r.exposure[100], 1))
        # 화면 분해: ride_time_min(v1, 대기 제외) + v2 대기 = v2 소요
        assert c["ride_time_min"] + c["transfer_wait_min"] == pytest.approx(c["actual_time_min"], abs=0.15)
    alt = out["alternative"]
    if alt:
        f = out["fastest"]
        assert alt["actual_time_min"] - f["actual_time_min"] <= 15.0 + 1e-9
        assert f["max_congestion"] - alt["max_congestion"] >= 15.0 - 1e-9
        assert alt["perceived_time_min"] - f["perceived_time_min"] <= 5.0 + 1e-9
    ta = out["time_alternative"]
    if ta:
        assert alt is None
        assert abs(ta["best_depart_min"] - t) <= 120
        assert ta["gain_pp"] >= 10.0
