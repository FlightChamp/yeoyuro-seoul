"""
tests/test_counterfactual.py
============================
Counterfactual card 가 v2.3 분석 판정과 같은지 확인한다 (앱 데이터 폴더 data/app_v26 기준, 1~9호선).

1. 무작위 OD × 피크 시각에서 카드의 경로/시간 성공 표시 = classify() 의 route_ok / time_ok
2. 대표 사례: 상봉→건대입구 08:00 = 구조적 (v2.3 7호선 회랑), 노원→수유 08:00 = 여유
3. 성공한 대안은 실제로 T1 감소 기준을 넘는다
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
APP = ROOT / "data" / "app_v26"


@pytest.fixture(scope="module")
def router():
    if not (APP / "data" / "marts" / "route_edge_mart.parquet").exists():
        pytest.skip("앱 데이터 폴더 없음 (python scripts/v2/30_build_app_data.py)")
    from yeoyuro_v2 import load_v1
    from yeoyuro_v2.time_dependent import TDRouter
    mod, sr, _ = load_v1(ROOT)
    return TDRouter(APP, mod, sr, sr.load_display_master(APP), "weekday", policy="step", k=5, multi_bin=True)


def test_card_matches_classify(router):
    from yeoyuro_v2.counterfactual import counterfactual
    from yeoyuro_v2.time_dependent import parse_hhmm
    keys = router.display.station_key.tolist()
    rnd = random.Random(31)
    checked = 0
    for _ in range(12):
        o, d = rnd.sample(keys, 2)
        for t in ("08:00", "18:00"):
            c = counterfactual(router, o, d, parse_hhmm(t), with_joint=False)
            if c is None or c["type"] in ("calm", "invalid"):
                continue
            assert c["route"].ok == bool(c["classify"]["route_ok"])
            assert c["time"].ok == bool(c["classify"]["time_ok"])
            checked += 1
    assert checked >= 5


def test_reference_cases(router):
    from yeoyuro_v2.counterfactual import counterfactual
    from yeoyuro_v2.time_dependent import parse_hhmm
    s = counterfactual(router, "상봉", "건대입구", parse_hhmm("08:00"))
    assert s["type"] == "structural" and not s["route"].ok and not s["time"].ok
    assert counterfactual(router, "노원", "수유", parse_hhmm("08:00"))["type"] == "calm"


def test_successful_options_pass_threshold(router):
    from yeoyuro_v2.counterfactual import counterfactual
    from yeoyuro_v2.shift_rules import T1, reduction_ok
    from yeoyuro_v2.time_dependent import parse_hhmm
    c = counterfactual(router, "강변", "여의도", parse_hhmm("08:30"))
    b = c["now"]
    for k in ("route", "time", "joint"):
        x = c[k]
        if x.ok:
            assert reduction_ok(b.exp100, x.exp100, b.max_cong, x.max_cong, T1)
    assert c["time"].ok and abs(c["time"].shift_min) <= T1.time_shift_primary_window
