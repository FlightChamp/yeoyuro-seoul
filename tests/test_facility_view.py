"""
tests/test_facility_view.py
===========================
하차 칸 화면 문구(yeoyuro_v2/facility_view)의 모든 상태 회귀 테스트.

배경 (2026-10-17): 역사 안내 '왜 이 칸인가요?' 영역이 '평소 붐비지 않음(quiet)' 상태에서 crowd["nearest"] 를 읽다가
KeyError 로 멈췄다. 4호선 엘리베이터 역 도착 × 시간대 1,367개 조합 중 1,243개(91%)가 이 상태였는데,
기존 테스트는 붐비는 예시 하나만 봐서 놓쳤다. 그래서 여기서는 실제 자료의 모든 조합을 훑는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.facility_match import alight_crowd_match, alight_elevators   # noqa: E402
from yeoyuro_v2.facility_view import decision, evidence, state_of            # noqa: E402

MARTS = ROOT / "data" / "app_v26" / "data" / "marts"
ELEV = [{"car": 7, "door": 1, "label": "7-1", "toward": "충무로", "crtr_ymd": "20241231"}]


def _pattern(train_mean):
    row = {"line": "4", "from_station": "동대문", "to_station": "동대문역사문화공원", "time_bin": "07:30",
           "train_mean": train_mean, "n_passages": 6, "n_days": 4}
    row.update({f"car_{i}": v for i, v in enumerate([90, 95, 100, 105, 110, 119, 121.5, 111.4, 100, 95], 1)})
    return pd.DataFrame([row])


@pytest.mark.parametrize("tm,expect", [(110.0, "compare"), (40.0, "quiet")])
def test_each_state_renders_without_keyerror(tm, expect):
    c = alight_crowd_match(_pattern(tm), "4", "동대문", "동대문역사문화공원", None, ELEV, time_bin="07:30")
    assert state_of(c) == expect
    d = decision("동대문역사문화공원", "4", ELEV, c)
    ev = evidence(c, d)
    assert ev and ev["expander"].startswith("왜 ")
    if expect == "quiet":
        assert d["pick"] is None and "근처에서 내리세요" in d["title"] and ev["help"] is None
    else:
        assert d["pick"] == 8 and ev["help"] and "칸 번호 기준" in ev["help"]


def test_no_sample_none_public_and_other_line():
    ns = alight_crowd_match(_pattern(110.0), "4", "동대문", "동대문역사문화공원", None, ELEV, time_bin="08:00")
    for c, line, pub in [(ns, "4", None), (None, "4", "공개판"), (None, "6", None), ({}, "4", None)]:
        d = decision("역", line, ELEV, c, pub)
        assert evidence(c, d) is None and d["pct"] is None and d["title"]


@pytest.mark.skipif(not ((MARTS / "fast_exit_elevator.parquet").exists() and (MARTS / "car_pattern_sample.parquet").exists()
                         and (MARTS / "route_edge_mart.parquet").exists()), reason="엘리베이터 표·칸별 샘플 없음")
def test_every_line4_arrival_and_time_bin():
    """실제 자료의 모든 '이전역 → 엘리베이터 역' × 30분 조합에서 결론·근거 문구가 오류 없이 만들어지고 숫자가 맞는다."""
    mart = pd.read_parquet(MARTS / "fast_exit_elevator.parquet")
    pat = pd.read_parquet(MARTS / "car_pattern_sample.parquet")
    e = pd.read_parquet(MARTS / "route_edge_mart.parquet")
    e = e[(e.edge_type == "ride") & (e.line_id.astype(str) == "4")].drop_duplicates(["from_station", "to_station"])
    seen = {"compare": 0, "quiet": 0}
    for r in e.itertuples():
        els = alight_elevators(mart, "4", r.to_station, r.from_station)
        if not els:
            continue
        for tb in pat[(pat.from_station == r.from_station) & (pat.to_station == r.to_station)].time_bin.unique():
            c = alight_crowd_match(pat, "4", r.from_station, r.to_station, None, els, time_bin=tb)
            s = state_of(c)
            d = decision(r.to_station, "4", els, c)
            ev = evidence(c, d)
            assert s in seen and ev is not None, (r.from_station, r.to_station, tb, s)
            seen[s] += 1
            assert c["near_pct"] - c["pick_pct"] == c["gain_pct"]
            if s == "compare" and d["pick"]:
                assert "(−%d%%p)" % c["gain_pct"] in d["lines"][0]
    assert seen["quiet"] > 0 and seen["compare"] > 0
