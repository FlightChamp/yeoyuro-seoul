"""Facility PoC (엘리베이터 칸-문 × 칸별 혼잡) 순수 로직 테스트 — 합성 데이터만 쓴다."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yeoyuro_v2.facility_match import elevator_cars, load_fast_exit, match, parse_position, survey   # noqa: E402

VEC = [90, 95, 100, 105, 110, 119, 121.5, 111.4, 100, 95]


def _fe(tmp_path):
    rows = [
        {"lineNm": "4호선", "stnNm": "동대문역사문화공원", "upbdnbSe": "하행", "drtnInfo": "충무로",
         "qckgffVhclDoorNo": "7-1", "plfmCmgFac": "엘리베이터", "elvtrNo": "0001-1",
         "facPstnNm": "동대문(4) 방면4-4, 충무로 방면7-1", "crtrYmd": "20241231"},
        {"lineNm": "4호선", "stnNm": "동대문역사문화공원", "upbdnbSe": "상행", "drtnInfo": "동대문",
         "qckgffVhclDoorNo": "4-4", "plfmCmgFac": "엘리베이터", "elvtrNo": "0001-1",
         "facPstnNm": "동대문(4) 방면4-4, 충무로 방면7-1", "crtrYmd": "20241231"},
        {"lineNm": "4호선", "stnNm": "동대문역사문화공원", "upbdnbSe": "하행", "drtnInfo": "충무로",
         "qckgffVhclDoorNo": "2-1", "plfmCmgFac": "계단", "elvtrNo": "", "facPstnNm": "", "crtrYmd": "20241231"},
        {"lineNm": "4호선", "stnNm": "창동", "upbdnbSe": "상행", "drtnInfo": "노원",
         "qckgffVhclDoorNo": "NA-NA", "plfmCmgFac": "계단", "elvtrNo": "", "facPstnNm": "", "crtrYmd": "20241231"},
    ]
    p = tmp_path / "fe.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    return load_fast_exit(p)


def test_load_and_direction(tmp_path):
    fe = _fe(tmp_path)
    assert len(fe) == 3                                              # 'NA-NA' 행은 버린다
    e = elevator_cars(fe, "동대문역사문화공원", "동대문")              # 동대문에서 와서 충무로 방면으로 가는 열차
    assert [x["car"] for x in e] == [7] and e[0]["toward"] == "충무로"
    e2 = elevator_cars(fe, "동대문역사문화공원", "충무로")             # 반대 방향
    assert [x["car"] for x in e2] == [4]


def test_match_trade_off():
    m = match(VEC, 7)
    assert m["nearest"] == 7 and m["pick"] == 8 and m["extra_cars"] == 1
    assert m["gain"] == 10.1 and m["all_crowded"] is True            # 후보 6~8칸 모두 100% 이상
    assert match(VEC, 7, mirrored=True)["candidates"] == {6: 110.0, 7: 105.0, 8: 100.0}


def test_match_keeps_elevator_car_when_gain_small():
    v = [80] * 10
    v[5] = 78                                                        # 옆 칸이 2%p 덜 붐빔 → 기준(5%p) 미만
    m = match(v, 7)
    assert m["pick"] == 7 and m["extra_cars"] == 0 and m["gain"] == 0


def test_match_edge_car():
    m = match(VEC, 1)
    assert set(m["candidates"]) == {1, 2}


def test_parse_position_mirrored_numbering():
    q = parse_position("회현 방면9-1, 숙대입구 방면2-4")
    assert q == [("회현", 9, 1), ("숙대입구", 2, 4)]
    assert q[0][1] + q[1][1] == 11 and q[0][2] + q[1][2] == 5        # 10칸 편성: 진행 방향 기준 번호


def test_survey(tmp_path):
    fe = _fe(tmp_path)
    pat = pd.DataFrame([{"line": "4", "from_station": "동대문", "to_station": "동대문역사문화공원", "time_bin": "07:30",
                         **{f"car_{i}": v for i, v in enumerate(VEC, 1)}, "n_passages": 6, "n_days": 3,
                         "train_mean": sum(VEC) / 10}])
    s = survey(fe, pat)
    assert len(s) == 1 and s.iloc[0].pick == 8 and s.iloc[0].mirror_pick == 8
