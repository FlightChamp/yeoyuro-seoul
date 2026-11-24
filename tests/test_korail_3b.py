"""
tests/test_korail_3b.py — Phase 3B-1 정차표 (D-105 3절·4-3 C1).

원본·상세 정차표는 저장소에 없다 (P4). 그래서
- 파서·자정 펼치기·C1 규칙은 작은 가짜 xlsx·가짜 표로 어디서나 검사하고,
- 실제 정차표 검사는 사용자 PC 에서 52 를 돌려 data/marts/v2/korail_3b 가 있을 때만 한다 (없으면 건너뜀).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from yeoyuro_v2 import korail_3b as K

ROOT = Path(__file__).resolve().parents[1]
MART = ROOT / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet"
REPORT = ROOT / "reports" / "v2" / "korail_3b_build.json"


def _fake_workbook(path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "경의중앙선 하행"
    ws.append(["열차번호", "K1", "K2", "K3"])
    ws.append(["시발역", "문산", "용산", "용산"])
    ws.append(["종착역", "용문", "망우", "용산"])
    ws.append(["용산", "23:50", None, None])       # 도착 행
    ws.append([None, "23:51", "05:10", "06:00"])   # 출발 행
    ws.append(["이촌", "23:53", "05:12", None])
    ws.append([None, "23:53:30", "05:12:30", None])
    ws.append(["옥수", "00:01", "05:20", None])
    ws.append([None, "00:01:30", None, None])
    ws.append(["비고", "", "", ""])
    wb.save(path)


def test_parse_and_unfold(tmp_path):
    p = tmp_path / "x.xlsx"
    _fake_workbook(p)
    df = K.parse_workbook(p)
    assert set(df.train) == {"K1", "K2", "K3"}
    k1 = df[df.train == "K1"]
    assert list(k1.station) == ["용산", "이촌", "옥수"] and not k1.passing.any()
    times = [x for r in k1.itertuples() for x in (r.arr, r.dep)]
    u = [x for x in K._unfold(times) if x is not None]
    assert u[-1] == pytest.approx(24 * 60 + 1.5)          # 자정 넘김 → 24:01:30
    assert all(b >= a for a, b in zip(u, u[1:]))
    assert K._tmin("---") == (None, True)
    with pytest.raises(ValueError):
        K._tmin("오전")


def test_codes_and_order():
    assert K.CODE["왕십리"] == 1013 and K.CODE["양원"] == 1204 and K.CODE["야탑"] == 1854
    assert K.AR_CODE["마곡나루"] == ("A042", 20420)
    assert set(K.ORDER["GC"]) <= set(K.ORDER["GJ"])                  # 경춘선 공용 구간은 경의중앙선 역
    assert len(K.ORDER["GJ"]) == 20 and len(K.ORDER["SB"]) == 17 and len(K.ORDER["AR"]) == 6
    assert sum(K.EXPECTED_T4.values()) == 89 + 90 + 72 + 73 + 86 + 85 + 72 + 72 + 60 + 61 + 43 + 43 + 168 * 2 + 130 * 2 + 366 + 318


def test_calibrate_rule():
    val = pd.DataFrame(dict(set=["3B"] * 3 + ["1-8"] * 3, observed=[1.6, 3.2, 4.8, 1.6, 3.2, 4.8], official=[1, 2, 3, 1, 2, 3]))
    c = K.calibrate(val)
    assert c["best_by_error"] == 1.60 and c["chosen"] == 1.60          # 1.81 과 차이 ≥ 0.10분
    val2 = val.assign(observed=val.official * 1.75)
    assert K.calibrate(val2)["chosen"] == 1.81                         # 차이 < 0.10분이면 1.81 유지


def test_validation_3b_set():
    ev = pd.read_csv(ROOT / "data" / "master" / "transfer_3b_evidence.csv", dtype={"line_a": str, "line_b": str})
    v = K.validation_3b(ev)
    assert len(v) == 12                                                # A 9쌍 + U 3쌍 (사전 등록 4-3 C1)
    assert not ((v.station == "왕십리") & (v.observed > 8)).any()     # 왕십리 수인분당 원천값 8.6·8.2 는 쓰지 않음


@pytest.mark.skipif(not MART.exists(), reason="3-B 상세 정차표 없음 (사용자 PC 에서 52 실행 뒤 검사, P4)")
def test_mart():
    s = pd.read_parquet(MART)
    assert list(s.columns) == ["day_type", "group", "line", "train", "direction", "stop_seq", "code", "arr_min", "dep_min"]
    assert set(s.day_type) == {"weekday", "saturday", "sunday"} and set(s.group) == {"GJ", "GC", "SB", "AR"}
    t = pd.concat([s.arr_min, s.dep_min]).dropna()
    assert t.min() >= K.SERVICE_START and t.max() <= K.SERVICE_END
    sat, sun = (s[s.day_type == d].drop(columns="day_type").reset_index(drop=True) for d in ("saturday", "sunday"))
    pd.testing.assert_frame_equal(sat, sun)                            # 토·일 = 같은 휴일 시각표
    first = s[s.stop_seq == 0]
    assert first.arr_min.isna().all() and first.dep_min.notna().all()


@pytest.mark.skipif(not REPORT.exists(), reason="3B-1 보고서 없음")
def test_report():
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    assert r["check"]["t4_mismatch"] == {} and r["check"]["t5_gj_weekday_gt_holiday"][2]
    assert r["c1"]["chosen"] in K.FACTORS
    assert set(r["sources"]) >= {n for n, _ in K.SOURCES.values()}
