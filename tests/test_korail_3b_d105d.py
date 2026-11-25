"""
tests/test_korail_3b_d105d.py — D-105d (사후 수정, 3B-4 첫 결과 4d8c0f9 를 본 뒤).

| 검사 | 내용 |
|---|---|
| 정차 빈칸 | 도착·출발이 둘 다 빈칸인 정차 행은 '안 섬'(D-105 T1)으로 읽을 때 뺀다 → 첫 결과 계산 오류 7건이 사라짐 |
| 이어 갈아타기 | 켬 계산기는 한 역에서 노선 셋 이상을 이어 갈아타는 후보를 만들지 않는다 (B3). 끔 계산기는 그대로 |
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
from yeoyuro_v2 import korail_app as KA          # noqa: E402
from yeoyuro_v2 import korail_engine as KE       # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
HAS_APP = (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists()
HAS_3B = KA.k3b_ready(ROOT)
need_3b = pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (P4)")
ST = lambda n: n.split("_", 1)[1].split("@")[0]      # noqa: E731


def test_drop_unstopped():
    st = pd.DataFrame({"train": ["a"] * 4, "stop_seq": [0, 1, 2, 3], "code": [1, 2, 3, 4],
                       "arr_min": [np.nan, 10.0, np.nan, np.nan], "dep_min": [5.0, 10.5, 12.0, np.nan]})
    out = KE.drop_unstopped(st)
    assert out.code.tolist() == [1, 2, 3]                       # 둘 다 빈칸인 4 만 빠짐 (출발만 있는 행은 정차)
    assert KE.drop_unstopped(st.iloc[:0]).empty


def test_multi_line_chain():
    assert KE.multi_line_chain("AR_마곡나루|AR_공덕|GJ_공덕|5_공덕|5_애오개".split("|"), ST) == ["공덕"]
    assert KE.multi_line_chain("5_청구|5_동대문역사문화공원|4_동대문역사문화공원|2_동대문역사문화공원|2_신당".split("|"), ST) \
        == ["동대문역사문화공원"]
    assert KE.multi_line_chain("SB_선정릉|9L_선정릉|9X_선정릉|9X_신논현".split("|"), ST) == []   # 9 일반↔급행 = 같은 노선
    assert KE.multi_line_chain("4_동대문|4_동대문역사문화공원|5_동대문역사문화공원|5_을지로4가".split("|"), ST) == []


@pytest.fixture(scope="module")
def planner():
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    return KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", query_date="2026-10-14", repo_root=ROOT, include_3b=True)


@need_3b
def test_mart_rows_and_index_clean(planner):
    mart = pd.read_parquet(KE.K3B_MART)
    assert int((KE.drop_unstopped(mart).pipe(lambda x: x.arr_min.isna() & x.dep_min.isna())).sum()) == 0
    st3 = planner.on._st3b
    assert int((st3.arr_min.isna() & st3.dep_min.isna()).sum()) == 0


@need_3b
@pytest.mark.parametrize("o,d,hm", [("태릉입구", "양원", "08:00"), ("내방", "수색", "18:00")])
def test_former_error_cases_plan(planner, o, d, hm):
    """3B-4 첫 결과에서 계산 오류였던 사례 (끝 역 도착 시각 빈칸)."""
    r = planner.plan(o, d, KA.service_day("2026-10-14", hm))
    assert r.best is not None and r.best.status == KA.CONNECT_OK
    assert all(x.arr for x in r.best.rides)


@need_3b
def test_no_chain_candidates(planner):
    for o, d in (("서강대", "상왕십리"), ("야탑", "쌍문"), ("하계", "회기")):
        r = planner.plan(o, d, KA.service_day("2026-10-14", "08:00"))
        assert r.candidates
        for c in r.candidates:
            assert KE.multi_line_chain(c.path, ST) == [], (o, d, c.path)
