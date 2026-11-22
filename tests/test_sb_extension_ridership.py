"""D-091/D-092 — 수인분당선 연장 종점(야탑) 근거 보고서 고정 검사.

원본 CSV 는 커밋하지 않으므로 커밋된 결과(reports/v2/sb_extension_ridership.csv)와 중복 제거 로직만 검사한다.
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "v2" / "sb_extension_ridership.csv"


def _mod():
    spec = importlib.util.spec_from_file_location("sb44", ROOT / "scripts" / "v2" / "44_sb_extension_ridership.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_yatap_is_top_bundang_station():
    t = pd.read_csv(CSV, encoding="utf-8-sig")
    bd = t[t.line == "분당선"].set_index("station")
    assert bd.loc["야탑", "rank_bundang"] == 1
    assert (bd.n_months == 12).all()
    # 야탑까지는 범위 밖 환승이 없고, 바로 다음 이매부터 나온다
    order = bd.sort_values("order")
    upto = order.loc[:"야탑"]
    assert (upto.out_of_scope_transfer.fillna("") == "").all()
    assert order.loc["이매", "out_of_scope_transfer"] == "경강선"
    # 복정 분당선 개찰구는 사실상 0 (8호선 쪽에 잡힘)
    assert bd.loc["복정", "daily_total"] < 100
    assert t[(t.line == "8호선") & (t.station == "복정")].daily_total.iloc[0] > 1000


def test_dedupe_removes_exact_duplicates_only():
    m = _mod()
    row = {"사용월": "202603", "호선명": "분당선", "지하철역": "야탑", "07시-08시 승차인원": 10, "07시-08시 하차인원": 1}
    other = dict(row, **{"07시-08시 승차인원": 20})
    d = pd.DataFrame([row, row, other])
    out, exact, split = m.dedupe(d)
    assert exact == 1 and split == 2 and len(out) == 2
