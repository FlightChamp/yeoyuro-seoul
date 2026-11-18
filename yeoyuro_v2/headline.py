"""
yeoyuro_v2/headline.py
======================
Phase 2-4 (D-084) — 앱 '프로젝트 & 검증' 화면의 v2 핵심 결과를 **커밋된 mart 에서 계산**한다.

화면에 숫자를 손으로 적으면 리포트·테스트와 어긋날 수 있다. 그래서 앱도 테스트(tests/test_headline_numbers.py)와
같은 정의로 mart 에서 직접 계산한다: '혼잡 이동(calm 제외) 안에서의 유형 비율, 소수 첫째 자리'.
Streamlit 에 의존하지 않는다.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

# 화면 표시 순서 (시간 → 둘 다 → 구조적 → 경로만)
TYPES = ("time_shiftable", "dual", "structural", "route_shiftable")
TYPE_LABEL = {
    "time_shiftable": "시간 조정으로 해결",
    "dual": "경로·시간 모두 가능",
    "structural": "회피 어려움",
    "route_shiftable": "경로 변경으로만 해결",
}


def type_shares(df: pd.DataFrame, col: str) -> dict:
    """혼잡 이동(calm 제외) 안의 유형 비율(%) — 리포트·테스트와 같은 정의."""
    hot = df[df[col] != "calm"]
    if hot.empty:
        return {t: 0.0 for t in TYPES}
    s = (hot[col].value_counts(normalize=True) * 100).round(1)
    return {t: float(s.get(t, 0.0)) for t in TYPES}


def v23_summary(mart: Optional[pd.DataFrame]) -> Optional[dict]:
    """v2.3 (1~8호선) 요약. mart 가 없으면 None."""
    if mart is None or "final_shiftability_type" not in mart:
        return None
    col = "final_shiftability_type"
    sh = type_shares(mart, col)
    n_od = mart.drop_duplicates(["origin", "destination"]).shape[0]
    return {"n_units": len(mart), "n_od": n_od, "n_hot": int((mart[col] != "calm").sum()),
            "n_structural": int((mart[col] == "structural").sum()), "shares": sh,
            "time_alternative": round(sh["time_shiftable"] + sh["dual"], 1)}


def v26_summary(mart: Optional[pd.DataFrame]) -> Optional[dict]:
    """v2.6 B5 (9호선 포함 별도 표본) 군별 요약. mart 가 없으면 None."""
    if mart is None or not {"group", "type"} <= set(mart.columns):
        return None
    out = {}
    for g in ("A", "B"):
        m = mart[mart.group == g]
        out[g] = {"n_od": m.drop_duplicates(["origin", "destination"]).shape[0],
                  "n_hot": int((m.type != "calm").sum()), "shares": type_shares(m, "type")}
    return out


def comparison_table(v23: Optional[dict], v26: Optional[dict]) -> pd.DataFrame:
    """유형 × 표본 비교표 (화면용 문자열). 없는 표본은 열을 만들지 않는다."""
    rows = []
    for t in TYPES:
        r = {"혼잡 이동 중 유형": TYPE_LABEL[t]}
        if v23:
            r["1~8호선 (v2.3)"] = "%.1f%%" % v23["shares"][t]
        if v26:
            r["9호선 포함 · A군 (9호선 역)"] = "%.1f%%" % v26["A"]["shares"][t]
            r["9호선 포함 · B군 (전체 균등)"] = "%.1f%%" % v26["B"]["shares"][t]
        rows.append(r)
    return pd.DataFrame(rows)
