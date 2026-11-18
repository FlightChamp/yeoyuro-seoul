"""
tests/test_headline_numbers.py
==============================
Phase 2-2 (D-082) — 문서·README·앱에 쓰는 v2 핵심 수치를 **커밋된 mart 에서 다시 계산**해 고정한다.

문서 수치는 reports/v2/*.md 에서만 인용한다는 규칙(docs/v2/README 규칙 1)이 있지만, 리포트와 mart 가
어긋나도 알아챌 장치가 없었다. 여기서는 mart → 수치를 직접 계산해 리포트에 적힌 값과 비교한다.
수치를 바꿔야 하는 정당한 이유(재분석·새 사전 등록)가 생기면 이 파일과 리포트를 같은 커밋에서 고친다.

| 수치 | 출처 리포트 | mart |
|---|---|---|
| v2.3 혼잡 이동 1,690건, 유형 비율 73.5/23.1/2.8/0.5% | reports/v2/od_shiftability_report.md | data/marts/v2/od_shiftability_mart.parquet |
| v2.6 B5 군별 유형 비율 (A 7.6%, B 5.1% Structural) | reports/v2/line9_evaluation_report.md | data/marts/v2/line9_shiftability_mart.parquet |
| LightGBM Skill Score = 1 − MAE_model / MAE_baseline | docs/model_card_congestion.md 7-2 표 | (표 안의 MAE 로 재계산) |
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MARTS = ROOT / "data" / "marts" / "v2"


def _shares(df: pd.DataFrame, col: str) -> dict:
    """혼잡 이동(calm 제외) 안의 유형 비율(%) — 리포트와 같은 정의, 소수 첫째 자리."""
    hot = df[df[col] != "calm"]
    return (hot[col].value_counts(normalize=True) * 100).round(1).to_dict()


# ------------------------------------------------------------------ v2.3 (1~8호선)
def test_v23_counts_and_shares():
    m = pd.read_parquet(MARTS / "od_shiftability_mart.parquet")
    assert len(m) == 3720                                   # 620 OD × 평일 6시각
    assert (m.tier == "random").sum() == 3600 and (m.tier != "random").sum() == 120   # 무작위 600 + 지정 20
    assert (m.final_shiftability_type != "calm").sum() == 1690
    s = _shares(m, "final_shiftability_type")
    assert s == {"time_shiftable": 73.5, "dual": 23.1, "structural": 2.8, "route_shiftable": 0.5}
    assert (m.final_shiftability_type == "structural").sum() == 48
    assert set(m.threshold_set_id) == {"T1_prereg_v23"}


def test_v23_time_alternative_share():
    """'96.6% 는 시간 대안이 있는 비율' (= Time-only 73.5 + Dual 23.1). 혼잡 감소율이 아님."""
    s = _shares(pd.read_parquet(MARTS / "od_shiftability_mart.parquet"), "final_shiftability_type")
    assert round(s["time_shiftable"] + s["dual"], 1) == 96.6


# ------------------------------------------------------------------ v2.6 (1~9호선)
L9 = MARTS / "line9_shiftability_mart.parquet"


@pytest.mark.skipif(not L9.exists(), reason="9호선 평가 mart 없음 (Phase 2-2 에서 커밋)")
def test_v26_b5_group_shares():
    m = pd.read_parquet(L9)
    assert set(m.threshold_set_id) == {"T1_prereg_v23"}
    assert not m.default_w0.any()                           # B1 결과: 최초 승차 대기 미포함이 기본값
    ods = m.drop_duplicates(["group", "origin", "destination"]).group.value_counts().to_dict()
    assert ods == {"A": 300, "B": 300}                      # 사전 등록 표본 크기
    assert len(m) == 600 * 6                                # × 평일 6개 시각
    expect = {"A": {"route_shiftable": 2.7, "time_shiftable": 69.8, "dual": 19.9, "structural": 7.6},
              "B": {"route_shiftable": 1.9, "time_shiftable": 72.0, "dual": 20.9, "structural": 5.1}}
    for g, exp in expect.items():
        assert _shares(m[m.group == g], "type") == exp, g


# ------------------------------------------------------------------ 모델 카드 Skill Score
def test_model_card_skill_matches_its_own_mae():
    """모델 카드 7-2 표: 같은 Regime 의 baseline MAE 와 모델 MAE 로 계산한 Skill 이 표에 적힌 값과 같아야 한다 (D-080)."""
    text = (ROOT / "docs" / "model_card_congestion.md").read_text(encoding="utf-8")
    text = text.split("### 7-2.", 1)[1].split("\n### ", 1)[0]          # 7-2 표만 (다른 표의 A/B 행 제외)
    rows = re.findall(r"^\| ([ABC])\. [^|]+\| ([^|]+?) \| \**([\d.]+)\** \| [\d.]+ \| \**([−\-\d.()기준 ]+?)\** \|$",
                      text, flags=re.M)
    by = {}
    for regime, model, mae, skill in rows:
        by.setdefault(regime, []).append((model.strip(), float(mae), skill.strip()))
    assert set(by) == {"A", "B", "C"}, by
    for regime, items in by.items():
        base = [mae for model, mae, s in items if "기준" in s]
        assert len(base) == 1, (regime, items)
        for model, mae, s in items:
            if "기준" in s:
                continue
            got = float(s.replace("−", "-"))
            assert round(1 - mae / base[0], 3) == got, (regime, model, mae, base[0], got)
