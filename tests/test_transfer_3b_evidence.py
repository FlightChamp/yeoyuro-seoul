"""
tests/test_transfer_3b_evidence.py — 3-B 환승 근거 표가 사전 등록(D-105 4-1절)과 맞는지 (구현 전 고정).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "data" / "master" / "transfer_3b_evidence.csv"


def _t():
    return pd.read_csv(F, dtype={"line_a": str, "line_b": str})


def test_counts_and_grades():
    t = _t()
    pairs = t.groupby(["station", "line_a", "line_b"]).grade.first()
    assert len(pairs) == 38
    assert pairs.value_counts().to_dict() == {"U": 13, "B": 14, "A": 11}
    assert set(t.grade) <= {"A", "B", "U"}
    assert t.minutes.notna().all() and (t.minutes >= 0).all()


def test_directional_gimpo():
    t = _t()
    g = t[(t.station == "김포공항") & (t.line_b == "AR")]
    assert set(g.line_a) == {"5", "9"} and len(g) == 8
    nine = {(r.dir_a, r.dir_b): r.minutes for r in g[g.line_a == "9"].itertuples()}
    assert nine[("중앙보훈병원", "서울역")] == 0 and nine[("개화", "인천공항")] == 0
    assert nine[("중앙보훈병원", "인천공항")] == 1 and nine[("개화", "서울역")] == 1
    assert (t[t.station != "김포공항"].dir_a == "*").all()


def test_b_grade_uses_factor():
    t = _t()
    b = t[t.grade == "B"]
    off = b.official_time.map(lambda s: int(s.split(":")[0]) + int(s.split(":")[1]) / 60)
    assert ((b.minutes - off * 1.81).abs() < 0.01).all()


def test_raw_and_detailed_mart_not_committed():
    """P4: 원본 xlsx·열차별 상세 mart 는 저장소에 넣지 않는다 (재배포 권한 미확인)."""
    import subprocess
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "data/raw", "data/marts/v2/korail_3b"],
                         capture_output=True, text=True).stdout
    assert ".xlsx" not in out and "korail_3b" not in out
    chk = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "data/marts/v2/korail_3b/x.parquet"])
    assert chk.returncode == 0


def test_source_values_not_called_errors():
    """원천값은 '채택 안 함/오류 의심' 으로만 기록하고 오류로 확정하지 않는다."""
    t = _t()
    for k in [("왕십리", "2", "SB"), ("왕십리", "5", "SB"), ("왕십리", "GJ", "SB")]:
        n = t[(t.station == k[0]) & (t.line_a == k[1]) & (t.line_b == k[2])].note.iloc[0]
        assert "채택 안 함" in n and "확정하지 않음" in n
    g = t[(t.station == "김포공항") & (t.line_a == "5")].note
    assert g.str.contains("오류 의심").all() and g.str.contains("확정하지 않음").all()


def test_prereg_rules_present():
    doc = (ROOT / "docs" / "v2" / "preregistration_3b.md").read_text(encoding="utf-8")
    for key in ["4-2. 방향별 환승과 후보 탐색 충분성", "4-3. B 등급 계수 확정 규칙", "×1.00·1.60·1.81·2.78",
                "현행 시각표 기반 확정 추천", "B12", "B13", "1.0분"]:
        assert key in doc, key
