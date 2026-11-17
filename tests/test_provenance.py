"""
tests/test_provenance.py
========================
Phase 1-3 (D-075) 계산 출처(provenance) 회귀 테스트.

단위 (데이터 불필요)
 1. v2 재계산 성공      → source = v2_time_dependent, 대체 안내 없음, 경고 로그 없음
 2. 평가기 None          → v1_static / evaluator_unavailable, 결과 값은 v1 원본 그대로
 3. 평가기 생성 예외     → v1_static / evaluator_error + 예외 이름
 4. 재계산 중간 실패     → v1_static / retime_error, **값이 섞이지 않음** (복사본에서 계산)
 5. 강제 대체 스위치     → v1_static / forced, 평가기를 아예 만들지 않음
 6. 대체 결과는 yeoyuro.route 로거에 WARNING
 7. 화면 문구 (D-076)    → 사용자용 문구에는 내부 용어(v1·v2·환경변수) 없음, 사유는 상세(툴팁) 문구에만

앱 (AppTest, 앱 데이터 있을 때만)
 8. YEOYURO_FORCE_DEGRADE=v1          → 경고 박스 + 오류 없음
 9. YEOYURO_FORCE_DEGRADE=precomputed → 대표 시나리오 경고 박스 (대표 결과가 있는 OD)
10. 강제 대체 없음                    → 대체 경고 박스 없음
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from yeoyuro_v2 import provenance as pv  # noqa: E402


def _v1_result():
    return {"ok": True,
            "recommended": {"actual_time_min": 30.0, "max_congestion": 120.0},
            "candidates": [{"actual_time_min": 30.0, "max_congestion": 120.0},
                           {"actual_time_min": 34.0, "max_congestion": 90.0}]}


def _ok_retime(result, factory, depart_min, mode):
    ev = factory(None)
    for c in result["candidates"]:
        c["actual_time_min"] += ev
    result["recommended"] = result["candidates"][0]
    return result


def _half_then_fail(result, factory, depart_min, mode):
    """첫 후보만 고친 뒤 실패 — 예전 코드였다면 v1·v2 값이 섞여 화면에 나갔다."""
    result["candidates"][0]["actual_time_min"] = 999.0
    raise ValueError("boom")


def _call(retime_fn=_ok_retime, factory=lambda: 1.5, **kw):
    return pv.retime_with_provenance(_v1_result(), factory, retime_fn,
                                     depart_min=480.0, mode="calm", **kw)


def test_success_is_v2():
    out = _call()
    assert out["provenance"] == {"source": "v2_time_dependent", "reason": None, "error_type": None}
    assert out["candidates"][0]["actual_time_min"] == 31.5
    assert pv.notice_ko(out["provenance"]) is None


def test_evaluator_none_keeps_v1_values():
    out = _call(factory=lambda: None)
    assert out["provenance"]["source"] == "v1_static"
    assert out["provenance"]["reason"] == "evaluator_unavailable"
    assert out["candidates"][0]["actual_time_min"] == 30.0


def test_evaluator_exception_records_type():
    def bad():
        raise KeyError("headway")
    out = _call(factory=bad)
    assert out["provenance"] == {"source": "v1_static", "reason": "evaluator_error", "error_type": "KeyError"}


def test_partial_retime_failure_does_not_mix_values():
    out = _call(retime_fn=_half_then_fail)
    assert out["provenance"]["reason"] == "retime_error"
    assert out["provenance"]["error_type"] == "ValueError"
    assert [c["actual_time_min"] for c in out["candidates"]] == [30.0, 34.0]


def test_force_degrade_skips_evaluator():
    called = []
    out = _call(factory=lambda: called.append(1) or 1.0, force_degrade=True)
    assert out["provenance"]["reason"] == "forced"
    assert called == []


def test_degraded_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="yeoyuro.route"):
        _call(retime_fn=_half_then_fail, context="od=A->B")
    rec = [r for r in caplog.records if r.name == "yeoyuro.route"]
    assert rec and "retime_error" in rec[0].getMessage() and "od=A->B" in rec[0].getMessage()
    assert rec[0].exc_info is not None


def test_success_does_not_log(caplog):
    with caplog.at_level(logging.WARNING, logger="yeoyuro.route"):
        _call()
    assert not [r for r in caplog.records if r.name == "yeoyuro.route"]


V1_ERR = {"source": "v1_static", "reason": "retime_error", "error_type": "ValueError"}
PRE_ERR = {"source": "precomputed", "reason": "search_error", "error_type": None}


@pytest.mark.parametrize("prov", [V1_ERR, PRE_ERR, {"source": "v1_static", "reason": "forced", "error_type": None}])
def test_user_notice_has_no_internal_terms(prov):
    text = pv.notice_ko(prov)
    assert text
    for term in ("v1", "v2", "YEOYURO", "Error", "reason", "source"):
        assert term not in text, (term, text)


def test_notice_text():
    assert "간이 계산" in pv.notice_ko(V1_ERR)
    assert "예시 결과" in pv.notice_ko(PRE_ERR)
    d1 = pv.detail_ko(V1_ERR)
    assert "v2" in d1 and "v1" in d1 and "v2 재계산 중 오류" in d1 and "ValueError" in d1
    assert "경로 탐색 중 오류" in pv.detail_ko(PRE_ERR)
    ok = {"source": "v2_time_dependent", "reason": None, "error_type": None}
    for f in (pv.notice_ko, pv.detail_ko):
        assert f(None) is None and f(ok) is None


# ---------------------------------------------------------------- 앱 (AppTest)
APP_DATA = ROOT / "data" / "app_v26" / "data" / "marts" / "route_edge_mart.parquet"
needs_app = pytest.mark.skipif(not APP_DATA.exists(), reason="앱 데이터 폴더 없음")


def _warnings(at):
    return [w.value for w in at.warning]


@needs_app
def test_app_force_v1_shows_warning(monkeypatch):
    import app_harness as H
    monkeypatch.setenv("YEOYURO_FORCE_DEGRADE", "v1")
    at = H.search(H.CASES[1])
    assert not at.exception
    w = _warnings(at)
    assert any("간이 계산" in x for x in w), w
    assert at.session_state["route_result"]["provenance"]["source"] == "v1_static"
    assert any(c.value == "ⓘ 계산 정보" for c in at.caption)


@needs_app
def test_app_force_precomputed_shows_warning(monkeypatch):
    import app_harness as H
    import pandas as pd
    marts = ROOT / "data" / "app_v26" / "data" / "marts"
    files = sorted(marts.glob("route_evaluation_mart_*.parquet")) or \
        sorted((ROOT / "data" / "marts").glob("route_evaluation_mart_*.parquet"))
    if not files:
        pytest.skip("대표 시나리오 마트 없음")
    rid = pd.read_parquet(files[0], columns=["request_id"])["request_id"]
    rid = rid[rid.str.contains(r"\|calm")].iloc[0]
    o, d = rid.split("|")[0].split("->")
    monkeypatch.setenv("YEOYURO_FORCE_DEGRADE", "precomputed")
    at = H.search((o, d, "2026-10-14", "08:30", "calm"))
    assert not at.exception
    assert at.session_state["route_fallback"] is True
    assert any("예시 결과" in x for x in _warnings(at)), _warnings(at)


@needs_app
def test_app_normal_has_no_degrade_warning(monkeypatch):
    import app_harness as H
    monkeypatch.delenv("YEOYURO_FORCE_DEGRADE", raising=False)
    at = H.search(H.CASES[1])
    assert at.session_state["route_result"]["provenance"]["source"] == "v2_time_dependent"
    assert not [x for x in _warnings(at) if "간이 계산" in x or "예시 결과" in x]
    assert not any(c.value == "ⓘ 계산 정보" for c in at.caption)
