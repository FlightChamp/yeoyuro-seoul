"""
tests/test_d109.py — D-109a: 광역철도 확장 카드 상태 문구·기준 표시 (사전 등록 preregistration_d109.md, 계산 불변)

| 검사 | 내용 |
|---|---|
| 문구 | 사용자 문구에 내부 기호 R4·(E) 없음 · 화면 이름(CONNECT_LABEL)과 내부 상태 값(CONNECT_OK) 분리 |
| 기준일 | 쓰인 노선의 날짜만 정확히 (1~9호선·코레일 1호선 2026-09-01, 경의중앙 2026-09-01, 수인분당 2026-08-22, 공항철도 2025-12-29) |
| 앱 | 로컬: 회기→수서 3-B 날짜·전체 연결 확인·**✅ 형식·계산 기준 펼침 · 공개 모드: 신도림→석계 1~9호선·코레일 1호선 날짜만 |
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
from yeoyuro_v2 import korail_3b as K3          # noqa: E402
from yeoyuro_v2 import korail_app as KA         # noqa: E402
from yeoyuro_v2 import korail_line1 as KL       # noqa: E402

APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"
HAS_APP = (ROOT / "data" / "app_v26" / "data" / "marts" / "route_edge_mart.parquet").exists()
HAS_3B = KA.k3b_ready(ROOT)
USER_TEXTS = ["BASIS_EXPECTED", "BASIS_TIMETABLE", "BASIS_STATIC", "FIRST_WAIT_NOTE", "FIRST_WAIT_RIDES",
              "EXCLUDED_PREFIX", "CALC_BASIS_TITLE", "NO_COMFORT", "BADGE_UNKNOWN"]


def test_no_internal_codes_in_user_texts():
    for name in USER_TEXTS:
        v = getattr(KA, name)
        assert "R4" not in v and "(E)" not in v, name
    assert set(KA.CONNECT_LABEL) == {KA.CONNECT_OK, KA.CONNECT_UNKNOWN, KA.CONNECT_FAIL}
    assert KA.CONNECT_OK == "연결 확인"                              # 내부 값은 그대로 (평가 도구·테스트가 씀)
    assert KA.CONNECT_OK in KA.CONNECT_LABEL[KA.CONNECT_OK]          # 화면 이름에 그 뜻이 들어 있음
    assert "2026-09-01" not in KA.BASIS_STATIC                       # 날짜는 노선별 줄이 맡음


def test_timetable_dates():
    assert KA.BASE_TT_DATE.replace("-", "") in KL.TT_FILE           # 1~9호선·코레일 1호선 원본 파일의 날짜
    assert K3.VALID_FROM == {"GJ": "2026-09-01", "GC": "2026-09-01", "SB": "2026-08-22", "AR": "2025-12-29"}
    leg = lambda ln: NS(line=ln)                                     # noqa: E731
    mixed = NS(legs=[leg("2"), leg("SB")], timetable_valid_from={"SB": "2026-08-22"})
    assert KA.timetable_dates(mixed) == "시각표 기준일: 1~9호선·코레일 1호선 2026-09-01 · 수인분당선 2026-08-22"
    only_3b = NS(legs=[leg("GJ"), leg("SB")], timetable_valid_from={"GJ": "2026-09-01", "SB": "2026-08-22"})
    assert KA.timetable_dates(only_3b) == "시각표 기준일: 경의중앙선 2026-09-01 · 수인분당선 2026-08-22"
    line1 = NS(legs=[leg("1")], timetable_valid_from={})
    assert KA.timetable_dates(line1) == "시각표 기준일: 1~9호선·코레일 1호선 2026-09-01"
    ar = NS(legs=[leg("9X"), leg("AR")], timetable_valid_from={"AR": "2025-12-29"})
    assert "공항철도 2025-12-29" in KA.timetable_dates(ar) and "2026-08-22" not in KA.timetable_dates(ar)
    assert KA.timetable_dates(NS(legs=[], timetable_valid_from={})) == ""


# ---------------------------------------------------------------------- 앱
def _fresh():
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()


def _search(o, d, date, hm):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=900)
    at.session_state["nav_page"] = "경로 추천"
    at.session_state["view_route"] = "경로"
    at.run()
    at.toggle(key="ext_on").set_value(True)
    at.run()
    at.selectbox(key="sel_origin").set_value(o)
    at.selectbox(key="sel_dest").set_value(d)
    at.date_input[0].set_value(dt.date.fromisoformat(date))
    at.time_input[0].set_value(dt.time(*map(int, hm.split(":"))))
    at.run()
    [b for b in at.button if b.label == "경로 검색"][0].click()
    at.run()
    return at


def _ext_texts(at):
    caps = [c.value for c in at.caption]
    mds = [m.value for m in at.markdown]
    helps = [getattr(m, "help", "") or "" for m in at.metric]
    i = next((k for k, x in enumerate(mds) if _KA_TITLE() in x), 0)
    return caps, mds[i:], helps


def _KA_TITLE():
    return KA.CARD_TITLE


@pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (P4)")
def test_app_local_3b_card():
    _fresh()
    try:
        at = _search("회기", "수서", "2026-10-14", "08:00")
        assert not at.exception
        caps, mds, helps = _ext_texts(at)
        line = [c for c in caps if c.startswith(KA.TT_DATES_PREFIX)]
        assert line and "경의중앙선 2026-09-01" in line[0] and "수인분당선 2026-08-22" in line[0]
        ok = [m for m in mds if m.startswith("**✅")]                   # 53 B13 이 쓰는 형식 그대로
        assert ok and KA.CONNECT_LABEL[KA.CONNECT_OK] in ok[0]
        assert KA.BASIS_EXPECTED in caps and KA.BASIS_TIMETABLE in caps and KA.BASIS_STATIC in caps
        assert any(e.label == KA.CALC_BASIS_TITLE for e in at.expander)
        for t in caps + mds + helps:
            assert "R4" not in t and "(E)" not in t, t[:80]
    finally:
        _fresh()


@pytest.mark.skipif(not HAS_APP, reason="app_v26 데이터 없음")
def test_app_public_line1_dates(monkeypatch):
    from yeoyuro_v2 import data_release as DR
    monkeypatch.setenv(DR.ENV_MODE, "public")
    _fresh()
    try:
        at = _search("신도림", "석계", "2026-10-14", "08:00")
        assert not at.exception
        caps, mds, _ = _ext_texts(at)
        line = [c for c in caps if c.startswith(KA.TT_DATES_PREFIX)]
        assert line == ["시각표 기준일: 1~9호선·코레일 1호선 2026-09-01"]
        assert any(c.startswith("시각표 확인 도착은 첫 열차까지 기다림") for c in caps)
        assert any(m.startswith("**✅") and KA.CONNECT_LABEL[KA.CONNECT_OK] in m for m in mds)
    finally:
        _fresh()
