"""
tests/test_app_ia.py
====================
v2 IA 리팩토링(D-057) 회귀 테스트. 앱을 AppTest 로 실제로 실행한다.

1. 계산 결과 불변   : 엔진 결과 + 경로 화면 metric·요약 줄이 리팩토링 전 기준값(tests/fixtures/app_baseline.json)과 같음
2. 사이드바         : 1차 메뉴가 정확히 5개 (공모전판: 역사 안내 추가)
3. 직접 진입        : 모든 페이지·subview·예전 메뉴 이름으로 들어가 3회 재실행해도 오류 없음
4. 공유 입력        : 경로 검색 뒤 다른 페이지를 다녀와도 OD·날짜·시각·결과 유지, 시간 비교 입력부도 같은 OD
5. CTA              : Time Shift 판정 성공일 때만 표시, 클릭 → '시간 바꿔 보기' 자동 계산, 1순위 시각 = CTA 시각
6. stale            : 검색 뒤 날짜·시각·선호 모드를 바꾸면 이전 결과와 CTA 를 즉시 숨김 (실제 버그 회귀)
7. 환승 inline      : 환승 구간 바로 아래 빠른 하차·예상 환승 + [자세히 보기], 자료 없으면 '미제공'
8. 이벤트 통합      : 시나리오 / 과거 데이터 두 subview 와 상호 이동, 시나리오 고유 경고 유지
9. 용어             : 화면에 '구조적'·'이벤트 혼잡 경보'·큰 실시간 안내 박스가 없음
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import app_harness as H  # noqa: E402

BASE = ROOT / "tests" / "fixtures" / "app_baseline.json"
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "app_v26" / "data" / "marts" / "route_edge_mart.parquet").exists() or not BASE.exists(),
    reason="앱 데이터 폴더 또는 기준값 파일 없음")

PAGES = ["경로 추천", "혼잡 탐색", "이벤트 분석", "역사 안내", "프로젝트 & 검증"]
VIEWS = [("경로 추천", "view_route", "경로"), ("경로 추천", "view_route", "시간 바꿔 보기"),
         ("혼잡 탐색", "view_explore", "역·노선별"), ("혼잡 탐색", "view_explore", "네트워크 지도"),
         ("이벤트 분석", "view_event", "시나리오"), ("이벤트 분석", "view_event", "과거 데이터"),
         ("역사 안내", "view_facility", "엘리베이터 하차 칸"), ("역사 안내", "view_facility", "역사 도면"),
         ("프로젝트 & 검증", None, None)]
LEGACY = ["쾌적 경로 찾기", "시간 분산", "혼잡 분산 지도", "이벤트 시나리오", "시간대별 혼잡 조회",
          "빠른 환승 안내", "이벤트 혼잡 경보", "프로젝트 소개 및 검증 리포트"]


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASE.read_text(encoding="utf-8"))["cases"]


def _fresh(nav=None, vkey=None, view=None, menu=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(H.APP), default_timeout=300)
    if nav:
        at.session_state["nav_page"] = nav
    if vkey:
        at.session_state[vkey] = view
    if menu:
        at.session_state["menu"] = menu
    return at


def _texts(at) -> list[str]:
    out = []
    for coll in (at.title, at.header, at.subheader, at.markdown, at.caption, at.info, at.warning,
                 at.success, at.error):
        out += [str(x.value) for x in coll]
    out += [m.label for m in at.metric] + [b.label for b in at.button] + [e.label for e in at.expander]
    return out


def _no_exc(at):
    assert not at.exception, [e.value[:300] for e in at.exception]


def _cta(at):
    return [b for b in at.button if b.label == "시간 비교 보기 →"]


# ---------------------------------------------------------------- 1. 계산 결과 불변
@pytest.mark.parametrize("case", H.CASES, ids=H.case_id)
def test_engine_unchanged(case, baseline):
    assert H.engine_snapshot(case) == baseline[H.case_id(case)]["engine"]


@pytest.mark.parametrize("case", H.CASES, ids=H.case_id)
def test_screen_numbers_unchanged(case, baseline):
    scr = H.screen_snapshot(H.search(case))
    ref = baseline[H.case_id(case)]["screen"]
    assert scr["exceptions"] == []
    assert [tuple(m) for m in scr["metrics"]] == [tuple(m) for m in ref["metrics"]]
    assert scr["summary"] == ref["summary"]


# ---------------------------------------------------------------- 2. 사이드바
def test_sidebar_exactly_five_menus():
    at = _fresh()
    at.run()
    _no_exc(at)
    assert [b.label for b in at.sidebar.button] == PAGES
    assert at.session_state["nav_page"] == "경로 추천"
    assert any("과거 패턴 기반 기대값 · 실시간 정보 아님" in c.value for c in at.sidebar.caption)


# ---------------------------------------------------------------- 3. 직접 진입 / 재실행
@pytest.mark.parametrize("nav,vkey,view", VIEWS, ids=lambda x: str(x))
def test_direct_entry_and_reruns(nav, vkey, view):
    at = _fresh(nav, vkey, view)
    for _ in range(3):
        at.run()
        _no_exc(at)
    assert at.title[0].value == nav
    if vkey:
        assert at.session_state[vkey] == view


@pytest.mark.parametrize("menu", LEGACY)
def test_legacy_menu_values(menu):
    at = _fresh(menu=menu)
    at.run()
    _no_exc(at)
    assert at.session_state["nav_page"] in PAGES
    at.run()
    _no_exc(at)


def test_time_view_direct_entry_shows_form_not_auto():
    at = _fresh("경로 추천", "view_route", "시간 바꿔 보기")
    at.run()
    _no_exc(at)
    assert any(b.label == "비교하기" for b in at.button)
    assert not at.metric          # 자동 계산하지 않음


# ---------------------------------------------------------------- 4. 공유 입력
def test_od_state_persists_across_pages():
    case = ("서울역", "잠실", "2026-10-14", "18:00", "calm")
    at = H.search(case)
    before = H.screen_snapshot(at)["metrics"]
    at.sidebar.button(key="nav_혼잡 탐색").click().run()
    _no_exc(at)
    at.sidebar.button(key="nav_이벤트 분석").click().run()
    at.sidebar.button(key="nav_경로 추천").click().run()
    _no_exc(at)
    assert at.selectbox(key="sel_origin").value == "서울역"
    assert at.selectbox(key="sel_dest").value == "잠실"
    assert at.date_input(key="w_date").value == dt.date(2026, 10, 14)
    assert at.time_input(key="w_time").value == dt.time(18, 0)
    assert H.screen_snapshot(at)["metrics"] == before          # 결과도 그대로 (다시 검색 불필요)

    # 같은 입력이 시간 비교 입력부에도 (직접 subview 전환, CTA 아님)
    at.session_state["view_route"] = "시간 바꿔 보기"
    at.session_state["ts_handoff"] = None
    at.run()
    _no_exc(at)
    assert at.selectbox(key="ts_origin").value == "서울역"
    assert at.selectbox(key="ts_dest").value == "잠실"
    assert at.time_input(key="ts_time").value == dt.time(18, 0)


# ---------------------------------------------------------------- 5. CTA
@pytest.mark.parametrize("case", [H.CASES[1], H.CASES[5], H.CASES[6]], ids=H.case_id)
def test_cta_to_time_view(case, baseline):
    cf = baseline[H.case_id(case)]["engine"]["counterfactual"]
    assert cf["time"]["ok"]
    at = H.search(case)
    cta = _cta(at)
    assert len(cta) == 1
    msg = [m.value for m in at.markdown if "⏰" in m.value][0]
    shift = cf["time"]["shift"]
    assert "%d분 %s" % (abs(shift), "일찍" if shift < 0 else "늦게") in msg
    cta[0].click().run()
    _no_exc(at)
    assert at.session_state["view_route"] == "시간 바꿔 보기"
    assert not any(b.label == "비교하기" for b in at.button)                 # 자동 계산
    assert dict((m.label, m.value) for m in at.metric)["판정"] == "시간 조정 효과 있음"
    top = at.dataframe[0].value
    assert top.iloc[0]["조정"] == "%d분 %s" % (abs(shift), "일찍" if shift < 0 else "늦게")
    o, d, date, hhmm, _ = case
    assert any(c.value.startswith("%s → %s" % (o, d)) and hhmm in c.value for c in at.caption)
    # 돌아가기
    [b for b in at.button if b.label == "← 경로로 돌아가기"][0].click().run()
    _no_exc(at)
    assert at.session_state["view_route"] == "경로"
    assert len(_cta(at)) == 1


@pytest.mark.parametrize("case", [H.CASES[0], H.CASES[3], H.CASES[7]], ids=H.case_id)
def test_no_cta_when_time_shift_fails(case, baseline):
    assert not baseline[H.case_id(case)]["engine"]["counterfactual"]["time"]["ok"]
    at = H.search(case)
    _no_exc(at)
    assert not _cta(at)


def test_structural_label_and_card_text():
    at = H.search(H.CASES[0])                    # 상봉→건대입구 08:00 = structural
    w = " ".join(x.value for x in at.warning)
    assert "**회피 어려움** · 현재 탐색한 경로·시간 대안 범위에서는 뚜렷한 개선안을 찾지 못했습니다." in w
    assert any("정해진 경로·시간 대안 범위에서 개선안이 발견되지 않은 이동" in c.value for c in at.caption)


# ---------------------------------------------------------------- 6. stale (회귀)
@pytest.mark.parametrize("what", ["date", "time", "mode"])
def test_stale_after_input_change(what):
    case = H.CASES[1]                           # 신림→강남 08:15, CTA 있음
    at = H.search(case)
    assert _cta(at) and at.metric
    if what == "date":
        at.date_input(key="w_date").set_value(dt.date(2026, 10, 15))
    elif what == "time":
        at.time_input(key="w_time").set_value(dt.time(9, 0))
    else:
        at.selectbox(key="w_mode").set_value("fast")
    at.run()
    _no_exc(at)
    assert not _cta(at), "조건이 바뀌었는데 CTA 가 남아 있음"
    assert not at.metric, "조건이 바뀌었는데 이전 결과 metric 이 남아 있음"
    assert any("조건" in w.value and "경로 검색" in w.value for w in at.warning)
    # stale 상태에서 시간 비교로 가도 자동 계산하지 않는다
    at.session_state["view_route"] = "시간 바꿔 보기"
    at.run()
    _no_exc(at)
    assert any(b.label == "비교하기" for b in at.button) and not at.metric
    # 다시 검색하면 결과·CTA 복귀
    at.session_state["view_route"] = "경로"
    at.run()
    [b for b in at.button if b.label == "경로 검색"][0].click().run()
    _no_exc(at)
    assert at.metric


def test_stale_cta_handoff_not_auto():
    """CTA 로 넘어간 뒤 조건을 바꾸면, 시간 비교는 자동 계산을 멈추고 입력부를 보여준다."""
    at = H.search(H.CASES[1])
    _cta(at)[0].click().run()
    assert at.metric
    at.session_state["q_time"] = dt.time(9, 0)        # 다른 화면에서 시각이 바뀐 상황
    at.run()
    _no_exc(at)
    assert any(b.label == "비교하기" for b in at.button) and not at.metric


# ---------------------------------------------------------------- 7. 환승 inline
def _transfer_blocks(at):
    out = []
    for m in at.markdown:
        if "mc-tf-h" in m.value and "↓" in m.value:
            x = re.sub(r"\s+", " ", re.sub("<style>.*?</style>", "", m.value, flags=re.S))
            out += [re.sub("<[^>]+>", " ", p) for p in x.split("↓")[1:]]
    return out


def test_transfer_inline_with_tip():
    at = H.search(("서울역", "잠실", "2026-10-14", "18:00", "calm"))
    blocks = _transfer_blocks(at)
    assert blocks and "동대문역사문화공원 · 4호선 → 2호선" in blocks[0]
    assert "빠른 하차" in blocks[0] and "예상 환승 약" in blocks[0]
    assert any(e.label == "자세히 보기 · 동대문역사문화공원 환승" for e in at.expander)


def test_transfer_inline_without_tip_line9():
    at = H.search(("가양", "여의나루", "2026-10-14", "10:00", "fast"))
    _no_exc(at)
    blocks = _transfer_blocks(at)
    assert any("여의도 · 9호선" in b and "빠른 환승 위치 정보 미제공" in b for b in blocks)
    assert any(e.label == "자세히 보기 · 여의도 환승" for e in at.expander)


# ---------------------------------------------------------------- 8. 이벤트 통합
def test_event_subviews_linked():
    at = _fresh("이벤트 분석", "view_event", "시나리오")
    at.run()
    _no_exc(at)
    assert any("가정된 시나리오" in w.value and "차내 혼잡 관측치가 아니라" in w.value for w in at.warning)
    at.button(key="evs_to_past").click().run()
    _no_exc(at)
    assert at.session_state["view_event"] == "과거 데이터"
    txt = " ".join(_texts(at))
    assert "DiD" in txt and "평행추세" in txt
    at.button(key="evp_to_scn").click().run()
    _no_exc(at)
    assert at.session_state["view_event"] == "시나리오"


# ---------------------------------------------------------------- 9. 용어
def test_terminology_all_views():
    for nav, vkey, view in VIEWS:
        at = _fresh(nav, vkey, view)
        at.run()
        _no_exc(at)
        txt = _texts(at)
        assert not [t for t in txt if "구조적" in t], (nav, view)
        assert not [t for t in txt if "이벤트 혼잡 경보" in t], (nav, view)
        assert not [i.value for i in at.info if "실시간 측정값이 아닌" in i.value], (nav, view)
    if (ROOT / "data" / "marts" / "v2" / "mobility_atlas_mart.parquet").exists():   # 지도 mart 가 있을 때만
        at = _fresh("혼잡 탐색", "view_explore", "네트워크 지도")
        at.run()
        assert any(m.label == "회피 어려움" for m in at.metric)


def test_project_scope_matches_app():
    at = _fresh("프로젝트 & 검증")
    at.run()
    _no_exc(at)
    df = at.dataframe[0].value
    scope = dict(zip(df["항목"], df["현재 앱"]))
    assert scope["버전"].startswith("v2")
    assert scope["지원 노선"].startswith("1~9호선")
    assert scope["Time-dependent 적용"].startswith("적용")
    assert "335 노드" in scope["경로 그래프"] and "승차 엣지 640" in scope["경로 그래프"] \
        and "환승 엣지 150" in scope["경로 그래프"]
