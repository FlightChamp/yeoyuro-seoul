"""
tests/test_contest_app.py
=========================
공모전판 화면 회귀 테스트 (AppTest 로 앱을 실제 실행).

1. 이동편의 우선: 끄면 하차 칸 결론이 없고, 켜면 마지막 승차 구간에 결론형 안내(추천 칸 → 근거 → 경고)와
   [역사 안내] 버튼이 붙는다. 버튼을 누르면 같은 역·방면·시간대로 역사 안내가 열린다.
2. 화면 숫자 산술 일치: 'A% → B% (−C%p)' 에서 A − B = C.
3. 구간 카드의 최대 기대 혼잡도 = 경로 지표(최대 기대 혼잡도) (한 노선 경로).
4. 역사 안내: 결론 카드가 칸 그림보다 위, 근거 그래프는 접힘. 다른 노선에서는 엘리베이터 위치만.
5. 역사 도면: 2D/3D 같은 자리 전환, 안전시설 원본이 없으면 스위치 비활성 · 위치를 그리지 않음.
6. 공개 배포판: 자료가 없으면 기능을 숨기지 않고 비활성 사유를 보여준다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import app_harness as H  # noqa: E402

MARTS = ROOT / "data" / "app_v26" / "data" / "marts"
HAS_ELEV = (MARTS / "fast_exit_elevator.parquet").exists()
HAS_CAR = (MARTS / "car_pattern_sample.parquet").exists()
pytestmark = pytest.mark.skipif(not (MARTS / "route_edge_mart.parquet").exists(), reason="앱 데이터 폴더 없음")
CASE = ("수유", "동대문역사문화공원", "2026-10-14", "07:20", "calm")
PAGE = "역사 안내"


def _acc_text(at) -> list[str]:
    out = []
    for m in at.markdown:
        out += [re.sub(r"<[^>]+>", " ", x) for x in re.findall(r'<div class="mc-acc".*?</div></div>', str(m.value), re.S)]
    return out


def _no_exc(at):
    assert not at.exception, [e.value[:300] for e in at.exception]


def _md(at) -> list[str]:
    return [str(m.value) for m in at.markdown]


def test_access_toggle_decision_and_cta():
    at = H.search(CASE)
    _no_exc(at)
    assert _acc_text(at) == []                                         # 기본은 꺼짐
    at.toggle(key="w_access").set_value(True).run()
    _no_exc(at)
    acc = _acc_text(at)
    assert len(acc) == 1
    if not HAS_ELEV:
        assert "이용허락범위" in acc[0]
        return
    if HAS_CAR:
        assert "추천: 8번째 칸" in acc[0] and "7-1" in acc[0] and "두 칸 모두" in acc[0]
        a, b, c = map(int, re.search(r"(\d+)% → \s*8번째 칸 (\d+)% \(−(\d+)%p\)", acc[0]).groups())
        assert a - b == c                                              # 화면 숫자끼리 산술 일치
    else:
        assert "7번째 칸 1번 문" in acc[0]
    cta = [b for b in at.button if b.label.startswith("♿ 역사 안내에서")]
    assert cta
    cta[0].click().run()
    _no_exc(at)
    assert at.session_state["nav_page"] == PAGE and at.session_state["view_facility"] == "엘리베이터 하차 칸"
    assert at.session_state["fac_station"] == "동대문역사문화공원" and at.session_state["fac_toward"] == "충무로"
    if HAS_CAR:
        assert at.session_state["fac_bin"] == "07:30"
    at.sidebar.button(key="nav_경로 추천").click().run()               # 돌아와도 스위치 유지
    _no_exc(at)
    assert at.toggle(key="w_access").value is True


def test_segment_max_matches_route_metric():
    at = H.search(CASE)
    metric = [m.value for m in at.metric if m.label == "최대 기대 혼잡도"][0]
    seg = [re.findall(r"최대 기대 혼잡도 (\d+)%", str(m.value)) for m in at.markdown if "mc-seg" in str(m.value)]
    assert seg and seg[0] and seg[0][0] + "%" == metric


def _page(view):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(H.APP), default_timeout=300)
    at.session_state["nav_page"] = PAGE
    at.session_state["view_facility"] = view
    at.run()
    _no_exc(at)
    return at


def test_facility_page_order_and_line_switch():
    at = _page("엘리베이터 하차 칸")
    md = _md(at)
    if not HAS_ELEV:
        assert any("이용허락범위" in x for x in md)
        return
    i_dec = next(i for i, x in enumerate(md) if "♿ 추천" in x or "근처에서 내리세요" in x)
    i_strip = next(i for i, x in enumerate(md) if "♿ EV" in x)
    assert i_dec < i_strip                                             # 결론 → 칸 그림
    if HAS_CAR:
        assert "추천: 8번째 칸" in md[i_dec]
        assert any(e.label.startswith("왜 8번째 칸인가요") for e in at.expander)
        assert "★ 추천" in md[i_strip] and re.search(r"−\d+%p", md[i_strip])
    at.selectbox(key="fac_line").set_value("6").run()
    _no_exc(at)
    at.selectbox(key="fac_station").set_value("동묘앞").run()
    _no_exc(at)
    assert not [b for b in at.button if b.key == "fac_to_map"][0].disabled
    assert any("4호선만 제공" in c.value for c in at.caption)


def test_spatial_page_same_slot_and_no_invented_facilities():
    at = _page("역사 도면")
    assert any("공간정보 PoC" in x for x in _md(at))
    tg = [t for t in at.toggle if t.key == "sp_fac"][0]
    if not (ROOT / "data" / "raw" / "safety_facility").exists():
        assert tg.disabled and tg.value is False                       # 원본 없으면 비활성
    if not list((ROOT / "data" / "raw" / "spatial_dongmyo").glob("*.shp")):
        assert any("이용조건" in x for x in _md(at))
        return
    at.session_state["sp_mode"] = "3D 모델"
    at.run()
    _no_exc(at)
    assert any("3D 공간정보 PoC" in x for x in _md(at))
    assert [e.label for e in at.expander][-2:] == ["데이터 범위 및 한계", "개발자 정보 보기"]
