"""
tests/test_contest_app.py
=========================
공모전판 화면 회귀 테스트 (AppTest 로 앱을 실제 실행).

1. 이동편의 안내: 끄면 엘리베이터 안내가 없고, 켜면 마지막 승차 구간에 도착역 엘리베이터 칸이 붙는다.
   4호선 평일 07:20 수유→동대문역사문화공원 은 칸별 샘플이 있으면 '1칸 더 걸으면 8번째 칸' 을 보여준다.
2. 구간 카드의 최대 기대 혼잡도 = 경로 지표(최대 기대 혼잡도) (한 노선 경로).
3. 역 시설·공간정보: 엘리베이터 하차 칸 / 역사 도면 두 화면이 오류 없이 열리고, 안전시설 원본이 없으면 위치를 그리지 않는다.
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
pytestmark = pytest.mark.skipif(not (MARTS / "route_edge_mart.parquet").exists(), reason="앱 데이터 폴더 없음")
CASE = ("수유", "동대문역사문화공원", "2026-10-14", "07:20", "calm")


def _acc_blocks(at) -> list[str]:
    out = []
    for m in at.markdown:
        out += [re.sub(r"<[^>]+>", " ", x) for x in re.findall(r'<div class="mc-acc".*?</div></div>', str(m.value), re.S)]
    return out


def _no_exc(at):
    assert not at.exception, [e.value[:300] for e in at.exception]


@pytest.mark.skipif(not (MARTS / "fast_exit_elevator.parquet").exists(), reason="엘리베이터 표 없음 (scripts/v2/38)")
def test_access_toggle_adds_elevator_car():
    at = H.search(CASE)
    _no_exc(at)
    assert _acc_blocks(at) == []                                      # 기본은 꺼짐
    at.toggle(key="w_access").set_value(True).run()
    _no_exc(at)
    blocks = _acc_blocks(at)
    assert len(blocks) == 1 and "7번째 칸 1번 문" in blocks[0]
    if (MARTS / "car_pattern_sample.parquet").exists():
        assert "1칸 더 걸으면 8번째 칸" in blocks[0]
    assert at.session_state["q_access"] is True
    at.sidebar.button(key="nav_혼잡 탐색").click().run()                # 다른 페이지를 다녀와도 유지
    at.sidebar.button(key="nav_경로 추천").click().run()
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
    at.session_state["nav_page"] = "역 시설·공간정보"
    at.session_state["view_facility"] = view
    at.run()
    _no_exc(at)
    return at


@pytest.mark.skipif(not (MARTS / "fast_exit_elevator.parquet").exists(), reason="엘리베이터 표 없음 (scripts/v2/38)")
def test_facility_page_default_example_and_line_switch():
    at = _page("엘리베이터 하차 칸")
    head = [str(m.value) for m in at.markdown if "엘리베이터와 가까운 위치" in str(m.value)]
    assert head and "충무로 방면" in head[0] and "7번째 칸 1번 문" in head[0]
    at.selectbox(key="fac_line").set_value("6").run()
    _no_exc(at)
    at.selectbox(key="fac_station").set_value("동묘앞").run()
    _no_exc(at)
    assert any("도면 보기" in b.label for b in at.button)
    assert any("4호선만 제공" in c.value for c in at.caption)


def test_spatial_page_never_invents_facilities():
    at = _page("역사 도면")
    if not list((ROOT / "data" / "raw" / "spatial_dongmyo").glob("*.shp")):
        assert any("포함하지 않았습니다" in i.value for i in at.info)
        return
    assert any("폴리곤" in c.value for c in at.caption)
    at.toggle(key="sp_fac").set_value(True).run()
    _no_exc(at)
    if not (ROOT / "data" / "raw" / "safety_facility").exists():
        assert any("임의로 만들어 그리지 않습니다" in i.value for i in at.info)
