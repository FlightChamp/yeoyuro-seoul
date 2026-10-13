"""
tests/app_harness.py
====================
앱 화면 기준값(baseline) 생성과 IA 리팩토링 회귀 테스트가 **같은 방식으로** 앱을 조작·관찰하기 위한 도우미.

- engine_snapshot : 앱과 같은 계산 경로(get_scorer → find_route_by_station → retime → counterfactual)를 직접 호출
- search          : AppTest 로 경로 검색을 실행 (리팩토링 전/후 모두 동작하도록 위젯을 라벨·순서로 찾는다)
- screen_snapshot : 화면에 표시된 metric·요약 줄을 모은다

기준 시나리오 CASES 는 1~8호선 / 9호선 급행 / 환승 / 여유 / 이벤트 날짜 / 빠른 도착 모드를 섞었다.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"

# (출발, 도착, 날짜, 시각, 모드)
CASES = [
    ("상봉", "건대입구", "2026-10-14", "08:00", "calm"),
    ("신림", "강남", "2026-10-14", "08:15", "calm"),
    ("강변", "여의도", "2026-10-14", "08:30", "calm"),
    ("노원", "수유", "2026-10-14", "08:00", "calm"),
    ("가양", "고속터미널", "2026-10-14", "08:00", "calm"),
    ("김포공항", "종합운동장", "2026-10-14", "08:00", "calm"),
    ("석촌", "잠실", "2026-10-14", "18:00", "calm"),
    ("서울역", "잠실", "2026-10-14", "18:00", "calm"),
    ("신림", "강남", "2026-10-14", "08:15", "fast"),
    ("서울역", "잠실", "2026-10-14", "18:00", "min_transfer"),
    ("공덕", "여의나루", "2025-09-27", "18:00", "calm"),      # 불꽃축제 (이벤트 배수 적용 날짜)
]


def case_id(c) -> str:
    return "%s>%s %s %s %s" % c


def _day_type(date: str) -> str:
    dow = dt.date.fromisoformat(date).weekday()
    return "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")


def engine_snapshot(case) -> dict:
    """앱의 경로 계산 경로를 Streamlit 없이 그대로 재현한다 (데이터 폴더 = 앱과 같은 data/app_v26)."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "app" / "streamlit"))
    import station_routing as sr
    from yeoyuro_v2.app_bridge import retime_result
    from yeoyuro_v2.counterfactual import counterfactual
    from yeoyuro_v2.line9 import register_v1_directions
    from yeoyuro_v2.time_dependent import TDRouter, TimeDependentEvaluator, parse_hhmm
    o, d, date, hhmm, mode = case
    data_root = ROOT / "data" / "app_v26"
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    rs = mod.RouteScorer(data_root, _day_type(date), hhmm, date)
    disp = sr.load_display_master(data_root)
    res = sr.find_route_by_station(rs, disp, o, d, mode)
    ev = TimeDependentEvaluator(rs, sr.load_headway(data_root), "step")
    res = retime_result(res, lambda b: ev, parse_hhmm(hhmm), mode)

    def card(x):
        if not x:
            return None
        return {"path": list(x["path"]), "actual": round(x["actual_time_min"], 3),
                "perceived": round(x["perceived_time_min"], 3), "max": round(x["max_congestion"] or 0, 3),
                "exp100": round(x.get("exposure_100_min") or 0, 3), "transfers": int(x["transfer_count"])}
    router = TDRouter(data_root, mod, sr, disp, _day_type(date), policy="step", k=5, multi_bin=True,
                      query_date=date)
    cf = counterfactual(router, o, d, parse_hhmm(hhmm))
    cfs = None if cf is None else {
        "type": cf["type"],
        **{k: {"ok": cf[k].ok, "shift": cf[k].shift_min, "exp100": None if cf[k].exp100 is None else round(cf[k].exp100, 3),
               "actual": None if cf[k].actual is None else round(cf[k].actual, 3), "path": cf[k].path}
           for k in ("now", "route", "time", "joint")}}
    return {"recommended": card(res["recommended"]), "alternative": card(res.get("alternative")),
            "fastest": card(res.get("fastest")), "counterfactual": cfs}


def _route_view(at):
    """리팩토링 전(menu) / 후(nav_page·view_route) 어느 쪽이든 경로 화면을 띄운다."""
    at.session_state["menu"] = "쾌적 경로 찾기"
    at.session_state["nav_page"] = "경로 추천"
    at.session_state["view_route"] = "경로"


def search(case, timeout: int = 300):
    """AppTest 로 경로 검색까지 실행하고 AppTest 객체를 돌려준다."""
    from streamlit.testing.v1 import AppTest
    o, d, date, hhmm, mode = case
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    _route_view(at)
    at.run()
    at.selectbox(key="sel_origin").set_value(o)
    at.selectbox(key="sel_dest").set_value(d)
    at.date_input[0].set_value(dt.date.fromisoformat(date))
    h, m = map(int, hhmm.split(":"))
    at.time_input[0].set_value(dt.time(h, m))
    [s for s in at.selectbox if s.label == "선호 모드"][0].set_value(mode)
    at.run()
    [b for b in at.button if b.label == "경로 검색"][0].click()
    at.run()
    return at


def screen_snapshot(at) -> dict:
    """화면의 metric(라벨·값·증감)과 경로 요약 줄. 순서까지 저장한다."""
    mets = [(m.label, m.value, m.delta or "") for m in at.metric]
    summary = [c.value for c in at.caption if " · " in c.value and "→" in c.value]
    return {"metrics": mets, "summary": summary[:3], "exceptions": [e.value[:200] for e in at.exception]}
