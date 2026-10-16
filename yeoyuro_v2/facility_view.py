"""
yeoyuro_v2/facility_view.py
===========================
공모전판 '하차 칸' 화면 문구 (경로 카드 · 역사 안내 공통). Streamlit 을 모른다 → 모든 상태를 단위 테스트로 훑을 수 있다.

판단은 facility_match.alight_crowd_match 가 하고, 여기서는 그 결과(state: compare / quiet / no_sample / None)를
결론 카드·근거 영역 문구로만 바꾼다. 상태마다 쓰는 키가 다르므로 dict 접근은 상태를 확인한 뒤에만 한다.
"""
from __future__ import annotations

import re
from html import escape as _esc

from yeoyuro_v2.facility_match import match

CAR_HINT_NOTE = "서울교통공사 칸별 혼잡도 샘플(4호선, 2026년 6월 평일)의 과거 패턴 · 시범 기능 · 실시간 아님"
NO_SAMPLE_NOTE = "칸별 혼잡 비교: 이 시간대는 샘플 표본(통과 3회·2일 이상)이 부족해 생략했습니다"
OTHER_LINE_NOTE = "칸별 혼잡도 샘플은 4호선만 제공되어 다른 노선은 엘리베이터 위치만 안내합니다"
BUSY = 60.0


def car_door_text(e: dict) -> str:
    """빠른하차정보 칸-문 → '7번째 칸 1번 문'. '3-2,3-3 사이' 같은 원문은 그대로 쓴다."""
    if re.fullmatch(r"\d+-\d+", str(e.get("label", ""))) and e.get("door") is not None:
        return "%d번째 칸 %d번 문" % (e["car"], e["door"])
    return "%s" % str(e.get("label", "")).replace(",", "·")


def state_of(crowd: dict | None) -> str:
    if not crowd:
        return "none"
    if crowd.get("no_sample"):
        return "no_sample"
    return "quiet" if crowd.get("quiet") else "compare"


def decision(station: str, line, elevators: list[dict], crowd: dict | None, public_reason: str | None = None) -> dict:
    """결론 카드. 반환: title, lines(html), warn, note, pick, pct, ev_labels, gain_pct, state."""
    ev_labels = {e["car"]: e["label"] for e in elevators}
    where = " · ".join(car_door_text(e) for e in elevators)
    stt = state_of(crowd)
    c = crowd or {}
    out = {"state": stt, "pick": None, "pct": c.get("pct") if stt in ("quiet", "compare") else None,
           "ev_labels": ev_labels, "gain_pct": None, "warn": None, "note": None}
    if stt == "compare":
        lab = c.get("elevator") or elevators[0]["label"]
        if c["pick"] != c["nearest"]:
            out.update(pick=c["pick"], gain_pct=c["gain_pct"])
            out["title"] = "♿ 추천: %d번째 칸" % c["pick"]
            out["lines"] = ["엘리베이터 <b>%s</b>에서 %d칸 이동하면 %d번째 칸 %d%% → <b>%d번째 칸 %d%% (−%d%%p)</b>"
                            % (_esc(lab), c["extra_cars"], c["nearest"], c["near_pct"], c["pick"], c["pick_pct"],
                               c["gain_pct"])]
            if c["near_pct"] >= 100 and c["pick_pct"] >= 100:
                out["warn"] = "두 칸 모두 평소 정원 100%를 넘어 혼잡합니다."
            elif c["pick_pct"] >= 100:
                out["warn"] = "추천 칸도 평소 정원 100%를 넘어 혼잡합니다."
        else:
            out["title"] = "♿ 추천: %d번째 칸 (엘리베이터 칸)" % c["nearest"]
            out["lines"] = ["엘리베이터 <b>%s</b> 바로 앞 칸입니다. 양옆 칸과 차이가 5%%p 미만이라 더 걸을 필요가 없습니다 "
                            "(평소 %d%%)" % (_esc(lab), c["near_pct"])]
            if c["near_pct"] >= 100:
                out["warn"] = "이 칸도 평소 정원 100%를 넘어 혼잡합니다."
        out["note"] = CAR_HINT_NOTE
        return out
    out["title"] = "♿ %s 근처에서 내리세요" % where
    out["lines"] = ["%s 엘리베이터와 가장 가까운 위치입니다" % _esc(station)]
    if stt == "quiet":
        out["lines"].append("평소 이 시간에는 붐비지 않는 구간이라 칸을 옮길 필요가 크지 않습니다 (열차 평균 %.0f%%)"
                            % c["train_mean"])
        out["note"] = CAR_HINT_NOTE
    elif stt == "no_sample":
        out["note"] = NO_SAMPLE_NOTE
    elif public_reason:
        out["note"] = public_reason
    elif str(line) != "4":
        out["note"] = OTHER_LINE_NOTE
    return out


def evidence(crowd: dict | None, dec: dict) -> dict | None:
    """'왜 이 칸인가요?' 근거 영역. 칸별 % 가 없으면 None (영역을 그리지 않는다).

    반환: expander(제목), caption(본문), help(툴팁 또는 None), pick(초록 테두리 칸 또는 None), ev_cars"""
    stt = state_of(crowd)
    if stt not in ("quiet", "compare") or not crowd.get("pct"):
        return None
    ev_cars = set(dec["ev_labels"])
    if stt == "quiet":
        return {"expander": "왜 이 칸인가요?", "pick": None, "ev_cars": ev_cars, "help": None,
                "caption": "평소 이 시간에는 열차 평균이 %.0f%%로 %.0f%% 미만이라, 칸을 옮기라고 권하지 않고 "
                           "엘리베이터와 가장 가까운 칸을 안내합니다." % (crowd["train_mean"], BUSY)}
    chosen = dec["pick"] or crowd["nearest"]
    # 칸별 혼잡도 샘플의 칸 번호 기준이 빠른하차정보와 반대일 가능성(미확인)을 사례마다 계산해 툴팁에 밝힌다
    mir = match(crowd["pct"], crowd["nearest"], mirrored=True)["pick"]
    mir_txt = ("칸 번호 기준을 반대로 보고 계산해도 이 경우 추천 칸은 같습니다." if mir == chosen else
               "칸 번호 기준을 반대로 보면 추천 칸이 %d번째 칸으로 달라집니다 — 공사 확인 전까지 시범 기능입니다." % mir)
    return {"expander": "왜 %s인가요?" % ("%d번째 칸" % dec["pick"] if dec["pick"] else "이 칸"),
            "pick": dec["pick"], "ev_cars": ev_cars,
            "caption": "엘리베이터 칸 양옆 1칸까지를 후보로 두고, 옆 칸이 5%p 이상 덜 붐빌 때만 추천합니다. "
                       "샘플 검증: 4호선 엘리베이터 역에 도착하는 붐비는 구간·시간대 129개 중 40%에서 옆 칸이 평균 10.8%p 덜 붐볐습니다.",
            "help": "칸별 혼잡도 샘플의 칸 번호 기준(진행 방향 여부)은 공사 확인이 필요합니다. " + mir_txt}
