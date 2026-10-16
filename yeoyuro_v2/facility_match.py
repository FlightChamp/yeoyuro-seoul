"""
yeoyuro_v2/facility_match.py
============================
공모전 Facility PoC — 교통약자 하차 칸 매칭 (실데이터 결합 시범 로직).

입력
- 서울교통공사_빠른하차정보 (공공데이터포털 OpenAPI, scripts/v2/34 로 수집)
  : 역 · 진행 방향(drtnInfo = 그 역 다음 정차역 방면) · 설비 종류 · 가장 가까운 칸-문 (예: "7-1")
  : 같은 엘리베이터가 방향마다 뒤집힌 번호로 적혀 있다 (예: "서울역 방면1-4, 명동 방면10-1") → 칸 번호는 진행 방향 기준
- 칸별 혼잡도 패턴 (car_pattern: 구간 × 30분 × 칸 평균)

로직
1. 하차역에 들어오는 마지막 역간(이전역 → 하차역)과 그 시간대의 칸별 혼잡을 가져온다.
2. 하차역에서 같은 진행 방향(이전역이 아닌 쪽 방면)의 엘리베이터 칸을 찾는다.
3. 엘리베이터 칸 ± window 칸을 후보로 두고, 가장 덜 붐비는 후보와 '더 걷는 칸 수'를 함께 돌려준다.

한계 (결과에 항상 함께 표기)
- 칸별 혼잡도 샘플의 칸 번호가 빠른하차정보와 같은 '진행 방향 기준'인지 공식 정의가 없다 (공사 확인 필요).
  mirrored=True 로 반대 기준일 때의 결과도 계산해 민감도를 보인다.
- 이 모듈은 Streamlit 을 모르고, 경로 계산에 관여하지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

N_CARS = 10
WINDOW = 1              # 엘리베이터 칸 양옆 몇 칸까지 후보로 볼지 (사용자가 바꿀 수 있는 값)
MEANINGFUL = 5.0        # 이 차이(%p) 이상일 때만 '옆 칸'을 권한다. 작으면 엘리베이터 칸 그대로


def load_fast_exit(path: Path, line: str | None = "4호선") -> pd.DataFrame:
    fe = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if line:
        fe = fe[fe.lineNm == line]
    m = fe.qckgffVhclDoorNo.str.extract(r"^(\d+)-(\d+)")
    fe = fe.assign(car=pd.to_numeric(m[0], errors="coerce"), door=pd.to_numeric(m[1], errors="coerce"))
    return fe.dropna(subset=["car"]).astype({"car": int}).reset_index(drop=True)


def elevator_cars(fe: pd.DataFrame, station: str, prev_station: str) -> list[dict]:
    """하차역 엘리베이터의 칸-문. 진행 방향 = 이전역의 반대쪽 방면 (drtnInfo != 이전역)."""
    f = fe[(fe.stnNm == station) & (fe.plfmCmgFac == "엘리베이터") & (fe.drtnInfo != prev_station)]
    return [{"car": int(r.car), "door": None if pd.isna(r.door) else int(r.door), "toward": r.drtnInfo,
             "label": r.qckgffVhclDoorNo, "position": r.facPstnNm, "elevator_no": r.elvtrNo}
            for r in f.itertuples()]


def match(car_vec, elevator_car: int, window: int = WINDOW, mirrored: bool = False,
          meaningful: float = MEANINGFUL) -> dict:
    """엘리베이터 칸 주변 후보 가운데 덜 붐비는 칸.

    car_vec  : 칸별 혼잡도 10개 (1번째 칸부터)
    mirrored : 혼잡도 칸 번호가 빠른하차정보와 반대 기준이라고 볼 때 (민감도용)
    반환: nearest(엘리베이터 칸), pick(권하는 칸), extra_cars(더 걷는 칸 수), gain(%p), candidates
    """
    v = np.asarray(car_vec, dtype=float)
    if mirrored:
        v = v[::-1]
    cand = [c for c in range(elevator_car - window, elevator_car + window + 1) if 1 <= c <= len(v)]
    vals = {c: float(v[c - 1]) for c in cand}
    best = min(cand, key=lambda c: (vals[c], abs(c - elevator_car)))
    gain = vals[elevator_car] - vals[best]
    pick = best if gain >= meaningful else elevator_car
    return {"nearest": elevator_car, "nearest_value": round(vals[elevator_car], 1),
            "pick": pick, "pick_value": round(vals[pick], 1), "extra_cars": abs(pick - elevator_car),
            "gain": round(vals[elevator_car] - vals[pick], 1),
            "all_crowded": bool(min(vals.values()) >= 100.0),
            "candidates": {c: round(x, 1) for c, x in vals.items()}}


def survey(fe: pd.DataFrame, pattern: pd.DataFrame, busy: float = 60.0, window: int = WINDOW) -> pd.DataFrame:
    """패턴에 있는 모든 '엘리베이터 역 도착 역간 × 30분' 조합에 매칭을 적용한다 (열차 평균 busy% 이상만)."""
    cols = [f"car_{i}" for i in range(1, N_CARS + 1)]
    rows = []
    for r in pattern[pattern.train_mean >= busy].itertuples():
        for e in elevator_cars(fe, r.to_station, r.from_station):
            vec = [getattr(r, c) for c in cols]
            a, b = match(vec, e["car"], window), match(vec, e["car"], window, mirrored=True)
            rows.append({"from": r.from_station, "to": r.to_station, "toward": e["toward"], "time_bin": r.time_bin,
                         "train_mean": round(r.train_mean, 1), "n_passages": int(r.n_passages),
                         "elevator": e["label"], "nearest": a["nearest"], "nearest_value": a["nearest_value"],
                         "pick": a["pick"], "gain": a["gain"], "extra_cars": a["extra_cars"],
                         "all_crowded": a["all_crowded"], "mirror_pick": b["pick"], "mirror_gain": b["gain"]})
    return pd.DataFrame(rows)


def parse_position(text: str) -> list[tuple[str, int, int]]:
    """'회현 방면9-1, 숙대입구 방면2-4' → [('회현', 9, 1), ('숙대입구', 2, 4)] (공모전 안전시설 '상세위치'와 같은 형식)."""
    out = []
    for m in re.finditer(r"([^,]+?)\s*방면\s*(\d+)-(\d+)", text or ""):
        out.append((m.group(1).strip(), int(m.group(2)), int(m.group(3))))
    return out


# ---------------------------------------------------------------- 공모전판 앱 연결 (1~8호선 엘리베이터 하차 칸)
STATION_ALIAS = {"총신대입구": "이수", "당고개": "불암산"}     # 빠른하차정보 역명 → 앱 역명
FAST_EXIT_SOURCE = "서울교통공사 빠른하차정보(공공데이터포털)"
LINE_CARS = {"1": 10, "2": 10, "3": 10, "4": 10, "5": 8, "6": 8, "7": 8, "8": 6}   # 노선별 편성 칸 수


def build_elevator_mart(fe_all: pd.DataFrame) -> pd.DataFrame:
    """빠른하차정보 원본(전 노선) → 앱용 엘리베이터 표.

    열: line('1'~'8'), station, toward(진행 방향 = 다음 정차역 방면), car, door, label(원문 칸-문), position, elevator_no, crtr_ymd
    칸-문이 '3-2,3-3 사이' 처럼 적힌 행은 앞 칸 번호를 car 로 쓰고 label 은 원문을 그대로 둔다.
    """
    f = fe_all[fe_all.plfmCmgFac == "엘리베이터"].copy()
    m = f.qckgffVhclDoorNo.astype(str).str.extract(r"^(\d+)-(\d+)")
    f["car"] = pd.to_numeric(m[0], errors="coerce")
    f["door"] = pd.to_numeric(m[1], errors="coerce")
    f = f.dropna(subset=["car"])
    out = pd.DataFrame({
        "line": f.lineNm.astype(str).str.replace("호선", "", regex=False).str.strip(),
        "station": f.stnNm.astype(str).str.strip().replace(STATION_ALIAS),
        "toward": f.drtnInfo.astype(str).str.strip().replace(STATION_ALIAS),
        "car": f.car.astype(int), "door": f.door.astype("Int64"),
        "label": f.qckgffVhclDoorNo.astype(str).str.strip(),
        "position": f.facPstnNm.astype(str), "elevator_no": f.elvtrNo.astype(str),
        "crtr_ymd": f.crtrYmd.astype(str)})
    # 편성 칸 수를 넘는 칸 번호는 원본 오류로 보고 안내하지 않는다 (예: 8량 5호선의 '10-4')
    cap = out.line.map(LINE_CARS)
    out = out[cap.isna() | (out.car <= cap)]
    return out.drop_duplicates(["line", "station", "toward", "label"]).reset_index(drop=True)


def alight_elevators(mart: pd.DataFrame | None, line, station: str, prev_station: str | None,
                     next_station: str | None = None) -> list[dict]:
    """하차역 엘리베이터와 가까운 칸-문. 방향을 하나로 정할 수 없으면 빈 목록 (틀린 칸을 안내하지 않는다).

    진행 방향 = 이전역이 아닌 쪽 방면. next_station 을 알면 그 방면을 우선한다.
    남은 방면이 둘 이상이면(분기·종착 등) 안내하지 않는다.
    """
    if mart is None or mart.empty or not station or not prev_station:
        return []
    s = mart[(mart.line.astype(str) == str(line)) & (mart.station == station)]
    s = s[s.toward != prev_station]
    if next_station and (s.toward == next_station).any():
        s = s[s.toward == next_station]
    if s.toward.nunique() != 1:
        return []
    s = s.sort_values(["car", "door"])
    return [{"car": int(r.car), "door": None if pd.isna(r.door) else int(r.door), "label": r.label,
             "toward": r.toward, "position": r.position, "crtr_ymd": r.crtr_ymd} for r in s.itertuples()]


def _bin30(minute_of_day: float) -> str:
    m = int(minute_of_day) % (24 * 60)
    return "%02d:%02d" % (m // 60, 30 * ((m % 60) // 30))


def alight_crowd_match(pattern: pd.DataFrame | None, line, prev_station: str, station: str,
                       arrive_min: float | None, elevators: list[dict], busy: float = 60.0,
                       time_bin: str | None = None) -> dict | None:
    """하차역 직전 역간(이전역→하차역)의 칸별 패턴으로 엘리베이터 칸 주변을 비교한다.

    시간대는 time_bin("07:30") 을 주면 그대로, 아니면 arrive_min(자정 기준 분)의 30분 구간.
    반환
      None                         : 이 노선 패턴이 없음 (비교 불가)
      {"no_sample": True}          : 노선 패턴은 있으나 이 역간·시간대 표본이 기준 미달
      {"quiet": True, ...}         : 평소 붐비지 않는 시간 (열차 평균 < busy)
      {"quiet": False, ...}        : 비교 결과 (match 결과 + 화면 표시용 정수 %)
    화면 표시용 값(pct, near_pct, pick_pct, gain_pct)은 원값을 한 번만 반올림한 정수다.
    gain_pct = near_pct − pick_pct 로 두어 화면의 숫자끼리 산술이 맞게 한다.
    여러 엘리베이터가 있으면 권하는 칸의 혼잡이 가장 낮은 쪽을 고른다.
    """
    if pattern is None or pattern.empty or not elevators:
        return None
    tb = time_bin or (_bin30(arrive_min) if arrive_min is not None else None)
    sub = pattern[(pattern.line.astype(str) == str(line)) & (pattern.from_station == prev_station)
                  & (pattern.to_station == station) & (pattern.time_bin == tb)]
    if sub.empty:
        # 그 노선 패턴은 있지만 이 역간·시간대 표본이 기준(통과 수·날짜 수)에 못 미침 → 비교를 생략했다고 알린다
        return {"no_sample": True, "time_bin": tb} if (pattern.line.astype(str) == str(line)).any() else None
    r = sub.iloc[0]
    vec = [float(r[f"car_{i}"]) for i in range(1, N_CARS + 1)]
    pct = [int(round(v)) for v in vec]
    base = {"time_bin": r.time_bin, "train_mean": round(float(r.train_mean), 1), "n_passages": int(r.n_passages),
            "pct": pct, "no_sample": False}
    if float(r.train_mean) < busy:
        return {**base, "quiet": True}
    best = None
    for e in elevators:
        if not 1 <= e["car"] <= N_CARS:
            continue
        m = match(vec, e["car"])
        if best is None or m["pick_value"] < best["pick_value"]:
            best = {**m, "elevator": e["label"]}
    if best is None:
        return None
    near_pct, pick_pct = pct[best["nearest"] - 1], pct[best["pick"] - 1]
    return {**base, **best, "quiet": False, "near_pct": near_pct, "pick_pct": pick_pct,
            "gain_pct": near_pct - pick_pct}
