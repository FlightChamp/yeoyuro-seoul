"""
yeoyuro_v2/korail_app.py — 광역철도 확장(코레일 1호선)을 앱에 붙이는 층 (D-101, 사전 등록 9절)
=========================================================================================

앱(Streamlit)과 엔진(korail_engine·train_timetable) 사이의 순수 파이썬 층이다. 화면 코드는 여기 함수만 부르고,
테스트·검증 스크립트(49)는 Streamlit 없이 같은 함수를 부른다 → 화면과 검증이 같은 계산을 본다.

- 끔(기본): 이 모듈은 아무것도 바꾸지 않는다. 앱은 지금과 똑같이 동작한다 (A2·W1).
- 켬: 기존 1~9호선 결과는 그대로 두고, 아래에 '광역철도 포함 경로 (소요시간 기준)' 카드를 따로 만든다 (A3).
  * 후보 = 코레일 켬 엔진(KorailTDRouter, 기본 k=10)의 경로를 기대 소요 순으로
  * 각 후보를 열차 단위로 따라가 3분류 (C2): 연결 확인 / 연결 미확인 / 연결 불가
  * 확정 추천 = 연결 확인 후보 중 기대 소요가 가장 짧은 것 (C3). 없으면 '확인하지 못함' (C4)
  * 비교 기준 = 같은 엔진에서 1~9호선만 쓴 최단 기대 소요 (A4)
- 심야: 00:00~04:29 입력은 전날 운행일의 24:00~28:29 (엔진 parse_hhmm 과 같은 경계, D1·D-102 정정)
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from yeoyuro_v2 import time_dependent as td

# 코레일·열차 단위 모듈(korail_line1·korail_engine·train_timetable)은 **켬일 때만** 불러온다.
# 끔이면 앱이 이 모듈을 import 해도 그 모듈들은 메모리에 올라오지 않는다 (D-093 P1 격리 유지, 테스트로 고정).


def _K():
    from yeoyuro_v2 import korail_line1 as K
    return K


def _T():
    from yeoyuro_v2 import train_timetable as T
    return T

TOGGLE_LABEL = "광역철도 확장 (시범) — 코레일 1호선"
TOGGLE_HELP = ("켜면 1호선 코레일 구간(서울 안 26개 역)을 포함한 경로를 기존 결과 아래에 따로 보여 줍니다. "
               "코레일 구간은 혼잡도 자료가 없어 소요시간 기준으로만 비교합니다.")
CARD_TITLE = "광역철도 포함 경로 (소요시간 기준)"
BADGE_UNKNOWN = "혼잡도 미확인"
NO_COMFORT = "혼잡도 미확인 구간이 있어 쾌적 판정을 하지 않습니다."
NOT_IN_BASE = "이 역은 1~9호선 경로 계산 대상이 아닙니다. 아래 광역철도 포함 경로만 보여 드립니다."
OFF_CLEARED = "광역철도 확장을 끄면 이 역은 선택할 수 없습니다. 출발·도착역을 다시 골라 주세요."
NONE_FOUND = "이 시각에 이용 가능한 광역철도 확장 경로를 확인하지 못했습니다."
NO_BENEFIT = "광역철도를 이용해도 더 빠른 경로가 없습니다."
BASIS_EXPECTED = ("기대 소요 = 첫 승차 대기 제외(R4) + 갈아타는 역의 대기는 그 시각 30분 평균 대기(E)로 계산한 값입니다.")
BASIS_TIMETABLE = ("시각표 확인 = 입력 시각 뒤 첫 열차부터 실제 열차번호로 이어 탄 결과입니다(첫 대기 포함). "
                   "두 값은 다를 수 있습니다.")
BASIS_STATIC = "정해진 시각표(2026-09-01) 기준이며 실시간 지연·운행 변경은 반영하지 않습니다."

CONNECT_OK, CONNECT_UNKNOWN, CONNECT_FAIL = "연결 확인", "연결 미확인", "연결 불가"


def _status_ko(status: str) -> str:
    T = _T()
    return {T.CONNECTED: CONNECT_OK, T.PARTIAL: CONNECT_UNKNOWN}.get(status, CONNECT_FAIL)


DAY_KO = "월화수목금토일"


def korail_codes() -> frozenset:
    """코레일 구간 역코드 (서울역~청량리 150~158 은 서울교통공사 구간이라 혼잡도가 있음)."""
    return frozenset(c for c in _K().SCOPE if c >= 1000)

LINE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


# ---------------------------------------------------------------------- 역 목록 (B)
def _scope_names() -> list[str]:
    out = []
    for n in _K().SCOPE.values():
        if n not in out:
            out.append(n)
    return out


def korail_only_stations(display: pd.DataFrame) -> list[str]:
    """지금 앱 목록에 없는 코레일 1호선 역 (B2). 순서 = 코레일 범위 정의 순서."""
    have = set(display["station_key"])
    return [n for n in _scope_names() if n not in have]


def common_stations(display: pd.DataFrame) -> list[str]:
    """앱 목록에 이미 있지만 1호선 배지가 없는 역 (B3) — 새 항목을 만들지 않고 배지만 더한다."""
    d = display.set_index("station_key")
    out = []
    for n in _scope_names():
        if n in d.index and "1" not in str(d.at[n, "available_lines"]).split(","):
            out.append(n)
    return out


def _sorted_lines(lines) -> str:
    xs = [x for x in lines if x]
    return ",".join(sorted(set(xs), key=lambda x: (LINE_ORDER.index(x) if x in LINE_ORDER else 99, x)))


def extended_display(display: pd.DataFrame) -> pd.DataFrame:
    """켬일 때 쓰는 역 목록 복사본. 원본은 건드리지 않는다.

    - 공통 역 8곳: available_lines 에 '1' 만 더함
    - 코레일 전용 역 18곳: 새 행 (구로는 엔진 안 두 노드를 한 역으로)
    """
    d = display.copy()
    for n in common_stations(display):
        i = d.index[d.station_key == n][0]
        d.at[i, "available_lines"] = _sorted_lines(str(d.at[i, "available_lines"]).split(",") + ["1"])
        if "n_lines" in d.columns:
            d.at[i, "n_lines"] = len(str(d.at[i, "available_lines"]).split(","))
        if "is_transfer_station" in d.columns:
            d.at[i, "is_transfer_station"] = True
    rows = []
    for n in korail_only_stations(display):
        uids = "1_구로@경부;1_구로@경인" if n == "구로" else f"1_{n}"
        row = {c: np.nan for c in d.columns}
        row.update({"station_key": n, "station_name": n, "display_name": n, "available_lines": "1",
                    "candidate_station_uids": uids})
        for c, v in (("n_lines", 1), ("is_transfer_station", False), ("is_event_station", False),
                     ("is_branch_node", False), ("alias_or_note", "코레일 1호선 (광역철도 확장)")):
            if c in d.columns:
                row[c] = v
        rows.append(row)
    if rows:
        d = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
    return d


def is_korail_only(key: str | None, display: pd.DataFrame) -> bool:
    return bool(key) and key in set(korail_only_stations(display))


# ---------------------------------------------------------------------- 운행일 (D)
@dataclass
class ServiceDay:
    date: dt.date            # 운행일 (심야 입력이면 전날)
    day_type: str            # weekday / saturday / sunday
    minute: float            # 운행일 기준 절대 분 (심야면 1440 이상)
    late_night: bool
    label: str               # 화면 문구


def day_type_of(date: dt.date) -> str:
    w = date.weekday()
    return "saturday" if w == 5 else ("sunday" if w == 6 else "weekday")


def service_day(date, hhmm: str) -> ServiceDay:
    """입력 날짜·시각 → 운행일. 시각 경계는 엔진 parse_hhmm 과 같다 (04:29 까지 전날 심야)."""
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    m = td.parse_hhmm(hhmm)
    late = m >= 1440
    sd = date - dt.timedelta(days=1) if late else date
    label = "운행일 기준: %d월 %d일(%s)%s" % (sd.month, sd.day, DAY_KO[sd.weekday()], " 심야" if late else "")
    return ServiceDay(sd, day_type_of(sd), float(m), late, label)


# ---------------------------------------------------------------------- 계획 (A·C)
@dataclass
class LegView:
    line: str
    board: str
    alight: str
    korail: bool             # 코레일 구간 포함 → '혼잡도 미확인'
    max_congestion: float | None   # 1~9호선 구간만 (코레일이면 None)


@dataclass
class RideView:
    train: str
    board: str
    dep: str
    alight: str
    arr: str
    wait_min: float


@dataclass
class CandidateView:
    path: list
    expected_min: float              # 기대 소요 (R4·E)
    expected_arrive: str
    transfers: int
    korail: bool
    legs: list = field(default_factory=list)
    status: str = ""                 # 연결 확인 / 연결 미확인 / 연결 불가
    reason: str | None = None
    train_arrive: str | None = None  # 시각표 확인 도착
    train_total_min: float | None = None
    first_wait_min: float | None = None
    rides: list = field(default_factory=list)
    line1_splits: list = field(default_factory=list)   # [(역 이름, 연결 여유 분)]
    congestion_judgement: str = ""


@dataclass
class ExtensionPlan:
    origin: str
    dest: str
    service: ServiceDay
    status: str                      # ok / no_benefit / none
    best: CandidateView | None
    baseline_min: float | None       # 같은 엔진, 1~9호선만 최단 기대 소요 (코레일 전용 역이면 None)
    candidates: list = field(default_factory=list)
    excluded_engine: Counter = field(default_factory=Counter)   # 엔진 단계 R3 제외 사유별 수
    excluded_train: Counter = field(default_factory=Counter)    # 열차 단위 제외(연결 불가·미확인) 사유별 수

    @property
    def main_reason(self) -> str | None:
        c = self.excluded_train + self.excluded_engine
        return c.most_common(1)[0][0] if c else None

    @property
    def saving_min(self) -> float | None:
        if self.best is None or self.baseline_min is None:
            return None
        return self.baseline_min - self.best.expected_min


def _reason_short(conn) -> str:
    """연결 불가·미확인 사유를 화면용 짧은 말로."""
    T, K = _T(), _K()
    if conn.status == T.IMPOSSIBLE:
        return "갈아탈 열차 없음"
    if conn.status == T.PARTIAL:
        return "열차 단위 자료 없음"
    r = conn.reason or ""
    if r == K.NO_SERVICE_AFTER_LAST:
        return "막차 종료"
    return r or conn.status


def _leg_congestion(trace, legs) -> list:
    """경로 trace(주행 간선)를 탑승 구간 순서대로 나눠 구간별 최대 혼잡도 (코레일 간선은 값 없음 → 무시)."""
    out, i = [], 0
    rides = [x for x in trace if x.kind == "ride"]
    for lg in legs:
        vals = []
        # 이 구간의 첫 간선 찾기
        while i < len(rides) and rides[i].u != lg.board_node:
            i += 1
        while i < len(rides):
            x = rides[i]
            if x.congestion_max is not None and not np.isnan(x.congestion_max):
                vals.append(float(x.congestion_max))
            i += 1
            if x.v == lg.alight_node:
                break
        out.append(max(vals) if vals else None)
    return out


class ExtensionPlanner:
    """하나의 운행 요일(day_type)에 대한 확장 경로 계산기. 앱은 요일·조회 날짜마다 하나를 만들어 둔다."""

    def __init__(self, root: Path, mod, sr, display: pd.DataFrame, day_type: str, query_date=None,
                 stop_times: pd.DataFrame | None = None, repo_root: Path | None = None):
        from yeoyuro_v2 import korail_engine as KE
        T = _T()
        kw = dict(policy="step", multi_bin=True, query_date=None if query_date is None else str(query_date))
        self.day_type = day_type
        self.display = display
        self.on = KE.KorailTDRouter(root, mod, sr, display, day_type, include_korail=True, **kw)   # k = 기본 10
        self.off = td.TDRouter(root, mod, sr, display, day_type, k=5, **kw)                      # 앱 대안 카드와 같은 설정
        if stop_times is None:
            base = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
            stop_times = pd.read_parquet(base / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet")
        self.tidx = T.TrainIndex(stop_times[stop_times.day_type == day_type])
        self._only = set(korail_only_stations(display))
        self._names: dict | None = None

    # --- 표시용 이름
    def _code_names(self, ev) -> dict:
        if self._names is None:
            T, K = _T(), _K()
            m = T._real_code_map(ev.s.ride)
            names = {}
            for node, c in m.items():
                names.setdefault(int(c), node.split("_", 1)[1].split("@")[0])
            for c, n in K.SCOPE.items():
                names[int(c)] = n
            self._names = names
        return self._names

    def _view(self, q, ev, minute: float, with_congestion: bool = False) -> CandidateView:
        T, K = _T(), _K()
        kc = korail_codes()
        legs = T.legs_from_path(ev, q.path)
        conn = T.verify_connection(self.tidx, legs, minute, self.day_type)
        names = self._code_names(ev)
        lv = []
        trace = []
        if with_congestion:                              # 확정 추천만: 같은 경로를 trace 와 함께 다시 평가 (값은 같음)
            trace = ev.evaluate(list(q.path), minute, keep_trace=True).trace
        cong = _leg_congestion(trace, legs)
        for lg, mx in zip(legs, cong):
            kor = any(int(c) in kc for c in (lg.codes or (lg.board_code, lg.alight_code)))
            lv.append(LegView(lg.group, lg.board_node.split("_", 1)[1].split("@")[0],
                              lg.alight_node.split("_", 1)[1].split("@")[0], kor, None if kor else mx))
        rides = [RideView(r.train, names.get(int(r.board_code), str(r.board_code)), td.fmt_min(r.dep_min),
                          names.get(int(r.alight_code), str(r.alight_code)), td.fmt_min(r.arr_min), float(r.wait_min))
                 for r in conn.rides]
        status = _status_ko(conn.status)
        return CandidateView(
            path=list(q.path), expected_min=float(q.actual_time_min), expected_arrive=td.fmt_min(q.arrive_min),
            transfers=int(q.transfer_count), korail=bool(getattr(q, "korail", False)), legs=lv, status=status,
            reason=None if status == CONNECT_OK else _reason_short(conn),
            train_arrive=None if conn.arrive_min is None or status != CONNECT_OK else td.fmt_min(conn.arrive_min),
            train_total_min=None if conn.arrive_min is None or status != CONNECT_OK else float(conn.arrive_min - minute),
            first_wait_min=conn.first_wait_min,
            rides=rides,
            line1_splits=[(K.SCOPE.get(c, str(c)), float(s)) for c, s in (conn.line1_splits or [])],
            congestion_judgement=getattr(q, "congestion_judgement", "") or "")

    def plan(self, origin: str, dest: str, service: ServiceDay) -> ExtensionPlan:
        m = float(service.minute)
        res = self.on.evaluate_od(origin, dest, m)
        excluded_engine = Counter(x.no_service_reason for x in getattr(self.on, "excluded", []) if x.no_service_reason)
        ev = self.on.evaluator(td.bin_index_of(m))
        cands, excluded_train, best = [], Counter(), None
        for q in res:                                    # 이미 기대 소요 순 (엔진 정렬)
            v = self._view(q, ev, m)
            if v.status != CONNECT_OK:
                excluded_train[v.reason or v.status] += 1
            elif best is None:
                v = self._view(q, ev, m, with_congestion=True)
                best = v
            cands.append(v)
        baseline = None
        if origin not in self._only and dest not in self._only:
            try:
                b = self.off.evaluate_od(origin, dest, m)
                baseline = float(b[0].actual_time_min) if b else None
            except Exception:                            # noqa: BLE001 — 기준값은 참고용
                baseline = None
        if best is None:
            status = "none"
        elif not best.korail:
            status = "no_benefit"
        else:
            status = "ok"
        return ExtensionPlan(origin, dest, service, status, best, baseline, cands, excluded_engine, excluded_train)


# ---------------------------------------------------------------------- 노선도 (B5)
def map_overlay(workbook: Path) -> dict:
    """노선도 엑셀(예정 행 포함)에서 3-A·3-B 그림 재료를 만든다. 엑셀은 읽기만 한다 (phase 열 그대로).

    반환:
      a_nodes / b_nodes : visual_node_id, station_key, x_px, y_px, is_transfer_station
      a_segs / b_segs   : x0, y0, x1, y1 (노선 순서대로 이웃 점)
      a_links           : 3-A 환승 연결 (x0, y0, x1, y1)
      a_click / a_labels: 3-A 에서 새로 생기는 역(코레일 전용)의 클릭 영역·역명 (3-B 는 클릭·역명 없음)
    """
    from yeoyuro_v2 import map_workbook as mw
    s = mw.materialize(Path(workbook), include_planned=True)
    vn, ls, tl = s["Station_Visual_Nodes"], s["Line_Sequences"], s["Transfer_Links"]
    pos = {r.visual_node_id: (float(r.x_px), float(r.y_px)) for r in vn.itertuples()
           if pd.notna(r.x_px) and pd.notna(r.y_px)}
    out = {}
    for ph, tag in (("3-A", "a"), ("3-B", "b")):
        n = vn[vn["phase"] == ph]
        out[f"{tag}_nodes"] = n[["visual_node_id", "station_key", "x_px", "y_px", "is_transfer_station"]].reset_index(drop=True)
        segs = []
        for _, g in ls[ls["phase"] == ph].groupby(["line_id", "branch_code"], sort=False):
            ids = g.sort_values("path_order").visual_node_id.tolist()
            for a, b in zip(ids, ids[1:]):
                if a in pos and b in pos:
                    segs.append((*pos[a], *pos[b]))
        out[f"{tag}_segs"] = pd.DataFrame(segs, columns=["x0", "y0", "x1", "y1"])
    links = []
    for r in tl[tl["phase"] == "3-A"].itertuples():
        a, b = r.from_visual_node_id, r.to_visual_node_id
        if a in pos and b in pos:
            links.append((*pos[a], *pos[b]))
    out["a_links"] = pd.DataFrame(links, columns=["x0", "y0", "x1", "y1"])
    new_a = [k for k in vn[vn["phase"] == "3-A"].station_key.unique()
             if k not in set(mw.materialize(Path(workbook))["Station_Click_Areas"].station_key)]
    ca, lb = s["Station_Click_Areas"], s["Station_Labels"]
    out["a_click"] = ca[ca.station_key.isin(new_a)].reset_index(drop=True)
    out["a_labels"] = lb[lb.station_key.isin(new_a)].reset_index(drop=True)
    out["a_new_stations"] = new_a
    return out
