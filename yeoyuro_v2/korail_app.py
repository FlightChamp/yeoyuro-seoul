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
- 심야: 00:00~04:29 입력은 전날 운행일의 24:00~28:29 (엔진 parse_hhmm 과 같은 경계, D1·D-102 정정, D-103 공통 모듈)
- 3B-3 (D-105·D-106): 같은 토글로 3-B(경의중앙·경춘·수인분당·공항철도)도 켠다 — 함수마다 include_3b 인자
  (기본 False = 3-A 만, 기존 검사와 같음). 앱은 3-B 정차표가 있을 때만 include_3b=True. 공항철도 구간 경로는
  열차 연결을 확인해도 '참고 경로' (확정 추천 아님, D-105 2절·B13).
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

TOGGLE_LABEL = "광역철도 확장 (시범)"
TOGGLE_HELP = ("켜면 코레일 1호선(서울 안 26개 역)과 경의중앙선·수인분당선·공항철도 구간을 포함한 경로를 기존 결과 "
               "아래에 따로 보여 줍니다. 확장 구간은 혼잡도 자료가 없어 소요시간 기준으로만 비교합니다.")
TOGGLE_HELP_3A_ONLY = ("켜면 코레일 1호선(서울 안 26개 역)을 포함한 경로를 기존 결과 아래에 따로 보여 줍니다. "
                       "경의중앙선·수인분당선·공항철도는 시각표 자료가 이 환경에 없어 계산하지 않습니다.")
K3B_MISSING = "경의중앙선·수인분당선·공항철도 시각표 자료가 이 환경에 없어 코레일 1호선만 계산합니다."
K3B_LINE_NAMES = {"GJ": "경의중앙선", "GC": "경춘선", "SB": "수인분당선", "AR": "공항철도"}
REFERENCE_TITLE = "참고 경로"
AR_REFERENCE = ("공항철도 2025-12-29 시각표 기준 (2026-10-23 개정 반영 여부 미확인) — 열차 연결은 계산했지만 "
                "현행 시각표로 확인한 확정 추천이 아닙니다.")
SATURDAY_3B = "토요일은 경의중앙선·경춘선·수인분당선·공항철도 휴일 시각표 기준입니다."
CARD_TITLE = "광역철도 포함 경로 (소요시간 기준)"
BADGE_UNKNOWN = "혼잡도 미확인"
NO_COMFORT = "혼잡도 미확인 구간이 있어 쾌적 판정을 하지 않습니다."
NOT_IN_BASE = "이 역은 1~9호선 경로 계산 대상이 아닙니다. 아래 광역철도 포함 경로만 보여 드립니다."
OFF_CLEARED = "광역철도 확장을 끄면 이 역은 선택할 수 없습니다. 출발·도착역을 다시 골라 주세요."
NONE_FOUND = "이 시각에 이용 가능한 광역철도 확장 경로를 확인하지 못했습니다."
NO_BENEFIT = "광역철도를 이용해도 더 빠른 경로가 없습니다."
NO_BENEFIT_WITH_REF = ("확정 추천할 수 있는 광역철도 경로 중에는 더 빠른 경로가 없습니다. "
                       "공항철도를 이용한 참고 경로는 아래에 따로 보여 드립니다.")
BASIS_EXPECTED = ("기대 소요 = 첫 승차 대기 제외(R4) + 갈아타는 역의 대기는 그 시각 30분 평균 대기(E)로 계산한 값입니다.")
BASIS_TIMETABLE = ("시각표 확인 = 입력 시각 뒤 첫 열차부터 실제 열차번호로 이어 탄 결과입니다(첫 대기 포함). "
                   "두 값은 다를 수 있습니다.")
BASIS_STATIC = "정해진 시각표(2026-09-01) 기준이며 실시간 지연·운행 변경은 반영하지 않습니다."
HOLIDAY_EXT = ("공휴일이라 휴일 시각표로 열차 연결을 확인했습니다. 1~9호선 구간의 혼잡·배차는 공휴일 관측값이 아니라 "
               "요일 기준 통계입니다. 토요일과 겹친 공휴일의 시각표, 선거일 특별 운행은 확인하지 못했습니다.")
HOLIDAY_BASE = ("공휴일 — 이 카드의 배차·혼잡은 요일 기준 통계입니다(공휴일 관측값·공휴일 시각표 아님).")
OUT_OF_RANGE = "공휴일 판정은 2025~2027년만 지원합니다. 이 날짜는 요일로만 판정했습니다."

CONNECT_OK, CONNECT_UNKNOWN, CONNECT_FAIL = "연결 확인", "연결 미확인", "연결 불가"


def _status_ko(status: str) -> str:
    T = _T()
    return {T.CONNECTED: CONNECT_OK, T.PARTIAL: CONNECT_UNKNOWN}.get(status, CONNECT_FAIL)


DAY_KO = "월화수목금토일"


def korail_codes() -> frozenset:
    """코레일 구간 역코드 (서울역~청량리 150~158 은 서울교통공사 구간이라 혼잡도가 있음)."""
    return frozenset(c for c in _K().SCOPE if c >= 1000)

LINE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "GJ", "SB", "AR"]


def k3b_ready(repo_root: Path | None = None) -> bool:
    """3-B 정차표가 있는가 (없으면 앱은 3-A 만). 엔진 모듈을 불러오지 않고 파일만 본다 (끔 격리)."""
    base = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    return (base / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet").exists()


def _k3b_stations() -> dict:
    """3-B 역 → 노선 묶음 목록 (노선도·역 목록용, 경춘선은 상봉·망우 승강장 노드만 따로)."""
    from yeoyuro_v2 import korail_3b as K3
    out: dict = {}
    for g in ("GJ", "SB", "AR"):
        for n in K3.ORDER[g]:
            out.setdefault(n, [])
            if g not in out[n]:
                out[n].append(g)
    return out


def _k3b_uids(name: str, groups) -> list:
    u = [f"{g}_{name}" for g in groups]
    if name in ("상봉", "망우"):
        u.append(f"GC_{name}")
    return u


# ---------------------------------------------------------------------- 역 목록 (B)
def _scope_names() -> list[str]:
    out = []
    for n in _K().SCOPE.values():
        if n not in out:
            out.append(n)
    return out


def korail_only_stations(display: pd.DataFrame, include_3b: bool = False) -> list[str]:
    """지금 앱 목록에 없는 확장 역 (B2). 코레일 1호선 범위 순서, include_3b 면 뒤에 3-B 전용 역 (D-105 A1)."""
    have = set(display["station_key"])
    out = [n for n in _scope_names() if n not in have]
    if include_3b:
        out += [n for n in _k3b_stations() if n not in have and n not in out]
    return out


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


def extended_display(display: pd.DataFrame, include_3b: bool = False) -> pd.DataFrame:
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
    if include_3b:                                   # 3B-3: 3-B 노선 배지·후보 노드, 3-B 전용 역 새 행
        k3 = _k3b_stations()
        rows = []
        for n, groups in k3.items():
            hit = d.index[d.station_key == n]
            if len(hit):
                i = hit[0]
                d.at[i, "available_lines"] = _sorted_lines(str(d.at[i, "available_lines"]).split(",") + groups)
                cur = [x for x in str(d.at[i, "candidate_station_uids"]).split(";") if x and x != "nan"]
                d.at[i, "candidate_station_uids"] = ";".join(cur + [u for u in _k3b_uids(n, groups) if u not in cur])
                if "n_lines" in d.columns:
                    d.at[i, "n_lines"] = len(str(d.at[i, "available_lines"]).split(","))
                if "is_transfer_station" in d.columns and len(str(d.at[i, "available_lines"]).split(",")) > 1:
                    d.at[i, "is_transfer_station"] = True
            else:
                row = {c: np.nan for c in d.columns}
                row.update({"station_key": n, "station_name": n, "display_name": n,
                            "available_lines": _sorted_lines(groups), "candidate_station_uids": ";".join(_k3b_uids(n, groups))})
                for c, v in (("n_lines", len(groups)), ("is_transfer_station", len(groups) > 1),
                             ("is_event_station", False), ("is_branch_node", False),
                             ("alias_or_note", " · ".join(K3B_LINE_NAMES[g] for g in groups) + " (광역철도 확장)")):
                    if c in d.columns:
                        row[c] = v
                rows.append(row)
        if rows:
            d = pd.concat([d, pd.DataFrame(rows)], ignore_index=True)
    return d


def is_korail_only(key: str | None, display: pd.DataFrame, include_3b: bool = False) -> bool:
    return bool(key) and key in set(korail_only_stations(display, include_3b))


# ---------------------------------------------------------------------- 운행일 (D)
@dataclass
class ServiceDay:
    date: dt.date            # 운행일 (심야 입력이면 전날)
    day_type: str            # weekday / saturday / sunday
    minute: float            # 운행일 기준 절대 분 (심야면 1440 이상)
    late_night: bool
    label: str               # 화면 문구
    timetable_day_type: str = ""     # D-104: 공휴일이면 sunday(END) — 확장 계산기는 이 요일의 시각표를 쓴다
    holiday: tuple = ()
    holiday_supported: bool = True


def day_type_of(date: dt.date) -> str:
    from yeoyuro_v2.service_day import day_type_of as f
    return f(date)


def service_day(date, hhmm: str) -> ServiceDay:
    """입력 날짜·시각 → 운행일. D-103: 공통 시간 모듈(service_day.service_clock)을 그대로 쓴다 (값은 D-101 과 같음)."""
    from yeoyuro_v2.service_day import service_clock
    c = service_clock(date, hhmm)
    return ServiceDay(c.service_date, c.day_type, float(c.minute), c.late_night, c.label,
                      c.timetable_day_type, c.holiday, c.holiday_supported)


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
    # 3B-3: 3-B 메타데이터 (엔진 결과 그대로). 연결 상태(status)와 따로 — 공항철도면 recommendation_status = '참고 경로'
    lines_3b: tuple = ()
    timetable_valid_from: dict = field(default_factory=dict)
    freshness_status: str = ""
    recommendation_status: str = ""

    @property
    def is_reference(self) -> bool:
        return self.recommendation_status == REFERENCE_TITLE


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
    reference: CandidateView | None = None   # 3B-3: 확정 추천보다 빠른(또는 확정 추천이 없을 때) 공항철도 '참고 경로'
    status_3b: str = "off"                   # on / unavailable / off

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
                 stop_times: pd.DataFrame | None = None, repo_root: Path | None = None, include_3b: bool = False,
                 router_kw: dict | None = None, keep_3a: bool = False):
        """router_kw: 3B-4 민감도(B5·B8·B12)용 KorailTDRouter 추가 인자 (b_factor·u_shift·zero_walk_margin·extra_margin·
        min_graph·k·k3b_stop_times). 앱은 넘기지 않는다 (기본 동작 불변).
        keep_3a: D-110 V5 — 3-A 켬 계산기(3-B 끔)의 후보를 더해 같은 3-B 켬 평가기·검증기로 다시 평가 (기본 꺼짐)."""
        from yeoyuro_v2 import korail_engine as KE
        T = _T()
        kw = dict(policy="step", multi_bin=True, query_date=None if query_date is None else str(query_date))
        self.day_type = day_type
        self.display = display
        self.include_3b = bool(include_3b)
        self.on = KE.KorailTDRouter(root, mod, sr, display, day_type, include_korail=True,   # k = 기본 10
                                    include_3b=self.include_3b, **kw, **(router_kw or {}))
        self.keep_3a = bool(keep_3a)
        self._ctor = (root, mod, sr, display, day_type, kw)
        self._r3a = None
        self.off = td.TDRouter(root, mod, sr, display, day_type, k=5, **kw)                      # 앱 대안 카드와 같은 설정
        if stop_times is None:
            base = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
            stop_times = pd.read_parquet(base / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet")
        if self.on.status_3b == "on":                        # 3-B: 엔진의 검증 색인(1~9 + 3-B)을 같이 쓴다
            self.tidx = self.on.verify_index()
        else:
            self.tidx = T.TrainIndex(stop_times[stop_times.day_type == day_type])
        self._only = set(korail_only_stations(display, self.on.status_3b == "on"))
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
            if self.on.status_3b == "on":
                from yeoyuro_v2 import korail_3b as K3
                for n, c in K3.CODE.items():
                    names.setdefault(int(c), n)
                for n, (_, c) in K3.AR_CODE.items():
                    names[int(c)] = n
            self._names = names
        return self._names

    def _view(self, q, ev, minute: float, with_congestion: bool = False) -> CandidateView:
        T, K = _T(), _K()
        kc = korail_codes()
        if self.on.status_3b == "on":                        # 3B-2: 방향별 보행·0분 환승 연결 여유
            legs = self.on.verification_legs(ev, q.path)
        else:
            legs = T.legs_from_path(ev, q.path)
        conn = T.verify_connection(self.tidx, legs, minute, self.day_type)
        names = self._code_names(ev)
        lv = []
        trace = []
        if with_congestion:                              # 확정 추천만: 같은 경로를 trace 와 함께 다시 평가 (값은 같음)
            trace = ev.evaluate(list(q.path), minute, keep_trace=True).trace
        cong = _leg_congestion(trace, legs)
        for lg, mx in zip(legs, cong):
            k3 = lg.group in ("GJ", "GC", "GJC", "SB", "AR")
            kor = k3 or (lg.group == "1" and any(int(c) in kc for c in (lg.codes or (lg.board_code, lg.alight_code))))
            lv.append(LegView("GJ" if lg.group == "GJC" else lg.group, lg.board_node.split("_", 1)[1].split("@")[0],
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
            congestion_judgement=getattr(q, "congestion_judgement", "") or "",
            lines_3b=tuple(getattr(q, "lines_3b", ()) or ()),
            timetable_valid_from=dict(getattr(q, "timetable_valid_from", {}) or {}),
            freshness_status=getattr(q, "freshness_status", "") or "",
            recommendation_status=getattr(q, "recommendation_status", "") or "")

    def router_3a(self):
        """D-110 V5: 3-A 켬 계산기 (include_korail=True, include_3b=False). 처음 부를 때 만든다."""
        if self._r3a is None:
            from yeoyuro_v2 import korail_engine as KE
            root, mod, sr, display, day_type, kw = self._ctor
            self._r3a = KE.KorailTDRouter(root, mod, sr, display, day_type, include_korail=True, include_3b=False, **kw)
        return self._r3a

    def extra_3a(self, origin: str, dest: str, m: float, have: set) -> tuple[list, Counter]:
        """D-110 V5: 3-A 후보 중 아직 없는 경로를 3-B 켬 평가기로 평가. (운행 있는 결과, 운행 없음 사유 수)"""
        try:
            paths = self.router_3a().candidates(origin, dest, m)
        except Exception:                                # noqa: BLE001 — 3-A 계산기에 없는 역(3-B 전용 역 등)
            paths = []
        ev = self.on.evaluator(td.bin_index_of(m))
        out, ex = [], Counter()
        for p in paths:
            if tuple(p) in have:
                continue
            have.add(tuple(p))
            r = ev.evaluate(list(p), m)
            if getattr(r, "no_service_reason", None):
                ex[r.no_service_reason] += 1
            else:
                out.append(r)
        return out, ex

    def plan(self, origin: str, dest: str, service: ServiceDay) -> ExtensionPlan:
        m = float(service.minute)
        res = self.on.evaluate_od(origin, dest, m)
        excluded_engine = Counter(x.no_service_reason for x in getattr(self.on, "excluded", []) if x.no_service_reason)
        if self.keep_3a:
            have = {tuple(q.path) for q in res} | {tuple(x.path) for x in getattr(self.on, "excluded", [])}
            more, ex = self.extra_3a(origin, dest, m, have)
            excluded_engine += ex
            res = sorted(res + more, key=lambda r: (round(r.actual_time_min, 6), r.perceived_time_min))
        ev = self.on.evaluator(td.bin_index_of(m))
        cands, excluded_train, best, reference = [], Counter(), None, None
        for q in res:                                    # 이미 기대 소요 순 (엔진 정렬)
            v = self._view(q, ev, m)
            if v.status != CONNECT_OK:
                excluded_train[v.reason or v.status] += 1
            elif v.is_reference:                         # 공항철도: 확정 추천 아님 (D-105 2절)
                if reference is None and best is None:
                    reference = self._view(q, ev, m, with_congestion=True)
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
            status = "reference" if reference is not None else "none"
        elif not best.korail:
            status = "no_benefit"
        else:
            status = "ok"
        return ExtensionPlan(origin, dest, service, status, best, baseline, cands, excluded_engine, excluded_train,
                             reference=reference, status_3b=self.on.status_3b)


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
        for (lid, _), g in ls[ls["phase"] == ph].groupby(["line_id", "branch_code"], sort=False):
            ids = g.sort_values("path_order").visual_node_id.tolist()
            for a, b in zip(ids, ids[1:]):
                if a in pos and b in pos:
                    segs.append((*pos[a], *pos[b], str(lid)))
        out[f"{tag}_segs"] = pd.DataFrame(segs, columns=["x0", "y0", "x1", "y1", "line_id"])
        links = []
        for r in tl[tl["phase"] == ph].itertuples():
            a, b = r.from_visual_node_id, r.to_visual_node_id
            if a in pos and b in pos:
                links.append((*pos[a], *pos[b]))
        out[f"{tag}_links"] = pd.DataFrame(links, columns=["x0", "y0", "x1", "y1"])
    new_a = [k for k in vn[vn["phase"] == "3-A"].station_key.unique()
             if k not in set(mw.materialize(Path(workbook))["Station_Click_Areas"].station_key)]
    ca, lb = s["Station_Click_Areas"], s["Station_Labels"]
    out["a_click"] = ca[ca.station_key.isin(new_a)].reset_index(drop=True)
    out["a_labels"] = lb[lb.station_key.isin(new_a)].reset_index(drop=True)
    out["a_new_stations"] = new_a
    base_keys = set(mw.materialize(Path(workbook))["Station_Click_Areas"].station_key)
    new_b = [k for k in vn[vn["phase"] == "3-B"].station_key.unique() if k not in base_keys and k not in set(new_a)]
    out["b_click"] = ca[ca.station_key.isin(new_b)].reset_index(drop=True)      # 3B-3: 켬 + 3-B 자료 있을 때만 씀
    out["b_labels"] = lb[lb.station_key.isin(new_b)].reset_index(drop=True)
    out["b_new_stations"] = new_b
    return out
