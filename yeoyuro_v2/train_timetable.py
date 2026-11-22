"""
yeoyuro_v2/train_timetable.py
=============================
Phase 3A-2a — 1~9호선 열차 단위 시각표 + **시각표 연결 검증** (D-093 사전 등록 2-1절).

왜 필요한가
-----------
경로 점수(기대 소요시간)는 30분 구간 기대값으로 계산한다 (1~9호선 배차÷2, 코레일 E).
기대값은 "평균적으로 이만큼"이지 "이 시각에 이 열차를 탈 수 있다"가 아니다.
그래서 화면에 열차·시각을 안내하거나 시간 역전 원인을 볼 때는, 같은 경로를 **열차 단위**로 따라가 본다.

규칙 (사전 등록 2-1절)
----------------------
- 탑승 구간(leg)마다: (그 역 도착 시각 + 환승 보행) 이후 출발하고, **같은 열차번호로** 그 구간의 내릴 역에
  정차하는 첫 열차를 탄다. 내린 시각 = 그 열차의 도착 시각.
- 열차 묶음(group): '1'(1호선 일반열차만, D-085), '2'~'8'(전체), '9L'(9호선 일반), '9X'(9호선 급행).
- 60분 안에 탈 열차가 없으면 연결 불가. 사유는 R3 세 가지 ('운행 시작 전'·'운행 종료'·'장시간 대기 제외').
- 그날 그 구간을 한 번에 가는 열차가 **하나도 없으면** '불가능한 연결' — 경로 엔진 오류 신호 (R6 오류 기준).

원천 표기 (3A-0, 3A-2a 감사)
-----------------------------
- 시각 'HH:MM:SS', 자정 이후는 24:xx·25:xx. 실제 00~02시 표기는 운행일 +1,440분 (korail_line1.to_minutes).
- 도착 '00:00:00': 급행이면 통과 자리표시(정차 아님). 일반열차면 **도착 시각 누락**으로 보고 출발 시각만 쓴다
  (예: 토요일 3호선 3305 열차 330역 도착 00:00:00·출발 21:08:30). 개수는 감사 보고서에 남긴다.
- 도착 > 출발(음수 정차): 그 행의 도착을 비우고 출발만 쓴다 (예: 평일 9호선 C9247 4122역 도착 24:12:35·출발 24:12:00).
  계산에 음수 정차가 들어가지 않게 하고, 전부 감사 보고서에 남긴다. NEG_DWELL_MAX 를 넘으면 멈춘다.

이 모듈은 경로 점수·순위를 바꾸지 않는다 (사전 등록 2-1절 마지막 줄).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from yeoyuro_v2 import korail_line1 as K

PASS_MARK = K.PASS_MARK
DAY_MAP = K.DAY_MAP
MAX_WAIT_MIN = 60.0
NEG_DWELL_MAX = 20          # 원천 42만 행 중 이보다 많으면 원천 이상으로 멈춤

# 경로 그래프 역코드 → 시각표 역코드 (같은 역, 코드만 다름). 3A-2a 감사에서 확인한 것만
CODE_ALIAS = {("2", 260): 200}       # 2호선 신정지선 까치산: 그래프 260, 시각표 200 (신정네거리 249 다음 종점)
# 원천에서 열차 기록이 끊기는 순환선 경계: 같은 방향 다음 기록으로 이어 탄다 (보행 0, 기다림은 시각표 그대로).
# 2호선 순환 열차는 성수(211)에서 한 바퀴 기록이 끝나고 다른 열차번호로 다시 시작한다 (3A-2a 감사).
# 6호선은 응암(2611)으로 들어오는 기록과 응암순환부터 시작하는 기록이 나뉜다
LOOP_SPLIT = {"2": {211}, "6": {2611}}   # 6호선 응암순환: 응암(2611)에서 기록이 끝나고 다음 기록이 순환부터 시작

# 연결 상태 (사전 등록 2-1절)
CONNECTED = "연결 확인"
PARTIAL = "부분 확인"
NOT_CONNECTED = "연결 불가"
IMPOSSIBLE = "불가능한 연결"          # 그날 그 구간을 한 열차로 가는 경우가 0 — 엔진 오류 신호


# ---------------------------------------------------------------------- 1. 정제
def train_rows(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """원천 전체(1~9호선) → 열차별 정차 행. (정차 행, 감사 수치) 를 돌려준다.

    - 급행 통과 행(도착 00:00:00, 급행) 제외
    - 일반열차 도착 00:00:00 = 도착 누락 → 도착을 비우고 출발만 사용
    - 1호선 급행은 제외 (D-085, 3A-3 전까지 엔진에 없음). 9호선 급행은 9X 묶음으로 유지
    - 음수 정차시간이면 멈춘다
    """
    d = raw.copy()
    d["code"] = d.code.astype(int)
    d["day_type"] = d.day.map(DAY_MAP)
    if d.day_type.isna().any():
        raise K.SourceError(f"알 수 없는 주중주말 값: {sorted(set(d.day[d.day_type.isna()]))}")
    d["express"] = d.exp.astype(int).astype(bool)
    zero = d.arr.eq(PASS_MARK)
    audit = {"rows_raw": int(len(d)),
             "pass_rows_express": int((zero & d.express).sum()),
             "missing_arrival_normal": int((zero & ~d.express).sum()),
             "missing_arrival_examples": d.loc[zero & ~d.express, ["line", "day", "train", "code", "arr", "dep"]]
             .head(5).to_dict("records")}
    d = d[~(zero & d.express)].copy()
    d.loc[d.arr.eq(PASS_MARK), "arr"] = np.nan
    d = d[~((d.line == "1") & d.express)]
    audit["line1_express_rows_dropped"] = int(audit["rows_raw"] - audit["pass_rows_express"] - len(d))
    d["group"] = np.where(d.line == "9", np.where(d.express, "9X", "9L"), d.line)
    d["arr_min"] = d.arr.map(K.to_minutes)
    d["dep_min"] = d.dep.map(K.to_minutes)
    d["t_min"] = d.dep_min.fillna(d.arr_min)
    d = d.sort_values(["day_type", "group", "train", "dir", "t_min"], kind="mergesort")
    d["stop_seq"] = d.groupby(["day_type", "group", "train", "dir"]).cumcount()
    # 음수 정차(도착 > 출발)는 계산에 넣지 않는다: 그 행의 도착을 비우고 출발만 쓴다, 전부 감사에 남김.
    # 원천 전체에서 드물게만 있어야 한다 (NEG_DWELL_MAX 넘으면 원천 문제로 보고 멈춤)
    dw = d.dep_min - d.arr_min
    neg = dw < -1e-9
    audit["negative_dwell_rows"] = d.loc[neg, ["day_type", "line", "train", "code", "arr", "dep"]].to_dict("records")
    if int(neg.sum()) > NEG_DWELL_MAX:
        raise K.SourceError(f"음수 정차시간 {int(neg.sum())}행 (허용 {NEG_DWELL_MAX}) — 멈춤: "
                            f"{audit['negative_dwell_rows'][:5]}")
    d.loc[neg, "arr_min"] = np.nan
    # 같은 열차 안에서 시각이 거꾸로 가는 행 (정렬 뒤 다음 역 도착 < 이번 역 출발) 도 멈춘다
    nxt_arr = d.groupby(["day_type", "group", "train", "dir"]).arr_min.shift(-1)
    back = (nxt_arr < d.dep_min - 1e-9)
    if back.any():
        ex = d.loc[back, ["day_type", "line", "train", "code", "dep"]].head(5).to_dict("records")
        raise K.SourceError(f"열차 안 시각 역행 {int(back.sum())}행 — 멈춤: {ex}")
    out = d.rename(columns={"dir": "direction"})[
        ["day_type", "group", "line", "train", "direction", "stop_seq", "code", "arr_min", "dep_min"]]
    audit["rows_kept"] = int(len(out))
    audit["trains"] = int(out.groupby(["day_type", "group", "train", "direction"]).ngroups)
    return out.reset_index(drop=True), audit


# ---------------------------------------------------------------------- 2. 열차 단위 다음 열차
@dataclass(frozen=True)
class Ride:
    group: str
    train: str
    direction: str
    board_code: int
    alight_code: int
    dep_min: float           # 탄 역 출발
    arr_min: float           # 내린 역 도착 (도착 누락이면 출발)
    wait_min: float          # 그 역 도착(+보행) 뒤 기다린 시간


class TrainIndex:
    """(요일, 묶음, 탄 역, 내릴 역) → 같은 열차로 가는 출발·도착 시각 목록 (필요할 때 만들고 기억).

    2호선 순환열차처럼 한 열차가 같은 역을 두 번 지나도, 탄 역 뒤에 처음 나오는 내릴 역을 쓴다.
    """

    def __init__(self, stops: pd.DataFrame):
        self._by_board: dict[tuple, list] = {}
        self._trains: dict[tuple, dict] = {}
        cols = ["day_type", "group", "train", "direction", "stop_seq", "code", "arr_min", "dep_min"]
        for k, g in stops[cols].groupby(["day_type", "group", "train", "direction"], sort=False):
            seq = g.sort_values("stop_seq")
            codes = seq.code.to_numpy()
            arrs = seq.arr_min.to_numpy(dtype=float)
            deps = seq.dep_min.to_numpy(dtype=float)
            self._trains[k] = (codes, arrs, deps)
            for i in range(len(codes)):
                if not np.isnan(deps[i]):
                    self._by_board.setdefault((k[0], k[1], int(codes[i])), []).append((deps[i], k, i))
        for v in self._by_board.values():
            v.sort(key=lambda x: x[0])
        self._pairs: dict[tuple, tuple[list, list, list]] = {}

    def has_station(self, day_type: str, group: str, code: int) -> bool:
        return (day_type, group, int(code)) in self._by_board

    def pair(self, day_type: str, group: str, board: int, alight: int,
             via: tuple | None = None) -> tuple[list, list, list]:
        """그날 board → alight 를 한 열차로 가는 (출발, 도착, 열차키) 목록, 출발 순.

        via: 탄 역부터 내릴 역까지 정차역 코드 순서 전체. 주면 열차의 정차 순서가 **그대로 이어져야** 한다
             (방향·분기·순환 방향을 경로와 맞추기 위해 — 2호선 반대 방향으로 한 바퀴 돌아 도착하는 열차 배제).
             주지 않으면 탄 역 뒤에 처음 나오는 내릴 역 (방향 확인 없음).
        """
        via = tuple(int(c) for c in via) if via else None
        key = (day_type, group, int(board), int(alight), via)
        if key not in self._pairs:
            deps, arrs, ks = [], [], []
            for dep, k, i in self._by_board.get((day_type, group, int(board)), []):
                codes, a, dd = self._trains[k]
                if via:
                    n = len(via)
                    if i + n > len(codes) or tuple(int(c) for c in codes[i:i + n]) != via:
                        continue
                    j = i + n - 1
                else:
                    hit = np.nonzero(codes[i + 1:] == int(alight))[0]
                    if not len(hit):
                        continue
                    j = i + 1 + int(hit[0])
                deps.append(dep)
                arrs.append(a[j] if not np.isnan(a[j]) else dd[j])
                ks.append(k)
            self._pairs[key] = (deps, arrs, ks)
        return self._pairs[key]

    def next_ride(self, day_type: str, group: str, board: int, alight: int, t_min: float,
                  max_wait: float = MAX_WAIT_MIN, via: tuple | None = None) -> tuple[Ride | None, str | None]:
        """t_min 뒤 처음 타는 열차. (Ride, None) 또는 (None, 사유)."""
        deps, arrs, ks = self.pair(day_type, group, board, alight, via)
        if not deps:
            return None, IMPOSSIBLE
        i = bisect_left(deps, t_min - 1e-9)
        if i >= len(deps):
            return None, K.NO_SERVICE_AFTER_LAST
        if deps[i] - t_min > max_wait + 1e-9:
            return None, (K.NO_SERVICE_BEFORE_FIRST if i == 0 else K.NO_SERVICE_LONG_GAP)
        k = ks[i]
        return Ride(group, k[2], k[3], int(board), int(alight), deps[i], arrs[i], deps[i] - t_min), None


# ---------------------------------------------------------------------- 3. 경로 → 탑승 구간
@dataclass(frozen=True)
class Leg:
    group: str
    board_code: int
    alight_code: int
    board_node: str
    alight_node: str
    walk_before: float = 0.0     # 이 구간을 타기 전 환승 보행 (첫 구간 0)
    codes: tuple = ()            # 탄 역 → 내릴 역 정차역 코드 순서 (방향·분기 확인용)


def group_of_line(line_id: str) -> str:
    return str(line_id)


def _real_code_map(ride: pd.DataFrame) -> dict:
    """노드 → 역코드. 분기 가상 노드(예: 2_성수@seongsu_branch, 코드 9002)는 같은 역 본선 노드의 코드로."""
    m = {}
    for a, b in ((ride.from_node, ride.from_code), (ride.to_node, ride.to_code)):
        for n, c in zip(a, b):
            m.setdefault(n, int(c))
    for n, c in list(m.items()):
        if "@" in n:
            base = n.split("@", 1)[0]
            if base in m and m[base] < 9000:
                m[n] = m[base]
    return m


def legs_from_path(evaluator, path: list[str]) -> list[Leg]:
    """TimeDependentEvaluator 가 보는 것과 같은 규칙으로 경로를 탑승 구간으로 나눈다.

    - 주행 간선이 이어지는 동안 = 한 구간 (같은 노선 묶음)
    - '같은 열차로 통과'인 환승 간선(강동 직결 등, is_through_pass)은 구간을 끊지 않는다
    - 그 밖의 환승 간선 = 구간 경계, 보행시간은 간선 time
    """
    adj = evaluator.s.adj
    ride = evaluator.s.ride
    line_of = {(u, v): str(l) for u, v, l in zip(ride.from_node, ride.to_node, ride.line_id)}
    code = _real_code_map(ride)
    for (ln, a), b in CODE_ALIAS.items():
        for n in [n for n, c in code.items() if c == a and n.split("_", 1)[0] == ln]:
            code[n] = b
    legs: list[Leg] = []
    cur = None                      # dict(group, nodes, codes, walk)
    walk = 0.0

    def close():
        nonlocal cur
        if cur is not None:
            legs.append(Leg(cur["group"], cur["codes"][0], cur["codes"][-1], cur["nodes"][0], cur["nodes"][-1],
                            cur["walk"], tuple(cur["codes"])))
            cur = None

    for i, (u, v) in enumerate(zip(path, path[1:])):
        e = next(x for x in adj[u] if x["to"] == v)
        if e["kind"] == "ride":
            g = group_of_line(line_of[(u, v)])
            if cur is not None and cur["group"] != g:
                close()
            if cur is None:
                cur = {"group": g, "nodes": [u], "codes": [code[u]], "walk": walk}
                walk = 0.0
            cur["nodes"].append(v)
            if code[v] != cur["codes"][-1]:
                cur["codes"].append(code[v])
        elif e["kind"] == "transfer" and evaluator._is_through(path, i, u, v):
            continue
        elif e["kind"] == "transfer":
            close()
            walk += float(e["time"])
    close()
    return legs


# ---------------------------------------------------------------------- 4. 시각표 연결 검증
@dataclass
class Connection:
    status: str                          # CONNECTED / PARTIAL / NOT_CONNECTED / IMPOSSIBLE
    depart_min: float
    arrive_min: float | None             # 연결 확인·부분 확인일 때 시각표 기준 도착
    rides: list = field(default_factory=list)
    reason: str | None = None            # 연결 불가 사유 (R3) 또는 불가능한 구간 설명
    unverified_legs: list = field(default_factory=list)   # 열차 단위 자료가 없어 기대 시간으로 넘긴 구간

    @property
    def first_wait_min(self) -> float | None:
        return self.rides[0].wait_min if self.rides else None


def _ride_leg(index: TrainIndex, day_type: str, leg: Leg, t: float, max_wait: float):
    """구간 하나를 열차로 탄다. 한 열차로 안 되면 순환선 기록 경계(LOOP_SPLIT)에서 한 번 이어 탄다.

    반환: (rides 목록, None) 또는 (None, 사유)
    """
    via = leg.codes or None
    r, why = index.next_ride(day_type, leg.group, leg.board_code, leg.alight_code, t, max_wait, via=via)
    if r is not None:
        return [r], None
    if why != IMPOSSIBLE or not via:
        return None, why
    for k, c in enumerate(via[1:-1], start=1):
        if c not in LOOP_SPLIT.get(leg.group, ()):
            continue
        a, b = via[:k + 1], via[k:]
        r1, w1 = index.next_ride(day_type, leg.group, a[0], a[-1], t, max_wait, via=a)
        if r1 is None:
            if w1 == IMPOSSIBLE:
                continue
            return None, w1
        r2, w2 = index.next_ride(day_type, leg.group, b[0], b[-1], r1.arr_min, max_wait, via=b)
        if r2 is None:
            if w2 == IMPOSSIBLE:
                continue
            return None, w2
        return [r1, r2], None
    return None, IMPOSSIBLE


def verify_connection(index: TrainIndex, legs: list[Leg], depart_min: float, day_type: str,
                      fallback_leg_minutes: dict | None = None,
                      max_wait: float = MAX_WAIT_MIN) -> Connection:
    """출발 시각부터 경로를 열차 단위로 따라간다 (사전 등록 2-1절).

    fallback_leg_minutes: {leg 번호: 기대 소요(분)} — 열차 단위 자료가 없는 구간을 기대 시간으로 넘길 때 (부분 확인)
    """
    t = float(depart_min)
    rides, unverified = [], []
    for n, leg in enumerate(legs):
        t += leg.walk_before
        if not index.has_station(day_type, leg.group, leg.board_code):
            if fallback_leg_minutes and n in fallback_leg_minutes:
                unverified.append(n)
                t += float(fallback_leg_minutes[n])
                continue
            return Connection(PARTIAL, depart_min, None, rides,
                              f"{leg.group} {leg.board_code} 열차 단위 자료 없음", unverified + [n])
        rs, why = _ride_leg(index, day_type, leg, t, max_wait)
        if rs is None:
            st = IMPOSSIBLE if why == IMPOSSIBLE else NOT_CONNECTED
            msg = (f"{leg.group} {leg.board_node}→{leg.alight_node} 를 열차로 이어 갈 수 없음"
                   if st == IMPOSSIBLE else why)
            return Connection(st, depart_min, None, rides, msg, unverified)
        rides.extend(rs)
        t = rs[-1].arr_min
    return Connection(PARTIAL if unverified else CONNECTED, depart_min, t, rides, None, unverified)


def impossible_legs(index: TrainIndex, legs: list[Leg], day_type: str) -> list[Leg]:
    """그날 어떤 시각에도 열차로 이어 갈 수 없는 구간 (R6 오류 기준: 실제로 불가능한 연결)."""
    out = []
    for lg in legs:
        if not index.has_station(day_type, lg.group, lg.board_code):
            continue
        rs, why = _ride_leg(index, day_type, lg, 0.0, max_wait=10_000.0)
        if rs is None and why == IMPOSSIBLE:
            out.append(lg)
    return out


def format_rides(conn: Connection) -> str:
    """화면용 한 줄: '시각표상 08:03 3305 → 08:20 도착' 처럼 (열차번호·시각)."""
    def hm(x):
        x = int(round(x)) % 1440
        return f"{x // 60:02d}:{x % 60:02d}"
    parts = [f"{hm(r.dep_min)} {r.train}({r.group}) → {hm(r.arr_min)}" for r in conn.rides]
    return " · ".join(parts)
