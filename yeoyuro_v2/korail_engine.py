"""
yeoyuro_v2/korail_engine.py
===========================
Phase 3A-2b — 코레일 1호선 서울 구간을 경로 엔진에 **켤 때만** 덧붙이는 확장 (D-093·D-095 사전 등록).

기존 엔진(time_dependent.TDRouter / TimeDependentEvaluator)과 v1 RouteScorer 는 **한 줄도 바꾸지 않는다.**
이 모듈의 KorailTDRouter 는 include_korail=False 면 TDRouter 와 똑같이 동작하고(상속만),
True 일 때만 새로 만든 복사본 객체에 코레일 노드·간선·표시 역을 붙인다 (E1·E2, T5).

설계 (preregistration_3a2.md 6절)
---------------------------------
- 노드: 코레일 26역 = '1_<역명>'. 구로는 '1_구로@경부'·'1_구로@경인' 두 노드, 사이에 갈아타기 1분 하나 (E3).
  서울역·청량리는 기존 '1_서울역'·'1_청량리' 와 주행 간선으로 잇는다 (E4).
- 주행·정차: 3A-1 일반열차 정차표의 요일·30분 구간 중앙값 → 기존 9호선 표(run_tab·dwell_tab)와 같은 자리 (E5).
- 혼잡도: 코레일 간선은 결측(NaN). v1 의 결측 채우기를 거친 값은 지우고, 노출은 확인된 구간만 센다 (E8).
  코레일 주행의 체감시간 = 주행시간 그대로(혼잡 가중 없음) — '혼잡 판정 제외' 표시.
- 대기: 코레일 코드를 포함하는 1호선 탑승 구간은 그 구간 **하차역 기준** 목적지 필터 E(30분 구간, wait_profile) (E6).
  후보 탐색(Yen) 간선 비용에는 대기를 넣지 않고, 완성된 후보를 구간으로 나눠 E 를 매긴다 (E7).
- 후보 생성: 소요시간 탐색('time') = 코레일 포함 그래프, 혼잡 기반 탐색('cost') = 코레일을 뺀 원래 그래프 (E8).
- 운행 없음 (R3, E10): 그 구간을 실제로 타는 분에 60분 안 열차가 없으면 후보 제외, 사유 3가지 기록.
- 새 환승역(코레일 ↔ 다른 노선) 보행 = 3A-1 환승 후보(노량진·석계 3.0분), 혼잡 페널티 0 = 자료 없음 (E9).
"""

from __future__ import annotations

import copy
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from yeoyuro_v2 import korail_line1 as K
from yeoyuro_v2 import time_dependent as td
from yeoyuro_v2 import train_timetable as T

ROOT = Path(__file__).resolve().parents[1]
KORAIL_MART = ROOT / "data" / "marts" / "v2" / "korail_line1"
GURO_GB = "1_구로@경부"
GURO_GI = "1_구로@경인"
GURO_NODES = (GURO_GB, GURO_GI)
GURO_WALK_MIN = K.SAME_LINE_TRANSFER_WALK[K.GURO]     # 1.0 (D-093)
MAX_WAIT = 60.0
KORAIL_DEFAULT_K = 10      # D-099 D1: 코레일 기능을 켜면 기본 목표 후보 수 (끄면 기존 k 그대로)


# ---------------------------------------------------------------------- 1. 코레일 그래프 조각
def _node(code: int, neighbor_codes: set[int]) -> list[str]:
    """역코드 → 노드. 구로는 이웃 역에 따라 경부/경인 노드 (신도림 쪽은 호출하는 쪽이 정함)."""
    name = K.SCOPE[code]
    if code != K.GURO:
        return [f"1_{name}"]
    out = []
    if neighbor_codes & set(K.GYEONGBU_SOUTH):
        out.append(GURO_GB)
    if neighbor_codes & set(K.GYEONGIN_SOUTH):
        out.append(GURO_GI)
    return out or list(GURO_NODES)


@dataclass
class KorailPieces:
    """기능을 켤 때 덧붙일 것 (요일 하나 기준)."""
    ride: pd.DataFrame              # route_edge_mart 와 같은 열 + direction, congestion(NaN) 등
    transfer: list[tuple]           # (from_node, to_node, walk_min, note)
    run_tab: dict                   # (u, v, hw_bin) -> 분
    dwell_tab: dict                 # (node, hw_bin) -> 분
    node_code: dict                 # 노드 -> 역코드
    korail_nodes: set
    stations: dict                  # 역명 -> [노드]


def _hw_bin(minute: float) -> str:
    return td.TimeDependentEvaluator._hw_bin(minute)


def build_pieces(day_type: str, base_nodes: set, stop_times: pd.DataFrame | None = None) -> KorailPieces:
    st = stop_times if stop_times is not None else pd.read_parquet(KORAIL_MART / "stop_times.parquet")
    st = st[st.day_type == day_type].sort_values(["train", "direction", "stop_seq"])
    kor = set(K.SEOUL_KORAIL)
    seg_rows, dwell_rows = [], []
    for (_, direction), g in st.groupby(["train", "direction"], sort=False):
        codes = g.code.to_list()
        arr = g.arr_min.to_numpy(float)
        dep = g.dep_min.to_numpy(float)
        for i in range(len(codes)):
            if codes[i] in kor and not np.isnan(arr[i]) and not np.isnan(dep[i]):
                nb = {codes[j] for j in (i - 1, i + 1) if 0 <= j < len(codes)}
                for n in _node(codes[i], nb):
                    dwell_rows.append((n, dep[i], dep[i] - arr[i]))
            if i + 1 >= len(codes):
                continue
            a, b = codes[i], codes[i + 1]
            if a not in kor and b not in kor:
                continue                                   # 서울교통공사 구간끼리는 기존 그래프 그대로
            if np.isnan(dep[i]) or np.isnan(arr[i + 1]):
                continue
            # 구로 노드: 이 열차가 구로 앞뒤로 어디서 오고 어디로 가는지로 정함
            nb_a = {codes[j] for j in (i - 1, i + 1) if 0 <= j < len(codes)}
            nb_b = {codes[j] for j in (i, i + 2) if 0 <= j < len(codes)}
            for u in _node(a, nb_a):
                for v in _node(b, nb_b):
                    seg_rows.append((u, v, a, b, direction.lower(), dep[i], arr[i + 1] - dep[i]))
    seg = pd.DataFrame(seg_rows, columns=["u", "v", "a", "b", "direction", "dep", "run"])
    seg["hw"] = seg.dep.map(_hw_bin)
    run_tab = {(r.u, r.v, r.hw): float(r.run) for r in
               seg.groupby(["u", "v", "hw"], as_index=False).run.median().itertuples()}
    allday = seg.groupby(["u", "v"]).agg(run=("run", "median"), a=("a", "first"), b=("b", "first"),
                                         direction=("direction", lambda s: s.mode().iloc[0])).reset_index()
    dw = pd.DataFrame(dwell_rows, columns=["node", "dep", "dwell"])
    dw["hw"] = dw.dep.map(_hw_bin)
    dwell_tab = {(r.node, r.hw): float(r.dwell) for r in
                 dw.groupby(["node", "hw"], as_index=False).dwell.median().itertuples()}
    ride = pd.DataFrame({
        "from_node": allday.u, "to_node": allday.v, "line_id": "1",
        "from_station": [K.SCOPE[int(c)] for c in allday.a], "to_station": [K.SCOPE[int(c)] for c in allday.b],
        "from_code": allday.a.astype(int), "to_code": allday.b.astype(int),
        "travel_time_min": allday.run.astype(float), "distance_km": np.nan, "edge_type": "ride",
        "branch_code": "korail", "from_branch": "korail", "to_branch": "korail", "is_bidirectional": 1,
        "direction_of_edge": "korail", "direction": allday.direction,
        "congestion": np.nan, "congestion_base": np.nan, "event_mult": 1.0,
        "perceived_time_min": allday.run.astype(float), "event_risk_min": 0.0,
    })
    node_code = {}
    for r in ride.itertuples():
        node_code[r.from_node] = int(r.from_code)
        node_code[r.to_node] = int(r.to_code)
    # 갈아타기: 구로 두 노드 사이 1분 (E3) + 코레일 ↔ 다른 노선 (E9). 다른 노선 노드가 그래프에 있을 때만
    transfer = [(GURO_GB, GURO_GI, GURO_WALK_MIN, "구로 경부↔경인 (D-093)")]
    for tc in K.transfer_candidates().itertuples():
        kn = f"1_{tc.station}"
        lines = ["9L", "9X"] if str(tc.other_line) == "9" else [str(tc.other_line)]
        for ln in lines:
            other = f"{ln}_{tc.station}"
            if other in base_nodes and kn in node_code:
                transfer.append((kn, other, float(tc.walk_min), f"{tc.station} 1↔{tc.other_line} ({tc.status})"))
    korail_nodes = {n for n in node_code if n.split("_", 1)[1].split("@")[0] in
                    {K.SCOPE[c] for c in kor}}
    stations: dict[str, list] = {}
    for n in sorted(korail_nodes):
        stations.setdefault(n.split("_", 1)[1].split("@")[0], []).append(n)
    return KorailPieces(ride, transfer, run_tab, dwell_tab, node_code, korail_nodes, stations)


# ---------------------------------------------------------------------- 2. 복사본 스코어러·표시 역
def augment_scorer(base, pieces: KorailPieces, assume_congestion: bool = False):
    """v1 RouteScorer 의 **복사본**에 코레일 간선을 붙인다. 원본(base)은 바꾸지 않는다 (T5).

    assume_congestion=True 는 **민감도 전용(T4)**: 코레일 간선의 'cost' 에 그 시간대 1~9호선 간선 혼잡 중앙값을
    가정해 넣는다 (평가·판정에는 쓰지 않음, 혼잡도는 여전히 결측).
    """
    s = copy.copy(base)
    s.adj = {u: [dict(e) for e in lst] for u, lst in base.adj.items()}
    s.ride = pd.concat([base.ride, pieces.ride], ignore_index=True)
    s.transfer = base.transfer.copy()

    def add(u, v, **kw):
        s.adj.setdefault(u, []).append({"to": v, **kw})

    mult = 1.0
    if assume_congestion:
        cm = float(np.nanmedian(base.ride["congestion"].to_numpy(float)))
        g = type(base).evaluate.__globals__
        mult = 1.0 + float(g["KAPPA"]) * max(cm - float(g["C0"]), 0.0) / 100.0
        s.assumed_congestion = cm
    for r in pieces.ride.itertuples():
        add(r.from_node, r.to_node, cost=float(r.travel_time_min) * mult, time=float(r.travel_time_min),
            cong=np.nan, event=0.0, kind="ride")
    for u, v, w, _ in pieces.transfer:
        add(u, v, cost=w, time=w, cong=np.nan, event=0.0, kind="transfer")
        add(v, u, cost=w, time=w, cong=np.nan, event=0.0, kind="transfer")
    s.nodes = set(s.adj) | {e["to"] for lst in s.adj.values() for e in lst}
    s.korail_nodes = set(pieces.korail_nodes)
    s._korail_base = base
    return s


def augment_display(display: pd.DataFrame, pieces: KorailPieces) -> pd.DataFrame:
    """표시 역 목록 복사본: 기존 역(신도림 등)에는 코레일 노드를 더하고, 새 역(남영 등)은 행을 만든다."""
    d = display.copy()
    for name, nodes in pieces.stations.items():
        hit = d.index[d.station_key == name]
        if len(hit):
            i = hit[0]
            cur = [x for x in str(d.at[i, "candidate_station_uids"]).split(";") if x and x != "nan"]
            d.at[i, "candidate_station_uids"] = ";".join(cur + [n for n in nodes if n not in cur])
        else:
            row = {c: np.nan for c in d.columns}
            row.update({"station_key": name, "station_name": name, "display_name": name,
                        "candidate_station_uids": ";".join(nodes)})
            d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    return d


# ---------------------------------------------------------------------- 3. 평가기
@dataclass
class KorailLegWait:
    board: str
    alight: str
    t_board: float
    E: float
    next_train_wait: float | None   # 시각표상 다음 열차까지 (R2 안내용, 판정 미사용)
    reason: str | None


@dataclass
class KorailTDResult(td.TDResult):
    korail: bool = False
    korail_ride_min: float = 0.0
    congestion_judgement: str = ""            # '혼잡 판정 제외(코레일 구간 혼잡도 미확인)' 또는 ''
    no_service_reason: str | None = None      # R3: 운행 시작 전 / 운행 종료 / 장시간 대기 제외
    korail_waits: list = field(default_factory=list)
    line1_splits: list = field(default_factory=list)  # D-097: 이 경로에서 쓴 같은 노선 갈아타기 역 (광운대·석계)


JUDGE_EXCLUDED = "혼잡 판정 제외(코레일 구간 혼잡도 미확인)"


class KorailTDEvaluator(td.TimeDependentEvaluator):
    """코레일 간선이 붙은 복사본 스코어러용 평가기. 1~9호선만 지나는 경로는 기존 평가기와 같은 값을 낸다."""

    def __init__(self, scorer, headway, policy="step", include_initial_wait=False, event_scale=1.0,
                 event_exclude_lines=(), pieces: KorailPieces | None = None, index: T.TrainIndex | None = None,
                 cap_wait: float | None = None, wait_mode: str = "E", line1_splits: dict | None = None,
                 timetable_day_type: str | None = None):
        super().__init__(scorer, headway, policy, include_initial_wait, event_scale, event_exclude_lines)
        # D-104b: 코레일 시각표 요일. 통계 요일(scorer.day_type, 조회 날짜의 요일)과 다를 수 있다 (평일 공휴일 → END).
        # 시각표 색인(index)은 이 요일 행만 갖고 있으므로 열차 조회는 반드시 이 값으로 한다.
        self.tt_day = timetable_day_type or self.day_type
        # D-097 B: 광운대·석계 같은 노선 갈아타기 설정 (None = 기본 LINE1_SPLITS_DEFAULT, {} = 끔)
        self.line1_splits = T.LINE1_SPLITS_DEFAULT if line1_splits is None else line1_splits
        self.pieces = pieces
        self.kidx = index
        self.cap_wait = cap_wait            # V6: E 를 12분에서 자르는 시나리오
        self.wait_mode = wait_mode          # V4/R7: 'E' 또는 'H'(그 역·방향 전체 열차 배차÷2)
        self.run_tab = dict(self.run_tab or {})
        self.dwell_tab = dict(self.dwell_tab or {})
        self.run_tab.update(pieces.run_tab)
        self.dwell_tab.update(pieces.dwell_tab)
        self.korail_edges = {(r.from_node, r.to_node) for r in pieces.ride.itertuples()}
        for (u, v) in self.korail_edges:                 # v1 결측 채우기로 들어간 값 지움 (E8)
            for b in range(td.N_BINS):
                self.cong_med[(u, v, b)] = float("nan")
                self.cong_p90[(u, v, b)] = float("nan")
        self.code = dict(T._real_code_map(scorer.ride))
        self.code.update(pieces.node_code)
        for (ln, a), b in T.CODE_ALIAS.items():
            for n in [n for n, c in self.code.items() if c == a and n.split("_", 1)[0] == ln]:
                self.code[n] = b
        self._E_cache: dict = {}

    # --- 탑승 구간
    def legs(self, path):
        """[(시작 간선 번호, 끝 간선 번호, 노선, 정차 코드 순서, 코레일 포함)]"""
        out, cur = [], None
        for i, (u, v) in enumerate(zip(path, path[1:])):
            e = next(x for x in self.s.adj[u] if x["to"] == v)
            if e["kind"] == "ride":
                ln = self.edge_meta[(u, v)]["line"]
                if cur is not None and cur["line"] != ln:
                    out.append(cur)
                    cur = None
                if cur is None:
                    cur = {"i0": i, "line": ln, "codes": [self.code[u]], "nodes": [u]}
                cur["i1"] = i
                cur["nodes"].append(v)
                if self.code[v] != cur["codes"][-1]:
                    cur["codes"].append(self.code[v])
            elif e["kind"] == "transfer" and self._is_through(path, i, u, v):
                continue
            elif e["kind"] == "transfer":
                if cur is not None:
                    out.append(cur)
                    cur = None
        if cur is not None:
            out.append(cur)
        kor = set(K.SEOUL_KORAIL)
        for lg in out:
            lg["korail"] = lg["line"] == "1" and any(c in kor for c in lg["codes"])
        return out

    def korail_wait(self, lg: dict, t: float) -> KorailLegWait:
        """코레일 포함 구간을 t(그 역 도착 분)에 탈 때: 30분 구간 E + 시각표상 다음 열차 + R3 사유."""
        day = self.tt_day
        via = tuple(lg["codes"])
        deps, _, _ = self.kidx.pair(day, "1", via[0], via[-1], via)
        r, why = self.kidx.next_ride(day, "1", via[0], via[-1], t, MAX_WAIT, via=via)
        b0 = td.BIN_START_MIN + td.bin_index_of(t) * td.BIN_WIDTH_MIN
        key = (via, b0, self.wait_mode)
        if key not in self._E_cache:
            if self.wait_mode == "H":
                allp = self.kidx._by_board.get((day, "1", via[0]), [])
                same_dir = [d for d, k, i in allp if i + 1 < len(self.kidx._trains[k][0])
                            and int(self.kidx._trains[k][0][i + 1]) == via[1]]
                E = K.headway_half(same_dir, b0, b0 + 30)
            else:
                E = K.wait_profile(deps, b0, b0 + 30, MAX_WAIT).E
            self._E_cache[key] = E
        E = self._E_cache[key]
        if self.cap_wait is not None and not np.isnan(E):
            E = min(E, self.cap_wait)
        return KorailLegWait(lg["nodes"][0], lg["nodes"][-1], t, E,
                             None if r is None else r.wait_min, None if r is not None else why)

    def _ride_med(self, via, b0) -> float:
        """그 30분 구간에 via 를 한 열차로 가는 열차들의 탑승 시간 중앙값 (없으면 그날 중앙값)."""
        deps, arrs, _ = self.kidx.pair(self.tt_day, "1", via[0], via[-1], tuple(via))
        if not deps:
            return float("nan")
        r = [a - d for d, a in zip(deps, arrs) if b0 <= d < b0 + 30]
        return float(np.median(r if r else [a - d for d, a in zip(deps, arrs)]))

    def plan_leg(self, lg: dict, t: float) -> dict:
        """D-097 B3: 코레일 포함 1호선 구간을 '바로 가기' vs '광운대·석계에서 1번 갈아타기' 기대 도착으로 비교.

        비교에는 첫 승차 대기도 넣는다(공정 비교). 반환: {'mode': 'direct'|'split', 'wait': KorailLegWait,
        'k': 갈아탈 위치(codes 기준), 'code', 'walk', 'b': 뒷 부분 구간 dict}
        """
        via = list(lg["codes"])
        w_d = self.korail_wait(lg, t)
        b0 = td.BIN_START_MIN + td.bin_index_of(t) * td.BIN_WIDTH_MIN
        best = None
        split_reason = None        # 바로 가기가 구조상 불가능하고 갈아타기는 구조상 가능하지만 그 시각에 안 될 때의 R3 사유
        if w_d.reason is None and not np.isnan(w_d.E):
            best = (t + w_d.E + self._ride_med(via, b0), {"mode": "direct", "wait": w_d})
        for k, c, walk in T.line1_split_points(via, self.line1_splits):
            a = {"codes": via[:k + 1], "nodes": lg["nodes"][:k + 1]}
            b = {"codes": via[k:], "nodes": lg["nodes"][k:]}
            w_a = self.korail_wait(a, t)
            if w_a.reason is not None or np.isnan(w_a.E):
                if (w_a.reason not in (None, T.IMPOSSIBLE)
                        and self.kidx.pair(self.tt_day, "1", b["codes"][0], b["codes"][-1], tuple(b["codes"]))[0]):
                    split_reason = split_reason or w_a.reason
                continue
            t_s = t + w_a.E + self._ride_med(a["codes"], b0) + walk
            if np.isnan(t_s):
                continue
            b1 = td.BIN_START_MIN + td.bin_index_of(t_s) * td.BIN_WIDTH_MIN
            w_b = self.korail_wait(b, t_s)
            if np.isnan(w_b.E):
                if w_b.reason not in (None, T.IMPOSSIBLE):
                    split_reason = split_reason or w_b.reason
                continue
            arr = t_s + w_b.E + self._ride_med(b["codes"], b1)
            if np.isnan(arr):
                continue
            if best is None or arr < best[0] - 1e-9:
                best = (arr, {"mode": "split", "wait": w_a, "k": k, "code": c, "walk": walk, "b": b})
        if best is None:
            if w_d.reason == T.IMPOSSIBLE and split_reason:
                # 사유 표기 (D-099 구현): 바로 가는 열차는 원래 없지만 같은 노선 갈아타기로는 이어지는 구간이
                # 그 시각에 운행이 없으면 '불가능한 연결' 이 아니라 R3 사유로 적는다 (후보 제외는 같음)
                w_d = KorailLegWait(w_d.board, w_d.alight, w_d.t_board, w_d.E, w_d.next_train_wait, split_reason)
            return {"mode": "direct", "wait": w_d}
        return best[1]

    # --- 평가 (기존 evaluate 와 같은 순서, 코레일 간선만 다르게)
    def evaluate(self, path, depart_min=None, keep_trace=False):
        if depart_min is None:
            depart_min = td.parse_hhmm(self.s.depart)
        legs = self.legs(path)
        leg_at = {lg["i0"]: lg for lg in legs}
        any_kor = any(lg["korail"] for lg in legs)
        if not any(self.edge_meta.get((u, v)) and (u, v) in self.korail_edges for u, v in zip(path, path[1:])):
            base = super().evaluate(path, depart_min, keep_trace)            # 1~9호선만: 기존 계산 그대로
            return KorailTDResult(**base.__dict__)
        adj = self.s.adj
        t = float(depart_min)
        trace = []
        thr = td.EXPOSURE_THRESHOLDS
        exp = {x: 0.0 for x in thr}
        exp90 = {x: 0.0 for x in thr}
        perceived = running = dwell = walk = wait = kor_min = 0.0
        congs_edge, cs, cd = [], 0.0, 0.0
        max_c = -np.inf
        bins_used, oow, n_tr, init_wait = set(), False, 0, 0.0
        waits, no_service = [], None

        def ride_block(u, v, t0, dur, kind):
            nonlocal perceived, cs, cd, max_c, oow, kor_min
            if (u, v) in self.korail_edges:
                if kind == "ride":
                    perceived += dur
                kor_min += dur
                if self.policy != "fixed" and (t0 < td.BIN_START_MIN - 1e-9 or t0 + dur > td.BIN_END_MIN + 1e-9):
                    oow = True
                if keep_trace:
                    trace.append(td.EdgeTrace(kind, u, v, t0, t0 + dur, None, None, ()))
                return float("nan")
            pcs = self._pieces(u, v, t0, dur)
            meta = self.edge_meta[(u, v)]
            mult = self._event_mult(meta["from_station"], t0, meta["line"])
            if kind == "ride" and meta["line"] in self.service_lines and self.policy != "fixed":
                if (meta["line"], meta["from_station"], meta["direction"], self._hw_bin(t0)) not in self.operating:
                    oow = True
            cw, cmax = 0.0, -np.inf
            if self.policy != "fixed" and (t0 < td.BIN_START_MIN - 1e-9 or t0 + dur > td.BIN_END_MIN + 1e-9):
                oow = True
            for d, c, c90, b in pcs:
                c, c90 = c * mult, c90 * mult
                bins_used.add(b)
                cw += d * c
                cmax = max(cmax, c)
                for x in thr:
                    if c >= x:
                        exp[x] += d
                    if c90 >= x:
                        exp90[x] += d
                if kind == "ride":
                    perceived += d * (1.0 + self.KAPPA * max(c - self.C0, 0.0) / 100.0)
            cavg = cw / dur if dur > 0 else float("nan")
            cs += cw
            cd += dur
            max_c = max(max_c, cmax)
            if keep_trace:
                trace.append(td.EdgeTrace(kind, u, v, t0, t0 + dur, round(cavg, 2), round(cmax, 2),
                                          tuple(sorted({p[3] for p in pcs}))))
            return cavg

        # 첫 탑승 구간: 대기는 넣지 않음(R4) — 코레일이면 운행 여부(R3)와 시각표상 다음 열차만 기록
        first = legs[0] if legs else None
        pending = {}                       # path 노드 번호 -> 그 역에서 하는 같은 노선 갈아타기 계획 (D-097)
        splits_used = []

        def schedule(lg, plan):
            if plan["mode"] == "split":
                node_idx = lg["i0"] + plan["k"]          # codes 와 nodes 는 코레일 구간에서 1:1
                pending[node_idx] = plan

        if first is not None and first["korail"]:
            plan = self.plan_leg(first, t)
            w = plan["wait"]
            waits.append(w)
            no_service = no_service or w.reason
            schedule(first, plan)
            if self.include_initial_wait and not np.isnan(w.E):
                init_wait = w.E
                t += init_wait
        elif self.include_initial_wait and first is not None:
            u, v = path[first["i0"]], path[first["i0"] + 1]
            m = self.edge_meta[(u, v)]
            init_wait = self._wait(m["line"], m["from_station"], m["direction"], t)
            t += init_wait

        prev_was_ride = False
        for i, (u, v) in enumerate(zip(path, path[1:])):
            e = next(x for x in adj[u] if x["to"] == v)
            if e["kind"] == "ride" and i in pending:
                # 같은 노선 갈아타기 (광운대·석계): 내려서 보행 + 뒷 부분 목적지행 E (R3 는 실제 도착 분 기준)
                plan = pending.pop(i)
                n_tr += 1
                walk += plan["walk"]
                perceived += plan["walk"]
                if keep_trace:
                    trace.append(td.EdgeTrace("transfer_walk", u, u, t, t + plan["walk"], None, None))
                t += plan["walk"]
                wb = self.korail_wait(plan["b"], t)
                waits.append(wb)
                no_service = no_service or wb.reason
                h = 0.0 if np.isnan(wb.E) else wb.E
                if keep_trace and h:
                    trace.append(td.EdgeTrace("transfer_wait", u, u, t, t + h, None, None))
                wait += h
                perceived += h
                t += h
                splits_used.append(K.SCOPE.get(plan["code"], str(plan["code"])))
                prev_was_ride = False
            if e["kind"] == "ride":
                dw = self._dwell(u, t) if prev_was_ride else 0.0
                if prev_was_ride and dw > 0:
                    ride_block(u, v, t, dw, "dwell")
                    perceived += dw
                    dwell += dw
                    t += dw
                d = self._run(u, v, t)
                c = ride_block(u, v, t, d, "ride")
                if not np.isnan(c):
                    congs_edge.append(c)
                running += d
                t += d
                prev_was_ride = True
            elif e["kind"] == "transfer" and self._is_through(path, i, u, v):
                continue
            else:
                prev_was_ride = False
                if e["kind"] != "transfer":
                    continue
                n_tr += 1
                a = float(e["time"])
                walk += a
                perceived += float(e["cost"])
                if keep_trace:
                    trace.append(td.EdgeTrace("transfer_walk", u, v, t, t + a, None, None))
                t += a
                nxt_leg = leg_at.get(i + 1)
                if nxt_leg is None:
                    continue
                if self.policy != "fixed" and not (td.BIN_START_MIN <= t < td.BIN_END_MIN):
                    oow = True
                if nxt_leg["korail"]:
                    plan = self.plan_leg(nxt_leg, t)
                    w = plan["wait"]
                    waits.append(w)
                    no_service = no_service or w.reason
                    schedule(nxt_leg, plan)
                    h = 0.0 if np.isnan(w.E) else w.E
                else:
                    nxt = path[i + 2]
                    m = self.edge_meta[(v, nxt)]
                    tl = m["line"]
                    sw = (self.switch_tab.get((u.split("_", 1)[0], tl, m["from_station"], m["direction"],
                                               self._hw_bin(t)))
                          if self.switch_tab and {u.split("_", 1)[0], tl} == {"9L", "9X"} else None)
                    h = sw if sw is not None else self._wait(m["line"], m["from_station"], m["direction"], t)
                if keep_trace and h:
                    trace.append(td.EdgeTrace("transfer_wait", v, v, t, t + h, None, None))
                wait += h
                perceived += h
                t += h

        known = [c for c in congs_edge if not np.isnan(c)]
        return KorailTDResult(
            path=list(path), policy=self.policy, depart_min=depart_min, arrive_min=t,
            actual_time_min=t - depart_min, perceived_time_min=perceived + init_wait,
            in_vehicle_min=running + dwell, running_time_min=running, dwell_time_min=dwell,
            transfer_walk_min=walk, transfer_wait_min=wait, initial_wait_min=init_wait,
            transfer_count=n_tr,
            max_congestion=float(max_c) if known else float("nan"),
            avg_congestion_edge=float(np.mean(known)) if known else float("nan"),
            avg_congestion_time=cs / cd if cd else float("nan"),
            exposure=exp, exposure_p90=exp90, bins_used=tuple(sorted(bins_used)),
            out_of_window=oow, trace=trace,
            korail=any_kor or kor_min > 0, korail_ride_min=kor_min,
            congestion_judgement=JUDGE_EXCLUDED, no_service_reason=no_service, korail_waits=waits,
            line1_splits=splits_used)


# ---------------------------------------------------------------------- 4. 라우터
_GI = {f"1_{K.SCOPE[c]}" for c in K.GYEONGIN_SOUTH}
_GB = {f"1_{K.SCOPE[c]}" for c in K.GYEONGBU_SOUTH}


def valid_guro_transfer(path) -> bool:
    """구로 갈아타기 간선은 '경인 쪽 역 → 구로 → 경부 쪽 역'(또는 반대)일 때만 의미가 있다 (D-095 E3·E3a).

    신도림 쪽에서 와서 다른 방면 노드로 갈아타는 것(= 구로 종착→시발)이나, 지선에서 와서 다른 노드로 갈아탄 뒤
    신도림 쪽으로 가는 것(= 같은 열차가 계속 가는데 내렸다 다시 타는 중복 환승)은 그래프에서 후보로 쓰지 않는다.
    """
    for i, (u, v) in enumerate(zip(path, path[1:])):
        if {u, v} != set(GURO_NODES):
            continue
        prev = path[i - 1] if i > 0 else None
        nxt = path[i + 2] if i + 2 < len(path) else None
        side = lambda n: "경인" if n in _GI else ("경부" if n in _GB else None)   # noqa: E731
        if prev is None or nxt is None or side(prev) is None or side(nxt) is None or side(prev) == side(nxt):
            return False
    return True


class KorailTDRouter(td.TDRouter):
    """include_korail=False 면 TDRouter 와 같다(상속만). True 면 복사본 그래프에서 후보를 찾고 평가한다."""

    def __init__(self, *args, include_korail: bool = False, cap_wait: float | None = None,
                 wait_mode: str = "E", guro_walk: float | None = None, assume_congestion: bool = False,
                 line1_splits: dict | None = None, max_k: int = 20, rep_wait: bool = False, **kw):
        k_explicit = "k" in kw or len(args) >= 7
        super().__init__(*args, **kw)
        self.include_korail = include_korail
        if include_korail and not k_explicit:
            self.k = KORAIL_DEFAULT_K                       # D-099 D1
        self.rep_wait = rep_wait                            # D-099 D2: 민감도 전용
        self.assume_congestion = assume_congestion          # T4 민감도 전용
        self.line1_splits = line1_splits                    # D-097 B (None = 기본, {} = 끔)
        self.max_k = max_k                                  # D-097 A2
        self.excluded: list = []          # 마지막 evaluate_od 에서 R3 로 뺀 후보
        if not include_korail:
            return
        self.cap_wait, self.wait_mode = cap_wait, wait_mode
        self._base_scorers: dict = {}
        self._aug: dict = {}
        base0 = super().scorer(0)
        self.pieces = build_pieces(self.day_type, base0.nodes)
        if guro_walk is not None:                                       # V5 민감도
            self.pieces.transfer = [(u, v, (guro_walk if {u, v} == set(GURO_NODES) else w), n)
                                    for u, v, w, n in self.pieces.transfer]
        st = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet")
        self.kidx = T.TrainIndex(st[(st.group == "1") & (st.day_type == self.day_type)])
        self.display = augment_display(self.display, self.pieces)

    def scorer(self, b: int):
        if not self.include_korail:
            return super().scorer(b)
        b = max(0, min(td.N_BINS - 1, b))
        if b not in self._aug:
            base = super().scorer(b)
            self._base_scorers[b] = base
            self._aug[b] = augment_scorer(base, self.pieces, self.assume_congestion)
            if self.rep_wait:
                self._add_rep_wait(self._aug[b], b)
        return self._aug[b]

    def evaluator(self, depart_bin: int, policy: str | None = None):
        if not self.include_korail:
            return super().evaluator(depart_bin, policy)
        policy = policy or self.policy
        key = (max(0, min(td.N_BINS - 1, depart_bin)), policy)
        if key not in self._evals:
            self._evals[key] = KorailTDEvaluator(
                self.scorer(depart_bin), self.headway, policy, self.include_initial_wait,
                self.event_scale, self.event_exclude_lines, pieces=self.pieces, index=self.kidx,
                cap_wait=self.cap_wait, wait_mode=self.wait_mode, line1_splits=self.line1_splits,
                timetable_day_type=self.day_type)
        return self._evals[key]

    def rep_E(self, node: str, b: int) -> float:
        """D-099 D2 대표 대기: 그 코레일 노드에서 인접 다음 역까지 가는 모든 열차의 30분 구간 E, 방향별 중앙값."""
        code = self.pieces.node_code[node]
        nxt = sorted({int(r.to_code) for r in self.pieces.ride.itertuples() if r.from_node == node})
        b0 = td.BIN_START_MIN + b * td.BIN_WIDTH_MIN
        es = []
        for c in nxt:
            deps = self.kidx.pair(self.day_type, "1", code, c, (code, c))[0]
            e = K.wait_profile(deps, b0, b0 + 30, MAX_WAIT).E if deps else float("nan")
            if not np.isnan(e):
                es.append(e)
        return float(np.median(es)) if es else 0.0

    def _add_rep_wait(self, s, b: int):
        """민감도 전용: 코레일 노드로 갈아타는 간선에 더할 대표 대기를 따로 기억 (간선 자체는 바꾸지 않음 — 평가 불변)."""
        extra, cache = {}, {}
        for u, lst in s.adj.items():
            for e in lst:
                v = e["to"]
                if e["kind"] == "transfer" and v in self.pieces.korail_nodes:
                    if v not in cache:
                        cache[v] = self.rep_E(v, b)
                    extra[(u, v)] = cache[v]
        s._rep_extra = extra

    def _time_yen(self, scorer, v_o, v_d, K):
        """소요시간 Yen. rep_wait(민감도)면 탐색하는 동안만 갈아타는 간선 비용에 대표 대기를 더한 복사본으로."""
        extra = getattr(scorer, "_rep_extra", None) if self.rep_wait else None
        if not extra:
            return scorer._yen(v_o, v_d, K=K, weight="time")
        saved = scorer.adj
        adj2 = dict(saved)
        for u in {u for u, _ in extra}:
            if u in saved:
                adj2[u] = [dict(e, time=float(e["time"]) + extra[(u, e["to"])]) if (u, e["to"]) in extra else e
                           for e in saved[u]]
        scorer.adj = adj2
        try:
            return scorer._yen(v_o, v_d, K=K, weight="time")
        finally:
            scorer.adj = saved

    def _clean(self, p):
        """Yen 결과 → 실제 노드 경로. 기존 규칙(출발·도착 같은 역 노드 자르기, 재방문 제외) + 구로 갈아타기 규칙."""
        sr = self.sr
        real = [n for n in p if not sr.is_virtual(n)]
        while len(real) >= 2 and sr.station_of(real[0]) == sr.station_of(real[1]):
            real = real[1:]
        while len(real) >= 2 and sr.station_of(real[-1]) == sr.station_of(real[-2]):
            real = real[:-1]
        if len(real) < 2 or sr.has_station_revisit(real):
            return None
        if not valid_guro_transfer(real):
            self.dropped_guro = getattr(self, "dropped_guro", 0) + 1
            return None
        return tuple(real)

    def _structurally_possible(self, path, ev) -> bool:
        """그날 어떤 시각에도 열차로 이어 갈 수 없는 코레일 구간이 있으면 False (바로 가기·같은 노선 갈아타기 모두 불가)."""
        for lg in ev.legs(list(path)):
            if not lg["korail"]:
                continue
            leg = T.Leg("1", lg["codes"][0], lg["codes"][-1], lg["nodes"][0], lg["nodes"][-1], 0.0, tuple(lg["codes"]))
            rs, why = T._ride_leg(self.kidx, self.day_type, leg, 0.0, 10_000.0, ev.line1_splits)
            if rs is None and why == T.IMPOSSIBLE:
                return False
        return True

    def _yen_split(self, scorer, v_o, v_d, k):
        """D-097 A1·A2: 소요시간 Yen(코레일 포함, 유효 후보 k 개까지 확대) ∪ 소요시간 Yen(원래 그래프) ∪ 혼잡 기반 Yen.

        혼잡 기반 탐색은 기본은 원래 그래프(E8), T4 민감도에서만 코레일 포함 그래프."""
        base = scorer._korail_base
        b = next((bb for bb, sc in self._aug.items() if sc is scorer), 0)
        ev = self.evaluator(b)
        K = k
        while True:
            t_paths = self._time_yen(scorer, v_o, v_d, K)
            valid = [p for p in t_paths if (q := self._clean(p)) is not None and self._structurally_possible(q, ev)]
            if len(valid) >= k or K >= self.max_k or len(t_paths) < K:
                break
            K = min(self.max_k, K * 2)
        self.last_K = K
        in_base = v_o in base.nodes and v_d in base.nodes
        tb = base._yen(v_o, v_d, K=k, weight="time") if in_base else []
        if self.assume_congestion:                         # T4 민감도: 혼잡 가정 'cost' 탐색도 코레일 포함 그래프
            c_paths = scorer._yen(v_o, v_d, K=k, weight="cost")
        else:
            c_paths = base._yen(v_o, v_d, K=k, weight="cost") if in_base else []
        return valid + tb + c_paths

    def raw_paths(self, scorer, origin, dest, k):
        if not self.include_korail:
            return super().raw_paths(scorer, origin, dest, k)
        sr = self.sr
        o_nodes = sr.candidates_of(self.display, origin)
        d_nodes = sr.candidates_of(self.display, dest)
        if not o_nodes or not d_nodes or origin == dest:
            return []
        v_o, v_d = sr.ORIGIN_PREFIX + origin, sr.DEST_PREFIX + dest
        saved = []
        for sc in (scorer, scorer._korail_base):
            aug = dict(sc.adj)
            aug[v_o] = [{"to": n, "cost": 0.0, "time": 0.0, "cong": float("nan"), "event": 0.0,
                         "kind": "origin_access"} for n in o_nodes if n in sc.nodes]
            for n in d_nodes:
                if n in sc.nodes:
                    aug[n] = list(aug.get(n, [])) + [{"to": v_d, "cost": 0.0, "time": 0.0, "cong": float("nan"),
                                                      "event": 0.0, "kind": "destination_exit"}]
            saved.append((sc, sc.adj, sc.nodes))
            sc.adj = aug
            sc.nodes = set(aug) | {e["to"] for lst in aug.values() for e in lst}
        try:
            paths = self._yen_split(scorer, v_o, v_d, k)
        finally:
            for sc, a, n in saved:
                sc.adj, sc.nodes = a, n
        seen, out = set(), []
        for p in paths:
            real = self._clean(p)
            if real is None or real in seen:
                continue
            seen.add(real)
            out.append(list(real))
        return out

    def evaluate_od(self, origin, dest, depart_min, policy=None, k=None, multi_bin=None):
        res = super().evaluate_od(origin, dest, depart_min, policy, k, multi_bin)
        if not self.include_korail:
            return res
        self.excluded = [r for r in res if getattr(r, "no_service_reason", None)]
        return [r for r in res if not getattr(r, "no_service_reason", None)]
