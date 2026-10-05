"""
yeoyuro_v2/time_dependent.py
============================
여유로 서울 v2.0 — 시간 진행형 경로 평가 엔진 (Time-dependent route evaluation)

v1 한계 (README §9-4)
--------------------
v1 `RouteScorer` 는 출발 시각의 time_bin 하나를 경로 전체에 적용했다.
08:20 에 출발해 50분 이동해도 마지막 구간까지 08:00~08:30 혼잡도를 쓰고,
환승 대기(`apply_transfer_wait`)도 출발 bin 의 배차간격으로 계산했다.

v2 방식
-------
edge 마다 진입 시각을 누적한다.

    t_i = t_dep + w_0 + Σ_{j<i} (주행 d_j + 정차 s_j + 환승도보 a_j + 환승대기 h_j)

- 혼잡도는 t_i 가 속한 bin 으로 조회한다 (정책은 BIN_POLICIES 참고).
- 환승 대기 h_j 는 **환승역 도착 시각의 bin** 배차간격으로 조회한다.
- 이벤트 배수도 edge 진입 시각의 '시(hour)' 로 조회한다.

설계 원칙
---------
1. v1 코드를 수정하지 않는다. `RouteScorer` 인스턴스에서 그래프·혼잡 lookup 을 읽기만 한다.
   → v1 결과 재현성 유지, v1/v2 를 같은 그래프 위에서 비교 가능.
2. `policy="fixed"` 는 v1 계산을 이 엔진 안에서 그대로 재현한다.
   → 회귀 테스트: fixed 결과 == v1 evaluate()+apply_transfer_wait() 결과.
3. 혼잡 지표는 '분(min)' 단위 노출 시간으로 낸다. 가중치 합산 점수는 만들지 않는다.

혼잡 노출(exposure) 정의
------------------------
exposure_X_min = 차내 시간(주행 + 중간역 정차) 중 기대 혼잡도 >= X 인 분의 합.
환승 도보·대기는 승강장 혼잡 데이터가 없어 제외한다.

시간 표기
---------
모든 시각은 '그날 00:00 부터 분' 정수/실수로 다룬다. 00:30 bin 은 24*60+30 = 1470.
bin index 0 = 05:30~06:00, 38 = 00:30~01:00.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- 상수
BIN_START_MIN = 5 * 60 + 30          # 05:30
BIN_WIDTH_MIN = 30
N_BINS = 39                          # 05:30 ~ 01:00
BIN_END_MIN = BIN_START_MIN + N_BINS * BIN_WIDTH_MIN   # 01:00 (=1500) — 혼잡 자료가 있는 마지막 시각
EXPOSURE_THRESHOLDS = (80, 100, 130)

# step   : 혼잡도를 bin 안에서 상수(계단함수)로 보고, edge 구간이 bin 경계를 넘으면 분할 적분. [기본값]
# floor  : edge 진입 시각이 속한 bin 하나를 적용.
# linear : bin 중심점 사이 선형 보간 (edge 구간 중점 기준). 관측 안 된 중간값을 만들므로 sensitivity 전용.
# fixed  : 출발 bin 을 경로 전체에 고정 (= v1 재현용).
BIN_POLICIES = ("step", "floor", "linear", "fixed")


def bin_index_of(minute: float) -> int:
    """절대 시각(분) -> bin index. 범위 밖은 -1 / N_BINS 그대로 돌려준다(호출 측이 clamp)."""
    return int(math.floor((minute - BIN_START_MIN) / BIN_WIDTH_MIN))


def bin_label(idx: int) -> str:
    s = BIN_START_MIN + idx * BIN_WIDTH_MIN
    e = s + BIN_WIDTH_MIN
    return f"{(s // 60) % 24:02d}:{s % 60:02d}~{(e // 60) % 24:02d}:{e % 60:02d}"


def parse_hhmm(text: str) -> int:
    """'08:20' -> 500. 00:00~04:59 는 전날 심야로 보고 24h 를 더한다."""
    hh, mm = (int(x) for x in text.split(":"))
    m = hh * 60 + mm
    return m + 24 * 60 if m < BIN_START_MIN - 60 else m


def fmt_min(minute: float) -> str:
    m = int(round(minute))
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


# --------------------------------------------------------------------------- 결과 타입
@dataclass
class EdgeTrace:
    """경로의 edge 1개를 시간 진행형으로 평가한 기록 (UI 곡선·검산용)."""
    kind: str                 # ride / dwell / transfer_walk / transfer_wait
    u: str
    v: str
    t_start: float
    t_end: float
    congestion: float | None  # ride/dwell 만. 구간 시간가중 평균
    congestion_max: float | None
    bins: tuple[int, ...] = ()


@dataclass
class TDResult:
    path: list[str]
    policy: str
    depart_min: float
    arrive_min: float
    actual_time_min: float
    perceived_time_min: float
    in_vehicle_min: float
    running_time_min: float
    dwell_time_min: float
    transfer_walk_min: float
    transfer_wait_min: float
    initial_wait_min: float
    transfer_count: int
    max_congestion: float
    avg_congestion_edge: float      # v1 과 같은 정의: ride edge 평균 (edge 수 기준)
    avg_congestion_time: float      # 시간가중 평균
    exposure: dict[int, float]      # {80: 분, 100: 분, 130: 분}
    exposure_p90: dict[int, float]  # 같은 계산을 congestion_p90 으로
    bins_used: tuple[int, ...]
    out_of_window: bool
    trace: list[EdgeTrace] = field(default_factory=list)

    def summary(self) -> dict:
        d = {
            "policy": self.policy,
            "depart": fmt_min(self.depart_min),
            "arrive": fmt_min(self.arrive_min),
            "actual_time_min": round(self.actual_time_min, 2),
            "perceived_time_min": round(self.perceived_time_min, 2),
            "in_vehicle_min": round(self.in_vehicle_min, 2),
            "transfer_count": self.transfer_count,
            "transfer_wait_min": round(self.transfer_wait_min, 2),
            "max_congestion": round(self.max_congestion, 1),
            "avg_congestion_edge": round(self.avg_congestion_edge, 1),
            "avg_congestion_time": round(self.avg_congestion_time, 1),
            "n_bins_crossed": len(self.bins_used),
            "out_of_window": self.out_of_window,
        }
        for x in EXPOSURE_THRESHOLDS:
            d[f"exposure_{x}_min"] = round(self.exposure[x], 2)
            d[f"exposure_{x}_min_p90"] = round(self.exposure_p90[x], 2)
        return d


# --------------------------------------------------------------------------- 엔진
class TimeDependentEvaluator:
    """v1 RouteScorer 의 그래프 위에서 경로를 시간 진행형으로 평가한다.

    Parameters
    ----------
    scorer : v1 RouteScorer 인스턴스 (요일유형·그래프·혼잡 lookup·이벤트 보정 출처)
    headway : station_routing.load_headway() 결과
    policy : BIN_POLICIES 중 하나
    include_initial_wait : True 면 최초 승차 전 대기(배차/2)를 더한다.
        v1 은 넣지 않았다. 기본 False — v1 과 비교 가능성을 우선한다(decision_log 참고).
    """

    def __init__(self, scorer, headway: dict, policy: str = "step",
                 include_initial_wait: bool = False, event_scale: float = 1.0,
                 event_exclude_lines: tuple = ()):
        if policy not in BIN_POLICIES:
            raise ValueError(f"policy 는 {BIN_POLICIES} 중 하나여야 합니다: {policy}")
        self.s = scorer
        self.headway = headway
        self.policy = policy
        self.include_initial_wait = include_initial_wait
        # v2.5: 이벤트 배수 m 을 1 + (m-1)*event_scale 로 조정 (λ sensitivity. 1.0 = v1 λ=0.3 그대로)
        self.event_scale = float(event_scale)
        # v2.6: 이벤트 배수를 적용하지 않을 노선 (9호선 적용/미적용 sensitivity, D-047)
        self.event_exclude_lines = set(event_exclude_lines)
        self.day_type = scorer.day_type
        self._load_line9_tables(Path(scorer.root))
        self._g = type(scorer).evaluate.__globals__     # v1 모듈 상수 (C0, KAPPA, DWELL ...)
        self.C0 = float(self._g["C0"])
        self.KAPPA = float(self._g["KAPPA"])
        self.DWELL = float(self._g["DEFAULT_DWELL_TIME_MIN"])
        self._is_through = self._g["is_through_pass"]
        self._build_lookup()

    # ------------------------------------------------------------ 조회표
    def _build_lookup(self) -> None:
        """(edge, bin) -> (median, p90). v1 의 미매칭 보정 규칙을 bin 마다 똑같이 적용한다."""
        lk = self.s.lookup
        lk = lk[lk["day_type"] == self.day_type][
            ["station_uid", "direction", "time_bin_index", "congestion_median", "congestion_p90"]
        ].drop_duplicates(subset=["station_uid", "direction", "time_bin_index"])

        ride = self.s.ride[["from_node", "to_node", "line_id", "direction",
                            "travel_time_min", "from_station"]].copy()
        grid = ride.assign(key=1).merge(
            pd.DataFrame({"time_bin_index": range(N_BINS), "key": 1}), on="key").drop(columns="key")
        grid = grid.merge(lk.rename(columns={"station_uid": "from_node"}),
                          on=["from_node", "direction", "time_bin_index"], how="left")
        # v1 과 동일: 노선·방향 중앙값 -> 전체 중앙값 (단, bin 마다)
        for col in ("congestion_median", "congestion_p90"):
            f1 = grid.groupby(["line_id", "direction", "time_bin_index"])[col].transform("median")
            f2 = grid.groupby("time_bin_index")[col].transform("median")
            grid[col] = grid[col].fillna(f1).fillna(f2)

        self.cong_med: dict[tuple, float] = {}
        self.cong_p90: dict[tuple, float] = {}
        for r in grid.itertuples(index=False):
            k = (r.from_node, r.to_node, int(r.time_bin_index))
            self.cong_med[k] = float(r.congestion_median)
            self.cong_p90[k] = float(r.congestion_p90)

        self.edge_meta = {
            (r.from_node, r.to_node): {
                "line": str(r.line_id),
                "direction": r.direction if pd.notna(r.direction) else None,
                "time": float(r.travel_time_min),
                "from_station": r.from_station,
            } for r in ride.itertuples(index=False)
        }
        self.fixed_bin = bin_index_of(parse_hhmm(self.s.depart))
        self.fixed_bin = max(0, min(N_BINS - 1, self.fixed_bin))

    # ------------------------------------------------------------ 9호선 확장 표 (v2.6)
    def _load_line9_tables(self, root: Path) -> None:
        """9호선 mart 가 있는 root 에서만 켜진다. 1~8호선 root 에서는 아무 영향 없음.

        dwell  : 역·방향·요일·시간대별 정차 (급행 대피 반영)            data/marts/v2_line9_dwell.parquet
        switch : 같은 역 일반↔급행 갈아타기 실제 연계 대기             data/marts/v2_line9_switch_wait.parquet
        service: 9L/9X 가 그 시간대에 운행하지 않으면 out_of_window 처리 (headway is_operating)
        """
        self.dwell_tab, self.switch_tab, self.service_lines, self.run_tab = {}, {}, set(), {}
        mart = root / "data" / "marts"
        f = mart / "v2_line9_run.parquet"
        if f.exists():
            r_ = pd.read_parquet(f)
            r_ = r_[r_.day_type == self.day_type]
            self.run_tab = {(r.from_node, r.to_node, r.time_bin): float(r.run_min) for r in r_.itertuples()}
        f = mart / "v2_line9_dwell.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            d = d[d.day_type == self.day_type]
            self.dwell_tab = {(r.station_uid, r.time_bin): float(r.dwell_min)
                              for r in d.itertuples()}
        f = mart / "v2_line9_switch_wait.parquet"
        if f.exists():
            w = pd.read_parquet(f)
            w = w[w.day_type == self.day_type]
            self.switch_tab = {(r.from_line, r.to_line, r.station, r.direction, r.time_bin): float(r.wait_min)
                               for r in w.itertuples()}
        f = mart / "headway_station_30min.parquet"
        if self.dwell_tab and f.exists():
            h = pd.read_parquet(f, columns=["line_id", "station_name", "direction", "day_type",
                                            "time_bin", "is_operating"])
            h = h[h.line_id.isin(["9L", "9X"]) & (h.day_type == self.day_type)]
            self.service_lines = {"9L", "9X"}
            self.operating = {(r.line_id, r.station_name, r.direction, r.time_bin)
                              for r in h.itertuples() if r.is_operating == 1}

    @staticmethod
    def _hw_bin(minute: float) -> str:
        """배차·정차 표의 30분 bin 이름 (05:00 기준)."""
        s = int((minute - 300) // 30) * 30 + 300
        e = s + 30
        return f"{(s // 60) % 24:02d}:{s % 60:02d}~{(e // 60) % 24:02d}:{e % 60:02d}"

    def _run(self, u: str, v: str, minute: float) -> float:
        """구간 운행시간. 9호선은 요일·시간대별 표 (없으면 평일 중앙값), 나머지는 v1 고정값."""
        if self.run_tab and self.policy != "fixed":
            return self.run_tab.get((u, v, self._hw_bin(minute)), self.edge_meta[(u, v)]["time"])
        return self.edge_meta[(u, v)]["time"]

    def _dwell(self, node: str, minute: float) -> float:
        if self.dwell_tab and self.policy != "fixed":
            return self.dwell_tab.get((node, self._hw_bin(minute)), self.DWELL)
        return self.DWELL

    # ------------------------------------------------------------ 혼잡도 조회
    def _event_mult(self, station: str, minute: float, line: str | None = None) -> float:
        if line is not None and line in self.event_exclude_lines:
            return 1.0
        by_hour = getattr(self.s, "event_effect_by_hour", None) or {}
        if not by_hour:
            return 1.0
        if self.policy == "fixed":
            eff = self.s.event_effect          # v1: 출발 시각의 hour 고정
        else:
            eff = by_hour.get(int(minute // 60), {})
        m = float(eff.get(station, {}).get("mult", 1.0))
        return 1.0 + (m - 1.0) * self.event_scale

    def _pieces(self, u: str, v: str, t0: float, dur: float):
        """edge 구간 [t0, t0+dur] 을 (분, 혼잡median, 혼잡p90, bin) 조각 리스트로 나눈다."""
        clamp = lambda b: max(0, min(N_BINS - 1, b))
        if dur <= 0:
            return []
        if self.policy == "fixed":
            b = self.fixed_bin
            return [(dur, self.cong_med[(u, v, b)], self.cong_p90[(u, v, b)], b)]
        if self.policy == "floor":
            b = clamp(bin_index_of(t0))
            return [(dur, self.cong_med[(u, v, b)], self.cong_p90[(u, v, b)], b)]
        if self.policy == "linear":
            mid = t0 + dur / 2
            x = (mid - BIN_START_MIN) / BIN_WIDTH_MIN - 0.5        # bin 중심 = 정수 좌표
            lo = clamp(int(math.floor(x)))
            hi = clamp(lo + 1)
            w = min(max(x - lo, 0.0), 1.0) if hi != lo else 0.0
            med = (1 - w) * self.cong_med[(u, v, lo)] + w * self.cong_med[(u, v, hi)]
            p90 = (1 - w) * self.cong_p90[(u, v, lo)] + w * self.cong_p90[(u, v, hi)]
            return [(dur, med, p90, lo if w < 0.5 else hi)]
        # step: bin 경계에서 분할
        out, t, end = [], t0, t0 + dur
        while t < end - 1e-9:
            b_raw = bin_index_of(t)
            b_end = BIN_START_MIN + (b_raw + 1) * BIN_WIDTH_MIN
            seg_end = min(end, b_end)
            b = clamp(b_raw)
            out.append((seg_end - t, self.cong_med[(u, v, b)], self.cong_p90[(u, v, b)], b))
            t = seg_end
        return out

    def _wait(self, line: str, station: str, direction, minute: float) -> float:
        """환승(또는 최초 승차) 대기. v1 expected_wait 를 그대로 쓰되 bin 만 시각 기준으로."""
        from station_routing import expected_wait      # app/streamlit 경로가 sys.path 에 있어야 함
        if self.policy == "fixed":
            tb = self.s.time_bin
        else:
            tb = bin_label(max(0, min(N_BINS - 1, bin_index_of(minute))))
        w = expected_wait(self.headway, line, station, direction, self.day_type, tb)
        return float(w) if w else 0.0

    # ------------------------------------------------------------ 평가
    def evaluate(self, path: list[str], depart_min: float | None = None,
                 keep_trace: bool = False) -> TDResult:
        """경로 1개를 시간 진행형으로 평가한다. depart_min 미지정 시 scorer 의 출발 시각."""
        if depart_min is None:
            depart_min = parse_hhmm(self.s.depart)
        adj = self.s.adj
        t = float(depart_min)
        trace: list[EdgeTrace] = []
        exp = {x: 0.0 for x in EXPOSURE_THRESHOLDS}
        exp90 = {x: 0.0 for x in EXPOSURE_THRESHOLDS}
        perceived = running = dwell = walk = wait = 0.0
        congs_edge, cong_time_sum, cong_time_den = [], 0.0, 0.0
        max_c = -np.inf
        bins_used: set[int] = set()
        oow = False
        n_tr = 0
        init_wait = 0.0

        def ride_block(u, v, t0, dur, kind):
            nonlocal perceived, cong_time_sum, cong_time_den, max_c, oow
            pcs = self._pieces(u, v, t0, dur)
            meta = self.edge_meta[(u, v)]
            mult = self._event_mult(meta["from_station"], t0, meta["line"])
            if kind == "ride" and meta["line"] in self.service_lines and self.policy != "fixed":
                # 9L/9X: 그 시각에 해당 패턴 열차가 없으면 계산은 하되 분석 제외 표시 (D-043)
                if (meta["line"], meta["from_station"], meta["direction"], self._hw_bin(t0)) not in self.operating:
                    oow = True
            cw, cmax = 0.0, -np.inf
            # 자료 범위(05:30~01:00) 밖 구간은 _pieces 가 경계 bin 으로 clamp 한다.
            # 값은 계산하되 out_of_window 로 표시하고, 분석(od_shiftability.classify)에서는 제외한다 (D-036).
            if self.policy != "fixed" and (t0 < BIN_START_MIN - 1e-9 or t0 + dur > BIN_END_MIN + 1e-9):
                oow = True
            for d, c, c90, b in pcs:
                c, c90 = c * mult, c90 * mult
                bins_used.add(b)
                cw += d * c
                cmax = max(cmax, c)
                for x in EXPOSURE_THRESHOLDS:
                    if c >= x:
                        exp[x] += d
                    if c90 >= x:
                        exp90[x] += d
                if kind == "ride":
                    perceived += d * (1.0 + self.KAPPA * max(c - self.C0, 0.0) / 100.0)
            cavg = cw / dur if dur > 0 else float("nan")
            cong_time_sum += cw
            cong_time_den += dur
            max_c = max(max_c, cmax)
            if keep_trace:
                trace.append(EdgeTrace(kind, u, v, t0, t0 + dur, round(cavg, 2),
                                       round(cmax, 2), tuple(sorted({p[3] for p in pcs}))))
            return cavg

        # 최초 승차 대기 (옵션)
        first_ride = next(((u, v) for u, v in zip(path, path[1:])
                           if (u, v) in self.edge_meta), None)
        if self.include_initial_wait and first_ride:
            m = self.edge_meta[first_ride]
            init_wait = self._wait(m["line"], m["from_station"], m["direction"], t)
            t += init_wait

        prev_was_ride = False
        for i, (u, v) in enumerate(zip(path, path[1:])):
            e = next(x for x in adj[u] if x["to"] == v)
            if e["kind"] == "ride":
                dw = self._dwell(u, t) if prev_was_ride else 0.0
                if prev_was_ride and dw > 0:
                    # 직전 역 정차: 차내 시간. 정차 중 열차 혼잡 = 다음 구간 출발 혼잡으로 본다.
                    # 체감시간에는 v1 과 같이 배수 없이 1회만 더한다. 9호선은 역·시간대별 정차 (대피 포함).
                    ride_block(u, v, t, dw, "dwell")
                    perceived += dw
                    dwell += dw
                    t += dw
                d = self._run(u, v, t)
                c = ride_block(u, v, t, d, "ride")
                congs_edge.append(c)
                running += d
                t += d
                prev_was_ride = True
            elif e["kind"] == "transfer" and self._is_through(path, i, u, v):
                continue                    # 강동 직결 통과: 같은 열차, 시간·환승 없음
            else:
                prev_was_ride = False
                if e["kind"] != "transfer":
                    continue
                n_tr += 1
                a = float(e["time"])
                walk += a
                perceived += float(e["cost"])          # v1: 환승 페널티(도보+혼잡환승) 그대로
                if keep_trace:
                    trace.append(EdgeTrace("transfer_walk", u, v, t, t + a, None, None))
                t += a
                nxt = path[i + 2] if i + 2 < len(path) else None
                if nxt and (v, nxt) in self.edge_meta:
                    m = self.edge_meta[(v, nxt)]
                    if self.policy != "fixed" and not (BIN_START_MIN <= t < BIN_END_MIN):
                        oow = True                      # 배차간격도 자료 범위 밖 → clamp 값
                    fl, tl = e.get("from_line") or u.split("_", 1)[0], m["line"]
                    sw = (self.switch_tab.get((u.split("_", 1)[0], tl, m["from_station"], m["direction"],
                                               self._hw_bin(t)))
                          if self.switch_tab and {u.split("_", 1)[0], tl} == {"9L", "9X"} else None)
                    h = sw if sw is not None else self._wait(m["line"], m["from_station"], m["direction"], t)
                    if keep_trace and h:
                        trace.append(EdgeTrace("transfer_wait", v, v, t, t + h, None, None))
                    wait += h
                    perceived += h
                    t += h

        in_vehicle = running + dwell
        actual = t - depart_min
        return TDResult(
            path=list(path), policy=self.policy, depart_min=depart_min, arrive_min=t,
            actual_time_min=actual, perceived_time_min=perceived + init_wait,
            in_vehicle_min=in_vehicle, running_time_min=running, dwell_time_min=dwell,
            transfer_walk_min=walk, transfer_wait_min=wait, initial_wait_min=init_wait,
            transfer_count=n_tr,
            max_congestion=float(max_c) if congs_edge else float("nan"),
            avg_congestion_edge=float(np.mean(congs_edge)) if congs_edge else float("nan"),
            avg_congestion_time=cong_time_sum / cong_time_den if cong_time_den else float("nan"),
            exposure=exp, exposure_p90=exp90, bins_used=tuple(sorted(bins_used)),
            out_of_window=oow, trace=trace,
        )


# --------------------------------------------------------------------------- 후보 생성
class TDRouter:
    """역 단위 OD 에 대해 후보 경로를 만들고 시간 진행형으로 평가한다.

    후보 생성 근사
    --------------
    Yen 은 정적 가중치에서만 돈다. 출발 bin 가중치로 만든 후보만 쓰면, 이동 중 넘어가는
    뒤쪽 bin 에서 더 나은 경로를 놓칠 수 있다. 그래서 출발 bin 과 '예상 이동 중 통과하는
    bin' 각각의 scorer 로 Yen 을 돌려 후보를 합집합한 뒤, 전부 시간 진행형으로 재평가한다.
    """

    def __init__(self, root: Path, scorer_module, station_routing_module,
                 display: pd.DataFrame, day_type: str = "weekday", policy: str = "step",
                 k: int = 5, multi_bin: bool = True, include_initial_wait: bool = False,
                 query_date: str | None = None, event_scale: float = 1.0,
                 event_exclude_lines: tuple = ()):
        self.root = Path(root)
        self.mod = scorer_module
        self.sr = station_routing_module
        self.display = display
        self.day_type = day_type
        self.policy = policy
        self.k = k
        self.multi_bin = multi_bin
        self.include_initial_wait = include_initial_wait
        self.query_date = query_date          # v2.5: 이벤트 날짜 (None = 평시)
        self.event_scale = event_scale
        self.event_exclude_lines = tuple(event_exclude_lines)
        self.headway = self.sr.load_headway(self.root)
        self._scorers: dict[int, object] = {}
        self._evals: dict[tuple, TimeDependentEvaluator] = {}

    def scorer(self, b: int):
        b = max(0, min(N_BINS - 1, b))
        if b not in self._scorers:
            hhmm = fmt_min(BIN_START_MIN + b * BIN_WIDTH_MIN)
            self._scorers[b] = self.mod.RouteScorer(self.root, self.day_type, hhmm, self.query_date)
        return self._scorers[b]

    def evaluator(self, depart_bin: int, policy: str | None = None) -> TimeDependentEvaluator:
        policy = policy or self.policy
        key = (max(0, min(N_BINS - 1, depart_bin)), policy)
        if key not in self._evals:
            self._evals[key] = TimeDependentEvaluator(
                self.scorer(depart_bin), self.headway, policy, self.include_initial_wait,
                self.event_scale, self.event_exclude_lines)
        return self._evals[key]

    def raw_paths(self, scorer, origin: str, dest: str, k: int) -> list[list[str]]:
        """station_routing.find_route_by_station 과 같은 가상노드 방식으로 원시 후보만 뽑는다."""
        sr = self.sr
        o_nodes = sr.candidates_of(self.display, origin)
        d_nodes = sr.candidates_of(self.display, dest)
        if not o_nodes or not d_nodes or origin == dest:
            return []
        v_o, v_d = sr.ORIGIN_PREFIX + origin, sr.DEST_PREFIX + dest
        orig_adj, orig_nodes = scorer.adj, scorer.nodes
        aug = dict(orig_adj)
        aug[v_o] = [{"to": n, "cost": 0.0, "time": 0.0, "cong": float("nan"), "event": 0.0,
                     "kind": "origin_access"} for n in o_nodes if n in orig_nodes]
        for n in d_nodes:
            if n in orig_nodes:
                aug[n] = list(aug.get(n, [])) + [{"to": v_d, "cost": 0.0, "time": 0.0,
                                                  "cong": float("nan"), "event": 0.0,
                                                  "kind": "destination_exit"}]
        try:
            scorer.adj = aug
            scorer.nodes = set(aug) | {e["to"] for lst in aug.values() for e in lst}
            paths = scorer._yen(v_o, v_d, K=k, weight="time") + scorer._yen(v_o, v_d, K=k, weight="cost")
        finally:
            scorer.adj, scorer.nodes = orig_adj, orig_nodes
        seen, out = set(), []
        for p in paths:
            real = [n for n in p if not sr.is_virtual(n)]
            # 출발·도착역에서의 같은 역 갈아타기(예: 9X_역 → 9L_역)는 의미가 없으므로 잘라낸다
            while len(real) >= 2 and sr.station_of(real[0]) == sr.station_of(real[1]):
                real = real[1:]
            while len(real) >= 2 and sr.station_of(real[-1]) == sr.station_of(real[-2]):
                real = real[:-1]
            real = tuple(real)
            if len(real) < 2 or real in seen or sr.has_station_revisit(list(real)):
                continue
            seen.add(real)
            out.append(list(real))
        return out

    def candidates(self, origin: str, dest: str, depart_min: float,
                   k: int | None = None, multi_bin: bool | None = None) -> list[list[str]]:
        k = k or self.k
        multi_bin = self.multi_bin if multi_bin is None else multi_bin
        b0 = bin_index_of(depart_min)
        paths = self.raw_paths(self.scorer(b0), origin, dest, k)
        if not paths or not multi_bin or self.policy == "fixed":
            return paths
        ev = self.evaluator(b0)
        arrive = max(ev.evaluate(p, depart_min).arrive_min for p in paths)
        seen = {tuple(p) for p in paths}
        for b in range(b0 + 1, bin_index_of(arrive) + 1):
            for p in self.raw_paths(self.scorer(b), origin, dest, k):
                if tuple(p) not in seen:
                    seen.add(tuple(p))
                    paths.append(p)
        return paths

    def evaluate_od(self, origin: str, dest: str, depart_min: float,
                    policy: str | None = None, k: int | None = None,
                    multi_bin: bool | None = None) -> list[TDResult]:
        """후보 전부를 평가해 실제 소요시간 오름차순으로 돌려준다. [0] = 시간 진행형 최속."""
        policy = policy or self.policy
        b0 = bin_index_of(depart_min)
        ev = self.evaluator(b0, policy)
        paths = self.candidates(origin, dest, depart_min, k=k,
                                multi_bin=(False if policy == "fixed" else multi_bin))
        res = [ev.evaluate(p, depart_min) for p in paths]
        res.sort(key=lambda r: (round(r.actual_time_min, 6), r.perceived_time_min))
        return res
