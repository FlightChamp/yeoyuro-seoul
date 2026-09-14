"""
09_route_scoring_prototype.py
=============================
이 프로젝트의 전제를 검증한다.

  "서울 지하철 1~8호선에서, 최단경로와 유의미하게 다른 '쾌적 대안 경로'가 실제로 존재하는가?"

존재하지 않는다면 혼잡도 예측 모델(Model B)에 3주를 쓰기 전에 알아야 한다.
프로토타입이므로 완성본이 아니다. 아래를 의도적으로 단순화했다.

  - 이벤트 위험(event_risk)은 Phase 2 미완이므로 0
  - 출발 시각의 time_bin 을 경로 전체에 고정 적용 (시간 전진 미반영)
  - 예측 혼잡도가 아니라 11개 스냅샷의 중앙값(관측 패턴)을 사용

핵심 설계 결정
--------------
1. seat bonus 를 edge cost 에서 빼지 않는다.
   음수 가중치가 생기면 Dijkstra 의 최적성이 깨진다.
     edge_cost   = perceived_time + transfer_penalty        (항상 양수)
     route_score = Σ edge_cost − eta × seat_chance_score    (경로 확정 후)

2. 엣지 -> 혼잡도 방향 라벨 매핑 (혼잡도 조인의 전제)
   혼잡도는 (역, 방향, 시간대) 단위, 그래프는 (A->B) 단위다.
   노선별로 '역번호 증가'가 뜻하는 방향이 다르며, 아래는 오전 피크
   도심 방면 부하로 실측 검증했다.
     1호선          : 코드 증가 = 상선(up)
     3~8호선        : 코드 증가 = 하선(down)
     2호선 본선     : 코드 증가 = 내선(inner)   (순환 폐합 구간은 반전)
     지선/순환      : 계통 순서로 판정 (BRANCH_SEQ)

3. 환승 페널티는 절대 기준을 쓴다.
   백분위 기반은 역이 하나 추가되면 모든 값이 바뀐다.
     volume_penalty_min = min(3.0, weekday_volume / 50000)   # 5만 명당 1분

4. 대안 경로 채택 조건 (초기값 — 결과를 보고 튜닝할 대상)
     time_loss <= 15분  AND  최대혼잡 감소 >= 15%p  AND  체감시간 <= 최속 + 5분
   미달이면 대안을 만들지 않고 '대안 없음'으로 응답한다.

사용법
------
    python scripts/09_route_scoring_prototype.py --root .
    python scripts/09_route_scoring_prototype.py --root . --from 2_신촌 --to 2_잠실 --time 08:30
"""

from __future__ import annotations

import argparse
import heapq
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENC = "utf-8-sig"

# 체감시간 환산 (04 스크립트와 동일 파라미터)
C0 = 80.0
KAPPA = 0.5

# 중간역 정차시간(분).
# 원본 역간거리 및 소요시간 의 소요시간은 순수 주행시간이며 정차시간이 없다.
# 열차운행현황 의 공표 소요시간·표정속도와 교차검증한 결과 중간역당 약 30초가
# 빠져 있었다(1호선 30.0초, 3호선 30.0초, 6호선 30.8초, 7호선 27.0초).
# 노선·시간대별 실제 정차시간은 다르므로 v1 한계로 남긴다.
DEFAULT_DWELL_TIME_MIN = 0.5

# 직결 분기역.
# 5호선은 방화~하남검단산 / 방화~마천 이 하나의 계통으로 직결 운행한다
# (열차운행현황 구간 표기 "방화~하남검단산/마천", 시격·운행횟수 단일 집계).
# 2호선 성수·신도림 지선은 별도 셔틀이라 환승이 필수지만
# (구간 표기 "성수~성수[성수지선/신정지선]", 시격·운행횟수 분리 집계),
# 강동은 본선에서 온 열차가 그대로 지선으로 들어간다.
#
# 그래프는 강동을 본선/지선 노드로 나누고 그 사이에 환승 엣지를 두었다.
# 이 엣지는 경로에 따라 의미가 다르다.
#   천호(본선) <-> 둔촌동(마천)  : 같은 열차로 통과. 환승 아님
#   길동(하남) <-> 둔촌동(마천)  : 강동에서 갈아타야 함. 환승 맞음
# 엣지 하나로는 구분할 수 없으므로 경로의 앞뒤 노드를 보고 판정한다.
# value 는 본선(trunk) 쪽 인접 노드다.
THROUGH_JUNCTIONS = {
    frozenset(("5_강동", "5_강동@macheon_branch")): "5_천호",
}


def through_junction_trunk(u: str, v: str):
    """(u, v) 가 직결 분기 엣지면 본선 쪽 인접 노드를, 아니면 None 을 돌려준다."""
    return THROUGH_JUNCTIONS.get(frozenset((u, v)))


def is_through_pass(path, i: int, u: str, v: str) -> bool:
    """경로의 i 번째 엣지 (u, v) 가 '같은 열차로 통과' 인지 판정한다.

    분기 엣지의 바로 앞 또는 바로 뒤 노드가 본선 쪽이면 통과다.
    최초 승차 전 대기는 이 프로젝트에서 반영하지 않으므로,
    통과 시에는 환승 횟수·도보·대기를 모두 더하지 않는다.
    """
    trunk = through_junction_trunk(u, v)
    if trunk is None:
        return False
    prev_n = path[i - 1] if i > 0 else None
    next_n = path[i + 2] if i + 2 < len(path) else None
    return trunk in (prev_n, next_n)

# 환승 혼잡 페널티: 5만 명당 1분, 최대 3분
TRANSFER_CROWD_PENALTY_RATE = 50000.0
TRANSFER_CROWD_PENALTY_MAX = 3.0

# 대안 채택 조건
MAX_TIME_LOSS_MIN = 15.0
MIN_CONGESTION_DROP_PP = 15.0
MAX_PERCEIVED_EXCESS_MIN = 5.0

# 시간대 대안: 출발 시각 기준 앞뒤 탐색 폭(분)과 최소 개선폭
TIME_ALT_WINDOW_MIN = 120
TIME_ALT_MIN_GAIN_PP = 10.0

# 이벤트 보정: 승하차 spike 배율을 그대로 혼잡도에 곱하지 않는다.
# spike 는 '역 이용객' 급증이고 혼잡도는 '열차 내 밀도'라 같은 크기로 움직이지 않는다.
# LAMBDA 는 그 감쇠 계수이며, 실측 대조 전까지는 보수적으로 잡는다(튜닝 대상).
EVENT_LAMBDA = 0.3

# 경로 다양성: 노드 자카드 유사도가 이 값 이상이면 사실상 같은 경로로 보고 제외
MAX_PATH_SIMILARITY = 0.65

# 모드별 가중치
MODES = {
    "fast":        {"time": 1.0, "perceived": 0.0, "max_cong": 0.00, "transfer": 0.3, "seat": 0.0},
    "calm":        {"time": 0.0, "perceived": 1.0, "max_cong": 0.05, "transfer": 0.5, "seat": 0.0},
    "seat":        {"time": 0.0, "perceived": 1.0, "max_cong": 0.02, "transfer": 0.5, "seat": 2.0},
    "min_transfer": {"time": 0.5, "perceived": 0.5, "max_cong": 0.00, "transfer": 3.0, "seat": 0.0},
    "balanced":    {"time": 0.3, "perceived": 0.7, "max_cong": 0.03, "transfer": 0.8, "seat": 0.5},
}

# 코드 증가 방향이 뜻하는 라벨 (오전 피크 도심 방면 부하로 실측 검증)
CODE_ASC_DIRECTION = {
    "1": "up", "2": "inner", "3": "down", "4": "down",
    "5": "down", "6": "down", "7": "down", "8": "down",
}
CODE_DESC_DIRECTION = {
    "1": "down", "2": "outer", "3": "up", "4": "up",
    "5": "up", "6": "up", "7": "up", "8": "up",
}

# 지선·순환 계통의 진행 순서. 리스트 방향으로 가면 'forward_direction'.
BRANCH_SEQ = {
    "seongsu_branch": (["성수", "용답", "신답", "용두", "신설동"], "outer", "inner"),
    "sinjeong_branch": (["신도림", "도림천", "양천구청", "신정네거리", "까치산"], "inner", "outer"),
    "macheon_branch": (["강동", "둔촌동", "올림픽공원", "방이", "오금", "개롱", "거여", "마천"],
                       "down", "up"),
}
EUNGAM_LOOP = ["응암", "역촌", "불광", "독바위", "연신내", "구산", "응암"]


def load_mart(mart_dir: Path, name: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv.gz"):
        p = mart_dir / f"{name}{ext}"
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"{name} 이 없습니다. 선행 스크립트를 실행하세요.")


def station_of(node: str) -> str:
    return node.split("_", 1)[1].split("@")[0]


def line_of(node: str) -> str:
    return node.split("_", 1)[0]


# --------------------------------------------------------------------------
class RouteScorer:

    def __init__(self, root: Path, day_type: str = "weekday", depart: str = "08:30",
                 query_date: str | None = None):
        self.root = root
        self.mart = root / "data" / "marts"
        self.query_date = query_date
        if query_date:
            dow = pd.Timestamp(query_date).dayofweek
            auto = "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")
            if auto != day_type:
                day_type = auto
        self.day_type = day_type
        self.depart = depart
        self.notes: list[str] = []

        self.ride = load_mart(self.mart, "route_edge_mart")
        self.transfer = load_mart(self.mart, "transfer_edge_mart")
        self.lookup = load_mart(self.mart, "congestion_edge_lookup")
        for df in (self.ride, self.transfer, self.lookup):
            if "line_id" in df.columns:
                df["line_id"] = df["line_id"].astype(str)

        self.time_bin = self._resolve_time_bin(depart)
        self.event_effect = self._load_event_effect()
        self._attach_congestion()
        self._build_adjacency()

    # ---------- 시간대 ----------
    def _resolve_time_bin(self, depart: str) -> str:
        hh, mm = (int(x) for x in depart.split(":"))
        total = hh * 60 + mm
        start = 5 * 60 + 30
        idx = max(0, min(38, (total - start) // 30))
        s = start + idx * 30
        e = s + 30
        return f"{s // 60:02d}:{s % 60:02d}~{e // 60:02d}:{e % 60:02d}"

    # ---------- 이벤트 ----------
    def _load_event_effect(self) -> dict[str, dict]:
        """조회 날짜에 활성화된 이벤트를 역별 혼잡 배수로 환산한다."""
        self.event_effect_by_hour: dict[int, dict[str, dict]] = {}
        if not self.query_date:
            return {}
        master = self.root / "data" / "master"
        cal_p = master / "event_calendar_master.csv"
        str_p = master / "event_strength_verified.csv"
        if not cal_p.exists():
            self.notes.append("event_calendar_master.csv 없음 → event_risk 0")
            return {}
        cal = pd.read_csv(cal_p)
        strength = pd.read_csv(str_p) if str_p.exists() else pd.DataFrame()
        if strength.empty:
            self.notes.append("event_strength_verified.csv 없음 → 05 를 먼저 실행하세요. event_risk 0")
            return {}

        smap = strength.set_index("event_id").to_dict("index")
        qd = str(self.query_date)
        effect_by_hour: dict[int, dict[str, dict]] = {h: {} for h in range(5, 26)}
        effect: dict[str, dict] = {}
        active = []
        for r in cal.itertuples():
            s0 = str(r.impact_start_date) if pd.notna(r.impact_start_date) else str(r.official_start_date)
            e0 = str(r.impact_end_date) if pd.notna(r.impact_end_date) else str(r.official_end_date)
            if not (s0 <= qd <= e0):
                continue
            info = smap.get(r.event_id)
            if not info:
                continue
            if int(info.get("exclude_from_training", 0)) == 1:
                self.notes.append(f"{r.event_id}: 학습 제외 이벤트라 보정에서 제외")
                continue
            w = float(info.get("use_weight", 0.0))
            ratio = float(info.get("spike_ratio", 1.0))
            if w <= 0 or ratio <= 1:
                continue
            # 야간 한정 이벤트는 해당 시간대에만 적용
            bins = str(r.peak_hour_bins) if pd.notna(r.peak_hour_bins) else ""
            if info.get("apply_scope") == "peak_hours_only" and bins:
                hrs = {(int(x) + 24 if int(x) < 5 else int(x)) for x in bins.split(";") if x.strip()}
            else:
                hrs = set(range(5, 26))
            mult = 1.0 + w * (ratio - 1.0) * EVENT_LAMBDA
            stations = [x for x in (str(r.anchor_stations) + ";" + str(r.impact_stations)).split(";")
                        if x and x != "nan"]
            for h in hrs:
                if h not in effect_by_hour:
                    continue
                for st in stations:
                    cur = effect_by_hour[h].get(st, {"mult": 1.0, "events": []})
                    cur["mult"] = max(cur["mult"], mult)
                    cur["events"].append(r.event_id)
                    effect_by_hour[h][st] = cur
            hr_txt = "종일" if len(hrs) >= 20 else f"{min(hrs)}~{max(hrs)}시"
            active.append(f"{r.event_name}(x{mult:.2f}, {len(stations)}역, {hr_txt})")
        if active:
            self.notes.append(f"활성 이벤트 {len(active)}건: " + "; ".join(active))
        else:
            self.notes.append(f"{qd}: 활성 이벤트 없음")
        self.event_effect_by_hour = effect_by_hour
        return effect_by_hour.get(int(self.time_bin[:2]), {})

    # ---------- 방향 라벨 ----------
    def _edge_direction(self, r) -> str | None:
        line = str(r.line_id)
        a, b = r.from_station, r.to_station

        # 응암순환 (단방향)
        if r.is_bidirectional == 0:
            return "down"

        # 지선
        for code, (seq, fwd, bwd) in BRANCH_SEQ.items():
            if r.branch_code == code and a in seq and b in seq:
                return fwd if seq.index(b) > seq.index(a) else bwd

        fc, tc = r.from_code, r.to_code
        if pd.isna(fc) or pd.isna(tc):
            return None
        fc, tc = int(fc), int(tc)

        # 2호선 순환 폐합 (충정로 243 <-> 시청 201) : 코드 규칙이 반전된다
        if line == "2" and abs(fc - tc) > 10 and fc < 9000 and tc < 9000:
            return "inner" if fc > tc else "outer"

        # 분기 노드가 끼면 계통 순서로 판정 (코드가 9xxx 라 비교 불가)
        if fc >= 9000 or tc >= 9000:
            for code, (seq, fwd, bwd) in BRANCH_SEQ.items():
                if a in seq and b in seq:
                    return fwd if seq.index(b) > seq.index(a) else bwd
            return None

        return CODE_ASC_DIRECTION.get(line) if tc > fc else CODE_DESC_DIRECTION.get(line)

    # ---------- 혼잡도 결합 ----------
    def _attach_congestion(self) -> None:
        r = self.ride.copy()
        r["direction"] = [self._edge_direction(x) for x in r.itertuples()]
        n_nodir = int(r["direction"].isna().sum())
        if n_nodir:
            self.notes.append(f"방향 판정 실패 엣지 {n_nodir}개")

        lk = self.lookup[(self.lookup["day_type"] == self.day_type)
                         & (self.lookup["time_bin"] == self.time_bin)]
        lk = lk[["station_uid", "direction", "congestion_median"]].drop_duplicates(
            subset=["station_uid", "direction"])

        # 엣지 혼잡도 = 출발역에서 그 방향으로 떠나는 열차의 혼잡도
        r = r.merge(lk.rename(columns={"station_uid": "from_node",
                                       "congestion_median": "congestion"}),
                    on=["from_node", "direction"], how="left")
        miss = int(r["congestion"].isna().sum())
        self.notes.append(
            f"혼잡도 매칭 {len(r) - miss}/{len(r)} 엣지 ({(1 - miss / len(r)) * 100:.1f}%)")
        if miss:
            sample = r[r["congestion"].isna()][["from_node", "to_node", "direction"]].head(8)
            self.unmatched = sample
        # 미매칭은 노선·방향 평균으로 보정
        fill = r.groupby(["line_id", "direction"])["congestion"].transform("median")
        r["congestion"] = r["congestion"].fillna(fill).fillna(r["congestion"].median())

        # 이벤트 보정 (혼잡도 배수). 단위 정합을 위해 '분'을 직접 더하지 않는다.
        r["congestion_base"] = r["congestion"]
        r["event_mult"] = [self.event_effect.get(st, {}).get("mult", 1.0)
                           for st in r["from_station"]]
        r["congestion"] = r["congestion_base"] * r["event_mult"]

        base_perceived = r["travel_time_min"] * (
            1.0 + KAPPA * np.clip(r["congestion_base"] - C0, 0, None) / 100.0)
        r["perceived_time_min"] = r["travel_time_min"] * (
            1.0 + KAPPA * np.clip(r["congestion"] - C0, 0, None) / 100.0)
        r["event_risk_min"] = (r["perceived_time_min"] - base_perceived).round(3)
        n_ev = int((r["event_mult"] > 1.0).sum())
        if n_ev:
            self.notes.append(f"이벤트 보정 적용 엣지 {n_ev}개 "
                              f"(추가 체감시간 합 {r['event_risk_min'].sum():.1f}분)")
        self.ride = r

        # 환승 페널티 재계산 (절대 기준)
        t = self.transfer.copy()
        vol = pd.to_numeric(t.get("weekday_volume"), errors="coerce")
        t["volume_penalty_min"] = np.minimum(
            TRANSFER_CROWD_PENALTY_MAX,
            (vol.fillna(0) / TRANSFER_CROWD_PENALTY_RATE)).round(3)
        t["transfer_penalty_min"] = (t["transfer_time_min"] + t["volume_penalty_min"]).round(3)
        self.transfer = t

    # ---------- 그래프 ----------
    def _build_adjacency(self) -> None:
        self.adj: dict[str, list[dict]] = {}
        def add(u, v, **kw):
            self.adj.setdefault(u, []).append({"to": v, **kw})

        for r in self.ride.itertuples():
            add(r.from_node, r.to_node, cost=float(r.perceived_time_min),
                time=float(r.travel_time_min), cong=float(r.congestion),
                event=float(getattr(r, "event_risk_min", 0.0) or 0.0), kind="ride")
        for r in self.transfer.itertuples():
            c = float(r.transfer_penalty_min)
            add(r.from_node, r.to_node, cost=c, time=float(r.transfer_time_min),
                cong=np.nan, event=0.0, kind="transfer")
            add(r.to_node, r.from_node, cost=c, time=float(r.transfer_time_min),
                cong=np.nan, event=0.0, kind="transfer")
        self.nodes = set(self.adj) | {e["to"] for lst in self.adj.values() for e in lst}

    def _dijkstra(self, src, dst, ban_nodes=frozenset(), ban_edges=frozenset(),
                  weight="cost"):
        if src not in self.nodes or dst not in self.nodes:
            return None
        dist = {src: 0.0}
        prev = {}
        pq = [(0.0, src)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == dst:
                break
            for e in self.adj.get(u, []):
                v = e["to"]
                if v in ban_nodes or (u, v) in ban_edges:
                    continue
                nd = d + e[weight]
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return None
        path, cur = [dst], dst
        while cur != src:
            cur = prev[cur]
            path.append(cur)
        return list(reversed(path))

    def _yen(self, src, dst, K=5, weight="cost"):
        first = self._dijkstra(src, dst, weight=weight)
        if not first:
            return []
        A = [first]
        B = []
        for _ in range(K - 1):
            last = A[-1]
            for i in range(len(last) - 1):
                spur = last[i]
                root = last[:i + 1]
                ban_edges = set()
                for p in A:
                    if p[:i + 1] == root and len(p) > i + 1:
                        ban_edges.add((p[i], p[i + 1]))
                ban_nodes = set(root[:-1])
                sp = self._dijkstra(spur, dst, ban_nodes, frozenset(ban_edges), weight)
                if sp:
                    cand = root[:-1] + sp
                    if cand not in A and cand not in B:
                        B.append(cand)
            if not B:
                break
            B.sort(key=lambda p: self._path_cost(p, weight))
            A.append(B.pop(0))
        return A

    def _path_cost(self, path, weight="cost") -> float:
        tot = 0.0
        for u, v in zip(path, path[1:]):
            e = next((x for x in self.adj[u] if x["to"] == v), None)
            tot += e[weight] if e else 1e9
        return tot

    # ---------- 경로 평가 ----------
    def evaluate(self, path) -> dict:
        t = p = tp = ev = 0.0
        walk = 0.0
        congs = []
        n_tr = 0
        # 연속한 ride 엣지 묶음(= 같은 열차를 타고 가는 구간)마다
        # 중간 정차역 수는 max(k - 1, 0) 이다. 출발역·도착역·환승역은
        # 이 정의에서 자동으로 빠진다.
        seg_edges = 0
        dwell_stops = 0
        for i, (u, v) in enumerate(zip(path, path[1:])):
            e = next(x for x in self.adj[u] if x["to"] == v)
            if e["kind"] == "ride":
                seg_edges += 1
                t += e["time"]
                p += e["cost"]
                congs.append(e["cong"])
                ev += e.get("event", 0.0)
            elif e["kind"] == "transfer" and is_through_pass(path, i, u, v):
                # 같은 열차로 통과한다. 승차 구간을 끊지 않으므로
                # 분기역은 중간 정차역으로 계산된다.
                continue
            else:
                dwell_stops += max(seg_edges - 1, 0)
                seg_edges = 0
                if e["kind"] == "transfer":
                    n_tr += 1
                    walk += e["time"]
                    tp += e["cost"]
                t += e["time"]
                p += e["cost"]
        dwell_stops += max(seg_edges - 1, 0)
        running = t - walk
        dwell = dwell_stops * DEFAULT_DWELL_TIME_MIN
        # 정차시간은 물리적 시간이라 혼잡도 배수를 곱하지 않고 한 번만 더한다.
        t += dwell
        p += dwell
        congs = [c for c in congs if not np.isnan(c)]
        p95 = self.lookup["congestion_median"].quantile(0.95)
        seat = self._seat_score(path)
        return {
            "path": path,
            "actual_time_min": round(t, 1),
            "perceived_time_min": round(p, 1),
            "running_time_min": round(running, 1),
            "dwell_time_min": round(dwell, 1),
            "dwell_stop_count": int(dwell_stops),
            "transfer_walk_min": round(walk, 1),
            "avg_congestion": round(float(np.mean(congs)), 1) if congs else np.nan,
            "max_congestion": round(float(np.max(congs)), 1) if congs else np.nan,
            "p95_exposure_count": int(sum(c >= p95 for c in congs)),
            "transfer_count": n_tr,
            "transfer_penalty_min": round(tp, 1),
            "event_risk_min": round(ev, 1),
            "seat_chance_score": round(seat, 2),
            "n_stops": len(path) - 1 - n_tr,
        }

    def _seat_score(self, path) -> float:
        """혼잡도가 떨어지는 구간을 지날수록 높다. 방향 정보가 있는 유일한 신호."""
        drops = []
        for u, v in zip(path, path[1:]):
            e = next(x for x in self.adj[u] if x["to"] == v)
            if e["kind"] != "ride" or np.isnan(e["cong"]):
                continue
            nxt = [x for x in self.adj.get(v, []) if x["kind"] == "ride" and not np.isnan(x["cong"])]
            if nxt:
                drops.append(e["cong"] - float(np.mean([x["cong"] for x in nxt])))
        return float(np.mean(drops)) / 10.0 if drops else 0.0

    # ---------- 다양성 ----------
    @staticmethod
    def _similar(a: list[str], b: list[str]) -> float:
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb)

    def diverse(self, paths, k=3):
        out = []
        for p in paths:
            if all(self._similar(p, q) < MAX_PATH_SIMILARITY for q in out):
                out.append(p)
            if len(out) >= k:
                break
        return out

    # ---------- 시간대 대안 ----------
    def time_alternative(self, path, window_min: int = TIME_ALT_WINDOW_MIN):
        """경로를 바꿀 수 없을 때, 같은 경로의 최대혼잡이 가장 낮은 출발 시간대를 찾는다.

        탐색 범위는 '출발 시각 기준' 앞뒤 window_min 분이다.
        18시에 출발하는 사람에게 06:30 을 권하면 안 되기 때문이다.
        """
        edges = []
        for u, v in zip(path, path[1:]):
            e = next(x for x in self.adj[u] if x["to"] == v)
            if e["kind"] == "ride":
                r = self.ride[(self.ride.from_node == u) & (self.ride.to_node == v)]
                if len(r):
                    edges.append((u, r.iloc[0]["direction"]))
        if not edges:
            return None

        lk = self.lookup[self.lookup["day_type"] == self.day_type]
        key = lk.set_index(["station_uid", "direction", "time_bin"])["congestion_median"]
        bin_idx = (lk[["time_bin", "time_bin_index"]].drop_duplicates()
                   .set_index("time_bin")["time_bin_index"].to_dict())

        cur_idx = bin_idx.get(self.time_bin)
        if cur_idx is None:
            return None
        span = max(1, window_min // 30)
        bins = [b for b, i in bin_idx.items() if abs(i - cur_idx) <= span]

        rows = []
        for b in sorted(bins, key=lambda x: bin_idx[x]):
            eff = self.event_effect_by_hour.get(int(b[:2]), {}) if self.event_effect_by_hour else {}
            vals = []
            for u, d in edges:
                v = key.get((u, d, b))
                if v is None or pd.isna(v):
                    continue
                vals.append(float(v) * eff.get(station_of(u), {}).get("mult", 1.0))
            if vals:
                rows.append({"time_bin": b, "max_congestion": round(max(vals), 1),
                             "avg_congestion": round(float(np.mean(vals)), 1)})
        if not rows:
            return None
        df = pd.DataFrame(rows)
        cur = df[df["time_bin"] == self.time_bin]
        if not len(cur):
            return None
        cur_max = float(cur["max_congestion"].iloc[0])
        best = df.loc[df["max_congestion"].idxmin()]
        gain = round(cur_max - float(best["max_congestion"]), 1)
        if best["time_bin"] == self.time_bin or gain < TIME_ALT_MIN_GAIN_PP:
            return None          # 지금 출발하는 게 이미 최선이거나 개선폭이 미미하다
        return {"best_time_bin": best["time_bin"],
                "best_max_congestion": float(best["max_congestion"]),
                "current_max_congestion": cur_max,
                "gain_pp": gain,
                "profile": df}

    # ---------- 추천 ----------
    def recommend(self, src, dst) -> dict:
        fastest_paths = self._yen(src, dst, K=5, weight="time")
        calm_paths = self._yen(src, dst, K=5, weight="cost")
        if not fastest_paths:
            return {"ok": False, "reason": "경로 없음"}

        fastest = self.evaluate(fastest_paths[0])
        cands = self.diverse(
            sorted({tuple(p) for p in fastest_paths + calm_paths},
                   key=lambda p: self._path_cost(list(p), "cost")), k=4)
        cands = [self.evaluate(list(p)) for p in cands]

        alt = None
        for c in cands:
            if c["path"] == fastest["path"]:
                continue
            time_loss = c["actual_time_min"] - fastest["actual_time_min"]
            cong_drop = fastest["max_congestion"] - c["max_congestion"]
            excess = c["perceived_time_min"] - fastest["perceived_time_min"]
            if (time_loss <= MAX_TIME_LOSS_MIN and cong_drop >= MIN_CONGESTION_DROP_PP
                    and excess <= MAX_PERCEIVED_EXCESS_MIN):
                alt = {**c, "time_loss_vs_fastest": round(time_loss, 1),
                       "comfort_gain_vs_fastest": round(cong_drop, 1)}
                break

        time_alt = None if alt else self.time_alternative(fastest["path"])
        return {"ok": True, "from": src, "to": dst, "fastest": fastest,
                "alternative": alt, "time_alternative": time_alt, "candidates": cands}


# --------------------------------------------------------------------------
DEFAULT_ODS = [
    ("2_신촌", "2_잠실"), ("1_서울역", "2_강남"), ("5_군자", "5_여의나루"),
    ("2_한양대", "3_고속터미널"), ("4_혜화", "2_사당"), ("1_종각", "6_이태원"),
    ("2_건대입구", "2_홍대입구"), ("6_안암", "2_삼성"), ("5_방화", "5_마천"),
    ("6_응암", "3_고속터미널"),
]


def describe(node: str) -> str:
    return f"{line_of(node)}호선 {station_of(node)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--day-type", default="weekday")
    ap.add_argument("--time", default="08:30")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD. 이벤트 보정을 적용할 날짜")
    ap.add_argument("--from", dest="src", default=None)
    ap.add_argument("--to", dest="dst", default=None)
    args = ap.parse_args(argv)

    rs = RouteScorer(Path(args.root), args.day_type, args.time, args.date)
    print("=" * 88)
    date_txt = f"{args.date} " if args.date else ""
    print(f" 09_route_scoring_prototype — {date_txt}{rs.day_type} {args.time} "
          f"(time_bin {rs.time_bin})")
    print("=" * 88)
    for n in rs.notes:
        print(f"  {n}")
    if hasattr(rs, "unmatched"):
        print("  미매칭 샘플:")
        print(rs.unmatched.to_string(index=False))
    print()

    ods = [(args.src, args.dst)] if args.src and args.dst else DEFAULT_ODS
    rows = []
    for s, d in ods:
        r = rs.recommend(s, d)
        if not r["ok"]:
            rows.append({"OD": f"{describe(s)} → {describe(d)}", "결과": r["reason"]})
            continue
        f = r["fastest"]
        a = r["alternative"]
        rows.append({
            "OD": f"{describe(s)}→{describe(d)}",
            "최속_시간": f["actual_time_min"], "최속_체감": f["perceived_time_min"],
            "최속_최대혼잡": f["max_congestion"], "최속_환승": f["transfer_count"],
            "대안": "있음" if a else "없음",
            "대안_시간손실": a["time_loss_vs_fastest"] if a else None,
            "대안_혼잡감소pp": a["comfort_gain_vs_fastest"] if a else None,
            "대안_체감": a["perceived_time_min"] if a else None,
            "대안_환승": a["transfer_count"] if a else None,
            "시간대대안": (r["time_alternative"]["best_time_bin"][:5]
                       if r.get("time_alternative") else None),
            "시간대_혼잡감소pp": (r["time_alternative"]["gain_pp"]
                             if r.get("time_alternative") else None),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if "대안" in df.columns:
        n_alt = int((df["대안"] == "있음").sum())
        print(f"\n  대안 경로 존재: {n_alt}/{len(df)} OD쌍")
        n_time = int(df["시간대대안"].notna().sum()) if "시간대대안" in df.columns else 0
        print(f"  시간대 대안 제시: {n_time}/{len(df)} OD쌍")
        if n_alt == 0:
            print("  → 경로 대안이 전혀 없다. '출발 시간대 조정 + 혼잡 경보 + 환승 팁'"
                  " 비중을 키우는 방향으로 재검토한다.")

    # 상세 1건
    if len(ods) == 1 or True:
        s, d = ods[0]
        r = rs.recommend(s, d)
        if r["ok"]:
            print(f"\n[상세] {describe(s)} → {describe(d)}")
            for i, c in enumerate(r["candidates"], 1):
                mark = " (최속)" if c["path"] == r["fastest"]["path"] else ""
                print(f"  {i}{mark} {c['actual_time_min']}분 / 체감 {c['perceived_time_min']}분 "
                      f"/ 평균 {c['avg_congestion']}% / 최대 {c['max_congestion']}% "
                      f"/ 환승 {c['transfer_count']}회 / 고혼잡노출 {c['p95_exposure_count']}"
                      + (f" / 이벤트 +{c['event_risk_min']}분" if c["event_risk_min"] else ""))
                print("      " + " → ".join(station_of(n) for n in c["path"]))
    print("=" * 88)

    out = Path(args.root) / "reports" / "model"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "route_prototype_result.csv", index=False, encoding=ENC)
    print(f"결과 저장: {out / 'route_prototype_result.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
