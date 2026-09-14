"""
10_evaluate_routes.py
=====================
경로 추천이 '실제로 의미 있는 대안'을 만들어내는지 정량 검증한다.

평가 프레임
-----------
경로 추천에는 정답 경로가 없다. 그래서 Accuracy 계열 지표를 쓸 수 없고,
다목적 최적화(multi-objective optimization)의 표준 도구를 쓴다.

  3축 Pareto optimality   후보 경로가 열등한지(dominated) 판정. **후보 보존 장치**이지
                          최종 순위가 아니다.
                            obj1 actual_time_min      (작을수록 좋음)
                            obj2 max_congestion       (작을수록 좋음)
                            obj3 transfer_penalty_min (작을수록 좋음)

  Stretch Factor          candidate_time / fastest_time. OD 간 비교가 가능하도록 정규화.

  Directed-edge Jaccard   경로 다양성. **노드 기준이 아니라 방향 있는 엣지 기준**이다.
                          순환선·왕복·지선 때문에 같은 역을 지나도 실제 구간이 다를 수 있다.

왜 3축인가
----------
2축(시간, 혼잡)만 쓰면 아래가 탈락한다.
    A: 30분 / 120% / 환승 2회
    B: 31분 / 122% / 환승 0회
B 는 시간도 혼잡도 A 보다 나쁘므로 dominated 다. 그러나 환승 없는 직통은
실제 이용자에게 좋은 대안일 수 있다. 환승 부담을 별도 축으로 두어 후보를 보존한다.

transfer_count 가 아니라 transfer_penalty_min 을 축으로 쓴다.
환승 1회라도 긴 환승이면 부담이 크고, 2회라도 짧으면 낮기 때문이다.
transfer_count 는 보조 컬럼으로 남긴다.

최종 순위는 Pareto 가 아니라 perceived_time_min 기반 모드별 scoring 으로 정한다.

출력
----
    data/marts/route_evaluation_mart.parquet      경로 단위
    data/marts/route_evaluation_summary.parquet   요청(OD x 모드) 단위
    reports/route/route_eval_cases.csv
    reports/route/route_eval_summary.csv
    reports/route/pareto_front_scatter.png
    reports/route/route_recommendation_eval.md

사용법
------
    python scripts/10_evaluate_routes.py --root . --time 08:30
    python scripts/10_evaluate_routes.py --root . --time 18:00 --date 2025-09-27
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENC = "utf-8-sig"
MAX_PATH_SIMILARITY = 0.65

# 유의미한 대안 판정 조건 (초기값 - 결과를 보고 튜닝할 대상)
MAX_TIME_LOSS_MIN = 15.0
MIN_CONGESTION_DROP_PP = 15.0
MAX_PERCEIVED_EXCESS_MIN = 5.0

MODES = ["fast", "calm", "min_transfer", "balanced"]

# 모드별 scoring 가중치. 최종 순위는 이 점수로 정한다.
MODE_WEIGHTS = {
    "fast":         {"actual": 1.0, "perceived": 0.0, "max_cong": 0.00, "tpen": 0.3, "seat": 0.0},
    "calm":         {"actual": 0.0, "perceived": 1.0, "max_cong": 0.05, "tpen": 0.5, "seat": 0.0},
    "min_transfer": {"actual": 0.5, "perceived": 0.5, "max_cong": 0.00, "tpen": 3.0, "seat": 0.0},
    "balanced":     {"actual": 0.3, "perceived": 0.7, "max_cong": 0.03, "tpen": 0.8, "seat": 0.5},
}

TEST_ODS = [
    ("2_신촌", "2_잠실"), ("1_서울역", "2_강남"), ("5_군자", "5_여의나루"),
    ("2_한양대", "3_고속터미널"), ("4_혜화", "2_사당"), ("1_종각", "6_이태원"),
    ("2_건대입구", "2_홍대입구"), ("6_안암", "2_삼성"), ("5_방화", "5_마천"),
    ("6_응암", "3_고속터미널"),
    # smoke test 성격
    ("2_신설동", "2_까치산"),      # 성수지선 -> 본선 -> 신정지선
    ("6_연신내", "6_응암"),        # 응암순환 단방향
    ("6_응암", "6_연신내"),        # 반대 방향 (경로가 달라야 정상)
]

VALID_LINES = {"1", "2", "3", "4", "5", "6", "7", "8"}


def station_of(node: str) -> str:
    return node.split("_", 1)[1].split("@")[0]


def line_of(node: str) -> str:
    return node.split("_", 1)[0]


def load_scorer_module(root: Path):
    """09 프로토타입을 재사용한다. 그래프/혼잡도/이벤트 결합 로직을 중복 구현하지 않는다."""
    cands = [root / "scripts" / "09_route_scoring_prototype.py",
             Path(__file__).with_name("09_route_scoring_prototype.py")]
    for cand in cands:
        if cand.exists():
            spec = importlib.util.spec_from_file_location("rs09", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("09_route_scoring_prototype.py 를 찾을 수 없습니다.")


class RouteEvaluator:

    def __init__(self, root: Path, day_type: str, depart: str, query_date):
        self.root = root
        self.mart = root / "data" / "marts"
        self.report = root / "reports" / "route"
        self.report.mkdir(parents=True, exist_ok=True)
        self.warnings = []
        self.notes = []

        mod = load_scorer_module(root)
        self.rs = mod.RouteScorer(root, day_type, depart, query_date)
        self.notes.extend(self.rs.notes)
        self.depart = depart
        self.query_date = query_date
        self.day_type = self.rs.day_type
        self.time_bin = self.rs.time_bin

        self.edge_meta = self._build_edge_meta()
        self._graph_sanity_checks()
        self.p95_cong = float(self.rs.lookup["congestion_median"].quantile(0.95))

    # ---------- 엣지 메타 ----------
    def _build_edge_meta(self):
        meta = {}
        for r in self.rs.ride.itertuples():
            meta[(r.from_node, r.to_node)] = {
                "kind": "ride", "line_id": str(r.line_id),
                "direction": r.direction if pd.notna(r.direction) else "unknown",
                "travel_time": float(r.travel_time_min),
                "congestion": float(r.congestion),
                "perceived": float(r.perceived_time_min),
                "event_risk": float(getattr(r, "event_risk_min", 0.0) or 0.0),
                "transfer_penalty": 0.0,
            }
        for r in self.rs.transfer.itertuples():
            for a, b in ((r.from_node, r.to_node), (r.to_node, r.from_node)):
                meta[(a, b)] = {
                    "kind": "transfer", "line_id": str(r.from_line) + "->" + str(r.to_line),
                    "direction": "transfer",
                    "travel_time": float(r.transfer_time_min),
                    "congestion": np.nan,
                    "perceived": float(r.transfer_penalty_min),
                    "event_risk": 0.0,
                    "transfer_penalty": float(r.transfer_penalty_min),
                }
        return meta

    # ---------- 그래프 방향 점검 ----------
    def _graph_sanity_checks(self):
        ride = self.rs.ride
        d2 = set(ride[ride.line_id.astype(str) == "2"]["direction"].dropna())
        if not {"inner", "outer"}.issubset(d2):
            self.warnings.append("2호선 방향 라벨 이상: " + str(sorted(d2)))
        loop = [("6_응암", "6_역촌"), ("6_역촌", "6_불광"), ("6_불광", "6_독바위"),
                ("6_독바위", "6_연신내"), ("6_연신내", "6_구산"), ("6_구산", "6_응암")]
        missing = [e for e in loop if e not in self.edge_meta]
        if missing:
            self.warnings.append("응암순환 정방향 엣지 누락: " + str(missing))
        rev = [(b, a) for a, b in loop if (b, a) in self.edge_meta]
        if rev:
            self.warnings.append("응암순환 역방향 엣지 발견(단방향이어야 함): " + str(rev))
        if ("5_강동@macheon_branch", "5_둔촌동") not in self.edge_meta:
            self.warnings.append("5호선 마천지선 분기 엣지 없음")
        n_nodir = int(ride["direction"].isna().sum())
        if n_nodir:
            self.warnings.append("방향 판정 실패 엣지 %d개" % n_nodir)

    # ---------- 후보 생성 ----------
    def candidates(self, src, dst, k=5):
        a = self.rs._yen(src, dst, K=k, weight="time")
        b = self.rs._yen(src, dst, K=k, weight="cost")
        seen, out = set(), []
        for p in a + b:
            t = tuple(p)
            if t not in seen:
                seen.add(t)
                out.append(p)
        return out

    # ---------- 경로 지표 ----------
    def path_edges(self, path):
        """directed edge 키. (from, to, line_id, direction) - 방향이 다르면 다른 엣지."""
        out = []
        for u, v in zip(path, path[1:]):
            m = self.edge_meta.get((u, v))
            if m is None:
                out.append((u, v, "?", "?"))
            else:
                out.append((u, v, m["line_id"], m["direction"]))
        return out

    def evaluate_path(self, path):
        """경로 평가. 비용 계산은 RouteScorer.evaluate() 하나만 쓴다.

        과거에는 이 메서드가 edge_meta 로 같은 계산을 따로 했다. 그 결과
        09 쪽에 중간역 정차시간과 강동 직결 분기 통과를 반영했을 때
        리포트만 옛 값을 유지해, 앱과 리포트가 갈라진 적이 있다
        (방화->마천 앱 0회 88.7분 vs 마트 1회 69.25분).
        계산을 한 곳에 두어 같은 종류의 불일치를 구조적으로 막는다.

        여기서는 10 스크립트에만 필요한 필드를 덧붙이는 일만 한다.
        """
        ev = self.rs.evaluate(path)

        violation = []
        for u, v in zip(path, path[1:]):
            if self.edge_meta.get((u, v)) is None:
                violation.append("missing_edge:%s->%s" % (u, v))
        for n in path:
            if line_of(n) not in VALID_LINES:
                violation.append("out_of_scope_line:" + n)

        # 환승역 목록. evaluate() 와 같은 규칙(직결 분기 통과는 환승 아님)을 쓴다.
        g = getattr(type(self.rs).evaluate, "__globals__", {})
        is_through = g.get("is_through_pass")
        tstations = []
        for i, (u, v) in enumerate(zip(path, path[1:])):
            m = self.edge_meta.get((u, v))
            if m is None or m["kind"] == "ride":
                continue
            if is_through and is_through(path, i, u, v):
                continue
            tstations.append(station_of(u))

        out = {
            "path_nodes": " -> ".join(path),
            "path_edges": len(path) - 1,
            "actual_time_min": ev["actual_time_min"],
            "perceived_time_min": ev["perceived_time_min"],
            "running_time_min": ev["running_time_min"],
            "dwell_time_min": ev["dwell_time_min"],
            "avg_congestion": ev["avg_congestion"],
            "max_congestion": ev["max_congestion"],
            "transfer_count": ev["transfer_count"],
            "transfer_stations": ";".join(tstations),
            "transfer_penalty_min": ev["transfer_penalty_min"],
            "event_risk_min": ev["event_risk_min"],
            "seat_chance_score": ev["seat_chance_score"],
            "high_congestion_exposure_count": ev["p95_exposure_count"],
            "constraint_violation": ";".join(sorted(set(violation))),
            "_edges": set(self.path_edges(path)),
        }
        assert out["transfer_count"] == len(tstations), (
            "환승 횟수(%d)와 환승역 수(%d)가 다르다: %s"
            % (out["transfer_count"], len(tstations), out["path_nodes"]))
        return out

    # ---------- 3축 Pareto ----------
    @staticmethod
    def pareto_flags(df, cols=("actual_time_min", "max_congestion", "transfer_penalty_min")):
        X = df[list(cols)].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=np.inf)
        n = len(X)
        flags = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(X[j] <= X[i]) and np.any(X[j] < X[i]):
                    flags[i] = False
                    break
        return flags

    @staticmethod
    def jaccard(a, b):
        if not a and not b:
            return 1.0
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    # ---------- 요청 단위 평가 ----------
    def evaluate_request(self, src, dst, mode):
        req_id = "%s->%s|%s" % (station_of(src), station_of(dst), mode)
        dep = ("%s %s" % (self.query_date or "", self.depart)).strip()
        paths = self.candidates(src, dst)
        if not paths:
            return pd.DataFrame(), {
                "request_id": req_id, "origin": src, "destination": dst,
                "departure_datetime": dep, "preference_mode": mode,
                "n_candidate_routes": 0, "n_pareto_routes_3d": 0, "pareto_front_size": 0,
                "best_stretch_factor": np.nan, "best_max_congestion_reduction": np.nan,
                "best_perceived_time_gain": np.nan, "avg_jaccard_similarity_top3": np.nan,
                "n_near_duplicate_removed": 0, "has_meaningful_alternative": False,
                "selected_route_id": "", "constraint_violation_count": 0,
                "selection_reason": "경로 없음",
            }

        rows = [self.evaluate_path(p) for p in paths]
        df = pd.DataFrame(rows)
        edge_sets = df.pop("_edges").tolist()
        df["route_id"] = ["%s#%d" % (req_id, i) for i in range(len(df))]

        fastest_i = int(df["actual_time_min"].idxmin())
        f_time = float(df.loc[fastest_i, "actual_time_min"])
        f_cong = float(df.loc[fastest_i, "max_congestion"])
        f_avg = float(df.loc[fastest_i, "avg_congestion"])
        f_perc = float(df.loc[fastest_i, "perceived_time_min"])

        df["stretch_factor"] = (df["actual_time_min"] / f_time).round(4)
        df["time_loss_vs_fastest"] = (df["actual_time_min"] - f_time).round(2)
        df["max_congestion_reduction_vs_fastest"] = (f_cong - df["max_congestion"]).round(2)
        df["avg_congestion_reduction_vs_fastest"] = (f_avg - df["avg_congestion"]).round(2)
        df["perceived_time_gain_vs_fastest"] = (f_perc - df["perceived_time_min"]).round(2)

        df["is_pareto_3d"] = self.pareto_flags(df)

        w = MODE_WEIGHTS[mode]
        df["route_score"] = (w["actual"] * df["actual_time_min"]
                             + w["perceived"] * df["perceived_time_min"]
                             + w["max_cong"] * df["max_congestion"].fillna(0)
                             + w["tpen"] * df["transfer_penalty_min"]
                             - w["seat"] * df["seat_chance_score"]).round(3)
        order = df["route_score"].to_numpy().argsort(kind="stable")
        df = df.iloc[order].reset_index(drop=True)
        edge_sets = [edge_sets[i] for i in order]
        df["route_rank"] = np.arange(1, len(df) + 1)

        base = edge_sets[0]
        df["jaccard_similarity_to_rank1"] = [round(self.jaccard(base, e), 4) for e in edge_sets]

        # near-duplicate 제거 순서는 Pareto optimal 을 먼저 본다.
        # route_score 순으로만 처리하면, 혼잡이 오히려 늘어나는 dominated 경로가
        # 먼저 자리를 잡고 Pareto 최적해를 중복으로 몰아낼 수 있다.
        # (실제로 방화->마천 18:00 에서 혼잡 -27.8%p 인 Pareto 해가 잘려나갔다)
        dedup_order = sorted(range(len(edge_sets)),
                             key=lambda i: (0 if df.loc[i, "is_pareto_3d"] else 1, i))
        kept = []
        dup = [False] * len(edge_sets)
        for i in dedup_order:
            e = edge_sets[i]
            if any(self.jaccard(e, k) > MAX_PATH_SIMILARITY for k in kept):
                dup[i] = True
            else:
                kept.append(e)
        df["is_near_duplicate"] = dup

        alt_mask = ((df["route_rank"] > 1)
                    & (~df["is_near_duplicate"])
                    & (df["time_loss_vs_fastest"] <= MAX_TIME_LOSS_MIN)
                    & (df["max_congestion_reduction_vs_fastest"] >= MIN_CONGESTION_DROP_PP)
                    & (df["perceived_time_min"] <= f_perc + MAX_PERCEIVED_EXCESS_MIN)
                    & (df["constraint_violation"] == ""))
        has_alt = bool(alt_mask.any())
        df["recommendation_policy"] = np.where(
            df["route_rank"] == 1, "primary",
            np.where(alt_mask, "meaningful_alternative",
                     np.where(df["is_near_duplicate"], "near_duplicate", "rejected")))
        if not has_alt:
            df.loc[df["route_rank"] > 1, "recommendation_policy"] = "no_meaningful_alternative"

        df.insert(0, "request_id", req_id)
        df.insert(1, "preference_mode", mode)

        # 임계 민감도: 어느 조건이 대안을 막고 있는지 분해한다.
        cond = {
            "not_near_dup": ~df["is_near_duplicate"],
            "time_loss": df["time_loss_vs_fastest"] <= MAX_TIME_LOSS_MIN,
            "cong_drop": df["max_congestion_reduction_vs_fastest"] >= MIN_CONGESTION_DROP_PP,
            "perceived": df["perceived_time_min"] <= f_perc + MAX_PERCEIVED_EXCESS_MIN,
        }
        rank_gt1 = df["route_rank"] > 1
        blockers = []
        for name, m in cond.items():
            others = rank_gt1.copy()
            for n2, m2 in cond.items():
                if n2 != name:
                    others &= m2
            # 이 조건 하나만 풀면 통과하는 후보 수
            n_unlock = int((others & ~m).sum())
            if n_unlock:
                blockers.append("%s(%d)" % (name, n_unlock))

        sel = df.iloc[0]
        pair_j = [self.jaccard(edge_sets[i], edge_sets[j])
                  for i in range(min(3, len(edge_sets)))
                  for j in range(i + 1, min(3, len(edge_sets)))]
        summary = {
            "request_id": req_id, "origin": src, "destination": dst,
            "departure_datetime": dep, "preference_mode": mode,
            "n_candidate_routes": len(df),
            "n_pareto_routes_3d": int(df["is_pareto_3d"].sum()),
            "pareto_front_size": int(df["is_pareto_3d"].sum()),
            "best_stretch_factor": (float(df.loc[alt_mask, "stretch_factor"].min())
                                    if has_alt else np.nan),
            "best_max_congestion_reduction": float(df["max_congestion_reduction_vs_fastest"].max()),
            "best_perceived_time_gain": float(df["perceived_time_gain_vs_fastest"].max()),
            "avg_jaccard_similarity_top3": (round(float(np.mean(pair_j)), 4) if pair_j else np.nan),
            "n_near_duplicate_removed": int(df["is_near_duplicate"].sum()),
            "has_meaningful_alternative": has_alt,
            "selected_route_id": sel["route_id"],
            "constraint_violation_count": int((df["constraint_violation"] != "").sum()),
            "blocking_conditions": ";".join(blockers),
            "selection_reason": ("%s 모드 최저 route_score. 체감 %.1f분, 최대혼잡 %.1f%%, 환승 %d회"
                                 % (mode, sel["perceived_time_min"],
                                    sel["max_congestion"], sel["transfer_count"])),
        }
        return df.drop(columns=["route_score"]), summary

    def run(self, ods, modes):
        cases, summaries = [], []
        for src, dst in ods:
            for mode in modes:
                df, s = self.evaluate_request(src, dst, mode)
                if len(df):
                    cases.append(df)
                summaries.append(s)
        return (pd.concat(cases, ignore_index=True) if cases else pd.DataFrame(),
                pd.DataFrame(summaries))


def make_scatter(cases, out_png, subtitle):
    """산점도를 그린다. matplotlib 이 없으면 조용히 건너뛴다.

    그림은 보조 산출물이다. 이것 때문에 마트·CSV·리포트가 날아가면 안 된다.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    d = cases[cases["stretch_factor"].notna()
              & cases["max_congestion_reduction_vs_fastest"].notna()].copy()
    if d.empty:
        return False
    # 짧은 OD 에서 Yen 이 만든 극단 우회(stretch 10배 이상)가 x축을 늘려
    # 정작 중요한 1.0~2.0 구간이 뭉개진다. 축을 자르고 제외 개수를 명시한다.
    xlim_hi = 2.2
    n_clip = int((d["stretch_factor"] > xlim_hi).sum())
    d = d[d["stretch_factor"] <= xlim_hi]
    if d.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 6.5))
    size = 25 + d["transfer_penalty_min"] * 9
    sc = ax.scatter(d["stretch_factor"], d["max_congestion_reduction_vs_fastest"],
                    c=d["transfer_count"], s=size, cmap="viridis",
                    alpha=0.6, edgecolors="none")
    par = d[d["is_pareto_3d"]]
    ax.scatter(par["stretch_factor"], par["max_congestion_reduction_vs_fastest"],
               s=25 + par["transfer_penalty_min"] * 9, facecolors="none",
               edgecolors="crimson", linewidths=1.6, label="3-axis Pareto optimal")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(1.0, color="grey", lw=0.8, ls="--")
    ax.axhline(MIN_CONGESTION_DROP_PP, color="steelblue", lw=1.0, ls=":",
               label="alternative threshold (+%.0f%%p)" % MIN_CONGESTION_DROP_PP)
    ax.set_xlabel("Stretch Factor  (candidate time / fastest time)")
    ax.set_ylabel("Max congestion reduction vs fastest (%p)")
    ax.set_title("Yeoyuro Seoul route trade-off: time cost vs congestion gain\n" + subtitle,
                 fontsize=11)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("transfer count")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0.95, xlim_hi)
    ax.annotate("better\n(less time, less crowding)", xy=(1.02, ax.get_ylim()[1] * 0.82),
                fontsize=9, color="seagreen", weight="bold")
    note = "marker size = transfer penalty (min)"
    if n_clip:
        note += "  |  %d routes with stretch > %.1f omitted" % (n_clip, xlim_hi)
    ax.text(0.01, 0.02, note, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color="dimgrey")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def write_report(ev, cases, summary, png, paths):
    L = []
    A = L.append
    A("# Route Recommendation Evaluation\n")

    A("## 1. 목적\n")
    A("추천 알고리즘이 최단경로와 다른 의미 있는 대안 경로를 만들어내는지 검증한다.")
    A("경로 추천에는 정답 경로가 없으므로 Accuracy 계열 지표를 쓸 수 없다."
      " 다목적 최적화의 표준 도구인 Pareto optimality 와 Stretch Factor 를 쓴다.\n")

    A("## 2. 평가 데이터\n")
    A("- OD쌍 %d개 x 선호 모드 %d종 = 요청 %d건"
      % (summary["origin"].nunique(), summary["preference_mode"].nunique(), len(summary)))
    A("- 후보 경로: **%d**개" % len(cases))
    A("- 출발 조건: %s / %s / %s (time_bin %s)"
      % (ev.query_date or "(날짜 미지정)", ev.day_type, ev.depart, ev.time_bin))
    A("- 혼잡도 기준: congestion_edge_lookup 의 스냅샷 **중앙값**"
      " (예측 모델이 아니라 관측 패턴. 08 판정에 따라 baseline 을 사용한다)")
    A("- 그래프: route_edge_mart %d 엣지 / transfer_edge_mart %d 엣지\n"
      % (len(ev.rs.ride), len(ev.rs.transfer)))

    A("## 3. 평가 지표\n")
    A("| 지표 | 정의 | 역할 |")
    A("|---|---|---|")
    A("| 3-axis Pareto | (actual_time, max_congestion, transfer_penalty) 비지배 | 후보 보존·열등 판정 |")
    A("| Stretch Factor | candidate_time / fastest_time | OD 간 비교 가능한 시간 손실 |")
    A("| Max congestion reduction | fastest_max - candidate_max (%p) | 혼잡 이득 |")
    A("| Perceived time gain | fastest_perceived - candidate_perceived (분) | 체감 이득 |")
    A("| Directed-edge Jaccard | (from,to,line,direction) 집합 유사도 | 경로 다양성 |")
    A("| Meaningful alternative rate | 채택 조건 통과 비율 | 서비스 가치 |")
    A("")
    A("> Pareto 는 **후보 보존 장치**이지 최종 순위가 아니다."
      " 최종 순위는 perceived_time_min 기반 모드별 scoring 으로 정한다.\n")

    A("## 4. 주요 결과\n")
    A("### 4-1. 모드별 요약\n")
    g = (summary.groupby("preference_mode")
         .agg(요청=("request_id", "size"),
              대안있음=("has_meaningful_alternative", "sum"),
              평균후보수=("n_candidate_routes", "mean"),
              평균파레토수=("n_pareto_routes_3d", "mean"),
              평균Jaccard_top3=("avg_jaccard_similarity_top3", "mean"),
              near_dup=("n_near_duplicate_removed", "sum"))
         .round(3).reset_index())
    A(g.to_markdown(index=False))
    A("")

    A("### 4-2. OD쌍별 결과 (calm 모드)\n")
    c = summary[summary["preference_mode"] == "calm"].copy()
    c["OD"] = c["origin"].map(station_of) + "->" + c["destination"].map(station_of)
    A(c[["OD", "n_candidate_routes", "n_pareto_routes_3d",
         "best_max_congestion_reduction", "best_stretch_factor",
         "avg_jaccard_similarity_top3", "has_meaningful_alternative"]].to_markdown(index=False))
    A("")
    yes = c[c["has_meaningful_alternative"]]
    no = c[~c["has_meaningful_alternative"]]
    A("- 유의미한 대안이 나온 OD쌍: **%d/%d**" % (len(yes), len(c)))
    if len(yes):
        A("  - " + ", ".join(yes["OD"].tolist()))
    A("- 대안이 없는 OD쌍: **%d/%d**" % (len(no), len(c)))
    if len(no):
        A("  - " + ", ".join(no["OD"].tolist()))
    A("")
    A("> front_size = 1 인 OD 는 trade-off 자체가 존재하지 않는다는 뜻이다."
      " 임의 임계값이 아니라 **수학적으로** 대안이 없음을 보인다.\n")

    A("### 4-3. 탈락 사유별 후보 수\n")
    A(cases["recommendation_policy"].value_counts().to_frame("routes").to_markdown())
    A("")

    rej = cases[cases["recommendation_policy"].isin(["rejected", "no_meaningful_alternative"])]
    if len(rej):
        r = (rej[rej["max_congestion_reduction_vs_fastest"] > 0]
             .nlargest(5, "max_congestion_reduction_vs_fastest")
             [["request_id", "stretch_factor", "time_loss_vs_fastest",
               "max_congestion_reduction_vs_fastest", "transfer_count",
               "transfer_penalty_min"]])
        A("**혼잡 감소는 있으나 조건 미달로 탈락한 사례 (상위 5)**\n")
        A(r.to_markdown(index=False) if len(r) else "해당 없음")
        A("")
        r2 = rej.nlargest(5, "transfer_penalty_min")[
            ["request_id", "transfer_count", "transfer_penalty_min",
             "transfer_stations", "stretch_factor"]]
        A("**환승 부담이 큰 탈락 사례 (transfer_penalty 상위 5)**\n")
        A(r2.to_markdown(index=False))
        A("")

    A("### 4-4. 임계 민감도 — 어느 조건이 대안을 막았나\n")
    bl = summary["blocking_conditions"].fillna("")
    from collections import Counter
    cnt = Counter()
    for row in bl:
        for tok in [t for t in row.split(";") if t]:
            cnt[tok.split("(")[0]] += 1
    if cnt:
        A("| 조건 | 이 조건만 풀면 대안이 생기는 요청 수 |")
        A("|---|---|")
        for k, v in cnt.most_common():
            A("| %s | %d |" % (k, v))
        A("")
        A("> 임계값(15분 / 15%p / 5분)은 근거 있는 상수가 아니다."
          " 어느 조건이 병목인지 밝혀두면 나중에 사용자 피드백으로 튜닝할 수 있다.\n")
    else:
        A("모든 요청에서 단일 조건 완화만으로는 대안이 생기지 않는다.\n")

    A("### 4-5. 모드 간 추천 경로가 달라지는가\n")
    piv = summary.pivot_table(index=["origin", "destination"], columns="preference_mode",
                              values="selected_route_id", aggfunc="first")
    diff = piv.apply(lambda r: r.nunique(), axis=1)
    A("- 모드에 따라 선택 경로가 달라진 OD쌍: **%d/%d**" % (int((diff > 1).sum()), len(diff)))
    A("- 달라지지 않은 OD쌍은 해당 구간에 대안 자체가 없다는 뜻이다."
      " 서울 지하철 구조상 직통 노선 하나뿐인 구간이 존재한다.\n")

    A("## 5. 대표 시각화\n")
    if png.exists():
        A("![pareto](%s)\n" % png.name)
    else:
        A("(산점도 미생성 — matplotlib 설치 후 재실행하면 생성된다)\n")
    A("- x축 Stretch Factor, y축 최대혼잡 감소(%p)")
    A("- 색상 transfer_count, 크기 transfer_penalty_min")
    A("- 붉은 테두리 = 3축 Pareto optimal")
    A("- 좌상단으로 갈수록 좋다(시간 손해 적고 혼잡 이득 큼)\n")

    A("## 6. 그래프 경고\n")
    if ev.warnings:
        for w in ev.warnings:
            A("- [warning] " + w)
    else:
        A("2호선 내선/외선, 6호선 응암순환 단방향, 5호선 마천지선 분기 모두 정상. 경고 없음.")
    A("")

    A("## 7. 한계\n")
    A("- 실시간 혼잡도가 아니라 **과거 스냅샷 기반 기대 혼잡도**다.")
    A("- 객차별 혼잡도가 아니다. 호차/문 안내는 환승 동선 기준이다.")
    A("- 착석 확률이 아니라 **착석 가능성 proxy** 점수다.")
    A("- 환승 호차/문은 원본 데이터에 있는 조합만 안내한다.")
    A("- 출발 시각의 time_bin 을 경로 전체에 고정 적용한다(시간 전진 미반영).")
    A("- 대안 채택 임계값(15분 / 15%p / 5분)은 근거 있는 상수가 아니라 **튜닝 대상 초기값**이다.")
    A("- 대안이 없는 OD쌍은 억지로 만들지 않고 `no_meaningful_alternative` 를 반환한다.\n")

    A("## 8. 면접용 요약\n")
    A("> 경로 추천은 단순 최단시간 문제가 아니라 시간, 최대 혼잡도, 환승 피로도를 함께"
      " 고려하는 다목적 최적화 문제로 정의했습니다. 수학적으로는 실제시간·최대혼잡·환승페널티"
      " 3축 Pareto optimality 로 후보 경로의 열등 여부를 판단하고, 포트폴리오 시각화에서는"
      " Stretch Factor 와 최대혼잡 감소폭의 2축 그래프에 환승 부담을 색상과 크기로"
      " 인코딩했습니다.\n")

    if ev.notes:
        A("## 9. 처리 노트\n")
        for n in ev.notes:
            A("- " + str(n))
        A("")

    A("## 10. 산출물\n")
    for k, v in paths.items():
        A("- `%s` : %s" % (k, v))

    p = ev.report / ("route_recommendation_eval_%s.md" % ev.tag)
    p.write_text("\n".join(L), encoding=ENC)
    return p


def save(df, base, name):
    try:
        p = base / (name + ".parquet")
        df.to_parquet(p, index=False)
    except Exception:
        p = base / (name + ".csv.gz")
        df.to_csv(p, index=False, encoding=ENC, compression="gzip")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--day-type", default="weekday")
    ap.add_argument("--time", default="08:30")
    ap.add_argument("--date", default=None)
    ap.add_argument("--modes", default=",".join(MODES))
    args = ap.parse_args(argv)

    root = Path(args.root)
    ev = RouteEvaluator(root, args.day_type, args.time, args.date)
    ev.tag = (args.date.replace("-", "") + "_" if args.date else "") + args.time.replace(":", "")
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODE_WEIGHTS]
    cases, summary = ev.run(TEST_ODS, modes)

    mart = root / "data" / "marts"
    mart.mkdir(parents=True, exist_ok=True)
    # 시각·날짜별로 파일을 분리한다. 같은 이름을 쓰면 08:30 리포트가 18:00 실행에 덮어써진다.
    tag = args.time.replace(":", "")
    if args.date:
        tag = args.date.replace("-", "") + "_" + tag
    paths = {
        "route_evaluation_mart": save(cases, mart, "route_evaluation_mart_" + tag),
        "route_evaluation_summary": save(summary, mart, "route_evaluation_summary_" + tag),
    }
    p1 = ev.report / ("route_eval_cases_%s.csv" % tag)
    p2 = ev.report / ("route_eval_summary_%s.csv" % tag)
    cases.to_csv(p1, index=False, encoding=ENC)
    summary.to_csv(p2, index=False, encoding=ENC)
    paths["route_eval_cases"] = p1
    paths["route_eval_summary"] = p2

    png = ev.report / ("pareto_front_scatter_%s.png" % tag)
    subtitle = ("%s %s %s | %d requests"
                % (args.date or "", ev.day_type, args.time, len(summary))).strip()
    drawn = make_scatter(cases, png, subtitle)
    if drawn:
        paths["pareto_front_scatter"] = png
    else:
        ev.notes.append("matplotlib 미설치 또는 그릴 데이터 없음 → 산점도 생략"
                        " (pip install matplotlib 후 재실행하면 생성된다)")
    paths["report"] = write_report(ev, cases, summary, png, paths)

    print("=" * 92)
    print(" 10_evaluate_routes - 완료")
    print("=" * 92)
    for n in ev.notes:
        print("  " + str(n))
    if ev.warnings:
        for w in ev.warnings:
            print("  [warning] " + w)
    else:
        print("  그래프 경고 없음 (2호선 내외선 / 응암순환 단방향 / 마천지선 정상)")
    print("\n  요청 %d건 / 후보 경로 %d개" % (len(summary), len(cases)))
    print("  제약 위반 경로: %d" % int((cases["constraint_violation"] != "").sum()))

    print("\n[모드별 요약]")
    g = (summary.groupby("preference_mode")
         .agg(요청=("request_id", "size"),
              대안있음=("has_meaningful_alternative", "sum"),
              평균후보=("n_candidate_routes", "mean"),
              평균파레토=("n_pareto_routes_3d", "mean"),
              Jaccard_top3=("avg_jaccard_similarity_top3", "mean"))
         .round(3))
    print(g.to_string())

    print("\n[OD쌍별 (calm 모드)]")
    c = summary[summary["preference_mode"] == "calm"].copy()
    c["OD"] = c["origin"].map(station_of) + "->" + c["destination"].map(station_of)
    print(c[["OD", "n_candidate_routes", "n_pareto_routes_3d",
             "best_max_congestion_reduction", "avg_jaccard_similarity_top3",
             "has_meaningful_alternative"]].to_string(index=False))

    print("\n[생성 파일]")
    for k, v in paths.items():
        print("  %-26s %s" % (k, v))
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
