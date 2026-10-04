"""
scripts/v2/21_build_route_profile_mart.py
=========================================
v2.1 — route_profile_mart 생성 + 혼잡 노출 지표 검증.

route_profile_mart : 1행 = (OD, 요일유형, 출발시각, 후보경로). 시간 진행형(step) 평가 결과.

이 스크립트가 답하는 질문 (docs/v2/exposure_metrics.md 근거)
------------------------------------------------------------
Q1. exposure_100 이 max_congestion 과 다른 정보를 주는가?
    -> 같은 OD 후보 중 '최대혼잡 최소 경로'와 'exposure_100 최소 경로'가 다른 비율
Q2. 130% 노출은 얼마나 희소한가?  -> exposure_130 > 0 인 행 비율
Q3. 중앙값 대신 p90 을 쓰면 얼마나 달라지는가? -> exposure_100 vs exposure_100_p90
Q4. 계산이 내부적으로 일관적인가? -> 불변식 검사 (아래 INVARIANTS)

OD 표본: Tier C 20쌍 + 무작위 N쌍(seed 고정). 출발: 평일 07:00~09:45, 17:00~19:45 15분 간격.

실행
----
    python scripts/v2/21_build_route_profile_mart.py                 # 기본 (약 10~20분)
    python scripts/v2/21_build_route_profile_mart.py --n-random 20   # 빠른 점검

산출물
------
    data/marts/v2/route_profile_mart.parquet
    reports/v2/exposure_metrics_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import load_v1                                   # noqa: E402
from yeoyuro_v2.time_dependent import (TDRouter, parse_hhmm, fmt_min,  # noqa: E402
                                       EXPOSURE_THRESHOLDS, bin_index_of)

TIER_C = [
    ("신촌", "잠실"), ("서울역", "강남"), ("군자", "여의나루"), ("한양대", "고속터미널"),
    ("혜화", "사당"), ("종각", "이태원"), ("건대입구", "홍대입구"), ("안암", "삼성"),
    ("방화", "마천"), ("응암", "고속터미널"), ("노원", "사당"), ("장암", "온수"),
    ("불암산", "남태령"), ("하남검단산", "방화"), ("암사역사공원", "모란"), ("지축", "오금"),
    ("신내", "응암"), ("잠실", "홍대입구"), ("까치산", "성수"), ("천호", "공덕"),
]
DEPARTS = [fmt_min(m) for m in list(range(7 * 60, 10 * 60, 15)) + list(range(17 * 60, 20 * 60, 15))]
EPS = 1e-6


def route_id(path: list[str]) -> str:
    return hashlib.md5(">".join(path).encode()).hexdigest()[:10]


def line_sequence(path: list[str]) -> str:
    seq = []
    for n in path:
        ln = n.split("_", 1)[0]
        if not seq or seq[-1] != ln:
            seq.append(ln)
    return ">".join(seq)


def check_invariants(r) -> list[str]:
    """한 경로 결과의 불변식. 위반 항목 이름 리스트를 돌려준다."""
    bad = []
    e = r.exposure
    if not (e[80] + EPS >= e[100] >= e[130] - EPS):
        bad.append("exposure_monotone")
    if e[80] > r.in_vehicle_min + EPS:
        bad.append("exposure_le_in_vehicle")
    if r.exposure_p90[100] + EPS < 0:
        bad.append("p90_negative")
    total = r.in_vehicle_min + r.transfer_walk_min + r.transfer_wait_min + r.initial_wait_min
    if abs(total - r.actual_time_min) > 1e-4:
        bad.append("time_decomposition")
    if r.max_congestion < 100 and e[100] > EPS:
        bad.append("max_lt100_but_exposure100")
    if r.max_congestion >= 100 and e[100] <= EPS:
        bad.append("max_ge100_but_no_exposure100")
    starts = [t.t_start for t in r.trace]
    if starts != sorted(starts):
        bad.append("trace_not_monotone")
    if r.trace and abs(r.trace[-1].t_end - r.arrive_min) > 1e-4:
        bad.append("trace_end_ne_arrival")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--day-type", default="weekday")
    args = ap.parse_args(argv)

    t0 = time.time()
    mod, sr, disp = load_v1(ROOT)
    keys = disp["station_key"].tolist()
    tier_c = [p for p in TIER_C if p[0] in keys and p[1] in keys]
    rnd = random.Random(args.seed)
    rand_ods = set()
    while len(rand_ods) < args.n_random:
        a, b = rnd.sample(keys, 2)
        rand_ods.add((a, b))
    ods = [(o, d, "tier_c") for o, d in tier_c] + [(o, d, "random") for o, d in sorted(rand_ods)]

    router = TDRouter(ROOT, mod, sr, disp, args.day_type, policy="step", k=5, multi_bin=True)
    rows, viol = [], []
    total = len(ods) * len(DEPARTS)
    done = 0
    for o, d, tier in ods:
        for dep in DEPARTS:
            done += 1
            tdep = parse_hhmm(dep)
            res = router.evaluate_od(o, d, tdep)
            if not res:
                continue
            # trace 가 필요하므로 같은 evaluator 로 trace 포함 재평가 (결과값은 동일)
            ev = router.evaluator(bin_index_of(tdep))
            res = [ev.evaluate(r.path, tdep, keep_trace=True) for r in res]
            for rank, r in enumerate(res, start=1):
                b = check_invariants(r)
                if b:
                    viol.append({"origin": o, "destination": d, "depart": dep, "rank": rank,
                                 "violations": ";".join(b)})
                s = r.summary()
                rows.append({
                    "tier": tier, "origin": o, "destination": d, "day_type": args.day_type,
                    "departure_time": dep, "route_id": route_id(r.path), "route_rank": rank,
                    "is_fastest": rank == 1,
                    "line_sequence": line_sequence(r.path),
                    "station_sequence": ">".join(r.path),
                    "n_ride_edges": sum(1 for t in r.trace if t.kind == "ride"),
                    "actual_time_min": s["actual_time_min"],
                    "perceived_time_min": s["perceived_time_min"],
                    "in_vehicle_min": s["in_vehicle_min"],
                    "transfer_count": r.transfer_count,
                    "transfer_wait_min": round(r.transfer_wait_min, 2),
                    "transfer_walk_min": round(r.transfer_walk_min, 2),
                    "max_expected_congestion": s["max_congestion"],
                    "avg_expected_congestion": s["avg_congestion_time"],
                    **{f"exposure_{x}_min": s[f"exposure_{x}_min"] for x in EXPOSURE_THRESHOLDS},
                    **{f"exposure_{x}_min_p90": s[f"exposure_{x}_min_p90"] for x in EXPOSURE_THRESHOLDS},
                    "n_bins_crossed": s["n_bins_crossed"],
                    "arrival_time": s["arrive"],
                    "bin_policy": r.policy,
                    "out_of_window_flag": r.out_of_window,
                })
            if done % 200 == 0:
                print(f"  ... {done}/{total}  ({time.time() - t0:.0f}s)")

    mart = pd.DataFrame(rows)
    mdir = ROOT / "data" / "marts" / "v2"
    mdir.mkdir(parents=True, exist_ok=True)
    mart.to_parquet(mdir / "route_profile_mart.parquet", index=False)
    vdf = pd.DataFrame(viol)

    # ------------------------------------------------------------------ 분석
    key = ["origin", "destination", "departure_time"]
    fast = mart[mart.is_fastest]
    # Q1: 최속 +15분 이내 후보 중, max 최소 경로 vs exposure_100 최소 경로
    m = mart.merge(fast[key + ["actual_time_min"]].rename(columns={"actual_time_min": "fast_t"}), on=key)
    m = m[m.actual_time_min - m.fast_t <= 15]
    multi = m.groupby(key).filter(lambda g: len(g) >= 2)
    q1 = []
    for _, g in multi.groupby(key):
        best_max = g.sort_values(["max_expected_congestion", "perceived_time_min"]).iloc[0].route_id
        best_exp = g.sort_values(["exposure_100_min", "perceived_time_min"]).iloc[0].route_id
        has_exp = g.exposure_100_min.max() > 0
        q1.append({"differ": best_max != best_exp, "has_exp": has_exp})
    q1 = pd.DataFrame(q1)
    q1_exp = q1[q1.has_exp]

    peak_am = fast.departure_time.between("07:30", "08:45")
    peak_pm = fast.departure_time.between("17:45", "19:00")
    by_dep = (fast.groupby("departure_time")
                  .agg(n=("route_id", "size"),
                       share_exp100_pos=("exposure_100_min", lambda s: (s > 0).mean()),
                       mean_exp100=("exposure_100_min", "mean"),
                       share_exp130_pos=("exposure_130_min", lambda s: (s > 0).mean()))
                  .round(3))
    d_p90 = fast.exposure_100_min_p90 - fast.exposure_100_min

    def pct(x):
        return f"{100 * float(np.mean(x)):.1f}%"

    elapsed = time.time() - t0
    L = [
        "# v2.1 혼잡 노출 지표 리포트 (route_profile_mart)",
        "",
        f"- 실행: `python scripts/v2/21_build_route_profile_mart.py --n-random {args.n_random} --seed {args.seed}`",
        f"- 표본: Tier C {len(tier_c)}쌍 + 무작위 {args.n_random}쌍 × {args.day_type} 출발 {len(DEPARTS)}개 "
        f"(07:00~09:45, 17:00~19:45, 15분 간격)",
        f"- mart: {len(mart):,}행 (OD×출발 {mart[key].drop_duplicates().shape[0]:,}건, 건당 후보 평균 "
        f"{mart.groupby(key).size().mean():.1f}개) → `data/marts/v2/route_profile_mart.parquet`",
        f"- 실행 시간: {elapsed / 60:.1f}분 · 평가 정책: step (D-002) · 혼잡 = 스냅샷 중앙값, 실시간 아님",
        "- 표본은 균등 무작위 OD 이며 수요 가중이 아니다. 아래 비율을 서울 전체 이동량 비율로 읽지 않는다.",
        "",
        "## Q4. 불변식 검사",
        "",
        f"- 검사 행 {len(mart):,}개, 위반 {len(vdf)}개",
        "- 검사 항목: 80≥100≥130 단조, 노출 ≤ 차내시간, 시간 분해(차내+도보+대기=소요), "
        "최대혼잡<100 이면 노출100=0, 최대혼잡≥100 이면 노출100>0, trace 시각 단조·도착 일치",
        "",
        "**판정**: " + ("통과" if vdf.empty else "실패 — 아래 위반 목록 확인"),
        "",
    ]
    if not vdf.empty:
        L += [vdf.head(20).to_markdown(index=False), ""]
    L += [
        "## Q1. exposure_100 은 max_congestion 과 다른 정보를 주는가",
        "",
        f"- **헤드라인**: 최속 +15분 이내 후보가 2개 이상이고 그중 하나라도 100% 이상 노출이 있는 "
        f"OD×출발 {len(q1_exp):,}건 중, '최대혼잡 최소 경로'와 'exposure_100 최소 경로'가 다른 비율: "
        f"{pct(q1_exp.differ) if len(q1_exp) else 'n/a'}",
        f"- 참고: 노출이 전혀 없는 건까지 포함한 {len(q1):,}건 기준 {pct(q1.differ) if len(q1) else 'n/a'} "
        "(노출 0 인 건은 체감시간으로 동률을 깨므로 차이에 의미가 적다. 헤드라인으로 쓰지 않는다)",
        "",
        "## Q2. 출발 시각별 노출 (최속 경로 기준)",
        "",
        by_dep.to_markdown(),
        "",
        f"- 오전 피크(07:30~08:45) 최속 경로 중 exposure_100 > 0: {pct(fast[peak_am].exposure_100_min > 0)}, "
        f"exposure_130 > 0: {pct(fast[peak_am].exposure_130_min > 0)}",
        f"- 오후 피크(17:45~19:00) 최속 경로 중 exposure_100 > 0: {pct(fast[peak_pm].exposure_100_min > 0)}, "
        f"exposure_130 > 0: {pct(fast[peak_pm].exposure_130_min > 0)}",
        "",
        "## Q3. 중앙값 vs p90",
        "",
        f"- 최속 경로 exposure_100: 중앙값 평균 {fast.exposure_100_min.mean():.2f}분, "
        f"p90 평균 {fast.exposure_100_min_p90.mean():.2f}분",
        f"- p90 이 중앙값보다 1분 이상 큰 비율: {pct(d_p90 >= 1)}, 차이의 p90: {d_p90.quantile(.9):.2f}분",
        "",
        "## 해석 가이드",
        "",
        "- Q4 가 통과해야 나머지 수치를 쓴다.",
        "- Q1 비율이 높을수록 '최대혼잡 한 점'과 '얼마나 오래 붐비는가'가 다른 경로를 고른다는 뜻 → exposure 를 헤드라인 지표로 쓰는 근거.",
        "- Q2 의 exposure_130 > 0 비율이 매우 낮으면 130 은 꼬리 지표로만 표시한다 (기획서 §5-2).",
        "- Q3 차이가 크면 '중앙값 기준 노출'이 보수적이지 않다는 뜻 → UI 에 p90 범위를 함께 보여줄 근거.",
    ]
    rdir = ROOT / "reports" / "v2"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "exposure_metrics_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[8:]))
    print(f"\n저장: {rdir / 'exposure_metrics_report.md'}")
    return 0 if vdf.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
