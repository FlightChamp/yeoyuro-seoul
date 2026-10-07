"""
scripts/v2/20_validate_td_engine.py
===================================
v2.0 게이트 검증: 시간 진행형 평가 엔진이 (1) v1 을 정확히 재현하고 (2) 실제로 차이를 만드는가.

검증 항목 (기획서 §8-1)
----------------------
A. 회귀        : policy="fixed" 결과 == v1 find_route_by_station 결과 (소요·체감 ±0.11분, 최대혼잡, 환승)
B. 편향 크기    : 같은 경로를 fixed(v1) vs step(v2) 로 평가했을 때 exposure_100·최대혼잡 차이, 소요시간 구간별
C. 판정 변화    : Route Shift 판정(성공/실패)이 fixed -> step 에서 뒤집히는 비율
D. bin 정책     : step vs floor vs linear 의 exposure_100 차이와 판정 일치율
E. 후보 근사    : 단일 bin K=5 / 다중 bin K=5 / 다중 bin K=10 의 '쾌적 최선 경로' 일치율

OD 표본
-------
- Tier C 고정셋 20쌍 (v1 기본 OD + 장거리·분기·순환 포함)
- 무작위 OD 150쌍 (seed 42)
- 출발 시각: 평일 07:40, 08:10, 08:25, 17:40, 18:15 (bin 경계를 일부러 피한 시각)

실행
----
    python scripts/v2/20_validate_td_engine.py              # 기본 (약 5~15분)
    python scripts/v2/20_validate_td_engine.py --n-random 30  # 빠른 점검

산출물
------
    reports/v2/td_engine_validation.md
    reports/v2/td_engine_validation_detail.csv
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import load_v1                                     # noqa: E402
from yeoyuro_v2.time_dependent import (TDRouter, TimeDependentEvaluator,  # noqa: E402
                                       parse_hhmm, bin_index_of)
from yeoyuro_v2.shift_rules import T0, route_shift, is_calm        # noqa: E402

DEPARTS = ["07:40", "08:10", "08:25", "17:40", "18:15"]
TIER_C = [
    ("신촌", "잠실"), ("서울역", "강남"), ("군자", "여의나루"), ("한양대", "고속터미널"),
    ("혜화", "사당"), ("종각", "이태원"), ("건대입구", "홍대입구"), ("안암", "삼성"),
    ("방화", "마천"), ("응암", "고속터미널"), ("노원", "사당"), ("장암", "온수"),
    ("불암산", "남태령"), ("하남검단산", "방화"), ("암사역사공원", "모란"), ("지축", "오금"),
    ("신내", "응암"), ("잠실", "홍대입구"), ("까치산", "성수"), ("천호", "공덕"),
]
TOL = 0.11


def comfort_best(res, base):
    """쾌적 최선 = base 대비 +15분·환승+1 이내 후보 중 (exposure_100, 체감시간) 최소."""
    ok = [r for r in res if r.actual_time_min - base.actual_time_min <= 15
          and r.transfer_count - base.transfer_count <= 1]
    return min(ok, key=lambda r: (round(r.exposure[100], 3), r.perceived_time_min)).path


def trip_bucket(m):
    return "<20분" if m < 20 else ("20~40분" if m < 40 else "40분+")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    t_start = time.time()
    mod, sr, disp = load_v1(ROOT)
    keys = disp["station_key"].tolist()
    unknown = [s for p in TIER_C for s in p if s not in keys]
    if unknown:
        print(f"[경고] display master 에 없는 Tier C 역명: {sorted(set(unknown))} -> 해당 OD 제외")
    tier_c = [p for p in TIER_C if p[0] in keys and p[1] in keys]
    rnd = random.Random(args.seed)
    rand_ods = set()
    while len(rand_ods) < args.n_random:
        a, b = rnd.sample(keys, 2)
        rand_ods.add((a, b))
    ods = [(o, d, "tier_c") for o, d in tier_c] + [(o, d, "random") for o, d in sorted(rand_ods)]

    router = TDRouter(ROOT, mod, sr, disp, "weekday", policy="step", k=5, multi_bin=True)
    headway = router.headway

    # ---------------- A. 회귀 (Tier C 전부 x 출발시각 전부)
    reg_rows = []
    for o, d, _ in ods[:len(tier_c)]:
        for dep in DEPARTS:
            rs = mod.RouteScorer(ROOT, "weekday", dep)
            r1 = sr.find_route_by_station(rs, disp, o, d, "calm")
            if not r1.get("ok"):
                continue
            ev = TimeDependentEvaluator(rs, headway, "fixed")
            for c in r1["candidates"]:
                r = ev.evaluate(c["path"], parse_hhmm(dep))
                reg_rows.append({
                    "od": f"{o}->{d}", "depart": dep,
                    "d_actual": abs(r.actual_time_min - c["actual_time_min"]),
                    "d_perceived": abs(r.perceived_time_min - c["perceived_time_min"]),
                    "d_maxcong": abs(round(r.max_congestion, 1) - c["max_congestion"]),
                    "d_transfer": abs(r.transfer_count - c["transfer_count"]),
                })
    reg = pd.DataFrame(reg_rows)
    reg_fail = reg[(reg.d_actual > TOL) | (reg.d_perceived > TOL)
                   | (reg.d_maxcong > 0.05) | (reg.d_transfer > 0)]
    print(f"[A] 회귀: {len(reg)}개 경로 중 불일치 {len(reg_fail)}개")

    # ---------------- B~E
    rows = []
    n_total = len(ods) * len(DEPARTS)
    done = 0
    for o, d, tier in ods:
        for dep in DEPARTS:
            done += 1
            tdep = parse_hhmm(dep)
            res_step = router.evaluate_od(o, d, tdep, policy="step")
            if not res_step:
                continue
            base = res_step[0]
            # 같은 후보 집합을 다른 정책으로 재평가 (경로 고정 비교)
            b0 = bin_index_of(tdep)
            evs = {p: router.evaluator(b0, p) for p in ("fixed", "floor", "linear")}
            same = {p: [evs[p].evaluate(r.path, tdep) for r in res_step] for p in evs}
            for p in same:
                same[p].sort(key=lambda r: (round(r.actual_time_min, 6), r.perceived_time_min))
            # v1 방식 그대로(단일 bin 후보 + fixed)
            res_v1 = router.evaluate_od(o, d, tdep, policy="fixed")
            base_v1 = res_v1[0]

            ok_step, _, bind_step = route_shift(base, res_step, T0)
            ok_v1, _, bind_v1 = route_shift(base_v1, res_v1, T0)
            ok_floor, _, _ = route_shift(same["floor"][0], same["floor"], T0)
            ok_lin, _, _ = route_shift(same["linear"][0], same["linear"], T0)

            # 같은 base 경로를 fixed 로 평가
            base_fixed_same = evs["fixed"].evaluate(base.path, tdep)
            base_floor_same = evs["floor"].evaluate(base.path, tdep)
            base_lin_same = evs["linear"].evaluate(base.path, tdep)

            # E. 후보 근사
            res_single = router.evaluate_od(o, d, tdep, policy="step", multi_bin=False)
            res_k10 = router.evaluate_od(o, d, tdep, policy="step", k=10)

            rows.append({
                "tier": tier, "origin": o, "destination": d, "depart": dep,
                "trip_min": round(base.actual_time_min, 2),
                "trip_bucket": trip_bucket(base.actual_time_min),
                "n_bins_crossed": len(base.bins_used),
                "n_cand_step": len(res_step), "n_cand_v1": len(res_v1),
                "fastest_changed": base.path != base_v1.path,
                "exp100_step": round(base.exposure[100], 2),
                "exp100_fixed": round(base_fixed_same.exposure[100], 2),
                "exp100_floor": round(base_floor_same.exposure[100], 2),
                "exp100_linear": round(base_lin_same.exposure[100], 2),
                "maxcong_step": round(base.max_congestion, 1),
                "maxcong_fixed": round(base_fixed_same.max_congestion, 1),
                "actual_step": round(base.actual_time_min, 2),
                "actual_fixed": round(base_fixed_same.actual_time_min, 2),
                "calm_step": is_calm(base), "calm_v1": is_calm(base_v1),
                "rs_step": ok_step, "rs_v1": ok_v1, "rs_floor": ok_floor, "rs_linear": ok_lin,
                "bind_step": bind_step, "bind_v1": bind_v1,
                "best_single_eq_multi": comfort_best(res_single, res_single[0]) == comfort_best(res_step, base),
                "best_multi_eq_k10": comfort_best(res_step, base) == comfort_best(res_k10, res_k10[0]),
            })
            if done % 50 == 0:
                print(f"  ... {done}/{n_total}  ({time.time() - t_start:.0f}s)")

    df = pd.DataFrame(rows)
    out_dir = ROOT / "reports" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "td_engine_validation_detail.csv", index=False, encoding="utf-8-sig")

    # ---------------- 집계
    df["d_exp100"] = df.exp100_step - df.exp100_fixed
    df["d_maxcong"] = df.maxcong_step - df.maxcong_fixed
    df["abs_d_exp100"] = df.d_exp100.abs()
    df["abs_d_maxcong"] = df.d_maxcong.abs()
    peak = df  # 모든 출발시각이 피크 부근

    def pct(x):
        return f"{100 * float(np.mean(x)):.1f}%"

    by_len = (df.groupby("trip_bucket")
                .agg(n=("trip_min", "size"),
                     mean_abs_d_exp100=("abs_d_exp100", "mean"),
                     p90_abs_d_exp100=("abs_d_exp100", lambda s: s.quantile(.9)),
                     mean_abs_d_maxcong=("abs_d_maxcong", "mean"),
                     share_maxcong_diff_ge5=("abs_d_maxcong", lambda s: (s >= 5).mean()))
                .reindex(["<20분", "20~40분", "40분+"]).round(2))
    by_bins = df.groupby("n_bins_crossed").agg(n=("trip_min", "size"),
                                                mean_abs_d_exp100=("abs_d_exp100", "mean"),
                                                mean_abs_d_maxcong=("abs_d_maxcong", "mean")).round(2)
    flip = df.rs_step != df.rs_v1
    bind_tab = pd.crosstab(df.bind_v1, df.bind_step, margins=True)

    elapsed = time.time() - t_start
    lines = [
        "# v2.0 시간 진행형 평가 엔진 검증",
        "",
        f"- 실행: `python scripts/v2/20_validate_td_engine.py --n-random {args.n_random} --seed {args.seed}`",
        f"- 표본: Tier C {len(tier_c)}쌍 + 무작위 {args.n_random}쌍 × 평일 출발 {', '.join(DEPARTS)} = 평가 {len(df)}건",
        f"- 실행 시간: {elapsed / 60:.1f}분",
        "- 혼잡도: congestion_edge_lookup 스냅샷 중앙값 (v1 baseline 과 동일). 실시간 아님.",
        "- 판정 규칙: `yeoyuro_v2/shift_rules.py` ThresholdSet T0 (v1 채택 조건 계승)",
        "",
        "## A. 회귀 — fixed 정책이 v1 을 재현하는가",
        "",
        f"- 비교 경로 {len(reg)}개, 불일치 {len(reg_fail)}개 (허용 오차: 소요·체감 ±{TOL}분, 최대혼잡 ±0.05)",
        f"- 소요시간 최대 차이 {reg.d_actual.max():.3f}분 (v1 이 중간 단계에서 1자리 반올림하는 데서 오는 차이)",
        "",
        "**판정**: " + ("통과" if len(reg_fail) == 0 else "실패 — td_engine_validation_detail 와 별도 점검 필요"),
        "",
        "## B. 고정 bin 편향의 크기 — 같은 경로, fixed(v1) vs step(v2)",
        "",
        by_len.to_markdown(),
        "",
        "통과 bin 수별:",
        "",
        by_bins.to_markdown(),
        "",
        f"- 최대 기대혼잡이 5%p 이상 달라진 비율: {pct(df.abs_d_maxcong >= 5)}",
        f"- exposure_100 이 1분 이상 달라진 비율: {pct(df.abs_d_exp100 >= 1)}",
        f"- 부호: step 이 fixed 보다 exposure_100 이 큰 비율 {pct(df.d_exp100 > 0)}, 작은 비율 {pct(df.d_exp100 < 0)}",
        "",
        "## C. 판정 변화 — v1 방식 vs v2 방식",
        "",
        f"- 시간 진행형 최속 경로가 v1 최속과 달라진 비율: {pct(df.fastest_changed)}",
        f"- Route Shift 성공 비율: v1 방식 {pct(df.rs_v1)} → v2 방식 {pct(df.rs_step)}",
        f"- 판정이 뒤집힌 비율: {pct(flip)} (실패→성공 {int((~df.rs_v1 & df.rs_step).sum())}건, "
        f"성공→실패 {int((df.rs_v1 & ~df.rs_step).sum())}건)",
        f"- Calm 게이트 해당 비율: v1 방식 {pct(df.calm_v1)} → v2 방식 {pct(df.calm_step)}",
        "",
        "binding constraint 교차표 (행 = v1 방식, 열 = v2 방식):",
        "",
        bind_tab.to_markdown(),
        "",
        "## D. bin 조회 정책 민감도",
        "",
        f"- exposure_100 평균 절대차: step vs floor {(df.exp100_step - df.exp100_floor).abs().mean():.2f}분, "
        f"step vs linear {(df.exp100_step - df.exp100_linear).abs().mean():.2f}분",
        f"- Route Shift 판정 일치율: step vs floor {pct(df.rs_step == df.rs_floor)}, "
        f"step vs linear {pct(df.rs_step == df.rs_linear)}",
        "",
        "## E. 후보 생성 근사",
        "",
        f"- 쾌적 최선 경로 일치율: 단일 bin K5 vs 다중 bin K5 = {pct(df.best_single_eq_multi)}",
        f"- 쾌적 최선 경로 일치율: 다중 bin K5 vs 다중 bin K10 = {pct(df.best_multi_eq_k10)}",
        f"- 평균 후보 수: v1 방식 {df.n_cand_v1.mean():.1f}개, v2 방식 {df.n_cand_step.mean():.1f}개",
        "",
        "## 해석 가이드",
        "",
        "- A 가 통과해야 B~E 의 차이를 '엔진 변경 효과'로 읽을 수 있다.",
        "- B 의 차이가 거의 0 이면 '30분 bin 해상도에서는 v1 고정 근사로 충분했다'가 결론이다. 그것도 유효한 결과다.",
        "- D 의 판정 일치율이 낮으면 bin 정책이 결론을 좌우한다는 뜻이므로 v2.3 sensitivity 에 반드시 포함한다.",
        "- E 의 K10 일치율이 낮으면 K 를 올리거나 후보 생성 방식을 바꿔야 한다.",
    ]
    (out_dir / "td_engine_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[8:]))
    print(f"\n저장: {out_dir / 'td_engine_validation.md'}")
    return 0 if len(reg_fail) == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
