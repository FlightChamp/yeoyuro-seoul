"""
scripts/v2/22_build_temporal_shift_mart.py
==========================================
v2.2 — temporal_shift_mart 생성 + Time Shift 리포트.

1행 = (OD, 기준 출발, 후보 시각, 경로 유형). 후보 시각 = 기준 ±30/60/90분.
경로 유형: 같은 경로(same_route=True) / 그 시각의 최속 경로(same_route=False, 다를 때만).

리포트가 답하는 질문
--------------------
Q0. 판정 산출물이 규칙과 일치하는가 (불변식)
Q1. 혼잡한 이동(Calm 아님) 중 출발 시각 조정으로 노출을 줄일 수 있는 비율은? 기준 시각별로?
Q2. 성공한 경우 최소 이동폭(30/60/90)과 방향(일찍/늦게)은?
Q3. 실패한 경우 무엇이 막는가 (binding constraint)?
Q4. 시각을 옮기면 최속 경로가 바뀌는가 (same_route 유지율)?
Q5. v1 시간대안(±120분, 최대혼잡 최소 bin)과 비교하면?

※ Route Shift × Time Shift 교차(= v2 중심 가설의 4유형 분포)는 이 리포트에서 **계산하지 않는다.**
   반증 조건과 임계값을 v2.3 착수 전에 고정한 뒤 처음 계산한다 (기획서 §3, decision_log D-017).

실행
----
    python scripts/v2/22_build_temporal_shift_mart.py                # 기본 (약 5~10분)
    python scripts/v2/22_build_temporal_shift_mart.py --n-random 20  # 빠른 점검

산출물
------
    data/marts/v2/temporal_shift_mart.parquet
    reports/v2/temporal_shift_report.md
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

from yeoyuro_v2 import load_v1                                       # noqa: E402
from yeoyuro_v2.shift_rules import T0                                 # noqa: E402
from yeoyuro_v2.temporal_shift import evaluate_time_shift            # noqa: E402
from yeoyuro_v2.time_dependent import TDRouter, parse_hhmm, bin_index_of  # noqa: E402

TIER_C = [
    ("신촌", "잠실"), ("서울역", "강남"), ("군자", "여의나루"), ("한양대", "고속터미널"),
    ("혜화", "사당"), ("종각", "이태원"), ("건대입구", "홍대입구"), ("안암", "삼성"),
    ("방화", "마천"), ("응암", "고속터미널"), ("노원", "사당"), ("장암", "온수"),
    ("불암산", "남태령"), ("하남검단산", "방화"), ("암사역사공원", "모란"), ("지축", "오금"),
    ("신내", "응암"), ("잠실", "홍대입구"), ("까치산", "성수"), ("천호", "공덕"),
]
BASE_TIMES = ["07:30", "08:00", "08:30", "17:30", "18:00", "18:30"]


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
    rows, od_rows, bad = [], [], []
    total, done = len(ods) * len(BASE_TIMES), 0
    for o, d, tier in ods:
        for bt in BASE_TIMES:
            done += 1
            tb = parse_hhmm(bt)
            res = router.evaluate_od(o, d, tb)
            if not res:
                continue
            base_path = res[0].path
            ev = router.evaluator(bin_index_of(tb))

            def reopt(t, o=o, d=d):
                r = router.evaluate_od(o, d, t)
                return r[0] if r else None

            ts = evaluate_time_shift(ev, base_path, tb, T0, reopt=reopt)
            base = ts["base"]

            # Q0 불변식
            top_keys = [(abs(x.shift_min), -x.exposure_100_drop_min, -x.shift_min) for x in ts["top"]]
            if top_keys != sorted(top_keys):
                bad.append((o, d, bt, "top_not_sorted"))
            if any(not x.success for x in ts["top"]):
                bad.append((o, d, bt, "top_contains_failure"))
            if ts["type"] == "time_shiftable" and not ts["top"]:
                bad.append((o, d, bt, "shiftable_without_top"))
            if ts["type"] == "calm" and ts["top"]:
                bad.append((o, d, bt, "calm_with_top"))
            if abs(base.actual_time_min - res[0].actual_time_min) > 1e-6:
                bad.append((o, d, bt, "base_mismatch"))

            top_rank = {(x.shift_min, x.same_route): i + 1 for i, x in enumerate(ts["top"])}
            for x in ts["rows"]:
                rows.append({"tier": tier, "origin": o, "destination": d, "day_type": args.day_type,
                             "base_time": bt, **x.as_dict(),
                             "recommended_rank": top_rank.get((x.shift_min, x.same_route)),
                             "threshold_set_id": ts["threshold_set_id"]})

            # v1 시간대안 (같은 base 경로, v1 scorer 의 time_alternative)
            v1 = router.scorer(bin_index_of(tb)).time_alternative(base_path)
            v1_shift = None
            if v1:
                bi = bin_index_of(parse_hhmm(v1["best_time_bin"][:5]))
                v1_shift = (bi - bin_index_of(tb)) * 30

            same_rows = [x for x in ts["rows"] if x.same_route]
            n_shift = len(same_rows)
            n_same_fast = n_shift - sum(1 for x in ts["rows"] if not x.same_route)
            od_rows.append({
                "tier": tier, "origin": o, "destination": d, "base_time": bt,
                "base_actual_min": round(base.actual_time_min, 2),
                "base_exposure_100_min": round(base.exposure[100], 2),
                "base_max_congestion": round(base.max_congestion, 1),
                "temporal_shift_type": ts["type"], "binding_constraint": ts["binding_constraint"],
                "min_shift_min": ts["min_shift_min"],
                "top1_drop_min": ts["top"][0].exposure_100_drop_min if ts["top"] else None,
                "n_shift_times": n_shift, "n_fastest_unchanged": n_same_fast,
                "v1_time_alt_shift_min": v1_shift,
            })
            if done % 100 == 0:
                print(f"  ... {done}/{total}  ({time.time() - t0:.0f}s)")

    mart = pd.DataFrame(rows)
    od = pd.DataFrame(od_rows)
    mdir = ROOT / "data" / "marts" / "v2"
    mdir.mkdir(parents=True, exist_ok=True)
    mart.to_parquet(mdir / "temporal_shift_mart.parquet", index=False)
    od.to_parquet(mdir / "temporal_shift_od_summary.parquet", index=False)

    def pct(x):
        x = list(x)
        return f"{100 * float(np.mean(x)):.1f}%" if x else "n/a"

    hot = od[od.temporal_shift_type != "calm"]
    succ = hot[hot.temporal_shift_type == "time_shiftable"]
    fail = hot[hot.temporal_shift_type == "not_time_shiftable"]
    by_bt = (od.groupby("base_time")
               .apply(lambda g: pd.Series({
                   "n": len(g),
                   "calm": (g.temporal_shift_type == "calm").mean(),
                   "time_shiftable": (g.temporal_shift_type == "time_shiftable").mean(),
                   "not_time_shiftable": (g.temporal_shift_type == "not_time_shiftable").mean(),
                   "shiftable_among_hot": ((g.temporal_shift_type == "time_shiftable").sum()
                                           / max(1, (g.temporal_shift_type != "calm").sum()))}),
                      include_groups=False).round(3))
    min_shift = succ.min_shift_min.value_counts().sort_index()
    bind = fail.binding_constraint.value_counts()
    keep_rate = od.n_fastest_unchanged.sum() / max(1, od.n_shift_times.sum())

    v1_has = hot.v1_time_alt_shift_min.notna()
    v1_far = hot.v1_time_alt_shift_min.abs() > 90
    both = hot[v1_has & (hot.temporal_shift_type == "time_shiftable")]
    same_dir = (np.sign(both.v1_time_alt_shift_min) == np.sign(both.min_shift_min))

    # Q6. 창 크기·감소 기준 민감도 (v2.3 사전 등록 결정용)
    key = ["origin", "destination", "base_time"]
    hot_keys = hot[key]
    mh = mart.merge(hot_keys, on=key)
    mh["ok_route"] = mh.same_route | (mh.jaccard >= T0.time_shift_similar_jaccard)
    mh["ok_time"] = mh.actual_delta_min <= T0.time_shift_max_time_increase_min
    mh["red_or"] = ((mh.exposure_100_drop_min >= T0.min_exposure_drop_min)
                    | (mh.max_cong_drop_pp >= T0.min_max_cong_drop_pp))
    mh["red_exp"] = mh.exposure_100_drop_min >= T0.min_exposure_drop_min
    sens = []
    for w in (30, 60, 90):
        sub = mh[mh.shift_min.abs() <= w]
        for crit, col in (("노출 또는 최대혼잡 (T0)", "red_or"), ("노출 감소만", "red_exp")):
            okk = sub[sub.ok_route & sub.ok_time & sub[col]][key].drop_duplicates()
            sens.append({"창": f"±{w}분", "감소 기준": crit,
                         "time_shiftable (Calm 제외)": f"{100 * len(okk) / max(1, len(hot_keys)):.1f}%"})
    sens = pd.DataFrame(sens)

    L = [
        "# v2.2 Time Shift 리포트 (temporal_shift_mart)",
        "",
        f"- 실행: `python scripts/v2/22_build_temporal_shift_mart.py --n-random {args.n_random} --seed {args.seed}`",
        f"- 표본: Tier C {len(tier_c)}쌍 + 무작위 {args.n_random}쌍 × {args.day_type} 기준 출발 "
        f"{', '.join(BASE_TIMES)} = {len(od):,}건, 후보 행 {len(mart):,}개",
        f"- 규칙: ThresholdSet `{T0.set_id}` — 후보 ±{'/'.join(map(str, T0.time_shift_windows))}분, "
        f"감소(노출 ≥{T0.min_exposure_drop_min:g}분 또는 최대혼잡 ≥{T0.min_max_cong_drop_pp:g}%p), "
        f"소요 증가 ≤{T0.time_shift_max_time_increase_min:g}분, Calm 게이트 {T0.calm_gate_exposure_min:g}분",
        f"- 실행 시간: {(time.time() - t0) / 60:.1f}분 · step 정책 · 혼잡 = 스냅샷 중앙값, 실시간 아님",
        "- 표본은 균등 무작위 OD 이며 수요 가중이 아니다. 비율을 서울 전체 이동량 비율로 읽지 않는다.",
        "- Route Shift × Time Shift 교차 분포는 의도적으로 계산하지 않았다 (D-017).",
        "",
        "## Q0. 불변식",
        "",
        f"- 위반 {len(bad)}건 (Top 정렬 규칙, Top 전원 성공, 유형-Top 정합, 기준 경로 일치)",
        "",
        "**판정**: " + ("통과" if not bad else "실패 — " + str(bad[:5])),
        "",
        "## Q1. 출발 시각 조정으로 노출을 줄일 수 있는가",
        "",
        f"- Calm 아닌 이동 {len(hot):,}건 중 time_shiftable: **{pct(hot.temporal_shift_type == 'time_shiftable')}**",
        f"- 전체 {len(od):,}건 중 Calm {pct(od.temporal_shift_type == 'calm')}",
        "",
        "기준 출발 시각별 (shiftable_among_hot = Calm 제외 중 성공 비율):",
        "",
        by_bt.to_markdown(),
        "",
        "## Q2. 성공한 경우 최소 이동폭과 방향",
        "",
        "최소 이동폭(분, 음수 = 일찍) 분포:",
        "",
        min_shift.to_frame("건수").to_markdown(),
        "",
        f"- 늦게 출발이 최선인 비율 {pct(succ.min_shift_min > 0)}, 일찍 {pct(succ.min_shift_min < 0)}",
        f"- ±30분 안에서 해결되는 비율 {pct(succ.min_shift_min.abs() <= 30)}",
        f"- Top1 의 exposure_100 감소 중앙값 {succ.top1_drop_min.median():.1f}분",
        "",
        "## Q3. 실패한 경우 무엇이 막는가",
        "",
        bind.to_frame("건수").to_markdown() if len(bind) else "(실패 없음)",
        "",
        "## Q4. 시각을 옮겨도 최속 경로가 유지되는가",
        "",
        f"- 후보 시각 {int(od.n_shift_times.sum()):,}개 중 최속 경로가 기준 경로와 같은 비율: {100 * keep_rate:.1f}%",
        "",
        "## Q5. v1 시간대안과 비교 (Calm 아닌 이동)",
        "",
        f"- v1 이 시간대안을 낸 비율 {pct(v1_has)}, v2.2 time_shiftable {pct(hot.temporal_shift_type == 'time_shiftable')}",
        f"- v1 추천이 90분 넘게 떨어진 시각인 비율 (v1 창 ±120분): {pct(v1_far[v1_has]) if v1_has.any() else 'n/a'}",
        f"- v1 은 추천했지만 v2.2 는 불가로 본 비율: "
        f"{pct((hot.temporal_shift_type == 'not_time_shiftable')[v1_has]) if v1_has.any() else 'n/a'}",
        f"- 둘 다 추천한 {len(both):,}건 중 방향(일찍/늦게) 일치: {pct(same_dir) if len(both) else 'n/a'}",
        "",
        "## Q6. 창 크기·감소 기준 민감도 (v2.3 사전 등록용)",
        "",
        sens.to_markdown(index=False),
        "",
        "Time 축의 성공률은 창 크기와 감소 기준에 크게 좌우될 수 있다. 이 표는 v2.3 의 주 기준을 "
        "정하기 위한 것이며, Route 축과의 교차는 여전히 계산하지 않는다.",
        "",
        "## 해석 가이드",
        "",
        "- Q0 가 통과해야 나머지를 쓴다.",
        "- Q1 의 shiftable_among_hot 이 시각대별로 크게 다르면, 피크 한가운데와 가장자리의 성격이 다르다는 뜻이다.",
        "- Q3 에서 insufficient_reduction 이 대부분이면 ±90분 안에 피크를 벗어나지 못하는 이동이 많다는 뜻이다.",
        "- Q5 에서 v1 추천의 상당수가 90분 밖이면, v2.2 에서 창을 ±90분으로 줄인 근거가 된다.",
    ]
    rdir = ROOT / "reports" / "v2"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "temporal_shift_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[9:]))
    print(f"\n저장: {rdir / 'temporal_shift_report.md'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
