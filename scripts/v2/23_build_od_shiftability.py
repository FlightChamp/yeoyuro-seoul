"""
scripts/v2/23_build_od_shiftability.py
======================================
v2.3 — OD Shiftability: 사전 등록(docs/v2/preregistration_v23.md) 기준으로
혼잡 이동을 Route / Time / Dual / Structural / Calm 으로 분류하고 가설 H1·H2·H3·H5 를 판정한다.

순서
----
0. 사전 등록 확인 : preregistration_v23.md 와 shift_rules.py 가 커밋된 상태인지 git 으로 확인
1. evidence 수집 : 엔진 계산 (멀티프로세싱). OD×시각마다 기준 경로, 대안 경로, ±30/60/90 평가값
2. 분류          : ThresholdSet T1 (순수 함수, 재계산 없음)
3. 판정          : H1·H2·H3·H5, bootstrap 95% 구간, 수요 proxy 가중
4. sensitivity   : 단일 요인 15개 + Monte Carlo 500회 + T0 비교

실행
----
    python scripts/v2/23_build_od_shiftability.py                 # 기본 (PC 코어 수에 따라 3~15분)
    python scripts/v2/23_build_od_shiftability.py --workers 4
    python scripts/v2/23_build_od_shiftability.py --reuse-evidence   # 1단계 생략, 분류·판정만 다시

산출물
------
    data/marts/v2/od_shiftability_evidence.pkl.gz
    data/marts/v2/od_shiftability_mart.parquet
    reports/v2/od_shiftability_report.md
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2.od_shiftability import TYPES, build_evidence, classify   # noqa: E402
from yeoyuro_v2.shift_rules import T0, T1                                # noqa: E402

PREREG = "docs/v2/preregistration_v23.md"
PREREG_CODE = "yeoyuro_v2/shift_rules.py"
TIER_C = [
    ("신촌", "잠실"), ("서울역", "강남"), ("군자", "여의나루"), ("한양대", "고속터미널"),
    ("혜화", "사당"), ("종각", "이태원"), ("건대입구", "홍대입구"), ("안암", "삼성"),
    ("방화", "마천"), ("응암", "고속터미널"), ("노원", "사당"), ("장암", "온수"),
    ("불암산", "남태령"), ("하남검단산", "방화"), ("암사역사공원", "모란"), ("지축", "오금"),
    ("신내", "응암"), ("잠실", "홍대입구"), ("까치산", "성수"), ("천호", "공덕"),
]
BASE_TIMES = ["07:30", "08:00", "08:30", "17:30", "18:00", "18:30"]
SHIFTS = (-90, -60, -30, 30, 60, 90)
TYPE_KO = {"calm": "Calm", "route_shiftable": "Route-shiftable", "time_shiftable": "Time-shiftable",
           "dual": "Dual", "structural": "Structurally constrained"}


# ---------------------------------------------------------------------- 0. 사전 등록 확인
def check_prereg() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()
    dirty = git("status", "--porcelain", "--", PREREG, PREREG_CODE)
    log = git("log", "-1", "--format=%H|%cI", "--", PREREG)
    return {"ok": (not dirty) and bool(log), "dirty": dirty,
            "commit": log.split("|")[0] if log else None, "date": log.split("|")[1] if log else None}


# ---------------------------------------------------------------------- 1. evidence (worker)
_W = {}


def _init_worker(day_type: str):
    from yeoyuro_v2 import load_v1
    from yeoyuro_v2.time_dependent import TDRouter
    mod, sr, disp = load_v1(ROOT)
    _W["router"] = TDRouter(ROOT, mod, sr, disp, day_type, policy="step", k=5, multi_bin=True)
    _W["sr"] = sr


def _edge_exposure(ev, base) -> dict:
    """기준 경로의 구간별 100%+ 노출(분). H3 집중도용."""
    from yeoyuro_v2.time_dependent import bin_index_of  # noqa: F401
    out = {}
    sr = _W["sr"]
    for t in base.trace:
        if t.kind not in ("ride", "dwell"):
            continue
        mins = sum(d for d, c, _c90, _b in ev._pieces(t.u, t.v, t.t_start, t.t_end - t.t_start) if c >= 100)
        if mins > 0:
            m = ev.edge_meta[(t.u, t.v)]
            k = f"{m['line']}호선 {sr.station_of(t.u)}→{sr.station_of(t.v)}"
            out[k] = out.get(k, 0.0) + mins
    return out


def _work(task):
    from yeoyuro_v2.time_dependent import parse_hhmm, bin_index_of, N_BINS
    o, d, tier = task
    router = _W["router"]
    cache = {}

    def fastest_at(t):
        if t not in cache:
            r = router.evaluate_od(o, d, t)
            cache[t] = r
        return cache[t]

    items = []
    for bt in BASE_TIMES:
        tb = parse_hhmm(bt)
        res = fastest_at(tb)
        if not res:
            continue
        base_q = res[0]
        ev = router.evaluator(bin_index_of(tb))
        base = ev.evaluate(base_q.path, tb, keep_trace=True)
        rows = []
        for s in SHIFTS:
            t = tb + s
            if not (0 <= bin_index_of(t) < N_BINS):
                continue
            rows.append((s, True, ev.evaluate(base.path, t)))
            f = fastest_at(t)
            if f and f[0].path != base.path:
                rows.append((s, False, f[0]))
        evd = build_evidence(base, res[1:], rows)
        evd.update({"origin": o, "destination": d, "tier": tier, "base_time": bt,
                    "edge_exp100": _edge_exposure(ev, base)})
        items.append(evd)
    return items


def collect(ods, workers: int, day_type: str):
    t0 = time.time()
    out = []
    if workers <= 1:
        _init_worker(day_type)
        for i, task in enumerate(ods, 1):
            out.extend(_work(task))
            if i % 50 == 0:
                print(f"  ... OD {i}/{len(ods)}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        with Pool(workers, initializer=_init_worker, initargs=(day_type,)) as pool:
            for i, items in enumerate(pool.imap_unordered(_work, ods, chunksize=4), 1):
                out.extend(items)
                if i % 50 == 0:
                    print(f"  ... OD {i}/{len(ods)}  ({time.time() - t0:.0f}s)", flush=True)
    out.sort(key=lambda x: (x["origin"], x["destination"], x["base_time"]))
    return out


# ---------------------------------------------------------------------- 수요 proxy
def demand_weights(items, beta: float = 0.0) -> np.ndarray:
    dem = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "demand_station_hour.parquet")
    dem = dem[dem.day_type == "weekday"]
    B = {(r.station_key, r.hour): r.boardings for r in dem.itertuples()}
    A = {(r.station_key, r.hour): r.alightings for r in dem.itertuples()}
    Atot = dem.groupby("hour").alightings.sum().to_dict()
    w = []
    for it in items:
        h = int(it["base_time"][:2])
        ha = min(24, int(it["base"]["arrive"] // 60))
        x = B.get((it["origin"], h), 0.0) * A.get((it["destination"], ha), 0.0) / max(1.0, Atot.get(ha, 1.0))
        if beta:
            x *= np.exp(-beta * it["base"]["actual"])
        w.append(x)
    return np.asarray(w, dtype=float)


# ---------------------------------------------------------------------- 분석 도우미
def shares(types, weights=None, hot_only=False):
    t = np.asarray(types)
    w = np.ones(len(t)) if weights is None else np.asarray(weights)
    if hot_only:
        m = t != "calm"
        t, w = t[m], w[m]
    tot = w.sum()
    return {k: (w[t == k].sum() / tot if tot else np.nan) for k in TYPES if not (hot_only and k == "calm")}


def bootstrap(types, od_ids, weights=None, hot_only=False, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    t = np.asarray(types)
    w = np.ones(len(t)) if weights is None else np.asarray(weights)
    ods = np.unique(od_ids)
    idx_by_od = {o: np.where(od_ids == o)[0] for o in ods}
    keys = [k for k in TYPES if not (hot_only and k == "calm")]
    per_od = np.zeros((len(ods), len(keys)))
    for i, o in enumerate(ods):
        ii = idx_by_od[o]
        for j, k in enumerate(keys):
            per_od[i, j] = w[ii][t[ii] == k].sum()
    draws = []
    for _ in range(n):
        s = per_od[rng.integers(0, len(ods), len(ods))].sum(axis=0)
        draws.append(s / s.sum() if s.sum() else s)
    draws = np.array(draws)
    return {k: (np.percentile(draws[:, j], 2.5), np.percentile(draws[:, j], 97.5)) for j, k in enumerate(keys)}


def sens_variants():
    v = []
    for a in (4.0, 6.0):
        v.append((f"노출 절대 감소 {a:g}분", replace(T1, set_id=f"S_abs{a:g}", min_exposure_drop_min=a)))
    for r in (0.4, 0.6):
        v.append((f"노출 상대 감소 {int(r * 100)}%", replace(T1, set_id=f"S_rel{r}", min_exposure_drop_rel=r)))
    for x in (12.0, 18.0):
        v.append((f"Route 시간손실 {x:g}분", replace(T1, set_id=f"S_tl{x:g}", max_time_loss_min=x)))
    for j in (0.55, 0.75):
        v.append((f"Route Jaccard {j}", replace(T1, set_id=f"S_jac{j}", max_edge_jaccard=j)))
    for c in (2.0, 5.0):
        v.append((f"Calm 게이트 {c:g}분", replace(T1, set_id=f"S_calm{c:g}", calm_gate_exposure_min=c)))
    for w in (30, 90):
        v.append((f"Time 주 창 ±{w}", replace(T1, set_id=f"S_win{w}", time_shift_primary_window=w)))
    for x in (4.0, 6.0):
        v.append((f"Time 소요 증가 {x:g}분", replace(T1, set_id=f"S_ti{x:g}", time_shift_max_time_increase_min=x)))
    v.append(("혼잡 통계량 p90", replace(T1, set_id="S_p90", use_p90=True)))
    return v


def mc_draw(rng):
    return replace(
        T1, set_id="MC",
        min_exposure_drop_min=float(rng.choice([4.0, 5.0, 6.0])),
        min_exposure_drop_rel=float(rng.choice([0.4, 0.5, 0.6])),
        max_time_loss_min=float(rng.choice([12.0, 15.0, 18.0])),
        max_edge_jaccard=float(rng.choice([0.55, 0.65, 0.75])),
        calm_gate_exposure_min=float(rng.choice([2.0, 3.0, 5.0])),
        time_shift_primary_window=int(rng.choice([30, 60, 90])),
        time_shift_max_time_increase_min=float(rng.choice([4.0, 5.0, 6.0])),
        use_p90=bool(rng.choice([False, True])))


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--reuse-evidence", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true",
                    help="사전 등록 커밋 확인을 건너뜀 (개발용, 리포트에 '미확인'으로 기록)")
    args = ap.parse_args(argv)
    t_start = time.time()

    pr = check_prereg()
    if not pr["ok"] and not args.allow_uncommitted:
        print("[중단] 사전 등록이 커밋되지 않았습니다.")
        print(f"  {PREREG}, {PREREG_CODE} 를 먼저 커밋하세요. 변경 상태: {pr['dirty'] or '(커밋 이력 없음)'}")
        return 2
    print(f"사전 등록: {'확인 ' + pr['commit'][:8] + ' (' + pr['date'] + ')' if pr['ok'] else '미확인'}")

    mdir = ROOT / "data" / "marts" / "v2"
    mdir.mkdir(parents=True, exist_ok=True)
    evp = mdir / "od_shiftability_evidence.pkl.gz"

    disp = pd.read_csv(ROOT / "data" / "master" / "station_display_master.csv")
    keys = disp.station_key.tolist()
    tier_c = [p for p in TIER_C if p[0] in keys and p[1] in keys]
    rnd = random.Random(args.seed)
    rand_ods = set()
    while len(rand_ods) < args.n_random:
        a, b = rnd.sample(keys, 2)
        if (a, b) not in tier_c:
            rand_ods.add((a, b))
    ods = [(o, d, "tier_c") for o, d in tier_c] + [(o, d, "random") for o, d in sorted(rand_ods)]

    if args.reuse_evidence and evp.exists():
        with gzip.open(evp, "rb") as f:
            items = pickle.load(f)
        print(f"evidence 재사용: {len(items):,}건")
    else:
        print(f"[1/4] evidence 수집: OD {len(ods)}개 × 기준 시각 {len(BASE_TIMES)}개, workers={args.workers}")
        items = collect(ods, args.workers, "weekday")
        with gzip.open(evp, "wb") as f:
            pickle.dump(items, f)
    t_ev = time.time() - t_start

    # ------------------------------------------------------------------ 2. 분류 (T1)
    print("[2/4] 분류")
    cls = [classify(it, T1) for it in items]
    types = np.array([c["type"] for c in cls])
    od_id = np.array([f"{it['origin']}>{it['destination']}" for it in items])
    w = demand_weights(items)
    w_g3 = demand_weights(items, 0.03)
    w_g6 = demand_weights(items, 0.06)
    hot = types != "calm"

    mart = pd.DataFrame([{
        "origin": it["origin"], "destination": it["destination"], "tier": it["tier"],
        "day_type": "weekday", "base_time": it["base_time"],
        "base_actual_min": round(it["base"]["actual"], 2),
        "base_exposure_100_min": round(it["base"]["exp100"], 2),
        "base_max_congestion": round(it["base"]["max"], 1),
        "route_shift_success": c["route_ok"], "route_shift_reason": c["route_bind"],
        "best_route_exposure_drop_min": None if c["route_drop"] is None else round(c["route_drop"], 2),
        "time_shift_success": c["time_ok"], "time_shift_reason": c["time_bind"],
        "best_time_shift_min": c["time_shift"],
        "best_time_exposure_drop_min": None if c["time_drop"] is None else round(c["time_drop"], 2),
        "final_shiftability_type": c["type"],
        "recommended_means": (("route" if (c["route_drop"] or 0) >= (c["time_drop"] or 0) else "time")
                              if c["type"] == "dual" else None),
        "demand_proxy_w": round(float(wi), 4), "demand_proxy_method": "B_o*A_d/sumA (no decay)",
        "threshold_set_id": T1.set_id, "event_sensitive_flag": None,
    } for it, c, wi in zip(items, cls, w)])
    mart.to_parquet(mdir / "od_shiftability_mart.parquet", index=False)

    # ------------------------------------------------------------------ 3. 판정
    print("[3/4] 가설 판정")
    sh_all = shares(types)
    sh_hot = shares(types, hot_only=True)
    sh_hot_w = shares(types, w, hot_only=True)
    sh_hot_g3 = shares(types, w_g3, hot_only=True)
    sh_hot_g6 = shares(types, w_g6, hot_only=True)
    ci_hot = bootstrap(types, od_id, hot_only=True)
    ci_hot_w = bootstrap(types, od_id, w, hot_only=True)
    ci_all = bootstrap(types, od_id)

    route_ok = np.array([bool(c["route_ok"]) for c in cls])
    time_ok = np.array([bool(c["time_ok"]) for c in cls])
    h1 = route_ok[hot].mean() if hot.any() else np.nan
    rf = hot & ~route_ok
    h2 = time_ok[rf].mean() if rf.any() else np.nan
    h1_pass = 0.01 <= h1 <= 0.50
    h2_pass = h2 >= 0.05

    # H3 집중도 순열 검정
    hot_idx = np.where(hot)[0]
    edges = sorted({e for i in hot_idx for e in items[i]["edge_exp100"]})
    eidx = {e: j for j, e in enumerate(edges)}
    M = np.zeros((len(hot_idx), len(edges)))
    for r, i in enumerate(hot_idx):
        for e, v in items[i]["edge_exp100"].items():
            M[r, eidx[e]] = v
    lab = (types[hot_idx] == "structural")

    def top10_share(mask):
        tot = M[mask].sum(axis=0)
        s = tot.sum()
        return np.sort(tot)[::-1][:10].sum() / s if s > 0 else np.nan

    n_struct = int(lab.sum())
    if n_struct >= 2 and len(edges) > 10:
        obs = top10_share(lab)
        rng = np.random.default_rng(11)
        perm = []
        for _ in range(1000):
            m = np.zeros(len(lab), bool)
            m[rng.choice(len(lab), n_struct, replace=False)] = True
            perm.append(top10_share(m))
        perm = np.array(perm)
        p3 = float((perm >= obs).mean())
        h3_pass = p3 < 0.05
        tot_s = M[lab].sum(axis=0)
        cnt_s = (M[lab] > 0).sum(axis=0)
        top_edges = [(edges[j], tot_s[j], int(cnt_s[j])) for j in np.argsort(tot_s)[::-1][:10] if tot_s[j] > 0]
    else:
        obs, p3, h3_pass, perm, top_edges = np.nan, np.nan, None, np.array([np.nan]), []

    # ------------------------------------------------------------------ 4. sensitivity
    print("[4/4] sensitivity")
    base_hot_types = types[hot]
    srows = []
    for label, th in sens_variants():
        tv = np.array([classify(it, th)["type"] for it in items])
        ret = (tv[hot] == base_hot_types).mean()
        per = {k: ((tv[hot] == k) & (base_hot_types == k)).sum() / max(1, (base_hot_types == k).sum())
               for k in ("route_shiftable", "time_shiftable", "dual", "structural")}
        sh = shares(tv, hot_only=True)
        srows.append({"요인": label, "유지율": ret, **{f"유지_{TYPE_KO[k]}": v for k, v in per.items()},
                      "Structural 비율(혼잡 이동)": sh["structural"], "Calm 비율(전체)": (tv == "calm").mean()})
    sens = pd.DataFrame(srows)
    n_fail = int((sens["유지율"] < 0.70).sum())
    h5_verdict = ("통과" if n_fail == 0 else
                  ("부분 통과 — 미달 요인을 결론에 명시" if n_fail < len(sens) / 2 else "미달 — 결론을 탐색적으로 격하"))

    rng = np.random.default_rng(23)
    mc = []
    for _ in range(500):
        th = mc_draw(rng)
        tv = np.array([classify(it, th)["type"] for it in items])
        mc.append(((tv[hot] == base_hot_types).mean(), shares(tv, hot_only=True)["structural"]))
    mc = np.array(mc)
    t0_types = np.array([classify(it, T0)["type"] for it in items])
    sh_t0 = shares(t0_types, hot_only=True)

    # ------------------------------------------------------------------ 리포트
    bt_tab = (pd.DataFrame({"base_time": [it["base_time"] for it in items], "type": types})
                .groupby("base_time").type.value_counts(normalize=True).unstack(fill_value=0)
                .reindex(columns=list(TYPES), fill_value=0).rename(columns=TYPE_KO).round(3))
    sb = mart[mart.final_shiftability_type == "structural"]
    bind_tab = pd.crosstab(sb.route_shift_reason, sb.time_shift_reason)
    dual = mart[mart.final_shiftability_type == "dual"]

    def type_table():
        rows = []
        for k in TYPES:
            r = {"유형": TYPE_KO[k], "전체 (OD-count)": pct(sh_all[k]),
                 "전체 95% 구간": f"{pct(ci_all[k][0])}~{pct(ci_all[k][1])}"}
            if k != "calm":
                r.update({"혼잡 이동 중 (OD-count)": pct(sh_hot[k]),
                          "95% 구간": f"{pct(ci_hot[k][0])}~{pct(ci_hot[k][1])}",
                          "혼잡 이동 중 (수요 proxy)": pct(sh_hot_w[k]),
                          "proxy 95% 구간": f"{pct(ci_hot_w[k][0])}~{pct(ci_hot_w[k][1])}"})
            rows.append(r)
        return pd.DataFrame(rows).fillna("—").to_markdown(index=False)

    sens_md = sens.copy()
    for c in sens_md.columns[1:]:
        sens_md[c] = sens_md[c].map(pct)
    L = [
        "# v2.3 OD Shiftability 리포트",
        "",
        f"- 사전 등록: `{PREREG}` "
        + (f"커밋 `{pr['commit'][:10]}` ({pr['date']}) 기준으로 실행" if pr["ok"] else "**미확인 (커밋 안 된 상태로 실행)**"),
        f"- ThresholdSet `{T1.set_id}` · 실행 `python scripts/v2/23_build_od_shiftability.py "
        f"--n-random {args.n_random} --seed {args.seed}`",
        f"- 단위: OD × 평일 기준 출발 {', '.join(BASE_TIMES)} = {len(items):,}건 "
        f"(OD {len(set(od_id)):,}개: Tier C {len(tier_c)} + 무작위 {args.n_random}), 혼잡 이동 {int(hot.sum()):,}건",
        f"- 실행 시간: evidence {t_ev / 60:.1f}분, 전체 {(time.time() - t_start) / 60:.1f}분",
        "- 혼잡 = 과거 스냅샷 중앙값 (실시간 아님). Route-shiftable 은 한 사람이 옮길 때 기준이며 수요 재배분은 반영하지 않는다.",
        "- OD-count 비율은 '무작위 역 쌍 기준'이다. 수요 proxy 는 승하차 기반 추정이며 실제 이동량이 아니다.",
        "",
        "## 1. 유형 분포",
        "",
        type_table(),
        "",
        f"- 수요 proxy 거리 감쇠 sensitivity (혼잡 이동 중 Structural): 감쇠 없음 {pct(sh_hot_w['structural'])}, "
        f"β=0.03 {pct(sh_hot_g3['structural'])}, β=0.06 {pct(sh_hot_g6['structural'])}",
        "",
        "기준 출발 시각별 (전체 대비 비율):",
        "",
        bt_tab.to_markdown(),
        "",
        "## 2. 가설 판정 (사전 등록 조건)",
        "",
        "| 가설 | 측정값 | 반증 조건 | 판정 |",
        "|---|---|---|---|",
        f"| H1 경로 변경은 일부에만 효과 | 혼잡 이동 중 Route 성공 {pct(h1)} | < 1% 또는 > 50% | {'유지' if h1_pass else '반증'} |",
        f"| H2 경로 실패 중 상당수는 시간으로 | Route 실패 혼잡 이동 중 Time 성공 {pct(h2)} | < 5% | {'유지' if h2_pass else '반증'} |",
        f"| H3 구조적 혼잡은 특정 구간에 집중 | 상위 10구간 비중 {pct(obs)} (순열 평균 {pct(np.nanmean(perm))}), p = {p3:.3f} | p ≥ 0.05 | "
        f"{'유지' if h3_pass else ('반증' if h3_pass is False else '판정 불가 (Structural 표본 부족)')} |",
        f"| H5 임계값 인공물 아님 | 단일 요인 {len(sens)}개 중 유지율 70% 미만 {n_fail}개 | 하나라도 미달 시 명시, 절반 이상 미달 시 격하 | {h5_verdict} |",
        "",
        "## 3. Structurally constrained 상세",
        "",
        "상위 구간 (Structural 이동의 100%+ 노출 합계):",
        "",
        (pd.DataFrame(top_edges, columns=["구간", "100%+ 노출 합계(분)", "해당 이동 수"]).round(1)
         .to_markdown(index=False) if top_edges else "(없음)"),
        "",
        "Structural 이 된 이유 (행 = Route 실패 사유, 열 = Time 실패 사유):",
        "",
        bind_tab.to_markdown() if len(bind_tab) else "(없음)",
        "",
        "## 4. Dual 에서 권장 수단",
        "",
        f"- Dual {len(dual):,}건 중 Route 권장 {pct((dual.recommended_means == 'route').mean()) if len(dual) else 'n/a'}, "
        f"Time 권장 {pct((dual.recommended_means == 'time').mean()) if len(dual) else 'n/a'}",
        "",
        "## 5. Sensitivity (H5)",
        "",
        "유지율 = T1 에서 혼잡 이동인 단위 중 같은 유형으로 남은 비율.",
        "",
        sens_md.to_markdown(index=False),
        "",
        f"- Monte Carlo 500회 (8개 요인 동시 무작위): 유지율 중앙값 {pct(np.median(mc[:, 0]))} "
        f"(5~95% {pct(np.percentile(mc[:, 0], 5))}~{pct(np.percentile(mc[:, 0], 95))}), "
        f"Structural 비율 중앙값 {pct(np.median(mc[:, 1]))} "
        f"(5~95% {pct(np.percentile(mc[:, 1], 5))}~{pct(np.percentile(mc[:, 1], 95))})",
        f"- 참고: T0 기준(최대혼잡 OR, Time ±90) 적용 시 혼잡 이동 중 "
        + ", ".join(f"{TYPE_KO[k]} {pct(v)}" for k, v in sh_t0.items()),
        "",
        "## 해석 원칙",
        "",
        "- 반증된 가설은 반증된 대로 보고한다. 기준을 바꿔 다시 돌리지 않는다 (사전 등록 §7).",
        "- 수치는 무작위 역 쌍 × 평일 피크 6개 시각 표본에서의 값이다. 서울 전체 이동량 비율로 일반화하지 않는다.",
    ]
    rdir = ROOT / "reports" / "v2"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "od_shiftability_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[9:]))
    print(f"\n저장: {rdir / 'od_shiftability_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
