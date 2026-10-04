"""
scripts/v2/25_event_stress_test.py
==================================
v2.5 — Event Stress Test: 평시 Route/Time/Structural 유형이 이벤트 시나리오에서 어떻게 바뀌는가.
사전 등록: docs/v2/preregistration_v25.md (분류 기준은 v2.3 T1 그대로)

이벤트 혼잡은 관측값이 아니라 v1 배수 가정(λ=0.3)으로 만든 시나리오다. λ = 0.15 / 0.6 도 함께 계산한다.

실행
----
    python scripts/v2/25_event_stress_test.py              # 기본 (PC 코어 수에 따라 3~15분)
    python scripts/v2/25_event_stress_test.py --reuse-evidence

산출물
------
    data/marts/v2/event_scenario_evidence.pkl.gz   (커밋 안 함)
    data/marts/v2/event_scenario_mart.parquet      (커밋, 앱 '이벤트 시나리오' 화면용)
    reports/v2/event_stress_test_report.md
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
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2.od_shiftability import TYPES, build_evidence, classify   # noqa: E402
from yeoyuro_v2.shift_rules import T1                                    # noqa: E402

PREREG = "docs/v2/preregistration_v25.md"
PREREG_CODE = "yeoyuro_v2/shift_rules.py"
EVENTS = [  # 사전 등록 §1
    {"event_id": "FW_2025", "name": "서울세계불꽃축제 2025", "date": "2025-09-27",
     "times": ["17:00", "18:00", "19:00"], "anchors": ["여의나루"]},
    {"event_id": "NYB_2025", "name": "제야의 종 2025", "date": "2025-12-31",
     "times": ["21:00", "22:00", "23:00"], "anchors": ["종각"]},
    {"event_id": "HW_2025", "name": "핼러윈 2025", "date": "2025-10-31",
     "times": ["19:00", "20:00", "21:00"], "anchors": ["이태원", "홍대입구"]},
    {"event_id": "CB_2025", "name": "여의도 벚꽃 2025", "date": "2025-04-11",
     "times": ["12:00", "15:00", "18:00"], "anchors": ["여의나루"]},
]
SCALES = {"0.15": 0.5, "0.3": 1.0, "0.6": 2.0}      # λ -> v1(λ=0.3) 배수 대비 배율
SHIFTS = (-60, -30, 30, 60)
TYPE_KO = {"calm": "Calm", "route_shiftable": "Route", "time_shiftable": "Time",
           "dual": "Dual", "structural": "Structural"}


def check_prereg() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()
    dirty = git("status", "--porcelain", "--", PREREG, PREREG_CODE)
    log = git("log", "-1", "--format=%H|%cI", "--", PREREG)
    return {"ok": (not dirty) and bool(log), "dirty": dirty,
            "commit": log.split("|")[0] if log else None, "date": log.split("|")[1] if log else None}


def day_type_of(date: str) -> str:
    d = pd.Timestamp(date).dayofweek
    return "saturday" if d == 5 else ("sunday" if d == 6 else "weekday")


# ---------------------------------------------------------------------- worker
_W = {}


def _init_worker():
    from yeoyuro_v2 import load_v1
    _W["v1"] = load_v1(ROOT)
    _W["routers"] = {}


def _router(day_type, qdate, scale):
    from yeoyuro_v2.time_dependent import TDRouter
    k = (day_type, qdate, scale)
    if k not in _W["routers"]:
        mod, sr, disp = _W["v1"]
        _W["routers"][k] = TDRouter(ROOT, mod, sr, disp, day_type, policy="step", k=5,
                                    multi_bin=True, query_date=qdate, event_scale=scale)
    return _W["routers"][k]


def _touched(ev, res) -> bool:
    """경로가 배수 > 1 인 구간을 지나는가 (구간 진입 시각 기준)."""
    for t in res.trace:
        if t.kind == "ride" and ev._event_mult(ev.edge_meta[(t.u, t.v)]["from_station"], t.t_start) > 1.0 + 1e-9:
            return True
    return False


def _evidence(router, o, d, tb, cache):
    from yeoyuro_v2.time_dependent import bin_index_of, N_BINS

    def fastest_at(t):
        if t not in cache:
            cache[t] = router.evaluate_od(o, d, t)
        return cache[t]
    res = fastest_at(tb)
    if not res:
        return None, None, None
    ev = router.evaluator(bin_index_of(tb))
    base = ev.evaluate(res[0].path, tb, keep_trace=True)
    rows = []
    for s in SHIFTS:
        t = tb + s
        if not (0 <= bin_index_of(t) < N_BINS):
            continue
        rows.append((s, True, ev.evaluate(base.path, t)))
        f = fastest_at(t)
        if f and f[0].path != base.path:
            rows.append((s, False, f[0]))
    return build_evidence(base, res[1:], rows), base, ev


def _work(task):
    from yeoyuro_v2.time_dependent import parse_hhmm
    eid, date, times, o, d, group = task
    dt = day_type_of(date)
    out = []
    caches = {k: {} for k in ["base", *SCALES]}
    for bt in times:
        tb = parse_hhmm(bt)
        evb, base_b, _ = _evidence(_router(dt, None, 1.0), o, d, tb, caches["base"])
        if evb is None:
            continue
        unit = {"event_id": eid, "origin": o, "destination": d, "group": group, "base_time": bt,
                "baseline": evb, "scenario": {}}
        for lam, sc in SCALES.items():
            ev_e, base_e, evaluator = _evidence(_router(dt, date, sc), o, d, tb, caches[lam])
            if ev_e is None:
                continue
            # 직접 영향: 이벤트 기준 경로 또는 평시 기준 경로가 배수 > 1 구간을 지남 (사전 등록 §3)
            base_b_under_e = evaluator.evaluate(base_b.path, tb, keep_trace=True)
            ev_e["treated"] = _touched(evaluator, base_e) or _touched(evaluator, base_b_under_e)
            unit["scenario"][lam] = ev_e
        out.append(unit)
    return out


# ---------------------------------------------------------------------- 분석 도우미
def paired_boot(od_ids, x_base, x_event, n=1000, seed=5):
    rng = np.random.default_rng(seed)
    ods = np.unique(od_ids)
    idx = {o: np.where(od_ids == o)[0] for o in ods}
    sb = np.array([x_base[idx[o]].sum() for o in ods])
    se = np.array([x_event[idx[o]].sum() for o in ods])
    cnt = np.array([len(idx[o]) for o in ods])
    diffs = []
    for _ in range(n):
        k = rng.integers(0, len(ods), len(ods))
        c = cnt[k].sum()
        diffs.append((se[k].sum() - sb[k].sum()) / c)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def pp(x):
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:+.1f}%p"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-background", type=int, default=150)
    ap.add_argument("--n-visitor", type=int, default=50, help="방향별 방문 이동 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--reuse-evidence", action="store_true")
    ap.add_argument("--allow-uncommitted", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()

    pr = check_prereg()
    if not pr["ok"] and not args.allow_uncommitted:
        print("[중단] 사전 등록이 커밋되지 않았습니다.")
        print(f"  {PREREG}, {PREREG_CODE} 를 먼저 커밋하세요. 변경 상태: {pr['dirty'] or '(커밋 이력 없음)'}")
        return 2
    print(f"사전 등록: {'확인 ' + pr['commit'][:8] + ' (' + pr['date'] + ')' if pr['ok'] else '미확인'}")

    mdir = ROOT / "data" / "marts" / "v2"
    evp = mdir / "event_scenario_evidence.pkl.gz"
    disp = pd.read_csv(ROOT / "data" / "master" / "station_display_master.csv")
    keys = disp.station_key.tolist()

    tasks = []
    for i, e in enumerate(EVENTS):
        rnd = random.Random(args.seed)
        bg = set()
        while len(bg) < args.n_background:
            a, b = rnd.sample(keys, 2)
            bg.add((a, b))
        rv = random.Random(args.seed + i + 1)
        vis = set()
        per = max(1, args.n_visitor // len(e["anchors"]))
        for anc in e["anchors"]:
            others = [k for k in keys if k not in e["anchors"]]
            for o in rv.sample(others, per):
                vis.add((o, anc))
            for d in rv.sample(others, per):
                vis.add((anc, d))
        for o, d in sorted(bg):
            tasks.append((e["event_id"], e["date"], e["times"], o, d, "background"))
        for o, d in sorted(vis - bg):
            tasks.append((e["event_id"], e["date"], e["times"], o, d, "visitor"))

    if args.reuse_evidence and evp.exists():
        with gzip.open(evp, "rb") as f:
            units = pickle.load(f)
        print(f"evidence 재사용: {len(units):,}단위")
    else:
        # 체크포인트: 50작업마다 저장. 중간에 끊기면 같은 명령으로 다시 실행하면 이어서 한다.
        part = mdir / "event_scenario_evidence.partial.pkl.gz"
        done = {}
        if part.exists():
            with gzip.open(part, "rb") as f:
                done = pickle.load(f)
            print(f"체크포인트에서 이어서: {len(done)}/{len(tasks)} 작업 완료 상태")
        todo = [t for t in tasks if (t[0], t[3], t[4]) not in done]
        print(f"[1/3] evidence 수집: 작업 {len(todo)}/{len(tasks)}개 (이벤트 {len(EVENTS)} × OD), workers={args.workers}")

        def save():
            with gzip.open(part, "wb") as f:
                pickle.dump(done, f)

        if args.workers <= 1:
            _init_worker()
            it = ((t, _work(t)) for t in todo)
        else:
            pool = Pool(args.workers, initializer=_init_worker)
            it = zip(todo, pool.imap(_work, todo, chunksize=4))
        for i, (t, u) in enumerate(it, 1):
            done[(t[0], t[3], t[4])] = u
            if i % 50 == 0:
                save()
                print(f"  ... {len(done)}/{len(tasks)}  ({time.time() - t0:.0f}s)", flush=True)
        if args.workers > 1:
            pool.close()
            pool.join()
        units = [x for t in tasks for x in done[(t[0], t[3], t[4])]]
        units.sort(key=lambda u: (u["event_id"], u["group"], u["origin"], u["destination"], u["base_time"]))
        with gzip.open(evp, "wb") as f:
            pickle.dump(units, f)
        part.unlink(missing_ok=True)
    t_ev = time.time() - t0

    # ------------------------------------------------------------------ 분류
    print("[2/3] 분류")
    rows = []
    for u in units:
        cb = classify(u["baseline"], T1)
        r = {"event_id": u["event_id"], "origin": u["origin"], "destination": u["destination"],
             "group": u["group"], "base_time": u["base_time"],
             "baseline_type": cb["type"], "baseline_time_ok": bool(cb["time_ok"]),
             "baseline_route_ok": bool(cb["route_ok"]),
             "baseline_exposure_100_min": round(u["baseline"]["base"]["exp100"], 2),
             "baseline_max_congestion": round(u["baseline"]["base"]["max"], 1)}
        for lam in SCALES:
            s = u["scenario"].get(lam)
            if s is None:
                continue
            ce = classify(s, T1)
            tag = lam.replace(".", "")
            r.update({f"event_type_l{tag}": ce["type"], f"event_time_ok_l{tag}": bool(ce["time_ok"]),
                      f"event_route_ok_l{tag}": bool(ce["route_ok"]), f"treated_l{tag}": bool(s["treated"]),
                      f"event_exposure_100_min_l{tag}": round(s["base"]["exp100"], 2),
                      f"event_max_congestion_l{tag}": round(s["base"]["max"], 1)})
        rows.append(r)
    df = pd.DataFrame(rows)
    main_tag = "03"
    df["event_shiftability_type"] = df[f"event_type_l{main_tag}"]
    df["treated"] = df[f"treated_l{main_tag}"]
    df["type_transition"] = df.baseline_type + "→" + df.event_shiftability_type
    df["event_sensitive_flag"] = df.treated & (df.baseline_type != df.event_shiftability_type)
    df["threshold_set_id"] = T1.set_id
    df["lambda_main"] = 0.3
    df.to_parquet(mdir / "event_scenario_mart.parquet", index=False)

    # ------------------------------------------------------------------ 판정
    print("[3/3] 판정")
    def h4(tag):
        pop = df[df[f"treated_l{tag}"] & (df.baseline_type != "calm")]
        n = len(pop)
        if n < 30:
            return {"n": n, "ok": False}
        ids = (pop.event_id + "|" + pop.origin + ">" + pop.destination).values
        tb_ = pop.baseline_time_ok.values.astype(float)
        te_ = pop[f"event_time_ok_l{tag}"].values.astype(float)
        sb_ = (pop.baseline_type == "structural").values.astype(float)
        se_ = (pop[f"event_type_l{tag}"] == "structural").values.astype(float)
        return {"n": n, "ok": True, "time_b": tb_.mean(), "time_e": te_.mean(), "d_time": te_.mean() - tb_.mean(),
                "ci_time": paired_boot(ids, tb_, te_), "str_b": sb_.mean(), "str_e": se_.mean(),
                "d_str": se_.mean() - sb_.mean(), "ci_str": paired_boot(ids, sb_, se_)}

    res = {lam: h4(lam.replace(".", "")) for lam in SCALES}
    m = res["0.3"]
    if not m["ok"]:
        h4a = h4b = "판정 불가 (모집단 < 30)"
    else:
        h4a = "유지" if (m["d_time"] <= -0.05 and m["ci_time"][1] < 0) else "반증"
        h4b = "유지" if (m["d_str"] >= 0.02 and m["ci_str"][0] > 0) else "반증"

    trt = df[df.treated]
    unt = df[~df.treated]
    order = [TYPE_KO[k] for k in TYPES]
    L = [
        "# v2.5 Event Stress Test 리포트",
        "",
        f"- 사전 등록: `{PREREG}` "
        + (f"커밋 `{pr['commit'][:10]}` ({pr['date']}) 기준" if pr["ok"] else "**미확인**"),
        f"- 분류 기준 `{T1.set_id}` · 이벤트 혼잡 = v1 배수 가정 (λ = 0.3 판정, 0.15·0.6 sensitivity). **관측값 아님.**",
        f"- 단위 {len(df):,}개 (이벤트 {len(EVENTS)}개 × OD × 기준 시각 3개), 직접 영향 {int(df.treated.sum()):,}개",
        f"- 실행 시간: evidence {t_ev / 60:.1f}분, 전체 {(time.time() - t0) / 60:.1f}분",
        "",
        "## 1. 가설 판정 (λ = 0.3, 직접 영향 & 평시 혼잡 이동)",
        "",
    ]
    if m["ok"]:
        L += [
            f"- 모집단 {m['n']:,}단위",
            "",
            "| 가설 | 평시 | 이벤트 | 차이 | 95% 구간 | 반증 조건 | 판정 |",
            "|---|---|---|---|---|---|---|",
            f"| H4a Time 성공률 감소 | {pct(m['time_b'])} | {pct(m['time_e'])} | {pp(m['d_time'])} | "
            f"{pp(m['ci_time'][0])} ~ {pp(m['ci_time'][1])} | 감소 < 5%p 또는 상한 ≥ 0 | {h4a} |",
            f"| H4b Structural 비율 증가 | {pct(m['str_b'])} | {pct(m['str_e'])} | {pp(m['d_str'])} | "
            f"{pp(m['ci_str'][0])} ~ {pp(m['ci_str'][1])} | 증가 < 2%p 또는 하한 ≤ 0 | {h4b} |",
        ]
    else:
        L += [f"- 모집단 {m['n']}단위 < 30 → H4a {h4a}, H4b {h4b}"]
    L += ["", "λ sensitivity (판정에 쓰지 않음):", "",
          "| λ | 모집단 | Time 성공률 차이 | Structural 비율 차이 |", "|---|---|---|---|"]
    for lam, r in res.items():
        L.append(f"| {lam} | {r['n']} | {pp(r.get('d_time', np.nan)) if r['ok'] else 'n/a'} | "
                 f"{pp(r.get('d_str', np.nan)) if r['ok'] else 'n/a'} |")
    L += ["", "## 2. 이벤트별 요약 (λ = 0.3)", ""]
    summ = []
    for e in EVENTS:
        g = df[df.event_id == e["event_id"]]
        gt = g[g.treated]
        summ.append({"이벤트": e["name"], "날짜": e["date"], "단위": len(g), "직접 영향": len(gt),
                     "평시 혼잡(직접)": int((gt.baseline_type != "calm").sum()),
                     "이벤트 혼잡(직접)": int((gt.event_shiftability_type != "calm").sum()),
                     "유형 변화(직접)": f"{100 * (gt.baseline_type != gt.event_shiftability_type).mean():.1f}%"
                     if len(gt) else "n/a",
                     "이벤트 Structural(직접)": int((gt.event_shiftability_type == "structural").sum())})
    L += [pd.DataFrame(summ).to_markdown(index=False), ""]
    L += ["## 3. 유형 전이 행렬 (직접 영향 단위, 행 = 평시, 열 = 이벤트, λ = 0.3)", ""]
    for e in EVENTS:
        g = trt[trt.event_id == e["event_id"]]
        if g.empty:
            continue
        ct = pd.crosstab(g.baseline_type.map(TYPE_KO), g.event_shiftability_type.map(TYPE_KO))
        ct = ct.reindex(index=[x for x in order if x in ct.index], columns=[x for x in order if x in ct.columns])
        L += [f"**{e['name']}**", "", ct.to_markdown(), ""]
    newly = trt[(trt.baseline_type == "calm") & (trt.event_shiftability_type != "calm")]
    L += ["## 4. 평시 Calm → 이벤트 혼잡 단위의 유형 (λ = 0.3)", "",
          (newly.event_shiftability_type.map(TYPE_KO).value_counts().to_frame("단위 수").to_markdown()
           if len(newly) else "(없음)"), "",
          "## 5. 간접 단위 (이벤트 구간을 지나지 않는 이동)", "",
          f"- {len(unt):,}단위 중 유형이 바뀐 비율 {pct((unt.baseline_type != unt.event_shiftability_type).mean())} "
          "(대안 경로가 이벤트 역을 지나며 대안이 사라진 경우)", "",
          "## 해석 원칙", "",
          "- 이벤트 혼잡은 승하차 급증을 배수로 바꾼 시나리오다. 결론은 'λ=0.3 가정 아래'로만 쓴다.",
          "- 반증·판정 불가는 그대로 보고한다 (사전 등록 §5)."]
    rdir = ROOT / "reports" / "v2"
    (rdir / "event_stress_test_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[6:]))
    print(f"\n저장: {rdir / 'event_stress_test_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
