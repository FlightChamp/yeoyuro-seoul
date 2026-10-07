"""
scripts/v2/23b_robustness.py
============================
v2.3b — 외부 교차검토 지적에 대한 강건성 검증. 기준: docs/v2/preregistration_v23b_robustness.md

R1 Holdout   : 새 seed(2026)·겹치지 않는 OD 600쌍으로 v2.3 재현
R2 Joint     : Structural 단위에서 경로와 시각을 함께 바꾸면 풀리는가
R3 K10       : Structural·Route 단위(+대조 150)를 K=10 으로 재평가해도 같은 유형인가
R4 w0        : 최초 승차 대기를 넣어도 유형이 유지되는가

각 단계 결과는 data/marts/v2/robust_*.pkl.gz 에 저장되어, 중간에 끊겨도 같은 명령으로 이어서 실행된다.

실행
----
    python scripts/v2/23b_robustness.py            # PC 코어 수에 따라 5~20분
    python scripts/v2/23b_robustness.py --workers 4

선행 조건
---------
    v2.3 evidence (data/marts/v2/od_shiftability_evidence.pkl.gz) — 23 스크립트 실행 시 생성
    preregistration_v23b_robustness.md 와 shift_rules.py 커밋

산출물
------
    reports/v2/robustness_report.md
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2.od_shiftability import TYPES, classify, is_oow              # noqa: E402
from yeoyuro_v2.shift_rules import T1, reduction_ok                         # noqa: E402

# 23 스크립트는 프로세스 안에서 한 번만 불러온다. 여러 스크립트가 각자 불러오면 sys.modules["s23"] 가
# 서로 다른 객체로 덮어써져 Windows(spawn) 멀티프로세싱에서 worker 함수를 찾지 못한다 (v2.6c 에서 발생).
if "s23" in sys.modules:
    S23 = sys.modules["s23"]
else:
    _spec = importlib.util.spec_from_file_location("s23", ROOT / "scripts" / "v2" / "23_build_od_shiftability.py")
    S23 = importlib.util.module_from_spec(_spec)
    sys.modules["s23"] = S23
    _spec.loader.exec_module(S23)
TAG = ""                          # --smoke 시 캐시 파일 이름 구분

PREREG = "docs/v2/preregistration_v23b_robustness.md"
PREREG_CODE = "yeoyuro_v2/shift_rules.py"
MDIR = ROOT / "data" / "marts" / "v2"
HOT_TYPES = [k for k in TYPES if k != "calm"]
TYPE_KO = {"calm": "Calm", "route_shiftable": "Route-shiftable", "time_shiftable": "Time-shiftable",
           "dual": "Dual", "structural": "Structural"}


# ---------------------------------------------------------------------- 공통
def check_prereg() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()
    dirty = git("status", "--porcelain", "--", PREREG, PREREG_CODE)
    log = git("log", "-1", "--format=%H|%cI", "--", PREREG)
    return {"ok": (not dirty) and bool(log), "dirty": dirty,
            "commit": log.split("|")[0] if log else None, "date": log.split("|")[1] if log else None}


def load(name):
    p = MDIR / f"robust{TAG}_{name}.pkl.gz"
    if p.exists():
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    return None


def save(name, obj):
    with gzip.open(MDIR / f"robust{TAG}_{name}.pkl.gz", "wb") as f:
        pickle.dump(obj, f)


def ukey(it):
    return (it["origin"], it["destination"], it["base_time"])


def classify_all(items):
    out = {}
    for it in items:
        c = classify(it, T1)
        if c["type"] != "invalid":
            out[ukey(it)] = (it, c)
    return out


def hypotheses(cl: dict, seed=11) -> dict:
    """v2.3 와 같은 정의로 H1·H2·H3 를 계산한다."""
    vals = list(cl.values())
    types = np.array([c["type"] for _, c in vals])
    hot = types != "calm"
    r_ok = np.array([bool(c["route_ok"]) for _, c in vals])
    t_ok = np.array([bool(c["time_ok"]) for _, c in vals])
    h1 = r_ok[hot].mean()
    rf = hot & ~r_ok
    h2 = t_ok[rf].mean() if rf.any() else np.nan
    hot_idx = np.where(hot)[0]
    edges = sorted({e for i in hot_idx for e in vals[i][0]["edge_exp100"]})
    eidx = {e: j for j, e in enumerate(edges)}
    M = np.zeros((len(hot_idx), len(edges)))
    for r, i in enumerate(hot_idx):
        for e, v in vals[i][0]["edge_exp100"].items():
            M[r, eidx[e]] = v
    lab = types[hot_idx] == "structural"

    def top10(mask):
        tot = M[mask].sum(axis=0)
        s = tot.sum()
        return np.sort(tot)[::-1][:10].sum() / s if s > 0 else np.nan

    n_s = int(lab.sum())
    obs = top10(lab) if n_s >= 2 else np.nan
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(1000):
        m = np.zeros(len(lab), bool)
        m[rng.choice(len(lab), n_s, replace=False)] = True
        perm.append(top10(m))
    p3 = float((np.array(perm) >= obs).mean()) if n_s >= 2 else np.nan
    tot_s = M[lab].sum(axis=0)
    top_edges = [edges[j] for j in np.argsort(tot_s)[::-1][:10] if tot_s[j] > 0]
    return {"n": len(vals), "n_hot": int(hot.sum()), "h1": h1, "h2": h2, "h3_obs": obs, "h3_p": p3,
            "h1_pass": 0.01 <= h1 <= 0.50, "h2_pass": h2 >= 0.05, "h3_pass": p3 < 0.05 if n_s >= 2 else None,
            "top_edges": top_edges, "types": types, "hot": hot,
            "od": np.array([f"{it['origin']}>{it['destination']}" for it, _ in vals])}


def share_diff_ci(a, b, n=1000, seed=3):
    """독립 두 표본 a(v2.3)·b(holdout)의 혼잡 이동 중 유형 비율 차이 (b − a), OD cluster bootstrap."""
    rng = np.random.default_rng(seed)

    def per_od(h):
        t, hot, od = h["types"][h["hot"]], h["hot"], h["od"][h["hot"]]
        ods = np.unique(od)
        mat = np.array([[np.sum((od == o) & (t == k)) for k in HOT_TYPES] for o in ods], float)
        return mat
    A, B = per_od(a), per_od(b)
    obs = B.sum(0) / B.sum() - A.sum(0) / A.sum()
    draws = []
    for _ in range(n):
        sa = A[rng.integers(0, len(A), len(A))].sum(0)
        sb = B[rng.integers(0, len(B), len(B))].sum(0)
        draws.append(sb / sb.sum() - sa / sa.sum())
    d = np.array(draws)
    return {k: (obs[j], np.percentile(d[:, j], 2.5), np.percentile(d[:, j], 97.5)) for j, k in enumerate(HOT_TYPES)}


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def pp(x):
    return "n/a" if x is None or np.isnan(x) else f"{100 * x:+.1f}%p"


# ---------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--allow-uncommitted", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="개발용 소규모 점검 (결과 파일 이름이 달라짐)")
    args = ap.parse_args(argv)
    global TAG
    TAG = "_smoke" if args.smoke else ""
    n_hold, n_ctrl = (8, 3) if args.smoke else (600, 75)
    t0 = time.time()

    pr = check_prereg()
    if not pr["ok"] and not args.allow_uncommitted:
        print("[중단] 사전 등록 부록이 커밋되지 않았습니다.")
        print(f"  {PREREG}, {PREREG_CODE} 를 먼저 커밋하세요. 변경 상태: {pr['dirty'] or '(커밋 이력 없음)'}")
        return 2
    print(f"사전 등록 부록: {'확인 ' + pr['commit'][:8] + ' (' + pr['date'] + ')' if pr['ok'] else '미확인'}")

    evp = MDIR / "od_shiftability_evidence.pkl.gz"
    if not evp.exists():
        print("[중단] v2.3 evidence 가 없습니다. 먼저 python scripts/v2/23_build_od_shiftability.py 를 실행하세요.")
        return 1
    with gzip.open(evp, "rb") as f:
        main_items = pickle.load(f)
    main_cl = classify_all(main_items)
    main_ods = sorted({(o, d) for o, d, _ in main_cl})

    disp = pd.read_csv(ROOT / "data" / "master" / "station_display_master.csv")
    keys = disp.station_key.tolist()

    # -------------------------------------------------- R1 Holdout
    hold_items = load("holdout")
    if hold_items is None:
        used = set(main_ods)
        rnd = random.Random(2026)
        hold = set()
        while len(hold) < n_hold:
            a, b = rnd.sample(keys, 2)
            if (a, b) not in used:
                hold.add((a, b))
        tasks = [(o, d, "holdout") for o, d in sorted(hold)]
        print(f"[R1] holdout evidence: OD {len(tasks)}개 (v2.3 과 겹침 0), workers={args.workers}")
        hold_items = S23.collect(tasks, args.workers, "weekday", checkpoint=MDIR / f"robust{TAG}_holdout.partial.pkl.gz")
        save("holdout", hold_items)
        (MDIR / f"robust{TAG}_holdout.partial.pkl.gz").unlink(missing_ok=True)
    hold_cl = classify_all(hold_items)
    assert not ({(o, d) for o, d, _ in hold_cl} & set(main_ods)), "holdout 이 v2.3 표본과 겹칩니다"
    Hm, Hh = hypotheses(main_cl), hypotheses(hold_cl)
    diffs = share_diff_ci(Hm, Hh)
    corridor = Hm["top_edges"][:7]
    n_cor = sum(e in Hh["top_edges"] for e in corridor)
    r1 = {"verdict_same": (Hm["h1_pass"], Hm["h2_pass"], Hm["h3_pass"]) == (Hh["h1_pass"], Hh["h2_pass"], Hh["h3_pass"]),
          "struct_ci0": diffs["structural"][1] <= 0 <= diffs["structural"][2], "corridor": n_cor}
    r1["pass"] = r1["verdict_same"] and r1["struct_ci0"] and n_cor >= 4
    print(f"[R1] 완료 ({time.time() - t0:.0f}s)")

    # -------------------------------------------------- R2 Joint
    joint = load("joint")
    if joint is None:
        from yeoyuro_v2 import load_v1
        from yeoyuro_v2.time_dependent import TDRouter, parse_hhmm
        mod, sr, disp_ = load_v1(ROOT)
        router = TDRouter(ROOT, mod, sr, disp_, "weekday", policy="step", k=5, multi_bin=True)
        joint = []
        targets = [("v2.3", k, v) for k, v in main_cl.items() if v[1]["type"] == "structural"] + \
                  [("holdout", k, v) for k, v in hold_cl.items() if v[1]["type"] == "structural"]
        print(f"[R2] joint 검사: Structural {len(targets)}단위")
        for src, (o, d, bt), (it, _c) in targets:
            b = it["base"]
            tb = parse_hhmm(bt)
            best = None
            for s in (-60, -30, 30, 60):
                for r in router.evaluate_od(o, d, tb + s):
                    if r.out_of_window:
                        continue
                    ok = (reduction_ok(b["exp100"], r.exposure[100], b["max"], r.max_congestion, T1)
                          and r.actual_time_min - b["actual"] <= T1.max_time_loss_min
                          and r.perceived_time_min - b["perceived"] <= T1.max_perceived_excess_min
                          and r.transfer_count - b["transfers"] <= T1.max_extra_transfer)
                    if ok:
                        drop = b["exp100"] - r.exposure[100]
                        if best is None or (abs(s), -drop) < (abs(best[0]), -best[1]):
                            best = (s, drop, r.path != it["base_path"])
            joint.append({"src": src, "origin": o, "destination": d, "base_time": bt,
                          "joint_ok": best is not None, "shift": best[0] if best else None,
                          "drop": best[1] if best else None, "route_changed": best[2] if best else None})
        save("joint", joint)
    jd = pd.DataFrame(joint)
    print(f"[R2] 완료 ({time.time() - t0:.0f}s)")

    # -------------------------------------------------- R3 K10
    k10 = load("k10")
    if k10 is None:
        rng = random.Random(7)
        sel = {}
        for src, cl in (("v2.3", main_cl), ("holdout", hold_cl)):
            focus = [k for k, v in cl.items() if v[1]["type"] in ("structural", "route_shiftable")]
            others = [k for k, v in cl.items() if v[1]["type"] not in ("structural", "route_shiftable")]
            ctrl = rng.sample(sorted(others), min(n_ctrl, len(others)))
            sel[src] = {"focus": focus, "control": ctrl}
        ods = sorted({(k[0], k[1]) for s in sel.values() for g in s.values() for k in g})
        print(f"[R3] K=10 evidence: OD {len(ods)}개")
        items = S23.collect([(o, d, "k10") for o, d in ods], args.workers, "weekday", k=10,
                            checkpoint=MDIR / f"robust{TAG}_k10.partial.pkl.gz")
        k10 = {"sel": sel, "items": items}
        save("k10", k10)
        (MDIR / f"robust{TAG}_k10.partial.pkl.gz").unlink(missing_ok=True)
    k10_cl = {ukey(it): classify(it, T1) for it in k10["items"]}
    rows = []
    for src, cl in (("v2.3", main_cl), ("holdout", hold_cl)):
        for grp, ks in k10["sel"][src].items():
            for k in ks:
                rows.append({"src": src, "group": grp, "type_k5": cl[k][1]["type"],
                             "type_k10": k10_cl[k]["type"] if k in k10_cl else "missing"})
    kd = pd.DataFrame(rows)
    kd["same"] = kd.type_k5 == kd.type_k10
    focus_agree = kd[kd.group == "focus"].same.mean()
    print(f"[R3] 완료 ({time.time() - t0:.0f}s)")

    # -------------------------------------------------- R4 w0
    w0_items = load("w0")
    if w0_items is None:
        tasks = [(o, d, "w0") for o, d in (main_ods[:8] if args.smoke else main_ods)]
        print(f"[R4] 최초 대기 포함 evidence: OD {len(tasks)}개")
        w0_items = S23.collect(tasks, args.workers, "weekday", w0=True,
                               checkpoint=MDIR / f"robust{TAG}_w0.partial.pkl.gz")
        save("w0", w0_items)
        (MDIR / f"robust{TAG}_w0.partial.pkl.gz").unlink(missing_ok=True)
    w0_cl = classify_all(w0_items)
    common = [k for k in main_cl if k in w0_cl and main_cl[k][1]["type"] != "calm"]
    w0_keep = np.mean([main_cl[k][1]["type"] == w0_cl[k][1]["type"] for k in common])
    Hw = hypotheses(w0_cl)
    w0_trans = pd.crosstab(pd.Series([TYPE_KO[main_cl[k][1]["type"]] for k in common], name="w0 미포함"),
                           pd.Series([TYPE_KO.get(w0_cl[k][1]["type"], w0_cl[k][1]["type"]) for k in common],
                                     name="w0 포함"))
    r4_pass = w0_keep >= 0.90 and (Hm["h1_pass"], Hm["h2_pass"]) == (Hw["h1_pass"], Hw["h2_pass"])
    print(f"[R4] 완료 ({time.time() - t0:.0f}s)")

    # -------------------------------------------------- 리포트
    def tshare(h):
        t = h["types"][h["hot"]]
        return {k: (t == k).mean() for k in HOT_TYPES}
    sm, sh = tshare(Hm), tshare(Hh)
    rep_rows = [{"유형 (혼잡 이동 중)": TYPE_KO[k], "v2.3": pct(sm[k]), "holdout": pct(sh[k]),
                 "차이": pp(diffs[k][0]), "95% 구간": f"{pp(diffs[k][1])} ~ {pp(diffs[k][2])}"} for k in HOT_TYPES]
    hy = pd.DataFrame([
        {"가설": "H1 Route 성공률 1~50%", "v2.3": pct(Hm["h1"]), "holdout": pct(Hh["h1"]),
         "v2.3 판정": "유지" if Hm["h1_pass"] else "반증", "holdout 판정": "유지" if Hh["h1_pass"] else "반증"},
        {"가설": "H2 Route 실패 중 Time 성공 ≥ 5%", "v2.3": pct(Hm["h2"]), "holdout": pct(Hh["h2"]),
         "v2.3 판정": "유지" if Hm["h2_pass"] else "반증", "holdout 판정": "유지" if Hh["h2_pass"] else "반증"},
        {"가설": "H3 Structural 구간 집중 (p<0.05)", "v2.3": f"{pct(Hm['h3_obs'])}, p={Hm['h3_p']:.3f}",
         "holdout": f"{pct(Hh['h3_obs'])}, p={Hh['h3_p']:.3f}",
         "v2.3 판정": "유지" if Hm["h3_pass"] else "반증", "holdout 판정": "유지" if Hh["h3_pass"] else "반증"}])
    jsum = jd.groupby("src").agg(Structural=("joint_ok", "size"), Joint로_풀림=("joint_ok", "sum")).reset_index()
    jsum["비율"] = (jsum.Joint로_풀림 / jsum.Structural).map(pct)
    j_rate = jd.joint_ok.mean() if len(jd) else np.nan
    ksum = kd.groupby(["src", "group"]).agg(단위=("same", "size"), 일치율=("same", "mean")).reset_index()
    ksum["일치율"] = ksum["일치율"].map(pct)
    kchg = kd[~kd.same].groupby(["type_k5", "type_k10"]).size().rename("단위").reset_index()

    L = [
        "# v2.3b 강건성 검증 리포트",
        "",
        f"- 사전 등록 부록: `{PREREG}` " + (f"커밋 `{pr['commit'][:10]}` ({pr['date']})" if pr["ok"] else "**미확인**"),
        f"- 기준 `{T1.set_id}` (변경 없음) · 실행 시간 {(time.time() - t0) / 60:.1f}분",
        "- 이 리포트는 v2.3 주 결과를 바꾸지 않는다. 강건성 보고다.",
        "",
        "## 요약",
        "",
        "| 검사 | 결과 | 사전 등록 기준 | 판정 |",
        "|---|---|---|---|",
        f"| R1 Holdout 재현 | 판정 동일 {'예' if r1['verdict_same'] else '아니오'}, Structural 차이 구간에 0 포함 "
        f"{'예' if r1['struct_ci0'] else '아니오'}, 회랑 {n_cor}/7 | 셋 모두 (회랑 ≥ 4) | {'재현' if r1['pass'] else '미재현'} |",
        f"| R2 Joint Shift | Structural {len(jd)}단위 중 {int(jd.joint_ok.sum())}단위 풀림 ({pct(j_rate)}) | ≥ 50% 면 표기 변경 | "
        f"{'표기 변경 필요' if j_rate >= 0.5 else '표기 유지'} |",
        f"| R3 K=10 | Structural·Route 단위 일치 {pct(focus_agree)} | ≥ 90% | {'K5 유지' if focus_agree >= 0.9 else 'v2.6 부터 K10'} |",
        f"| R4 최초 대기 포함 | 혼잡 이동 유형 유지 {pct(w0_keep)}, H1·H2 판정 "
        f"{'동일' if (Hm['h1_pass'], Hm['h2_pass']) == (Hw['h1_pass'], Hw['h2_pass']) else '다름'} | ≥ 90% & 동일 | "
        f"{'w0 미포함 유지' if r4_pass else 'v2.6 부터 w0 포함'} |",
        "",
        "## R1. Holdout (seed 2026, v2.3 과 겹치는 OD 0)",
        "",
        f"- 단위: v2.3 {Hm['n']:,} (혼잡 {Hm['n_hot']:,}) / holdout {Hh['n']:,} (혼잡 {Hh['n_hot']:,})",
        "",
        pd.DataFrame(rep_rows).to_markdown(index=False),
        "",
        hy.to_markdown(index=False),
        "",
        "Structural 상위 구간 비교:",
        "",
        pd.DataFrame({"순위": range(1, 11),
                      "v2.3": (Hm["top_edges"] + [""] * 10)[:10],
                      "holdout": (Hh["top_edges"] + [""] * 10)[:10]}).to_markdown(index=False),
        "",
        "## R2. Joint Shift (Structural 단위에서 경로와 시각을 함께 바꾸기)",
        "",
        jsum.to_markdown(index=False) if len(jsum) else "(Structural 없음)",
        "",
        (f"- 풀린 단위의 최소 이동폭 분포: {jd[jd.joint_ok]['shift'].value_counts().sort_index().to_dict()}, "
         f"경로가 바뀐 비율 {pct(jd[jd.joint_ok].route_changed.mean())}") if jd.joint_ok.any() else "- 풀린 단위 없음",
        "",
        "## R3. K=10 재평가",
        "",
        ksum.to_markdown(index=False),
        "",
        ("바뀐 단위 (K5 → K10):\n\n" + kchg.to_markdown(index=False)) if len(kchg) else "바뀐 단위 없음",
        "",
        "## R4. 최초 승차 대기 포함",
        "",
        f"- 혼잡 이동 {len(common):,}단위의 유형 유지율 {pct(w0_keep)}",
        f"- H1 {pct(Hm['h1'])} → {pct(Hw['h1'])}, H2 {pct(Hm['h2'])} → {pct(Hw['h2'])}, "
        f"H3 p {Hm['h3_p']:.3f} → {Hw['h3_p']:.3f}",
        "",
        w0_trans.to_markdown(),
        "",
        "## 해석 원칙",
        "",
        "- 기준 미달은 그대로 보고하고, 사전 등록 부록에 적은 후속 조치만 따른다.",
        "- holdout 도 균등 무작위 OD × 평일 피크 6개 시각이다. 이동량 기준 일반화는 하지 않는다.",
    ]
    out = ROOT / "reports" / "v2" / ("robustness_report_smoke.md" if args.smoke else "robustness_report.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[6:]))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
