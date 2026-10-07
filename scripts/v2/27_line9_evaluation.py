"""
scripts/v2/27_line9_evaluation.py
=================================
v2.6 Phase B — 9호선 통합 평가. 기준: docs/v2/preregistration_v26.md (Phase A 커밋에 포함)

B1 (먼저) 최초 승차 대기 포함/미포함 → v2.6 기본값 결정 (D-003 재검토)
B2 환승 계수 ×1.00 / ×1.60 / ×2.78 (기본 ×1.81)
B3 9호선 혼잡 max_3y
B4 이벤트 배수의 9호선 적용/미적용 (불꽃축제·벚꽃)
B5 기본값으로 A군(9호선 관련)·B군(전체) 분류, v2.3 결론 유지 여부

표본: seed 2027, v2.0~v2.3b 에서 쓴 OD 와 겹치지 않음. A군 300 (출발 또는 도착이 9호선 역), B군 300 (270역 균등).
B4 표본은 A군 앞쪽 100쌍 (사전 등록에 표본 크기가 없어 실행 전 이 문서에서 고정).
각 단계 결과는 data/marts/v2/l9_*.pkl.gz 로 캐시되어, 끊겨도 같은 명령으로 이어서 실행된다.

실행
----
    python scripts/v2/27_line9_evaluation.py              # 8 worker 기준 약 10~15분
    python scripts/v2/27_line9_evaluation.py --workers 4

산출물
------
    reports/v2/line9_evaluation_report.md
    data/marts/v2/line9_shiftability_mart.parquet
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

from yeoyuro_v2 import line9 as L9                                   # noqa: E402
from yeoyuro_v2.od_shiftability import TYPES, classify               # noqa: E402
from yeoyuro_v2.shift_rules import T1                                 # noqa: E402

# 23 스크립트는 프로세스 안에서 한 번만 불러온다. 여러 스크립트가 각자 불러오면 sys.modules["s23"] 가
# 서로 다른 객체로 덮어써져 Windows(spawn) 멀티프로세싱에서 worker 함수를 찾지 못한다 (v2.6c 에서 발생).
if "s23" in sys.modules:
    S23 = sys.modules["s23"]
else:
    _spec = importlib.util.spec_from_file_location("s23", ROOT / "scripts" / "v2" / "23_build_od_shiftability.py")
    S23 = importlib.util.module_from_spec(_spec)
    sys.modules["s23"] = S23
    _spec.loader.exec_module(S23)

PREREG = "docs/v2/preregistration_v26.md"
MDIR = ROOT / "data" / "marts" / "v2"
SROOT = ROOT / "data" / "interim" / "v26"
TAG = ""
HOT = [k for k in TYPES if k != "calm"]
EVENTS = [("FW_2025", "2025-09-27", ("17:00", "18:00", "19:00")),
          ("CB_2025", "2025-04-11", ("12:00", "15:00", "18:00"))]


def check_prereg():
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True,
                              encoding="utf-8").stdout.strip()
    dirty = git("status", "--porcelain", "--", PREREG)
    log = git("log", "-1", "--format=%H|%cI", "--", PREREG)
    return {"ok": (not dirty) and bool(log), "commit": log.split("|")[0] if log else None,
            "date": log.split("|")[1] if log else None, "dirty": dirty}


def cached(name, fn):
    p = MDIR / f"l9{TAG}_{name}.pkl.gz"
    if p.exists():
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    obj = fn()
    with gzip.open(p, "wb") as f:
        pickle.dump(obj, f)
    for q in MDIR.glob(f"l9{TAG}_{name}.partial*.pkl.gz"):
        q.unlink(missing_ok=True)
    return obj


def run(tasks, workers, scenario, name, **kw):
    root = SROOT / scenario
    if not (root / "data" / "marts" / "route_edge_mart.parquet").exists():
        L9.build_shadow_root(ROOT, root, scenario)
    return cached(name, lambda: S23.collect(tasks, workers, kw.pop("day_type", "weekday"), root=str(root),
                                            checkpoint=MDIR / f"l9{TAG}_{name}.partial.pkl.gz", **kw))


def classify_map(items):
    return {(it["origin"], it["destination"], it["base_time"]): (it, classify(it, T1)) for it in items
            if classify(it, T1)["type"] != "invalid"}


def uses(path, pref):
    return any(n.startswith(pref) for n in path)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def build_sample(nA: int, nB: int):
    """Phase B 표본 (seed 2027). v2.3(Tier C + seed 42 600쌍)·v2.3b holdout(seed 2026 600쌍)과 겹치지 않게."""
    disp18 = pd.read_csv(ROOT / "data" / "master" / "station_display_master.csv").station_key.tolist()
    disp9 = pd.read_csv(MDIR / "line9" / "station_display_master_with9.csv")
    keys = disp9.station_key.tolist()
    st9 = disp9[disp9.available_lines.astype(str).str.contains("9")].station_key.tolist()
    used = set(S23.TIER_C)
    r = random.Random(42)
    while len(used) < 620:
        a, b = r.sample(disp18, 2)
        used.add((a, b))
    r = random.Random(2026)
    h = 0
    while h < 600:
        a, b = r.sample(disp18, 2)
        if (a, b) not in used:
            used.add((a, b))
            h += 1
    r = random.Random(2027)
    A, B = set(), set()
    while len(A) < nA:
        a, b = r.sample(keys, 2)
        if (a in st9 or b in st9) and (a, b) not in used:
            A.add((a, b))
    while len(B) < nB:
        a, b = r.sample(keys, 2)
        if (a, b) not in used and (a, b) not in A:
            B.add((a, b))
    tasks = [(o, d, "A") for o, d in sorted(A)] + [(o, d, "B") for o, d in sorted(B)]
    return tasks, A, B


def main(argv=None) -> int:
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--allow-uncommitted", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    TAG = "_smoke" if args.smoke else ""
    nA, nB, nE = (6, 6, 3) if args.smoke else (300, 300, 100)
    t0 = time.time()
    pr = check_prereg()
    if not pr["ok"] and not args.allow_uncommitted:
        print(f"[중단] {PREREG} 가 커밋되지 않았습니다. Phase A 커밋을 먼저 하세요.")
        return 2
    print(f"사전 등록: {'확인 ' + pr['commit'][:8] if pr['ok'] else '미확인'}")

    # ------------------------------------------------------------ 표본
    tasks, A, B = build_sample(nA, nB)
    grp = {(o, d): g for o, d, g in tasks}
    print(f"표본: A {len(A)} + B {len(B)} OD, 기존 표본과 겹침 0")

    # ------------------------------------------------------------ B1 최초 대기
    print("[B1] 최초 승차 대기 미포함 / 포함")
    w0f = classify_map(run(tasks, args.workers, "base", "b1_w0off"))
    w0t = classify_map(run(tasks, args.workers, "base", "b1_w0on", w0=True))

    def x9_share(cm):
        v = [uses(it["base_path"], "9X_") for (o, d, _), (it, _) in cm.items() if grp[(o, d)] == "A"]
        return float(np.mean(v)) if v else np.nan

    def retention(a, b):
        ks = [k for k in a if k in b and a[k][1]["type"] != "calm"]
        return float(np.mean([a[k][1]["type"] == b[k][1]["type"] for k in ks])) if ks else np.nan, len(ks)

    x_off, x_on = x9_share(w0f), x9_share(w0t)
    keep_b1, n_b1 = retention(w0f, w0t)
    switch_w0 = abs(x_on - x_off) >= 0.05 or keep_b1 < 0.90
    default, dname = (w0t, "w0 포함") if switch_w0 else (w0f, "w0 미포함")
    wkw = {"w0": True} if switch_w0 else {}
    print(f"   9X 비율 {pct(x_off)} → {pct(x_on)}, 유지율 {pct(keep_b1)} → 기본값 {dname}")

    # ------------------------------------------------------------ B2·B3
    sc = {}
    for s in ("tf100", "tf160", "tf278", "max3y"):
        print(f"[B2/B3] {s}")
        sc[s] = classify_map(run(tasks, args.workers, s, f"{s}{'_w0' if switch_w0 else ''}", **wkw))

    def nine_share(cm):
        v = [uses(it["base_path"], "9") for (o, d, _), (it, _) in cm.items() if grp[(o, d)] == "A"]
        return float(np.mean(v)) if v else np.nan

    def struct_share(cm, g=None):
        t = [c["type"] for (o, d, _), (_, c) in cm.items() if (g is None or grp[(o, d)] == g)]
        t = [x for x in t if x != "calm"]
        return float(np.mean([x == "structural" for x in t])) if t else np.nan

    b2 = []
    for s, f in (("tf100", 1.00), ("tf160", 1.60), ("base", 1.81), ("tf278", 2.78)):
        cm = default if s == "base" else sc[s]
        k_, n_ = retention(default, cm) if s != "base" else (1.0, 0)
        b2.append({"환승 계수": f"×{f:.2f}", "A군 최속 경로 9호선 이용": pct(nine_share(cm)),
                   "혼잡 이동 유형 유지율 (기본 대비)": pct(k_)})
    keep160 = retention(default, sc["tf160"])[0]
    keep_max, _ = retention(default, sc["max3y"])

    # ------------------------------------------------------------ B4 이벤트
    print("[B4] 이벤트 9호선 적용/미적용")
    ev_rows = []
    ev_tasks = [t for t in tasks if t[2] == "A"][:nE]
    for eid, date, times in EVENTS:
        dt = pd.Timestamp(date).dayofweek
        day = "saturday" if dt == 5 else ("sunday" if dt == 6 else "weekday")
        on = classify_map(run(ev_tasks, args.workers, "base", f"b4_{eid}_on", day_type=day, query_date=date,
                              base_times=times, **wkw))
        off = classify_map(run(ev_tasks, args.workers, "base", f"b4_{eid}_off", day_type=day, query_date=date,
                               base_times=times, event_exclude=("9L", "9X"), **wkw))
        ks = [k for k in on if k in off]
        on9 = [k for k in ks if uses(on[k][0]["base_path"], "9")]
        ch = [k for k in ks if on[k][1]["type"] != off[k][1]["type"]]
        ev_rows.append({"이벤트": eid, "단위": len(ks), "기준 경로가 9호선 이용": len(on9),
                        "유형이 바뀐 단위 (적용 vs 미적용)": len(ch),
                        "적용 시 Structural": sum(on[k][1]["type"] == "structural" for k in ks),
                        "미적용 시 Structural": sum(off[k][1]["type"] == "structural" for k in ks)})

    # ------------------------------------------------------------ B5 분류
    print("[B5] 분류")
    rows = []
    for (o, d, bt), (it, c) in default.items():
        rows.append({"origin": o, "destination": d, "base_time": bt, "group": grp[(o, d)],
                     "type": c["type"], "uses_9": uses(it["base_path"], "9"), "uses_9X": uses(it["base_path"], "9X_"),
                     "base_exposure_100_min": round(it["base"]["exp100"], 2), "route_ok": c["route_ok"],
                     "time_ok": c["time_ok"], "default_w0": switch_w0, "threshold_set_id": T1.set_id})
    mart = pd.DataFrame(rows)
    mart.to_parquet(MDIR / f"line9_shiftability_mart{TAG}.parquet", index=False)
    dist = (mart[mart.type != "calm"].groupby("group").type.value_counts(normalize=True).unstack(fill_value=0)
            .reindex(columns=HOT, fill_value=0).map(pct))
    Bcm = {k: v for k, v in default.items() if grp[(k[0], k[1])] == "B"}
    spec = importlib.util.spec_from_file_location("s23b", ROOT / "scripts" / "v2" / "23b_robustness.py")
    s23b = importlib.util.module_from_spec(spec)
    sys.modules["s23b"] = s23b
    spec.loader.exec_module(s23b)
    HB = s23b.hypotheses(Bcm) if len(Bcm) > 20 else None
    allcm = s23b.hypotheses(default) if len(default) > 20 else None
    corridor7 = ["7호선 중곡→군자", "7호선 용마산→중곡", "7호선 사가정→용마산", "7호선 어린이대공원→건대입구",
                 "7호선 면목→사가정", "7호선 상봉→면목", "7호선 군자→어린이대공원"]

    L = ["# v2.6 Phase B — 9호선 통합 평가 리포트", "",
         f"- 사전 등록: `{PREREG}` " + (f"커밋 `{pr['commit'][:10]}` ({pr['date']})" if pr["ok"] else "**미확인**"),
         f"- 표본: A군 {len(A)} (9호선 역 출발/도착) + B군 {len(B)} (270역 균등) × 평일 6개 시각, seed 2027, 기존 표본과 겹침 0",
         "- 가정: 9호선 환승 = 거리시간 × 1.81 (보정 proxy, 실측 아님) · 일반↔급행 도보 0분 (가정) · 9호선 p90 미정의",
         f"- 실행 시간 {(time.time() - t0) / 60:.1f}분", "",
         "## B1. 최초 승차 대기 (먼저 수행, D-003 재검토)", "",
         f"- A군 최속 경로 중 급행(9X) 이용: 미포함 {pct(x_off)} → 포함 {pct(x_on)} (차이 {100 * (x_on - x_off):+.1f}%p)",
         f"- 혼잡 이동 {n_b1}단위 유형 유지율 {pct(keep_b1)}",
         f"- 기준: 9X 차이 ≥ 5%p 또는 유지율 < 90% 이면 w0 포함으로 변경 → **v2.6 기본값: {dname}**", "",
         "## B2. 환승 계수 sensitivity (보정 proxy)", "", pd.DataFrame(b2).to_markdown(index=False), "",
         f"- tf160(IQR 하한) 유지율 {pct(keep160)} → " +
         ("**9호선 결과는 환승 가정에 민감 (표기 필요)**" if keep160 < 0.90 else "환승 가정 민감 표기 불필요"), "",
         "## B3. 9호선 혼잡 max_3y", "",
         f"- 유형 유지율 {pct(keep_max)}, 혼잡 이동 중 Structural: 기본 {pct(struct_share(default))} → max_3y "
         f"{pct(struct_share(sc['max3y']))}", "",
         "## B4. 이벤트 배수의 9호선 적용 여부 (A군 100쌍)", "", pd.DataFrame(ev_rows).to_markdown(index=False), "",
         "## B5. 9호선 포함 분류 (기본값)", "",
         "혼잡 이동 중 유형 비율 (군별):", "", dist.to_markdown(), "",
         f"- A군 최속 경로의 9호선 이용 {pct(nine_share(default))}, 급행 이용 {pct(x9_share(default))}"]
    if HB:
        same = (HB["h1_pass"], HB["h2_pass"], HB["h3_pass"]) == (True, True, True)
        L += ["", "B군에서 v2.3 가설 재확인 (v2.3: H1·H2·H3 모두 유지):", "",
              f"- H1 Route 성공 {pct(HB['h1'])} ({'유지' if HB['h1_pass'] else '반증'}), "
              f"H2 {pct(HB['h2'])} ({'유지' if HB['h2_pass'] else '반증'}), "
              f"H3 상위 10구간 {pct(HB['h3_obs'])}, p={HB['h3_p']:.3f} ({'유지' if HB['h3_pass'] else '반증'})",
              f"- **{'9호선 추가 후에도 v2.3 결론 유지' if same else 'v2.3 결론 일부 변화 — 아래 확인'}**"]
    if allcm:
        te = allcm["top_edges"]
        L += ["", "Structural 상위 구간 (A+B, 기본값):", "",
              pd.DataFrame({"순위": range(1, len(te) + 1), "구간": te,
                            "9호선": ["예" if e.startswith("9") else "" for e in te],
                            "v2.3 7호선 회랑": ["예" if e in corridor7 else "" for e in te]}).to_markdown(index=False)]
    L += ["", "## 해석 원칙", "",
          "- 9호선 결과는 환승 보정 proxy·도보 0분 가정·p90 미정의 위에서의 값이다.",
          "- 기준 미달은 그대로 보고하고, 사전 등록에 적은 후속 조치만 따른다."]
    rp = ROOT / "reports" / "v2" / f"line9_evaluation_report{TAG}.md"
    rp.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n저장: {rp}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
