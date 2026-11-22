"""
scripts/v2/47_evaluate_korail_3a2c.py
=====================================
Phase 3A-2c — 코레일 확장 평가 (사전 등록 preregistration_3a2.md 4절 V2~V8, 6-1절 T4·V9). 판정 기준은 사전 등록 그대로.

| V | 무엇 | 표본 |
|---|---|---|
| V2 | 기존 v2.3 3,720건: 기능 켬 vs 끔 기준 경로 변경 목록·원인 | 1~8호선 그래프(v2.3 과 같음) + 코레일 |
| T4 | 혼잡 가정 탐색(민감도) vs 기본(E8): 1순위 코레일 비율·1순위 변경률 | V2 표본 + K 표본 |
| V3 | 시간 역전: 출퇴근 1분 간격, 발생률·규모·원인 5분류, 오류 | K 표본 (켬) + 1~9호선 기준선 (끔, v2.3 OD 412쌍) |
| V4 | E vs H: 1순위 변경률·소요 차이·환승 횟수·후보 순위 상관 | K 표본 × 평일 6개 출발 |
| V5 | 구로 보행 0·1·3·5분 + 연결 검증의 구로 종착→시발 이어 타기 건수 | 경인↔경부 24개 방향쌍 + 지정 12쌍 × 6개 출발 |
| V6 | E 12분 상한 시나리오 | K 표본 × 6개 출발 |
| V7 | 운행 없음 사유별 후보 제외 | K 표본 × 평일 05:30~00:30 30분마다 |
| V8 | FIFO: 탐색 구조, 비FIFO 간선, 기다림 이득 | 그래프 전체 + V3 결과 |
| V9 | 후보 충분성: k=5/10 (K×6), k=5/10/20 (대표 36쌍×6), 단순 경로 열거 기준값 | |

K 표본 = 한쪽 끝이 코레일 서울 26역인 무작위 400쌍(시드 42) + 지정 12쌍 (사전 등록 4절).

실행 (사용자 PC — 작업자 8개 기준 전체 약 30~60분, 작업자 하나에 메모리 약 0.5GB)
----
    python scripts/v2/47_evaluate_korail_3a2c.py                 # 전부
    python scripts/v2/47_evaluate_korail_3a2c.py --only V3       # 하나만
    python scripts/v2/47_evaluate_korail_3a2c.py --workers 6
    python scripts/v2/47_evaluate_korail_3a2c.py --dry           # 작은 표본으로 형식만 확인 (결과 저장 안 함, 보고 금지)

중간에 끊기면 다시 실행하면 이미 끝난 V 는 건너뛴다 (data/marts/v2/korail_3a2c/*.parquet). 다시 계산은 --force.

산출물
------
    data/marts/v2/korail_3a2c/<V>.parquet   계산 원자료
    reports/v2/korail_3a2c_evaluation.md    보고서 · reports/v2/korail_3a2c_evaluation.json 요약
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import korail_line1 as K           # noqa: E402
from yeoyuro_v2 import korail_engine as KE         # noqa: E402
from yeoyuro_v2 import time_dependent as td        # noqa: E402
from yeoyuro_v2 import train_timetable as T        # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
OUT = ROOT / "data" / "marts" / "v2" / "korail_3a2c"
REPORT = ROOT / "reports" / "v2" / "korail_3a2c_evaluation.md"
SUMMARY = ROOT / "reports" / "v2" / "korail_3a2c_evaluation.json"
BASE_TIMES = ["07:30", "08:00", "08:30", "17:30", "18:00", "18:30"]
MIN_WINDOWS = [(420, 540), (1050, 1170)]               # 07:00~09:00, 17:30~19:30 (끝 제외), 1분 간격
DAY_BINS = list(range(330, 1471, 30))                  # 05:30 ~ 24:30
DESIGNATED = [("구일", "가산디지털단지"), ("독산", "개봉"), ("회기", "가산디지털단지"), ("석계", "독산"),
              ("외대앞", "금천구청"), ("신이문", "독산"), ("노량진", "여의도"), ("석계", "태릉입구"),
              ("용산", "서울역"), ("남영", "시청"), ("도봉산", "창동"), ("온수", "신도림")]
INV_TOL = 0.5
ENUM_NODES_LINES = {
    "1": ["영등포", "신도림", "구로@경부", "구로@경인", "가산디지털단지", "독산", "구일", "개봉", "오류동", "온수"],
    "2": ["신도림", "대림", "구로디지털단지"],
    "7": ["온수", "천왕", "광명사거리", "철산", "가산디지털단지", "남구로", "대림"],
}
ENUM_OD = [("구일", "가산디지털단지"), ("개봉", "대림"), ("온수", "독산"), ("오류동", "남구로"), ("독산", "구일"),
           ("영등포", "가산디지털단지"), ("신도림", "온수"), ("대림", "개봉")]


# ---------------------------------------------------------------------- 표본
def korail_names():
    return sorted({K.SCOPE[c] for c in K.SEOUL_KORAIL})


def k_sample(display_keys) -> list[tuple]:
    rng = random.Random(42)
    kor = korail_names()
    allst = sorted(display_keys)
    seen, out = set(), []
    while len(out) < 400:
        a, b = rng.choice(kor), rng.choice(allst)
        if rng.random() < 0.5:
            a, b = b, a
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b, "random"))
    for a, b in DESIGNATED:
        if (a, b) not in seen:
            out.append((a, b, "designated"))
    return out


def branch_pairs() -> list[tuple]:
    pairs = [(K.SCOPE[a], K.SCOPE[b]) for a in K.GYEONGIN_SOUTH for b in K.GYEONGBU_SOUTH]
    return [(a, b, "branch") for a, b in pairs + [(b, a) for a, b in pairs]] + [(a, b, "designated") for a, b in DESIGNATED]


def v23_pairs(n=412) -> list[tuple]:
    m = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "od_shiftability_mart.parquet")
    u = m[["origin", "destination"]].drop_duplicates().sample(n=n, random_state=42)
    return [(o, d, "v23") for o, d in u.itertuples(index=False)]


# ---------------------------------------------------------------------- 작업자
_R: dict = {}


def _mods():
    if "mods" not in _R:
        sys.path[:0] = [str(ROOT / "app" / "streamlit")]
        import station_routing as sr
        from yeoyuro_v2.line9 import register_v1_directions
        mod = sr.load_scorer_class(ROOT)
        register_v1_directions(mod)
        _R["mods"] = (mod, sr)
    return _R["mods"]


def router(spec: tuple):
    """spec = (net, on, wait_mode, cap, guro_walk, assume, k)"""
    if spec not in _R:
        net, on, wait_mode, cap, gw, assume, k = spec
        mod, sr = _mods()
        root = DATA9 if net == "app9" else ROOT
        disp = sr.load_display_master(root)
        _R[spec] = KE.KorailTDRouter(root, mod, sr, disp, "weekday", policy="step", k=k, multi_bin=True,
                                     include_korail=on, cap_wait=cap, wait_mode=wait_mode, guro_walk=gw,
                                     assume_congestion=assume)
    return _R[spec]


def spec(net="app9", on=True, wait_mode="E", cap=None, gw=None, assume=False, k=5):
    return (net, on, wait_mode, cap, gw, assume, k)


def _row(r, res, o, d, t, tag):
    if not res:
        return {"tag": tag, "o": o, "d": d, "t": t, "ok": False,
                "excluded": ";".join(x.no_service_reason or "" for x in getattr(r, "excluded", []))}
    q = res[0]
    # 소요에 들어간 코레일 대기 = 갈아탄 뒤 탄 코레일 구간의 E (첫 탑승 구간 대기는 R4 로 소요에 없음)
    kw = sum(w.E for w in (getattr(q, "korail_waits", []) or []) if w.board != q.path[0] and not np.isnan(w.E))
    kride = getattr(q, "korail_ride_min", 0.0) or 0.0
    return {"tag": tag, "o": o, "d": d, "t": t, "ok": True, "arr": q.arrive_min, "actual": q.actual_time_min,
            "path": "|".join(q.path), "korail": bool(getattr(q, "korail", False)), "tr": q.transfer_count,
            "walk": q.transfer_walk_min, "wait": q.transfer_wait_min, "kwait": kw, "kride": kride,
            "ivt": q.in_vehicle_min, "n_cand": len(res),
            "cands": ("#".join("|".join(x.path) + "@" + f"{x.actual_time_min:.4f}" for x in res[:5])
                      if tag in ("E", "H") else ""),
            "excluded": ";".join(x.no_service_reason or "" for x in getattr(r, "excluded", []))}


def task(args):
    sp, tag, items = args            # items: [(o, d, t, k_override)]
    r = router(sp)
    rows = []
    for o, d, t, kk in items:
        try:
            res = r.evaluate_od(o, d, float(t), k=kk)
        except Exception as e:        # 역 이름이 그 그래프에 없음 등 — 기록만
            rows.append({"tag": tag, "o": o, "d": d, "t": t, "ok": False, "excluded": f"error:{type(e).__name__}"})
            continue
        rows.append(_row(r, res, o, d, t, tag))
    return rows


def run_tasks(jobs, workers) -> pd.DataFrame:
    """같은 라우터 설정끼리 묶어 설정마다 새 작업자 풀을 쓴다 (작업자 하나가 라우터 하나만 들고 있게 — 메모리)."""
    rows = []
    t0 = time.time()
    groups: dict = {}
    for j in jobs:
        groups.setdefault(j[0], []).append(j)
    done, total = 0, len(jobs)
    for sp, js in groups.items():
        if workers <= 1:
            for j in js:
                rows += task(j)
                done += 1
            continue
        with Pool(min(workers, len(js))) as pool:
            for rs in pool.imap_unordered(task, js, chunksize=1):
                rows += rs
                done += 1
                if done % max(1, total // 10) == 0:
                    print(f"    {done}/{total} 작업 ({time.time() - t0:.0f}초)", flush=True)
    return pd.DataFrame(rows)


def jobs_for(sp, tag, ods, times, k=None, chunk=1):
    out = []
    for o, d, _ in ods:
        items = [(o, d, t, k) for t in times]
        for i in range(0, len(items), max(1, len(items) // chunk)):
            out.append((sp, tag, items[i:i + max(1, len(items) // chunk)]))
    return out


# ---------------------------------------------------------------------- 계산
def compute(name, args, ods_k, ods_b, ods_23, workers) -> pd.DataFrame:
    bt = [td.parse_hhmm(x) for x in BASE_TIMES]
    minutes = [t for a, b in MIN_WINDOWS for t in range(a, b)]
    if args.dry:
        minutes = minutes[::40]
    J = []
    if name == "V2":
        m = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "od_shiftability_mart.parquet")
        if args.dry:
            m = m.head(60)
        items = [(r.origin, r.destination, td.parse_hhmm(r.base_time), None) for r in m.itertuples()]
        for tag, sp in [("off", spec("root", False)), ("on", spec("root", True)), ("assume", spec("root", True, assume=True))]:
            for i in range(0, len(items), 60):
                J.append((sp, tag, items[i:i + 60]))
    elif name == "T4":
        J = jobs_for(spec(), "on", ods_k, bt) + jobs_for(spec(assume=True), "assume", ods_k, bt)
    elif name == "V3":
        J = jobs_for(spec(), "on", ods_k, minutes, chunk=4) + jobs_for(spec("root", False), "off", ods_23, minutes, chunk=4)
    elif name == "V4":
        J = jobs_for(spec(), "E", ods_k, bt) + jobs_for(spec(wait_mode="H"), "H", ods_k, bt)
    elif name == "V5":
        for gw in (0.0, 1.0, 3.0, 5.0):
            J += jobs_for(spec(gw=gw), f"g{gw:g}", ods_b, bt)
    elif name == "V6":
        J = jobs_for(spec(), "E", ods_k, bt) + jobs_for(spec(cap=12.0), "cap12", ods_k, bt)
    elif name == "V7":
        J = jobs_for(spec(), "day", ods_k, DAY_BINS if not args.dry else DAY_BINS[::8])
    elif name == "V9":
        J = (jobs_for(spec(), "k5", ods_k, bt) + jobs_for(spec(k=10), "k10", ods_k, bt)
             + jobs_for(spec(), "b5", ods_b, bt) + jobs_for(spec(k=10), "b10", ods_b, bt)
             + jobs_for(spec(k=20), "b20", ods_b, bt))
    else:
        raise ValueError(name)
    print(f"  {name}: 작업 {len(J)}개", flush=True)
    return run_tasks(J, workers)


# ---------------------------------------------------------------------- 분석
def _edges(p):
    n = p.split("|")
    return set(zip(n, n[1:]))


def _cause_v2(path_on: str, pieces) -> str:
    n = path_on.split("|")
    kor = pieces.korail_nodes
    if any({a, b} == set(KE.GURO_NODES) for a, b in zip(n, n[1:])):
        return "구로 갈아타기"
    if any((a in kor) != (b in kor) for a, b in zip(n, n[1:])):
        return "코레일 환승역 새로 생김"
    if any(x in kor for x in n):
        return "코레일 구간 사용"
    return "기타(코레일 없음)"


def analyze_v2(df) -> dict:
    off = df[df.tag == "off"].set_index(["o", "d", "t"])
    on = df[df.tag == "on"].set_index(["o", "d", "t"])
    asm = df[df.tag == "assume"].set_index(["o", "d", "t"])
    m = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "od_shiftability_mart.parquet")
    m["t"] = m.base_time.map(td.parse_hhmm).astype(float)
    typ = m.set_index(["origin", "destination", "t"]).final_shiftability_type
    pieces = router(spec("root", True)).pieces
    rows = []
    for k in off.index:
        if k not in on.index:
            continue
        a, b = off.loc[k], on.loc[k]
        if not (a.ok and b.ok):
            continue
        if a.path != b.path:
            rows.append({"o": k[0], "d": k[1], "t": k[2], "off_min": a.actual, "on_min": b.actual,
                         "diff": b.actual - a.actual, "cause": _cause_v2(b.path, pieces), "korail": bool(b.korail),
                         "v23_type": typ.get((k[0], k[1], float(k[2])), None), "off_path": a.path, "on_path": b.path})
    ch = pd.DataFrame(rows)
    n = int(len(set(off.index) & set(on.index)))
    out = {"cases": n, "changed": int(len(ch)),
           "by_cause": ch.cause.value_counts().to_dict() if len(ch) else {},
           "changed_with_korail_judgement_excluded": int(ch.korail.sum()) if len(ch) else 0,
           "v23_type_of_changed": ch.v23_type.value_counts().to_dict() if len(ch) else {},
           "diff_median": round(float(ch["diff"].median()), 2) if len(ch) else None,
           "diff_min": round(float(ch["diff"].min()), 2) if len(ch) else None,
           "examples": ch.sort_values("diff").head(15)[["o", "d", "t", "off_min", "on_min", "cause", "v23_type"]]
           .round(2).to_dict("records") if len(ch) else []}
    # T4 (V2 표본): 혼잡 가정 탐색 vs 기본
    both = [k for k in on.index if k in asm.index and on.loc[k].ok and asm.loc[k].ok]
    out["T4_v23"] = {"cases": len(both),
                     "top1_korail_default": int(sum(bool(on.loc[k].korail) for k in both)),
                     "top1_korail_assume": int(sum(bool(asm.loc[k].korail) for k in both)),
                     "top1_changed": int(sum(on.loc[k].path != asm.loc[k].path for k in both))}
    return out, ch


def analyze_pairwise(df, a_tag, b_tag) -> dict:
    A = df[df.tag == a_tag].set_index(["o", "d", "t"])
    B = df[df.tag == b_tag].set_index(["o", "d", "t"])
    ks = [k for k in A.index if k in B.index and A.loc[k].ok and B.loc[k].ok]
    if not ks:
        return {"cases": 0}
    d = np.array([B.loc[k].actual - A.loc[k].actual for k in ks])
    chg = np.array([A.loc[k].path != B.loc[k].path for k in ks])
    tr = np.array([B.loc[k].tr - A.loc[k].tr for k in ks])
    corr = []
    for k in ks:
        ca = [c.split("@")[0] for c in str(A.loc[k].cands).split("#")]
        cb = [c.split("@")[0] for c in str(B.loc[k].cands).split("#")]
        common = [p for p in ca if p in cb]
        if len(common) >= 3:
            ra = [ca.index(p) for p in common]
            rb = [cb.index(p) for p in common]
            corr.append(pd.Series(ra).corr(pd.Series(rb), method="spearman"))
    return {"cases": len(ks), "top1_changed": int(chg.sum()), "top1_changed_rate": round(float(chg.mean()), 4),
            "diff_median": round(float(np.median(d)), 3), "diff_p90": round(float(np.quantile(d, 0.9)), 3),
            "diff_abs_p90": round(float(np.quantile(np.abs(d), 0.9)), 3), "diff_max": round(float(d.max()), 3),
            "diff_min": round(float(d.min()), 3), "transfer_change": {int(x): int((tr == x).sum()) for x in np.unique(tr)},
            "rank_spearman_median": round(float(np.nanmedian(corr)), 3) if corr else None,
            "rank_pairs": len(corr)}


def analyze_v3(df) -> dict:
    out = {}
    from yeoyuro_v2.train_timetable import TrainIndex
    st = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet")
    idx = TrainIndex(st[st.day_type == "weekday"])
    for tag in ("on", "off"):
        g = df[(df.tag == tag) & df.ok].sort_values(["o", "d", "t"])
        r = router(spec() if tag == "on" else spec("root", False))
        ev_cache = {}
        n_pairs = n_inv = 0
        ods_inv = set()
        sizes, causes, wait_gain = [], {}, 0
        n_wait = 0
        errors = []
        for (o, d), s in g.groupby(["o", "d"]):
            for a, b in MIN_WINDOWS:
                w = s[(s.t >= a) & (s.t < b)].reset_index(drop=True)
                if len(w) < 2:
                    continue
                arr = w.arr.to_numpy()
                n_pairs += len(w) - 1
                runmax_i = 0
                for j in range(1, len(w)):
                    if arr[runmax_i] < arr[j - 1]:
                        runmax_i = j - 1
                    i = runmax_i
                    if arr[i] > arr[j] + INV_TOL:
                        n_inv += 1
                        ods_inv.add((o, d))
                        sz = arr[i] - arr[j]
                        sizes.append(sz)
                        cs = []
                        pi, pj = w.path[i], w.path[j]
                        if pi != pj:
                            cs.append("① 경로 변경")
                        else:
                            k_i, k_j = w.kwait[i] + w.kride[i], w.kwait[j] + w.kride[j]
                            n_i = (w.wait[i] - w.kwait[i]) + (w.ivt[i] - w.kride[i])
                            n_j = (w.wait[j] - w.kwait[j]) + (w.ivt[j] - w.kride[j])
                            if arr[j] + (k_i - k_j) >= arr[i] - INV_TOL:
                                cs.append("③ 코레일 시간대 값(E·주행)")
                            if arr[j] + (n_i - n_j) >= arr[i] - INV_TOL:
                                cs.append("② 1~9호선 시간대 값")
                            # ④ 열차 연결도 역전되는가
                            key = int(td.bin_index_of(w.t[j]))
                            ev = ev_cache.setdefault(key, r.evaluator(key))
                            legs = T.legs_from_path(ev, pi.split("|"))
                            ci = T.verify_connection(idx, legs, float(w.t[i]), "weekday")
                            cj = T.verify_connection(idx, legs, float(w.t[j]), "weekday")
                            if T.impossible_legs(idx, legs, "weekday"):
                                errors.append(f"{o}→{d} {w.t[i]:.0f}: 불가능한 연결")
                            if ci.arrive_min is not None and cj.arrive_min is not None \
                                    and ci.arrive_min > cj.arrive_min + INV_TOL:
                                cs.append("④ 열차 연결도 역전")
                            if not cs:
                                cs.append("⑤ 설명 안 됨")
                        for c in cs:
                            causes[c] = causes.get(c, 0) + 1
                # V8(c) 기다림 이득: (t, t+30] 안에 0.5분 넘게 일찍 도착하는 출발이 있는가
                for j in range(len(w)):
                    later = arr[j + 1: j + 31]
                    if len(later) == 30:
                        n_wait += 1
                        if later.min() < arr[j] - INV_TOL:
                            wait_gain += 1
        n_od = g.groupby(["o", "d"]).ngroups
        sz = np.array(sizes) if sizes else np.array([0.0])
        out[tag] = {"od": int(n_od), "pairs": int(n_pairs), "inversions": int(n_inv),
                    "inversion_rate": round(n_inv / n_pairs, 5) if n_pairs else None,
                    "od_with_inversion": len(ods_inv), "od_rate": round(len(ods_inv) / n_od, 4) if n_od else None,
                    "size_median": round(float(np.median(sz)), 2), "size_p90": round(float(np.quantile(sz, 0.9)), 2),
                    "size_max": round(float(sz.max()), 2), "causes": causes, "errors": errors[:10], "n_errors": len(errors),
                    "wait_gain": int(wait_gain), "wait_gain_den": int(n_wait),
                    "wait_gain_rate": round(wait_gain / n_wait, 4) if n_wait else None}
    return out


def analyze_v5(df) -> dict:
    out = {}
    base = df[df.tag == "g1"].set_index(["o", "d", "t"])
    st = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet")
    idx = T.TrainIndex(st[st.day_type == "weekday"])
    for gw in ("g0", "g1", "g3", "g5"):
        g = df[(df.tag == gw)].set_index(["o", "d", "t"])
        ks = [k for k in g.index if k in base.index and g.loc[k].ok and base.loc[k].ok]
        guro = [k for k in ks if any({a, b} == set(KE.GURO_NODES)
                                     for a, b in itertools.pairwise(g.loc[k].path.split("|")))]
        out[gw] = {"cases": len(ks), "top1_uses_guro_transfer": len(guro),
                   "top1_changed_vs_1min": int(sum(g.loc[k].path != base.loc[k].path for k in ks)),
                   "actual_median": round(float(np.median([g.loc[k].actual for k in ks])), 2) if ks else None,
                   "diff_vs_1min_median": round(float(np.median([g.loc[k].actual - base.loc[k].actual for k in ks])), 3) if ks else None}
    # 연결 검증의 구로 종착→시발 이어 타기 (E3a)
    r = router(spec())
    n_conn = n_split = 0
    for k in base.index:
        q = base.loc[k]
        if not q.ok:
            continue
        ev = r.evaluator(td.bin_index_of(k[2]))
        c = T.verify_connection(idx, T.legs_from_path(ev, q.path.split("|")), float(k[2]), "weekday")
        n_conn += 1
        n_split += c.guro_splits
    out["E3a_guro_split_in_connection"] = {"connections": n_conn, "splits": n_split}
    return out


def analyze_v7(df) -> dict:
    d = df[df.tag == "day"]
    reasons = {}
    for x in d.excluded.fillna(""):
        for y in str(x).split(";"):
            if y:
                reasons[y] = reasons.get(y, 0) + 1
    no_route = d[~d.ok]
    return {"od_times": int(len(d)), "excluded_candidates_by_reason": reasons,
            "no_route_left": int(len(no_route)),
            "no_route_by_hour": no_route.t.map(lambda t: f"{int(t // 60) % 24:02d}시").value_counts().sort_index().to_dict()}


def analyze_v9(df) -> dict:
    def comp(a, b):
        A = df[df.tag == a].set_index(["o", "d", "t"])
        B = df[df.tag == b].set_index(["o", "d", "t"])
        ks = [k for k in A.index if k in B.index and A.loc[k].ok and B.loc[k].ok]
        diff = np.array([B.loc[k].actual - A.loc[k].actual for k in ks])
        better = diff < -INV_TOL
        return {"cases": len(ks), "top1_changed": int(sum(A.loc[k].path != B.loc[k].path for k in ks)),
                "better_found": int(better.sum()),
                "better_gain_median": round(float(-np.median(diff[better])), 3) if better.any() else None,
                "better_gain_max": round(float(-diff[better].min()), 3) if better.any() else None}
    return {"K_k5_vs_k10": comp("k5", "k10"), "branch_k5_vs_k10": comp("b5", "b10"),
            "branch_k5_vs_k20": comp("b5", "b20"), "branch_k10_vs_k20": comp("b10", "b20")}


def enumeration_check() -> dict:
    """V9(c): 구로 주변 실제 부분 그래프에서 단순 경로를 모두 열거해 기대 소요 최솟값과 엔진 1순위 비교."""
    r = router(spec())
    sr = _mods()[1]
    rows = []
    for o, d in ENUM_OD:
        for bt in ("08:00", "18:00"):
            t = td.parse_hhmm(bt)
            b = td.bin_index_of(t)
            sc = r.scorer(b)
            ev = r.evaluator(b)
            allowed = {f"{ln}_{s}" for ln, ss in ENUM_NODES_LINES.items() for s in ss}
            allowed = {n for n in allowed if n in sc.nodes}
            starts = [n for n in sr.candidates_of(r.display, o) if n in allowed]
            ends = set(n for n in sr.candidates_of(r.display, d) if n in allowed)
            best, n_paths = None, 0
            stack = [(s, [s]) for s in starts]
            while stack:
                u, p = stack.pop()
                if u in ends and len(p) >= 2:
                    real = list(p)
                    while len(real) >= 2 and sr.station_of(real[0]) == sr.station_of(real[1]):
                        real = real[1:]
                    while len(real) >= 2 and sr.station_of(real[-1]) == sr.station_of(real[-2]):
                        real = real[:-1]
                    if len(real) >= 2 and not sr.has_station_revisit(real) and KE.valid_guro_transfer(real):
                        q = ev.evaluate(real, t)
                        if not q.no_service_reason:
                            n_paths += 1
                            if best is None or q.actual_time_min < best[0]:
                                best = (q.actual_time_min, real)
                for e in sc.adj.get(u, []):
                    v = e["to"]
                    if v in allowed and v not in p and len(p) < 16:
                        stack.append((v, p + [v]))
            res = r.evaluate_od(o, d, t)
            eng = (res[0].actual_time_min, res[0].path) if res else (None, None)
            inside = eng[1] is not None and all(n in allowed for n in eng[1])
            rows.append({"od": f"{o}→{d}", "time": bt, "enum_paths": n_paths,
                         "enum_best": None if best is None else round(best[0], 2),
                         "engine": None if eng[0] is None else round(eng[0], 2), "engine_inside_subgraph": inside,
                         "engine_not_worse": bool(best is None or (eng[0] is not None and eng[0] <= best[0] + INV_TOL)),
                         "enum_best_path": None if best is None else "→".join(best[1])})
    return {"rows": rows, "all_engine_not_worse": all(x["engine_not_worse"] for x in rows)}


def fifo_structure() -> dict:
    """V8(a)(b): 탐색 구조 기록 + 30분 경계에서 '진입 시각 + 소요'가 줄어드는 간선 수."""
    r = router(spec())
    ev = r.evaluator(td.bin_index_of(480))
    n_edges = n_nonfifo = 0
    by_line = {}
    for (u, v), m in ev.edge_meta.items():
        for b in range(td.N_BINS - 1):
            B = td.BIN_START_MIN + (b + 1) * td.BIN_WIDTH_MIN
            before = ev._run(u, v, B - 0.01)
            after = ev._run(u, v, B)
            n_edges += 1
            if (B - 0.01) + before > B + after + 1e-9:
                n_nonfifo += 1
                ln = m["line"] + ("(코레일)" if (u, v) in getattr(ev, "korail_edges", set()) else "")
                by_line[ln] = by_line.get(ln, 0) + 1
    # 대기: 1~9호선 배차÷2 표 (역·노선·방향) 와 코레일 E (인접 역 구간)
    hw = ev.headway
    n_w = n_wn = 0
    keys = {(m["line"], m["from_station"], m["direction"]) for m in ev.edge_meta.values() if m["line"] != "1"}
    for ln, stn, dr in list(keys)[:5000]:
        for b in range(td.N_BINS - 1):
            B = td.BIN_START_MIN + (b + 1) * td.BIN_WIDTH_MIN
            w0 = ev._wait(ln, stn, dr, B - 0.01)
            w1 = ev._wait(ln, stn, dr, B)
            n_w += 1
            if (B - 0.01) + w0 > B + w1 + 1e-9:
                n_wn += 1
    n_kw = n_kwn = 0
    for r_ in r.pieces.ride.itertuples():
        via = (int(r_.from_code), int(r_.to_code))
        for b in range(td.N_BINS - 1):
            B = td.BIN_START_MIN + (b + 1) * td.BIN_WIDTH_MIN
            lg = {"codes": list(via), "nodes": [r_.from_node, r_.to_node]}
            e0 = ev.korail_wait(lg, B - 0.01).E
            e1 = ev.korail_wait(lg, B).E
            if np.isnan(e0) or np.isnan(e1):
                continue
            n_kw += 1
            if (B - 0.01) + e0 > B + e1 + 1e-9:
                n_kwn += 1
    return {"search_structure": "후보 = 출발 30분 구간(과 이동 중 지나는 구간)의 정적 가중치로 Yen 탐색한 합집합 → 시간 진행 평가 → 기대 소요 최소. "
                                "시간 의존 Dijkstra(FIFO 가정 필요)는 쓰지 않음. 비FIFO 는 탐색이 아니라 평가 단계의 '늦게 떠나면 일찍 도착' 으로 나타남",
            "run_boundaries": n_edges, "run_nonfifo": n_nonfifo, "run_nonfifo_by_line": by_line,
            "wait19_boundaries": n_w, "wait19_nonfifo": n_wn, "korail_E_boundaries": n_kw, "korail_E_nonfifo": n_kwn}


# ---------------------------------------------------------------------- 보고서
def _pct(x):
    return "-" if x is None else f"{x * 100:.1f}%"


def write_report(S: dict) -> None:
    L = ["# Phase 3A-2c — 코레일 1호선 확장 평가 (V2~V9, T4)", "",
         "- 판정 기준: [사전 등록 4절·6-1절](../../docs/v2/preregistration_3a2.md) (D-093·D-095). 기준에 없는 해석은 붙이지 않는다.",
         "- 엔진: `KorailTDRouter(include_korail=True)` ([3A-2b 구조 검증](korail_engine_3a2b.md)). 코레일 구간 혼잡도는 결측.",
         f"- 재현: `python scripts/v2/47_evaluate_korail_3a2c.py` · 원자료 `data/marts/v2/korail_3a2c/`", ""]
    if "V2" in S:
        v = S["V2"]
        L += ["## V2 기존 v2.3 3,720건 — 기능 켬 vs 끔", "",
              f"- 비교 {v['cases']:,}건 중 기준 경로가 바뀐 건 **{v['changed']}** (기존 v2.3 결과 파일은 덮어쓰지 않음)",
              "- 원인: " + (", ".join(f"{k} {n}" for k, n in v["by_cause"].items()) or "없음"),
              f"- 바뀐 건 중 코레일 구간을 지나 '혼잡 판정 제외'가 되는 건: {v['changed_with_korail_judgement_excluded']}",
              "- 바뀐 건의 v2.3 유형: " + (", ".join(f"{k} {n}" for k, n in v["v23_type_of_changed"].items()) or "없음"),
              f"- 소요 변화(켬 − 끔): 중앙 {v['diff_median']}분 · 가장 크게 준 건 {v['diff_min']}분", ""]
        if v["examples"]:
            L += ["| 출발 | 도착 | 시각 | 끔(분) | 켬(분) | 원인 | v2.3 유형 |", "|---|---|---|---|---|---|---|"]
            for e in v["examples"]:
                L.append(f"| {e['o']} | {e['d']} | {td.fmt_min(e['t'])} | {e['off_min']} | {e['on_min']} | {e['cause']} | {e['v23_type']} |")
            L.append("")
        t4 = v["T4_v23"]
        L += [f"- T4 (같은 3,720건): 1순위가 코레일을 지나는 건 기본 {t4['top1_korail_default']} · 혼잡 가정 탐색 {t4['top1_korail_assume']}, "
              f"1순위 변경 {t4['top1_changed']} / {t4['cases']}", ""]
    if "T4" in S:
        t = S["T4"]
        L += ["## T4 후보 탐색 편향 (K 표본 × 평일 6개 출발)", "",
              f"- 1순위 변경률 {_pct(t.get('top1_changed_rate'))} ({t.get('top1_changed')} / {t.get('cases')}) · "
              f"소요 차이(가정 − 기본) 중앙 {t.get('diff_median')}분, |차이| 90% {t.get('diff_abs_p90')}분",
              f"- 1순위 코레일 포함: 기본 {t.get('korail_default')} · 혼잡 가정 {t.get('korail_assume')}",
              "- 혼잡 가정 탐색은 민감도 전용 — 결과를 쾌적 경로 판정에 쓰지 않음 (E8)", ""]
    if "V3" in S:
        L += ["## V3 시간 역전 (평일 07:00~09:00·17:30~19:30, 1분 간격)", "",
              "| | OD | 출발 쌍 | 역전 | 역전 비율 | 역전 있는 OD | 규모 중앙·90%·최대(분) | 기다림 이득 (V8c) |", "|---|---|---|---|---|---|---|---|"]
        for tag, lab in (("on", "켬 (K 표본)"), ("off", "끔 1~9호선 기준선 (v2.3 OD 412쌍)")):
            v = S["V3"].get(tag)
            if not v:
                continue
            L.append(f"| {lab} | {v['od']} | {v['pairs']:,} | {v['inversions']:,} | {_pct(v['inversion_rate'])} | "
                     f"{v['od_with_inversion']} ({_pct(v['od_rate'])}) | {v['size_median']}·{v['size_p90']}·{v['size_max']} | "
                     f"{_pct(v['wait_gain_rate'])} |")
        L.append("")
        for tag in ("on", "off"):
            v = S["V3"].get(tag)
            if v:
                L.append(f"- 원인 ({tag}): " + ", ".join(f"{k} {n}" for k, n in sorted(v["causes"].items())))
                L.append(f"- 오류(실제로 불가능한 열차 연결) ({tag}): **{v['n_errors']}** " + " · ".join(v["errors"]))
        L += ["- 역전 = 먼저 떠난 출발 가운데 가장 늦게 도착한 것보다 0.5분 넘게 일찍 도착. 원인 ②·③은 같은 경로에서 한쪽 값을 앞선 출발의 값으로 바꾸면 역전이 사라지는지(구성요소 가산 분해로 근사)",
              "- ④ 열차 연결: 같은 경로를 열차 단위로 따라가도 역전되는지 (열차 단위는 앞지르기 없는 일반열차라 원래 역전이 없어야 함)", ""]
    for name, title in (("V4", "V4 E vs H (K 표본 × 평일 6개 출발, H 대비 E)"), ("V6", "V6 E 12분 상한 (상한 − 기본)")):
        if name in S:
            v = S[name]
            L += [f"## {title}", "",
                  f"- 비교 {v.get('cases')}건 · 1순위 변경 {v.get('top1_changed')} ({_pct(v.get('top1_changed_rate'))})",
                  f"- 소요 차이 중앙 {v.get('diff_median')}분 · 90% {v.get('diff_p90')}분 · 최소 {v.get('diff_min')} · 최대 {v.get('diff_max')}",
                  f"- 환승 횟수 변화 분포: {v.get('transfer_change')} · 후보 5개 순위 상관(스피어만) 중앙 {v.get('rank_spearman_median')} ({v.get('rank_pairs')}건)", ""]
    if "V5" in S:
        L += ["## V5 구로 보행 민감도 (경인↔경부 24개 방향쌍 + 지정 12쌍 × 평일 6개 출발)", "",
              "| 구로 보행 | 비교 | 1순위가 구로 갈아타기 | 1분 대비 1순위 변경 | 소요 중앙(분) | 1분 대비 소요 차이 중앙 |", "|---|---|---|---|---|---|"]
        for gw in ("g0", "g1", "g3", "g5"):
            v = S["V5"][gw]
            L.append(f"| {gw[1:]}분 | {v['cases']} | {v['top1_uses_guro_transfer']} | {v['top1_changed_vs_1min']} | "
                     f"{v['actual_median']} | {v['diff_vs_1min_median']} |")
        e = S["V5"]["E3a_guro_split_in_connection"]
        L += ["", f"- 열차 연결 검증에서 구로 종착→시발 이어 타기(E3a)를 쓴 횟수: {e['splits']} / 연결 {e['connections']}", ""]
    if "V7" in S:
        v = S["V7"]
        L += ["## V7 운행 없음 (K 표본 × 평일 05:30~24:30 30분마다)", "",
              f"- OD·시각 {v['od_times']:,}개 · R3 로 뺀 후보: " + (", ".join(f"{k} {n}" for k, n in v["excluded_candidates_by_reason"].items()) or "없음"),
              f"- 후보가 모두 빠져 경로가 없는 경우 {v['no_route_left']} · 시간대별 {v['no_route_by_hour']}", ""]
    if "V8" in S:
        v = S["V8"]
        L += ["## V8 FIFO", "", f"- (a) {v['search_structure']}",
              f"- (b) 주행 간선 × 30분 경계 {v['run_boundaries']:,}곳 중 '진입 + 소요'가 줄어드는 곳 {v['run_nonfifo']} · 노선별 {v['run_nonfifo_by_line']}",
              f"- (b) 1~9호선 배차÷2 대기 경계 {v['wait19_boundaries']:,}곳 중 {v['wait19_nonfifo']} · 코레일 E 경계 {v['korail_E_boundaries']:,}곳 중 {v['korail_E_nonfifo']}",
              "- (c) 기다림 이득 = V3 표의 마지막 열 (출발을 최대 30분 늦추면 0.5분 넘게 일찍 도착하는 출발 비율)",
              "- 알고리즘은 바꾸지 않음 (3A-2 범위 밖, 필요하면 새 D 번호로 제안)", ""]
    if "V9" in S:
        v = S["V9"]
        L += ["## V9 후보 충분성", "", "| 비교 | 건수 | 1순위 변경 | 더 빠른 경로 새로 찾음 | 줄어든 시간 중앙·최대(분) |", "|---|---|---|---|---|"]
        for k, lab in (("K_k5_vs_k10", "K 표본 k=5 → 10"), ("branch_k5_vs_k10", "대표 36쌍 k=5 → 10"),
                       ("branch_k5_vs_k20", "대표 36쌍 k=5 → 20"), ("branch_k10_vs_k20", "대표 36쌍 k=10 → 20")):
            x = v[k]
            L.append(f"| {lab} | {x['cases']} | {x['top1_changed']} | {x['better_found']} | {x['better_gain_median']}·{x['better_gain_max']} |")
        en = v.get("enum")
        if en:
            L += ["", "열거 기준값 (구로 주변 부분 그래프의 단순 경로 전부, 평일 08:00·18:00):", "",
                  "| OD | 시각 | 열거 경로 | 열거 최소(분) | 엔진 1순위(분) | 엔진이 더 나쁘지 않음 |", "|---|---|---|---|---|---|"]
            for x in en["rows"]:
                L.append(f"| {x['od']} | {x['time']} | {x['enum_paths']} | {x['enum_best']} | {x['engine']} | {'예' if x['engine_not_worse'] else '**아니오**'} |")
        L.append("")
    REPORT.write_text("\n".join(L), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2))),
                    help="작업자 수 (작업자 하나에 메모리 약 0.5GB)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    names = a.only or ["V2", "T4", "V3", "V4", "V5", "V6", "V7", "V9", "V8"]
    out_dir = OUT if not a.dry else Path(os.environ.get("TEMP", "/tmp")) / "korail_3a2c_dry"
    out_dir.mkdir(parents=True, exist_ok=True)
    r9 = router(spec())
    ods_k = k_sample(r9.display.station_key.unique())
    ods_b = branch_pairs()
    ods_23 = v23_pairs()
    if a.dry:
        ods_k, ods_b, ods_23 = ods_k[:8] + ods_k[-4:], ods_b[:4] + ods_b[-2:], ods_23[:6]
    print(f"K 표본 {len(ods_k)} · 대표 {len(ods_b)} · v2.3 기준선 {len(ods_23)} · 작업자 {a.workers}", flush=True)
    S = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() and not a.dry else {}
    for name in names:
        t0 = time.time()
        if name == "V8":
            S["V8"] = fifo_structure()
            print(f"V8 완료 ({time.time() - t0:.0f}초)", flush=True)
            continue
        f = out_dir / f"{name}.parquet"
        if f.exists() and not a.force and not a.dry:
            df = pd.read_parquet(f)
            print(f"{name}: 저장된 결과 사용 ({len(df):,}행)")
        else:
            df = compute(name, a, ods_k, ods_b, ods_23, a.workers)
            df.to_parquet(f, index=False)
            print(f"{name}: 계산 {len(df):,}행 ({time.time() - t0:.0f}초)", flush=True)
        if name == "V2":
            S["V2"], ch = analyze_v2(df)
            ch.to_csv((ROOT / "reports" / "v2" if not a.dry else out_dir) / "korail_3a2c_V2_changed_routes.csv",
                      index=False, encoding="utf-8-sig", lineterminator="\n")
        elif name == "T4":
            S["T4"] = analyze_pairwise(df, "on", "assume")
            S["T4"]["korail_default"] = int(df[(df.tag == "on") & df.ok].korail.sum())
            S["T4"]["korail_assume"] = int(df[(df.tag == "assume") & df.ok].korail.sum())
        elif name == "V3":
            S["V3"] = analyze_v3(df)
        elif name == "V4":
            S["V4"] = analyze_pairwise(df, "H", "E")
        elif name == "V5":
            S["V5"] = analyze_v5(df)
        elif name == "V6":
            S["V6"] = analyze_pairwise(df, "E", "cap12")
        elif name == "V7":
            S["V7"] = analyze_v7(df)
        elif name == "V9":
            S["V9"] = analyze_v9(df)
            S["V9"]["enum"] = enumeration_check()
        print(f"  → {name} 분석 끝 ({time.time() - t0:.0f}초)", flush=True)
    if a.dry:
        global REPORT
        REPORT = out_dir / "dry_report.md"
        write_report(S)
        print(f"연습 보고서: {REPORT}")
        print("\n[연습 실행] 결과는 저장하지 않았습니다 (보고 금지).")
        return 0
    SUMMARY.write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    write_report(S)
    print(f"저장: {REPORT.relative_to(ROOT)}, {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
