"""
scripts/v2/53_evaluate_korail_3b.py — Phase 3B-4 정식 평가 B1~B13 (D-105 사전 등록 6절, 구현 기록 D-105c)
===================================================================================================

| B | 검증 | 기준 (사전 등록) | 이 스크립트가 하는 일 |
|---|---|---|---|
| B1 | 끔 보존 | 불일치 0 | ① v2.3 3,720건 재계산 = 커밋된 od_shiftability_mart (46 V1 과 같은 계산) ② 앱 W1 14사례: D-105 사전 등록 커밋의 앱 ↔ 지금 앱, 끔. 화면 전체 + 노선도 그림 (D-106 회색 점선 층 2개만 빼고 비교, D-105c) |
| B2 | 데이터 검사 T4·T5 | 3B-0 수치와 같음 | 상세 정차표(mart)에서 다시 셈 + 3B-1 보고서(korail_3b_build.json)와 비교 |
| B3 | 구조 | 0 | 기본 실행의 모든 후보: 불가능한 연결 · 역 재방문 · 중복 환승(같은 역에서 서로 다른 노선 셋 이상을 잇는 연속 환승, D-105c) |
| B4 | 켬 3-A 결과 변화 (P5) | 보고 | 3-A K 표본 412쌍 × 평일 6시각: 3-B 끔(include_3b=False) ↔ 켬 — 확정 추천·상태 변화 목록과 원인 |
| B5 | 환승 민감도 | 보고 + C3 목록 | B ×1.00·1.60·2.78 (기본 1.81), U ±1분, 0분 환승 연결 여유 0.5·2.0분 (기본 1.0), 0분 초과 환승 +1.0분 |
| B6 | 토요일 휴일 시각표 | 보고 | 토 14:00 결과 분포 (+ 일 14:00 과 비교) |
| B7 | 분포 붕괴 점검 | 조건 없음 | 묶음(GJ·SB·AR·GC) × 시각: 그 시각 1시간 안에 열차가 있는데 그 묶음 후보 '연결 확인' 0, 또는 '불가능한 연결' 1% 초과 → 판정 보류 |
| B8 | 경춘선 포함 vs 제외 | 보고 | 정차표에서 GC 열차를 뺀 계산기 |
| B9 | 김포공항 방향별 vs 단일값 | 보고 | 방향별 표를 비우고(최댓값 하나) 최솟값 그래프 후보도 끈 계산기 |
| B10 | 확정 추천 연결 불가 | 0 | 기본 실행의 확정 추천 상태 |
| B11 | 성능 plan p95 | ≤ 2.6초 | 기본 실행 plan 1회 시간 |
| B12 | 후보 충분성 | 보고 | k=10 ↔ k=20, 합집합 ↔ 최댓값 그래프 단독 |
| B13 | 공항철도 표시 | 0 | 공항철도 구간 경로가 확정 추천이 된 건수 + 앱 화면 사례 (참고 경로 표시) |

표본 (D-105 6절): K3B = 3-B 역이 한쪽 이상인 무작위 300쌍(시드 42) + 지정 12쌍.
시각: 평일 08:00·18:00·23:30 (2026-10-14), 토 14:00 (2026-10-17), 일 14:00 (2026-10-18).
공항철도는 D-105 2절대로 **정밀 시각표 엔진**(지금 코드) 기준. 집계 모델(D-108 예정)은 이 평가에 쓰지 않는다.

실행 (사용자 PC, 작업자 8개 기준 약 40~70분 (미확인), 작업자 하나 메모리 약 0.7GB)
----
    python scripts/v2/53_evaluate_korail_3b.py --workers 8
    python scripts/v2/53_evaluate_korail_3b.py --dry          # 작은 표본 연습 (결과 저장 안 함, 보고 금지)
    python scripts/v2/53_evaluate_korail_3b.py --only B1      # 일부만 (B1·B2·plans·B4·APP)
중간에 끊기면 다시 실행: 끝난 계산(data/marts/v2/korail_3b_eval/*.parquet, git 무시)은 건너뛴다. 다시 계산은 --force.

산출: reports/v2/korail_3b_eval.{md,json}
후보 탐색 (D-111a): --cand v0(기본, D-105e 재현) | v4(D-110b 채택). v4 는 --tag 필수 — D-111 정식 실행:
    python scripts/v2/53_evaluate_korail_3b.py --workers 8 --cand v4 --tag d111
    (3-B 켬 계산기와 앱 화면 사례(B1_app·B13_app)까지 같은 설정. 3-B 끔(B4_3a)은 언제나 v0)
재계산 (D-105d 사후 수정 뒤): --tag b → reports/v2/korail_3b_eval_b.{md,json}, 원자료 data/marts/v2/korail_3b_eval_b/.
    첫 보고서(태그 없음)는 덮어쓰지 않는다. 같은 태그 보고서가 이미 있으면 멈춘다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit"), str(ROOT / "tests")]
from yeoyuro_v2 import korail_app as KA          # noqa: E402
from yeoyuro_v2 import time_dependent as td      # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"
OUT = ROOT / "data" / "marts" / "v2" / "korail_3b_eval"
REP = ROOT / "reports" / "v2"
NAME = "korail_3b_eval"
PREREG_SUBJECT = "Phase 3-B D-105 사전 등록"
K3B_MART = ROOT / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet"

DESIGNATED = [("김포공항", "서울역"), ("마곡나루", "홍대입구"), ("디지털미디어시티", "왕십리"), ("공덕", "청량리"),
              ("용산", "망우"), ("상봉", "용산"), ("망우", "왕십리"), ("왕십리", "야탑"), ("야탑", "강남"),
              ("선릉", "모란"), ("회기", "수서"), ("서울역", "김포공항")]
SLOTS = [("2026-10-14", "08:00"), ("2026-10-14", "18:00"), ("2026-10-14", "23:30"),
         ("2026-10-17", "14:00"), ("2026-10-18", "14:00")]
B4_DATE = "2026-10-14"
B4_TIMES = ["07:30", "08:00", "08:30", "17:30", "18:00", "18:30"]   # 3-A 평가(47)와 같은 6시각
GROUPS = ("GJ", "SB", "AR", "GC")
NEW3B = ("GJ", "GC", "GJC", "SB", "AR")
P95_LIMIT = 2.6                       # 3-A W8 p95 1.30초의 2배
SLACK_REPORT = 2.0                    # 4-2: 연결 여유 2분 미만 확정 추천 연결은 따로 센다

# 변형 이름 → (라우터 추가 인자, 정차표 처리, 방향별 환승 끔)
VARIANTS = {
    "base": ({}, None, False),
    "b1.00": ({"b_factor": 1.00}, None, False), "b1.60": ({"b_factor": 1.60}, None, False),
    "b2.78": ({"b_factor": 2.78}, None, False),
    "u-1": ({"u_shift": -1.0}, None, False), "u+1": ({"u_shift": 1.0}, None, False),
    "m0.5": ({"zero_walk_margin": 0.5}, None, False), "m2.0": ({"zero_walk_margin": 2.0}, None, False),
    "x1.0": ({"extra_margin": 1.0}, None, False),
    "noGC": ({}, "noGC", False),
    "nodir": ({"min_graph": False}, None, True),
    "k20": ({"k": 20}, None, False),
    "maxonly": ({"min_graph": False}, None, False),
}
B5_VARIANTS = ["b1.00", "b1.60", "b2.78", "u-1", "u+1", "m0.5", "m2.0", "x1.0"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------- 표본
def stations_3b() -> list[str]:
    return sorted(KA._k3b_stations())


def all_stations() -> list[str]:
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    return sorted(set(KA.extended_display(disp, include_3b=True).station_key))


def k3b_sample() -> list[tuple]:
    """D-105 6절: 한쪽 이상이 3-B 역인 무작위 300쌍 (시드 42) + 지정 12쌍 (겹치면 하나)."""
    rng = random.Random(42)
    s3, allst = stations_3b(), all_stations()
    seen, out = set(), []
    while len(out) < 300:
        a, b = rng.choice(s3), rng.choice(allst)
        if rng.random() < 0.5:
            a, b = b, a
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b, "random"))
    for a, b in DESIGNATED:
        if (a, b) not in seen:
            seen.add((a, b))
            out.append((a, b, "designated"))
    return out


# ---------------------------------------------------------------------- 작업자
_W: dict = {}


def _init_worker(cand: str) -> None:
    """작업자 프로세스 시작 (Windows 는 새로 import 하므로 설정을 다시 넣는다, D-111a)."""
    _W["cand"] = cand
    KA.DEFAULT_CAND = cand


def _rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:                                    # noqa: BLE001 — psutil 없음 = 미측정
        return None


def _mods():
    if "mods" not in _W:
        import station_routing as sr
        from yeoyuro_v2.line9 import register_v1_directions
        mod = sr.load_scorer_class(ROOT)
        register_v1_directions(mod)
        _W["mods"] = (mod, sr)
    return _W["mods"]


def planner(variant: str, tt_day: str, date: str, include_3b: bool = True):
    """작업자 하나 = 계산기 하나 (설정이 바뀌면 새로 만든다, 메모리)."""
    cand = _W.get("cand", "v0")
    key = (variant, tt_day, date, include_3b, cand)
    if _W.get("key") != key:
        _W.pop("p", None)
        mod, sr = _mods()
        rk, st_mode, nodir = VARIANTS[variant]
        rk = dict(rk)
        if st_mode == "noGC":
            st = pd.read_parquet(K3B_MART)
            rk["k3b_stop_times"] = st[st.group != "GC"].reset_index(drop=True)
        disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
        t0 = time.perf_counter()
        p = KA.ExtensionPlanner(DATA9, mod, sr, disp, tt_day, query_date=date, repo_root=ROOT,
                                include_3b=include_3b, router_kw=rk, cand=cand)
        if nodir:                                     # B9: 김포공항 방향별 표를 비움 → 평가·검증 모두 최댓값 하나
            p.on.pieces.dir_transfer = {}
        _W.update(key=key, p=p, build=time.perf_counter() - t0)
    return _W["p"]


def _names(path) -> list[str]:
    return [n.split("_", 1)[1].split("@")[0] for n in path]


def _line(node: str) -> str:
    ln = node.split("_", 1)[0]
    return "9" if ln in ("9L", "9X") else ln          # 9호선 일반↔급행 전환은 같은 노선 (기존 그래프 설계)


def structure(path) -> dict:
    """B3: 역 재방문 · 중복 환승 (D-105c 정의: 같은 역에 머무는 동안 **서로 다른 노선 셋 이상**을 환승 간선으로 잇는 경우 —
    예: 공덕 AR → GJ → 5. 9호선 일반↔급행 전환(9L↔9X)과 분기 노드(@)는 같은 노선으로 본다)."""
    st = _names(path)
    runs, lines, dbl = [st[0]], [{_line(path[0])}], []
    for n, s in zip(path[1:], st[1:]):
        if s == runs[-1]:
            lines[-1].add(_line(n))
        else:
            runs.append(s)
            lines.append({_line(n)})
    for s, ls in zip(runs, lines):
        if len(ls) >= 3:
            dbl.append(s)
    rev = [s for s, c in Counter(runs).items() if c > 1]
    return {"revisit": rev, "double_transfer": dbl}


def _slacks(p, ev, path, minute) -> list[float]:
    """확정 추천의 3-B 새 환승 연결 여유: 다음 열차 출발 − 앞 열차 도착 − 실제 보행 (분). 4-2 보고용."""
    from yeoyuro_v2 import train_timetable as T
    raw = T.legs_from_path(ev, list(path), walk_of=lambda i, u, v, e: ev.walk_actual(path, i, u, v, e))
    legs = p.on.verification_legs(ev, path) if p.on.status_3b == "on" else raw
    conn = T.verify_connection(p.tidx, legs, minute, p.day_type)
    out = []
    for j in range(1, len(conn.rides)):
        r0, r1 = conn.rides[j - 1], conn.rides[j]
        if r0.group == r1.group or not ({r0.group, r1.group} & set(NEW3B)):
            continue
        k = next((i for i, lg in enumerate(raw) if lg.group == r1.group and lg.board_code == r1.board_code and i > 0), None)
        if k is None:
            continue
        out.append(round(float(r1.dep_min - r0.arr_min - raw[k].walk_before), 3))
    return out


def plan_task(args):
    variant, include_3b, tt_day, date, hm, ods = args
    p = planner(variant, tt_day, date, include_3b)
    sd = KA.service_day(date, hm)
    rows = []
    for o, d in ods:
        t0 = time.perf_counter()
        try:
            r = p.plan(o, d, sd)
        except Exception as e:                   # noqa: BLE001 — 역 이름이 그 그래프에 없음 등, 기록만
            rows.append({"variant": variant, "date": date, "hhmm": hm, "o": o, "d": d, "status": "error",
                         "err": f"{type(e).__name__}: {e}"[:200]})
            continue
        sec = time.perf_counter() - t0
        b, ref = r.best, r.reference
        cands = [{"path": "|".join(c.path), "status": c.status, "reason": c.reason or "", "lines": list(c.lines_3b),
                  "expected": round(c.expected_min, 3), "transfers": c.transfers, "ref": bool(c.is_reference)}
                 for c in r.candidates]
        slack = []
        if b is not None and b.lines_3b and variant in ("base", "m0.5", "m2.0", "x1.0"):
            ev = p.on.evaluator(td.bin_index_of(sd.minute))
            slack = _slacks(p, ev, b.path, sd.minute)
        rows.append({
            "variant": variant, "cand": p.cand, "rss_mb": _rss_mb(), "include_3b": include_3b, "date": date, "hhmm": hm, "tt_day": tt_day,
            "day_type": sd.day_type, "o": o, "d": d, "sec": sec, "build": _W.get("build"), "status": r.status,
            "status_3b": r.status_3b, "n_cand": len(r.candidates),
            "best_path": "" if b is None else "|".join(b.path), "best_status": "" if b is None else b.status,
            "best_expected": None if b is None else round(b.expected_min, 3),
            "best_train_total": None if b is None or b.train_total_min is None else round(b.train_total_min, 3),
            "best_transfers": None if b is None else b.transfers, "best_korail": None if b is None else b.korail,
            "best_lines": "" if b is None else ",".join(b.lines_3b),
            "best_trains": "" if b is None else "|".join(x.train for x in b.rides),
            "ref_path": "" if ref is None else "|".join(ref.path), "ref_status": "" if ref is None else ref.status,
            "ref_expected": None if ref is None else round(ref.expected_min, 3),
            "baseline": r.baseline_min, "slack": json.dumps(slack),
            "excl_train": json.dumps(dict(r.excluded_train), ensure_ascii=False),
            "excl_engine": json.dumps(dict(r.excluded_engine), ensure_ascii=False),
            "cands": json.dumps(cands, ensure_ascii=False)})
    return rows


def run_jobs(jobs, workers) -> pd.DataFrame:
    """같은 계산기 설정끼리 묶어 설정마다 작업자 풀을 새로 (작업자 하나가 계산기 하나만 들고 있게)."""
    groups: dict = {}
    for j in jobs:
        groups.setdefault((j[0], j[1], j[2], j[3]), []).append(j)
    rows, done, total, t0 = [], 0, len(jobs), time.time()
    for key, js in groups.items():
        if workers <= 1:
            for j in js:
                rows += plan_task(j)
                done += 1
            continue
        with Pool(min(workers, len(js)), initializer=_init_worker, initargs=(_W.get("cand", "v0"),)) as pool:
            for rs in pool.imap_unordered(plan_task, js, chunksize=1):
                rows += rs
                done += 1
                if done % max(1, total // 20) == 0:
                    print(f"    {done}/{total} 작업 ({time.time() - t0:.0f}초)", flush=True)
    return pd.DataFrame(rows)


def jobs_for(variant, include_3b, slots, ods, chunk=8) -> list:
    J = []
    for date, hm in slots:
        tt = KA.service_day(date, hm).timetable_day_type
        for i in range(0, len(ods), chunk):
            J.append((variant, include_3b, tt, date, hm, ods[i:i + chunk]))
    return J


def compute(name, J, workers, force, dry) -> pd.DataFrame:
    out = _W.get("out_dir", OUT)
    out.mkdir(parents=True, exist_ok=True)
    f = out / (f"{name}_dry.parquet" if dry else f"{name}.parquet")
    if f.exists() and not force and not dry:
        print(f"  {name}: 저장본 사용 ({f.name})", flush=True)
        return pd.read_parquet(f)
    t0 = time.time()
    print(f"  {name}: 작업 {len(J)}개", flush=True)
    df = run_jobs(J, workers)
    if not dry:
        df.to_parquet(f)
    print(f"  → {name} 끝 ({time.time() - t0:.0f}초)", flush=True)
    return df


# ---------------------------------------------------------------------- 분석
KEY = ["date", "hhmm", "o", "d"]


def _cands(df):
    for r in df.itertuples():
        for c in json.loads(r.cands) if isinstance(r.cands, str) else []:
            yield r, c


def analyze_b3(base: pd.DataFrame) -> dict:
    n = imp = 0
    rev, dbl, dbl3b = [], [], []
    for r, c in _cands(base):
        n += 1
        if c["reason"] == "갈아탈 열차 없음":
            imp += 1
        s = structure(c["path"].split("|"))
        if s["revisit"]:
            rev.append((r.o, r.d, r.hhmm, s["revisit"], c["path"]))
        if s["double_transfer"]:
            item = (r.o, r.d, r.date, r.hhmm, s["double_transfer"], c["path"], c["status"], r.best_path == c["path"])
            dbl.append(item)
            if c["lines"]:
                dbl3b.append(item)
    best_dbl = [x for x in dbl if x[7]]
    by_station = Counter(s for x in dbl for s in x[4])
    return {"candidates": n, "impossible": imp, "revisit": len(rev), "double_transfer": len(dbl),
            "double_transfer_3b": len(dbl3b), "double_transfer_in_best": len(best_dbl),
            "double_transfer_by_station": dict(by_station.most_common()),
            "revisit_examples": rev[:5], "double_transfer_examples": dbl[:10]}


def analyze_b7(base: pd.DataFrame) -> dict:
    st = pd.read_parquet(K3B_MART)
    out, hold = [], []
    for (date, hm), g in base.groupby(["date", "hhmm"]):
        sd = KA.service_day(date, hm)
        rows = st[st.day_type == sd.timetable_day_type]
        for grp in GROUPS:
            x = rows[(rows.group == grp) & rows.dep_min.between(sd.minute, sd.minute + 60)]
            has_trains = len(x) > 0
            stat = Counter()
            for _, c in _cands(g):
                if grp in c["lines"]:
                    stat["impossible" if c["reason"] == "갈아탈 열차 없음" else c["status"]] += 1
            n = sum(stat.values())
            conn = stat.get(KA.CONNECT_OK, 0)
            imp = stat.get("impossible", 0)
            flag = bool(has_trains and n > 0 and (conn == 0 or imp > 0.01 * n))
            out.append({"date": date, "hhmm": hm, "group": grp, "trains_within_60min": int(len(x)), "candidates": n,
                        "connected": conn, "impossible": imp, "status": dict(stat), "hold": flag})
            if flag:
                hold.append(f"{date} {hm} {grp}")
    return {"table": out, "hold": hold}


def _best(df) -> pd.DataFrame:
    return df[df.status != "error"].set_index(KEY)


def compare(a: pd.DataFrame, b: pd.DataFrame, label: str) -> dict:
    """같은 (시각, OD) 에서 확정 추천(best)·참고 경로·상태 비교."""
    A, B = _best(a), _best(b)
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    path_ch = A.best_path != B.best_path
    ref_ch = A.ref_path != B.ref_path
    st_ch = A.status != B.status
    both = (A.best_path != "") & (B.best_path != "")
    dexp = (B.best_expected - A.best_expected)[both]
    conn_ch = (A.best_status != B.best_status) & both
    tr_ch = (A.best_transfers != B.best_transfers) & both
    # 후보 밀림: A 의 확정 추천 경로가 B 의 후보 목록에 아예 없는 건 (k 자리 경쟁, B4·B8 원인 조사)
    pushed = []
    for k in A[path_ch & (A.best_path != "")].index:
        bc = json.loads(B.loc[k].cands) if isinstance(B.loc[k].cands, str) else []
        if all(c["path"] != A.loc[k].best_path for c in bc):
            pushed.append({"key": list(k), "a_exp": A.loc[k].best_expected, "b_exp": B.loc[k].best_expected,
                           "b_lines": B.loc[k].best_lines, "b_n_cand": len(bc)})
    ch = A[path_ch | ref_ch].reset_index()[KEY].copy()
    ch["a_best"] = A[path_ch | ref_ch].best_path.values
    ch["b_best"] = B[path_ch | ref_ch].best_path.values
    ch["a_exp"] = A[path_ch | ref_ch].best_expected.values
    ch["b_exp"] = B[path_ch | ref_ch].best_expected.values
    return {"label": label, "compared": int(len(idx)), "best_changed": int(path_ch.sum()),
            "best_changed_od": int(A[path_ch].reset_index()[["o", "d"]].drop_duplicates().shape[0]),
            "reference_changed": int(ref_ch.sum()), "status_changed": int(st_ch.sum()),
            "best_status_changed": int(conn_ch.sum()), "transfers_changed": int(tr_ch.sum()),
            "expected_diff_median": None if dexp.empty else round(float(dexp.median()), 3),
            "expected_diff_max_abs": None if dexp.empty else round(float(dexp.abs().max()), 3),
            "examples": ch.head(12).to_dict("records"),
            "a_best_missing_from_b": len(pushed),
            "a_best_missing_worse": int(sum(1 for x in pushed if x["b_exp"] is not None and x["a_exp"] is not None
                                            and x["b_exp"] > x["a_exp"] + 1e-6)),
            "a_best_missing_examples": pushed[:12]}


def _b_transfers(path) -> list[str]:
    """경로가 지나는 B 등급 환승 (근거 표 기준, '역 노선↔노선')."""
    ev = pd.read_csv(ROOT / "data" / "master" / "transfer_3b_evidence.csv", dtype={"line_a": str, "line_b": str})
    b = ev[ev.grade == "B"]
    keys = {(r.station, r.line_a, r.line_b) for r in b.itertuples()}
    out = []
    p = path.split("|")
    for u, v in zip(p, p[1:]):
        su, sv = u.split("_", 1)[1].split("@")[0], v.split("_", 1)[1].split("@")[0]
        if su != sv:
            continue
        lu, lv = u.split("_", 1)[0], v.split("_", 1)[0]
        for la, lb in ((lu, lv), (lv, lu)):
            if (su, la, lb) in keys:
                out.append(f"{su} {la}↔{lb}")
    return out


def analyze_b5(R: dict, n_od: int) -> dict:
    out = {v: compare(R["base"], R[v], v) for v in B5_VARIANTS if v in R}
    # C3: 계수에 따라 1순위가 바뀐 OD 가 지나는 B 등급 환승 (바뀐 OD 5건 이상 또는 K3B 의 2% 이상)
    cnt: dict = {}
    for v in ("b1.00", "b1.60", "b2.78"):
        if v not in R:
            continue
        A, B = _best(R["base"]), _best(R[v])
        idx = A.index.intersection(B.index)
        for k in idx:
            pa, pb = A.loc[k].best_path, B.loc[k].best_path
            if pa != pb:
                for t in set(_b_transfers(pa)) | set(_b_transfers(pb)):
                    cnt.setdefault(t, set()).add((k[2], k[3]))
    thr = max(5, int(np.ceil(0.02 * n_od)))
    c3 = sorted(((t, len(s)) for t, s in cnt.items()), key=lambda x: -x[1])
    out["C3"] = {"threshold_od": thr, "by_transfer": c3, "flagged": [t for t, n in c3 if n >= thr]}
    return out


def analyze_base(base: pd.DataFrame) -> dict:
    ok = base[base.status != "error"]
    has = ok[ok.best_path != ""]
    sl = [x for s in has.slack for x in json.loads(s)]
    ex = Counter()
    for s in ok.excl_train:
        ex.update(json.loads(s))
    return {"plans": int(len(base)), "errors": int((base.status == "error").sum()),
            "error_examples": [f"{r.date} {r.hhmm} {r.o}→{r.d}: {getattr(r, 'err', '')}"
                               for r in base[base.status == "error"].head(10).itertuples()],
            "status": ok.status.value_counts().to_dict(),
            "status_by_slot": {f"{a} {b}": g.status.value_counts().to_dict() for (a, b), g in ok.groupby(["date", "hhmm"])},
            "best_not_connected": int((has.best_status != KA.CONNECT_OK).sum()),           # B10
            "best_with_ar": int(has.best_lines.str.contains("AR").sum()),                  # B13
            "reference_plans": int((ok.ref_path != "").sum()),
            "reference_connected": int((ok.ref_status == KA.CONNECT_OK).sum()),
            "best_3b": int((has.best_lines != "").sum()),
            "best_lines": Counter(x for s in has.best_lines for x in s.split(",") if x).most_common(),
            "excluded_train": dict(ex.most_common()),
            "slack_n": len(sl), "slack_lt2": int(sum(x < SLACK_REPORT for x in sl)),
            "slack_min": min(sl) if sl else None,
            "sec_mean": float(ok.sec.mean()), "sec_p95": float(ok.sec.quantile(0.95)), "sec_max": float(ok.sec.max()),
            "build_max": float(ok.build.max())}


def analyze_b6(base: pd.DataFrame) -> dict:
    ok = base[base.status != "error"]
    sat, sun = ok[ok.date == "2026-10-17"], ok[ok.date == "2026-10-18"]
    m = sat.merge(sun, on=["o", "d"], suffixes=("_sat", "_sun"))
    shown = sat[(sat.best_lines != "") | (sat.ref_path != "")]
    return {"plans": int(len(sat)), "status": sat.status.value_counts().to_dict(),
            "tt_day": sat.tt_day.unique().tolist(), "with_3b_shown": int(len(shown)),
            "best_same_as_sunday": int((m.best_path_sat == m.best_path_sun).sum()), "compared_sunday": int(len(m)),
            "trains_same_as_sunday": int((m.best_trains_sat == m.best_trains_sun).sum())}


def analyze_b4(a3: pd.DataFrame, a3b: pd.DataFrame) -> dict:
    c = compare(a3, a3b, "3-B 끔 → 켬 (3-A K 표본)")
    A, B = _best(a3), _best(a3b)
    idx = A.index.intersection(B.index)
    cause = Counter()
    for k in idx:
        a, b = A.loc[k], B.loc[k]
        if a.best_path == b.best_path and a.ref_path == b.ref_path:
            continue
        if b.best_lines:
            cause["3-B 경로가 확정 추천이 됨"] += 1
        elif b.ref_path and not a.ref_path:
            cause["공항철도 참고 경로가 생김"] += 1
        elif a.best_path and not b.best_path:
            cause["확정 추천이 없어짐"] += 1
        else:
            cause["후보 구성 변화 (3-B 아닌 경로끼리)"] += 1
    c["by_cause"] = dict(cause.most_common())
    return c


# ---------------------------------------------------------------------- B1·B2
def b1_v23(dry) -> dict:
    m46 = _load("m46", "scripts/v2/46_check_korail_engine.py")
    return m46.v1_check(30 if dry else None)


def _before_app() -> Path:
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-F", f"--grep={PREREG_SUBJECT}"],
                         capture_output=True, text=True, encoding="utf-8").stdout.split()
    if not sha:
        raise SystemExit(f"'{PREREG_SUBJECT}' 커밋을 찾지 못했습니다")
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha[-1]}:app/streamlit/yeoyuro_seoul_app.py"],
                         capture_output=True, encoding="utf-8").stdout
    p = APP.parent / "_b1_before_app.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return p


def _d106_trace(t: dict) -> bool:
    """D-106 이 끔 노선도에 더한 층: 확장 구간 회색 점선 + 그 위 안내 점 (클릭 없음)."""
    return ((t.get("line") or {}).get("dash") == "dot"
            or "광역철도 확장 구간" in str(t.get("hovertemplate", "")))


def b1_app(dry) -> dict:
    m49 = _load("m49b", "scripts/v2/49_check_korail_app.py")
    cases = m49.W1_CASES[:2] if dry else m49.W1_CASES
    before = _before_app()
    out = {"cases": len(cases), "same": 0, "diffs": [], "d106_traces_removed": []}
    try:
        for c in cases:
            a, b = m49._at(before), m49._at(APP)
            m49._search(a, *c)
            m49._search(b, *c)
            sa, sb = m49._screen(a), m49._screen(b)
            keys = [k for k in sa if k not in ("map_spec_len", "map_spec_hash") and sa[k] != sb[k]]
            ja = json.loads(a.get("plotly_chart")[0].proto.spec) if a.get("plotly_chart") else {}
            jb = json.loads(b.get("plotly_chart")[0].proto.spec) if b.get("plotly_chart") else {}
            rest = [t for t in jb.get("data", []) if not _d106_trace(t)]
            removed = len(jb.get("data", [])) - len(rest)
            if rest != ja.get("data", []):
                keys.append("map_traces")
            if jb.get("layout") != ja.get("layout"):
                keys.append("map_layout")
            out["d106_traces_removed"].append(removed)
            if keys:
                out["diffs"].append({"case": " ".join(c), "keys": keys})
            else:
                out["same"] += 1
    finally:
        before.unlink(missing_ok=True)
    return out


def b2_data() -> dict:
    from yeoyuro_v2 import korail_3b as K3
    st = pd.read_parquet(K3B_MART)
    S = st[st.day_type.isin(["weekday", "saturday"])].assign(day=lambda x: x.day_type.map(
        {"weekday": "weekday", "saturday": "holiday"}))
    cn = {**{v: k for k, v in K3.CODE.items()}, **{v[1]: k for k, v in K3.AR_CODE.items()}}
    t4 = {}
    for seg, (g, sts) in K3.SEGMENTS.items():
        x = S[S.group == g].assign(s=lambda y: y.code.map(cn))
        x = x[x.s.isin(sts)]
        n = x.groupby(["day", "direction", "train"]).size()
        n = n[n >= 2].reset_index()
        if g == "AR":
            for d, c in n.groupby("day").train.nunique().items():
                t4[(seg, d, "*")] = int(c)
        else:
            for (d, dr), c in n.groupby(["day", "direction"]).train.nunique().items():
                t4[(seg, d, dr)] = int(c)
    mism = {"|".join(k): (t4.get(k), v) for k, v in K3.EXPECTED_T4.items() if t4.get(k) != v}
    wk = st[(st.group == "GJ") & (st.day_type == "weekday")].train.nunique()
    ho = st[(st.group == "GJ") & (st.day_type == "saturday")].train.nunique()
    rev = dup = 0
    for _, g in st.groupby(["day_type", "group", "train", "direction"]):
        t = g.sort_values("stop_seq")
        v = np.concatenate([np.column_stack([t.arr_min, t.dep_min]).ravel()])
        v = v[~np.isnan(v)]
        rev += int((np.diff(v) < -1e-9).any())
        dup += int(t.code.duplicated().any())
    both = st[st.arr_min.isna() & st.dep_min.isna()]
    from yeoyuro_v2 import korail_engine as KE
    rep = json.loads((REP / "korail_3b_build.json").read_text(encoding="utf-8"))["check"]
    same_build = {"t4": rep["t4"] == {"|".join(k): v for k, v in sorted(t4.items())},
                  "rows": rep["rows"] == int(len(st)), "t5": tuple(rep["t5_gj_weekday_gt_holiday"]) == (int(wk), int(ho), bool(wk > ho))}
    return {"t4_mismatch": mism, "t5": (int(wk), int(ho), bool(wk > ho)), "rows": int(len(st)),
            "time_reversal_trains": rev, "dup_station_trains": dup, "same_as_3b1_report": same_build,
            "both_missing_rows": int(len(both)),                                     # D-105d: 정차표 파일 안
            "both_missing_by": {f"{a}|{b}|{c}": int(n) for (a, b, c), n in both.groupby(["day_type", "group", "code"]).size().items()},
            "both_missing_after_load": int((KE.drop_unstopped(st).pipe(lambda x: x.arr_min.isna() & x.dep_min.isna())).sum())}


# ---------------------------------------------------------------------- B13 앱
B13_CASES = [("서울역", "김포공항", "2026-10-14", "08:00"), ("마곡나루", "공덕", "2026-10-14", "18:00"),
             ("김포공항", "홍대입구", "2026-10-17", "14:00")]


def b13_app(dry) -> dict:
    m49 = _load("m49c", "scripts/v2/49_check_korail_app.py")
    out = []
    for o, d, date, hm in (B13_CASES[:1] if dry else B13_CASES):
        at = m49._at(APP)
        at.toggle(key="ext_on").set_value(True)
        at.run()
        m49._search(at, o, d, date, hm)
        md = [m.value for m in at.markdown]
        caps = [c.value for c in at.caption]
        ref_i = next((i for i, x in enumerate(md) if KA.REFERENCE_TITLE in x), None)
        ok_i = next((i for i, x in enumerate(md) if x.startswith("**✅")), None)
        # 확정 추천 블록(✅ … 다음 블록 전까지)의 노선 배지에 공항철도가 있으면 위반
        best_has_ar = False
        if ok_i is not None:
            end = ref_i if ref_i is not None and ref_i > ok_i else len(md)
            best_has_ar = any(">공항</span>" in x or "#0090D2" in x for x in md[ok_i + 1:end])   # 공항철도 노선 배지
        out.append({"case": f"{o}→{d} {date} {hm}", "reference_shown": ref_i is not None,
                    "ar_caption": any(KA.AR_REFERENCE in c for c in caps), "confirmed_block": ok_i is not None,
                    "confirmed_has_ar": best_has_ar, "exceptions": [e.value[:200] for e in at.exception]})
    return {"cases": out, "violations": sum(x["confirmed_has_ar"] for x in out)}


# ---------------------------------------------------------------------- 보고서
def _v(x, ok):
    return f"**{x}**" + ("" if ok else " ⚠")


def write_report(S: dict, dest: Path, tag: str = "") -> None:
    B = S.get("base", {})
    b1, b1a, b2, b3, b7 = S.get("B1_v23", {}), S.get("B1_app", {}), S.get("B2", {}), S.get("B3", {}), S.get("B7", {})
    b5, b4, b6 = S.get("B5", {}), S.get("B4", {}), S.get("B6", {})
    cmp_ = S.get("CMP", {})
    p95 = B.get("sec_p95")
    name = NAME + (f"_{tag}" if tag else "")
    L = ["# Phase 3B-4 정식 평가 — D-105 B1~B13 (경의중앙·경춘·수인분당·공항철도)" + (f" — 재계산 {tag}" if tag else ""), "",
         "- 규칙: [사전 등록 6절](../../docs/v2/preregistration_3b.md) D-105 (계산 전 고정) · 평가 도구 구현 기록 D-105c"
         + (" · 사후 수정 D-105d ([판정 기록](../../docs/v2/korail_3b_eval_verdict.md))" if tag else ""),
         "- 재현: `python scripts/v2/53_evaluate_korail_3b.py --workers 8" + (f" --tag {tag}`" if tag else "`")]
    cand = S.get("cand", "v0")
    if cand != "v0":
        L[-1] = L[-1].rstrip("`") + f" --cand {cand}`"
        L.append(f"- 후보 탐색 **{cand}** {S.get('cand_spec')} — D-111 ([사전 등록](../../docs/v2/preregistration_d111.md)). "
                 "3-B 켬 계산기와 앱 화면 사례만 이 설정, 3-B 끔(B4 의 3-A 쪽)은 v0. 결과 변화·G0~G4 는 55 보고서")
        L.append(f"- 첫 결과 [{NAME}.md]({NAME}.md) 와 재계산 [{NAME}_b.md]({NAME}_b.md)(D-105e 판정 기준) 는 그대로 둔다")
    elif tag:
        L.append(f"- 첫 결과 [{NAME}.md]({NAME}.md) 는 그대로 둔다 (정차표 빈칸 정차 9행·한 역 노선 셋 이어 갈아타기가 있던 계산)")
    L += [
         f"- 표본: K3B {S.get('n_od')}쌍 × {len(SLOTS)}시각 = {B.get('plans', 0):,}건 (평일 08:00·18:00·23:30, 토·일 14:00)",
         "- 공항철도: 정밀 시각표 엔진(2025-12-29 시각표) 기준 — 집계 모델(D-108 예정) 아님. A 와 비교하는 기준은 **계획 시각표**이지 실제 운행이 아님",
         f"- 판정 보류(B7): {'**있음 — ' + ', '.join(b7.get('hold', [])) + '**' if b7.get('hold') else '없음'}",
         f"- 계산 오류: {B.get('errors')}건" + (f" — {B.get('error_examples')}" if B.get('errors') else ""), "",
         "| B | 검증 | 결과 | 기준 |", "|---|---|---|---|",
         f"| B1 | 끔 보존 ① v2.3 {b1.get('cases', 0):,}건 ② 앱 W1 {b1a.get('cases', 0)}사례 | ① 불일치 {_v(b1.get('mismatch'), b1.get('mismatch') == 0)} ({b1.get('evaluated')}건 계산) ② 같음 {_v(str(b1a.get('same')) + '/' + str(b1a.get('cases')), b1a.get('same') == b1a.get('cases'))} (노선도는 D-106 층 {sorted(set(b1a.get('d106_traces_removed', [])))}개 빼고 비교) | 불일치 0 |",
         f"| B2 | 데이터 T4·T5 | T4 불일치 {_v(len(b2.get('t4_mismatch', {})), not b2.get('t4_mismatch'))} · T5 {b2.get('t5')} · 시각 역행 {b2.get('time_reversal_trains')} · 같은 역 중복 {b2.get('dup_station_trains')} · 3B-1 보고와 같음 {b2.get('same_as_3b1_report')} · 도착·출발 둘 다 빈칸 정차: 파일 {b2.get('both_missing_rows')}행 → 읽은 뒤 {_v(b2.get('both_missing_after_load'), b2.get('both_missing_after_load') == 0)} (D-105d) | 3B-0 수치와 같음 |",
         f"| B3 | 구조 (후보 {b3.get('candidates', 0):,}개) | 불가능한 연결 {_v(b3.get('impossible'), b3.get('impossible') == 0)} · 역 재방문 {_v(b3.get('revisit'), b3.get('revisit') == 0)} · 중복 환승 {_v(b3.get('double_transfer'), b3.get('double_transfer') == 0)} (3-B 포함 {b3.get('double_transfer_3b')}, 확정 추천 {b3.get('double_transfer_in_best')}) | 0 |",
         f"| B4 | 켬 3-A 결과 변화 (412쌍 × 평일 6시각) | 확정 추천 바뀜 {b4.get('best_changed')} / {b4.get('compared')} · 원인 {b4.get('by_cause')} | 보고 (P5) |",
         f"| B5 | 환승 민감도 | 아래 표 · C3 추가 검증 대상 {b5.get('C3', {}).get('flagged')} | 보고 |",
         f"| B6 | 토요일 휴일 시각표 (토 14:00) | 상태 {b6.get('status')} · 일 14:00 과 확정 추천 같음 {b6.get('best_same_as_sunday')}/{b6.get('compared_sunday')} | 보고 |",
         f"| B7 | 분포 붕괴 점검 | 보류 조건 {_v(len(b7.get('hold', [])), not b7.get('hold'))}건 | 없음 |",
         f"| B8 | 경춘선 포함 vs 제외 | 확정 추천 바뀜 {cmp_.get('noGC', {}).get('best_changed')} ({cmp_.get('noGC', {}).get('best_changed_od')}쌍) · 참고 경로 바뀜 {cmp_.get('noGC', {}).get('reference_changed')} | 보고 |",
         f"| B9 | 김포공항 방향별 vs 단일값 | 확정 추천 바뀜 {cmp_.get('nodir', {}).get('best_changed')} · 참고 경로 바뀜 {cmp_.get('nodir', {}).get('reference_changed')} | 보고 |",
         f"| B10 | 확정 추천 연결 불가 | {_v(B.get('best_not_connected'), B.get('best_not_connected') == 0)} | 0 |",
         f"| B11 | plan p95 | {_v(f'{p95:.2f}초' if p95 is not None else '-', p95 is not None and p95 <= P95_LIMIT)} (평균 {B.get('sec_mean', 0):.2f}, 최대 {B.get('sec_max', 0):.2f}, 계산기 준비 최대 {B.get('build_max', 0):.1f}초) | ≤ {P95_LIMIT}초 |",
         f"| B12 | 후보 충분성 | k=20: 확정 추천 바뀜 {cmp_.get('k20', {}).get('best_changed')} (소요 차이 중앙 {cmp_.get('k20', {}).get('expected_diff_median')}) · 최댓값 그래프 단독: 바뀜 {cmp_.get('maxonly', {}).get('best_changed')} | 보고 |",
         f"| B13 | 공항철도 표시 | 확정 추천에 공항철도 {_v(B.get('best_with_ar'), B.get('best_with_ar') == 0)} · 앱 사례 위반 {_v(S.get('B13_app', {}).get('violations'), S.get('B13_app', {}).get('violations') == 0)} | 0 |",
         "", "## 기본 실행 상세", "",
         f"- 상태: {B.get('status')}", f"- 시각별: {B.get('status_by_slot')}",
         f"- 확정 추천에 3-B 노선: {B.get('best_3b')}건 · 노선별 {B.get('best_lines')}",
         f"- 공항철도 참고 경로: {B.get('reference_plans')}건 (그중 열차 연결 확인 {B.get('reference_connected')})",
         f"- 열차 단위로 뺀 후보 사유: {B.get('excluded_train')}",
         f"- 3-B 새 환승 연결 여유(확정 추천): {B.get('slack_n')}곳 · 2분 미만 {B.get('slack_lt2')} · 최소 {B.get('slack_min')}분 (4-2 보고)",
         f"- 오류 {B.get('errors')}: {B.get('error_examples')}",
         "", "## B5 환승 민감도 (기본 = B×1.81, U 그대로, 0분 환승 여유 1.0분)", "",
         "| 변형 | 확정 추천 바뀜 (OD) | 참고 경로 바뀜 | 상태 바뀜 | 확정 추천 연결 상태 바뀜 | 환승 횟수 바뀜 | 소요 차이 중앙 / 최대 |",
         "|---|---|---|---|---|---|---|"]
    for v in B5_VARIANTS:
        c = b5.get(v)
        if c:
            L.append(f"| {v} | {c['best_changed']} ({c['best_changed_od']}) | {c['reference_changed']} | {c['status_changed']} | "
                     f"{c['best_status_changed']} | {c['transfers_changed']} | {c['expected_diff_median']} / {c['expected_diff_max_abs']} |")
    c3 = b5.get("C3", {})
    L += ["", f"- C3 (계수에 따라 1순위가 바뀐 OD 가 지나는 B 등급 환승, 기준 {c3.get('threshold_od')}쌍 이상): {c3.get('by_transfer')}",
          f"- **추가 검증 대상: {c3.get('flagged') or '없음'}** — C4: 이 결과로 C1 기본값(×1.81)·규칙을 바꾸지 않는다",
          "", "## B3 중복 환승 사례", "",
          f"- 역별: {b3.get('double_transfer_by_station')}"]
    L += [f"- {x}" for x in b3.get("double_transfer_examples", [])] or ["- 없음"]
    L += ["", "## B4 변경 예", "",
          f"- 3-A 확정 추천 경로가 3-B 켬 후보 목록에서 아예 빠진 건 (k 자리 경쟁): {b4.get('a_best_missing_from_b')} · 그중 소요가 늘어난 건 {b4.get('a_best_missing_worse')}",
          f"- 예: {b4.get('a_best_missing_examples', [])[:8]}"] + [f"- {x}" for x in b4.get("examples", [])] + [
        "", "## B7 묶음 × 시각", "", "| 시각 | 묶음 | 1시간 안 열차 | 후보 | 연결 확인 | 불가능 | 보류 |", "|---|---|---|---|---|---|---|"]
    for r in b7.get("table", []):
        L.append(f"| {r['date']} {r['hhmm']} | {r['group']} | {r['trains_within_60min']} | {r['candidates']} | "
                 f"{r['connected']} | {r['impossible']} | {'**보류**' if r['hold'] else ''} |")
    L += ["", "## B8·B9·B12 변경 예", ""]
    for k in ("noGC", "nodir", "k20", "maxonly"):
        c = cmp_.get(k, {})
        L.append(f"- {k}: 바뀜 {c.get('best_changed')} · 기본 확정 추천이 그 변형 후보에 없음 {c.get('a_best_missing_from_b')} · 예 {c.get('examples', [])[:5]}")
    L += ["", "## B1 차이", ""] + ([f"- {x['case']}: {x['keys']}" for x in b1a.get("diffs", [])] or ["- 없음"])
    L += [f"- v2.3 불일치 예: {b1.get('examples')}" if b1.get("mismatch") else "- v2.3 불일치 없음"]
    L += ["", "## B13 앱 사례", ""] + [f"- {x}" for x in S.get("B13_app", {}).get("cases", [])]
    L += ["", "## 한계", "",
          "- 비교 기준은 계획 시각표(2026-09-01·2026-08-22·2025-12-29)이며 실제 운행·지연은 반영하지 않는다",
          "- 공항철도 2026-10-23 개정 반영 여부 미확인 (D-105 2절) — 공항철도 경로는 참고 경로로만",
          "- 3-B 구간 혼잡도는 결측 (P3) — 쾌적 판정 비교 대상 아님",
          "- B1 노선도 비교에서 D-106 회색 점선 층을 뺀 것은 D-105c 에 미리 기록한 방법", ""]
    (dest / f"{name}.md").write_text("\n".join(str(x) for x in L) + "\n", encoding="utf-8", newline="\n")
    (dest / f"{name}.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str) + "\n",
                                       encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------- 실행
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="B1 B2 plans B4 APP 중 일부")
    ap.add_argument("--tag", default="", help="재계산 꼬리표 (예: b → korail_3b_eval_b.md, 원자료도 따로). 첫 보고서는 덮어쓰지 않음")
    ap.add_argument("--cand", choices=sorted(KA.CAND_SPECS), default="v0",
                    help="3-B 켬 후보 탐색 설정 (D-111a). v0 = D-105e 그대로, v4 = D-110b 채택. v0 이 아니면 --tag 필수")
    a = ap.parse_args()
    if a.cand != "v0" and not a.tag:
        raise SystemExit("--cand v0 가 아니면 --tag 를 주세요 (첫 보고서·재계산 b 덮어쓰기 금지). 예: --cand v4 --tag d111")
    _init_worker(a.cand)                                    # 이 프로세스(작업자 1·앱 화면 사례)도 같은 설정
    name = NAME + (f"_{a.tag}" if a.tag else "")
    _W["out_dir"] = OUT.parent / name
    if a.tag and not a.dry and not a.only and (REP / f"{name}.md").exists():
        raise SystemExit(f"{name}.md 가 이미 있습니다 — 다른 --tag 를 쓰세요 (보고서 덮어쓰기 금지)")
    if not K3B_MART.exists():
        raise SystemExit(f"3-B 상세 정차표가 없습니다: {K3B_MART} (52_build_korail_3b.py 먼저)")
    only = set(a.only or ["B1", "B2", "plans", "B4", "APP"])
    ods_k3b = [(o, d) for o, d, _ in k3b_sample()]
    slots = SLOTS
    if a.dry:
        ods_k3b, slots = ods_k3b[:4] + ods_k3b[-2:], SLOTS[:2] + SLOTS[3:4]
    workers = 1 if a.dry else a.workers
    S: dict = {"n_od": len(ods_k3b), "slots": slots, "variants": {k: str(v) for k, v in VARIANTS.items()},
               "cand": a.cand, "cand_spec": KA.CAND_SPECS[a.cand]}
    prev = REP / f"{name}.json"
    if a.only and prev.exists() and not a.dry:                  # 일부만 다시 돌리면 나머지는 지난 결과 유지
        S = {**json.loads(prev.read_text(encoding="utf-8")), **S}
    T0 = time.time()
    if "B2" in only:
        S["B2"] = b2_data()
        print(f"  → B2 끝 (T4 불일치 {len(S['B2']['t4_mismatch'])})", flush=True)
    if "plans" in only:
        R = {}
        for v in VARIANTS:
            R[v] = compute(v, jobs_for(v, True, slots, ods_k3b), workers, a.force, a.dry)
        base = R["base"]
        S["base"] = analyze_base(base)
        S["B3"] = analyze_b3(base)
        S["B7"] = analyze_b7(base)
        S["B6"] = analyze_b6(base)
        S["B5"] = analyze_b5(R, len(ods_k3b))
        S["CMP"] = {v: compare(base, R[v], v) for v in ("noGC", "nodir", "k20", "maxonly")}
    if "B4" in only:
        e47 = _load("e47", "scripts/v2/47_evaluate_korail_3a2c.py")
        ods_k = [(o, d) for o, d, _ in e47.k_sample(e47.router(e47.spec(on=True)).display.station_key.unique())]
        times = B4_TIMES
        if a.dry:
            ods_k, times = ods_k[:4], B4_TIMES[:2]
        sl = [(B4_DATE, t) for t in times]
        a3 = compute("B4_3a", jobs_for("base", False, sl, ods_k), workers, a.force, a.dry)
        a3b = compute("B4_3b", jobs_for("base", True, sl, ods_k), workers, a.force, a.dry)
        S["B4"] = analyze_b4(a3, a3b)
    if "B1" in only:
        t0 = time.time()
        S["B1_v23"] = b1_v23(a.dry)
        print(f"  → B1 v2.3 끝 ({time.time() - t0:.0f}초, 불일치 {S['B1_v23']['mismatch']})", flush=True)
    if "APP" in only:
        t0 = time.time()
        S["B1_app"] = b1_app(a.dry)
        S["B13_app"] = b13_app(a.dry)
        print(f"  → 앱 B1·B13 끝 ({time.time() - t0:.0f}초)", flush=True)
    S["elapsed_sec"] = round(time.time() - T0)
    if a.dry:
        d = Path(tempfile.gettempdir()) / f"{NAME}_dry"
        d.mkdir(exist_ok=True)
        write_report(S, d, a.tag)
        print(f"연습 보고서: {d / (name + '.md')}")
        print("[연습 실행] 결과는 저장하지 않았습니다 (보고 금지).")
    else:
        write_report(S, REP, a.tag)
        print(f"저장: {REP / (name + '.md')}, {REP / (name + '.json')}")
    return 0


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
