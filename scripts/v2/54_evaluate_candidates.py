"""
scripts/v2/54_evaluate_candidates.py — D-110 후보 탐색 품질 개선 평가 (사전 등록 docs/v2/preregistration_candidates.md, 구현 기록 D-110a)
====================================================================================================================

단계 (--stage)
-------------
    quality   품질 평가: 고유 입력마다 V0~V5 후보를 만들고 합집합을 한 번씩 평가·열차 검증, 방법별로 확정 추천을 고른다 (3-2)
    xcheck    교차 검증 40질의(시드 1101): 방법마다 따로 plan() 한 결과 = 합집합에서 고른 결과 (3-2, 정식 평가 전 필수)
    audit     k=40 심층 감사 (4-2 D1~D6), k=80 은 D2·D3·D5 (quality 결과 필요)
    perf      독립 성능 평가 300질의(시드 1102) + 스트레스 30 — 방법마다 새 계산기, 작업자 1, 순차 (3-3)
    b1        끔 보존 (v2.3 3,720건, 앱 W1 14사례 — 53 과 같은 방법)
    report    결과 모아 reports/v2/candidates_d110.{md,json} — 7절 채택 수식을 기계적으로 적용한 값까지 (판정 기록은 따로)

실행 (사용자 PC)
----
    python scripts/v2/54_evaluate_candidates.py --stage xcheck
    python scripts/v2/54_evaluate_candidates.py --stage quality --workers 8
    python scripts/v2/54_evaluate_candidates.py --stage audit --workers 8
    python scripts/v2/54_evaluate_candidates.py --stage perf
    python scripts/v2/54_evaluate_candidates.py --stage b1
    python scripts/v2/54_evaluate_candidates.py --stage report
    --dry : 작은 표본 연습 (원자료 폴더 _dry, 보고 금지)

중단·재개 (3-4): data/marts/v2/candidates_d110/ 의 묶음별 parquet + manifest.json(설정 해시). 같은 명령을 다시 실행하면
끝난 묶음은 건너뛴다. 설정 해시(코드 커밋·정차표 SHA-256·방법 정의·스크립트 해시)가 다르면 멈춘다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
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
from yeoyuro_v2 import korail_engine as KE       # noqa: E402
from yeoyuro_v2 import time_dependent as td      # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
REP = ROOT / "reports" / "v2"
NAME = "candidates_d110"
OUT0 = ROOT / "data" / "marts" / "v2" / NAME

# 방법 정의 (사전 등록 2절) — 라우터 속성·V5 여부
METHODS = {
    "V0": {"k": 10, "rep_union": False, "group_gc": False, "keep_3a": False},
    "V1": {"k": 20, "rep_union": False, "group_gc": False, "keep_3a": False},
    "V2": {"k": 10, "rep_union": True, "group_gc": False, "keep_3a": False},
    "V3": {"k": 10, "rep_union": False, "group_gc": True, "keep_3a": False},
    "V4": {"k": 10, "rep_union": True, "group_gc": True, "keep_3a": False},
    "V5": {"k": 10, "rep_union": False, "group_gc": False, "keep_3a": True},
}
IDS = list(METHODS)
SIMPLICITY = ["V1", "V5", "V2", "V3", "V4"]          # 7절 ④-③ (앞이 단순)
LOSS_TOL = 0.5
P95_LIMIT = 2.6
DELTA_P95 = 0.05
BUDGET_QUALITY_SEC = 8 * 3600
BUDGET_PERF_SEC = 3600
QUERY_TIMEOUT_SEC = 60.0
SEED_XCHECK, SEED_PERF, SEED_AUDIT = 1101, 1102, 1103
N_XCHECK, N_PERF, N_STRESS, N_AUDIT_RANDOM = 40, 300, 30, 200
B8_KEYS = [("2026-10-14", "08:00", "홍대입구", "망우"), ("2026-10-14", "18:00", "홍대입구", "망우"),
           ("2026-10-17", "14:00", "홍대입구", "망우"), ("2026-10-18", "14:00", "홍대입구", "망우"),
           ("2026-10-14", "23:30", "청담", "회기"), ("2026-10-14", "23:30", "월계", "도곡")]
D5_KEYS = [("2026-10-14", "08:00", "수색", "강남구청")]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------- 입력 (3-1)
def inputs() -> dict:
    """요청 = K3B 312×5 + B4 412×6. 키 = (날짜, 시각, 출발, 도착) — 나머지 키 요소(운행일·요일·켬 상태·옵션·코드·정차표)는
    한 실행 안에서 같으므로 매니페스트 해시로 고정한다."""
    m53 = _load("m53", "scripts/v2/53_evaluate_korail_3b.py")
    e47 = _load("e47", "scripts/v2/47_evaluate_korail_3a2c.py")
    k3b = [(d, h, o, t) for o, t, _ in m53.k3b_sample() for d, h in m53.SLOTS]
    ods_k = [(o, t) for o, t, _ in e47.k_sample(e47.router(e47.spec(on=True)).display.station_key.unique())]
    b4 = [(m53.B4_DATE, h, o, t) for o, t in ods_k for h in m53.B4_TIMES]
    req = k3b + b4
    uniq = sorted(set(req))
    return {"k3b": k3b, "b4": b4, "requested": len(req), "unique": uniq, "dup": len(req) - len(uniq)}


def config_hash(dry: bool) -> dict:
    """이어 계산해도 되는지 보는 설정 지문. D-111a: 현재 커밋(HEAD)은 넣지 않는다 — 보고서만 커밋해도 해시가 바뀌어
    이어하기가 막히던 결함 (D-110 판정 기록 1절). 실행에 쓰는 파일 해시·정차표·방법 정의만 넣는다."""
    mart = KE.K3B_MART
    sha = hashlib.sha256(mart.read_bytes()).hexdigest() if mart.exists() else ""
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    eng = hashlib.sha256((ROOT / "yeoyuro_v2" / "korail_engine.py").read_bytes()).hexdigest()[:12]
    app = hashlib.sha256((ROOT / "yeoyuro_v2" / "korail_app.py").read_bytes()).hexdigest()[:12]
    d = {"k3b_sha256": sha, "script": me, "engine": eng, "app": app,
         "methods": json.dumps(METHODS, sort_keys=True), "dry": dry}
    d["hash"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    return d


def out_dir(dry: bool) -> Path:
    return OUT0.parent / (NAME + ("_dry" if dry else ""))


def check_manifest(dry: bool, stage: str) -> dict:
    d = out_dir(dry)
    d.mkdir(parents=True, exist_ok=True)
    f = d / "manifest.json"
    cur = config_hash(dry)
    if f.exists():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old["hash"] != cur["hash"]:
            raise SystemExit(f"[중단] 설정이 바뀌었습니다 (이전 {old['hash']} / 지금 {cur['hash']}). "
                             f"코드·정차표·방법 정의가 달라진 상태로 이어 계산하지 않습니다. 새 폴더로 다시 하려면 {d} 를 옮기세요.")
    else:
        f.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    return cur


# ---------------------------------------------------------------------- 작업자
_W: dict = {}


def _mods():
    if "mods" not in _W:
        import station_routing as sr
        from yeoyuro_v2.line9 import register_v1_directions
        mod = sr.load_scorer_class(ROOT)
        register_v1_directions(mod)
        _W["mods"] = (mod, sr)
    return _W["mods"]


def _planner(tt_day: str, date: str, router_kw: dict | None = None, keep_3a: bool = False):
    mod, sr = _mods()
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    return KA.ExtensionPlanner(DATA9, mod, sr, disp, tt_day, query_date=date, repo_root=ROOT, include_3b=True,
                               router_kw=router_kw or {}, keep_3a=keep_3a,
                               cand="v0")          # D-111a: 앱 기본값이 바뀌어도 D-110 방법 정의(router_kw)가 그대로 재현되게


def exp_planner(tt_day: str, date: str):
    """품질 평가용 계산기 하나: 대표 대기(전체)를 미리 만들어 두고 방법마다 라우터 속성만 바꾼다."""
    key = (tt_day, date)
    if _W.get("key") != key:
        _W.pop("p", None)
        _W.update(key=key, p=_planner(tt_day, date, {"rep_union": True}))
    return _W["p"]


def _set(R, spec: dict):
    R.k, R.rep_union, R.group_gc = spec["k"], spec["rep_union"], spec["group_gc"]
    R.max_k = max(20, spec["k"])
    R.k_trace = []


def method_paths(P, o, d, m) -> tuple[dict, dict, dict]:
    """방법마다 후보 경로 목록 (실제 노드 순서). 반환 (paths, sec, maxK)."""
    R = P.on
    paths, sec, maxk = {}, {}, {}
    for mid, spec in METHODS.items():
        _set(R, spec)
        t0 = time.perf_counter()
        ps = [tuple(p) for p in R.candidates(o, d, m)]
        if spec["keep_3a"]:
            try:
                a3 = [tuple(p) for p in P.router_3a().candidates(o, d, m)]
            except Exception:                            # noqa: BLE001 — 3-A 계산기에 없는 역
                a3 = []
            paths["A3"] = list(dict.fromkeys(a3))        # 3-A 켬 계산기 후보 (감사 D1·D2 용, 방법 아님)
            ps += a3
        sec[mid] = time.perf_counter() - t0
        maxk[mid] = max(R.k_trace) if R.k_trace else None
        seen, uniq = set(), []
        for p in ps:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        paths[mid] = uniq
    _set(R, METHODS["V0"])
    return paths, sec, maxk


def evaluate_union(P, paths: list, m: float) -> list[dict]:
    """합집합 경로마다 한 번: 평가기(기대 소요·R3) → 운행 있으면 열차 연결 검증 (ExtensionPlanner._view 그대로).
    재사용 키 = 실제 노드 순서 (질의 안에서만) — 물리적 역 순서만 같은 다른 노드 경로는 따로 평가한다."""
    ev = P.on.evaluator(td.bin_index_of(m))
    out = []
    for p in paths:
        q = ev.evaluate(list(p), m)
        row = {"path": "|".join(p), "actual": float(q.actual_time_min), "perceived": float(q.perceived_time_min),
               "excluded": getattr(q, "no_service_reason", None) or ""}
        if not row["excluded"]:
            v = P._view(q, ev, m)
            row.update(status=v.status, ref=bool(v.is_reference), korail=bool(v.korail),
                       lines=",".join(v.lines_3b), chain=bool(KE.multi_line_chain(list(p), lambda n: n.split("_", 1)[1].split("@")[0])))
        out.append(row)
    return out


def select(evals: list[dict], subset: set) -> dict:
    """ExtensionPlanner.plan 과 같은 선택: 기대 소요(소수 6자리) → 체감 순, '연결 확인' 이고 공항철도 아님 = 확정 추천,
    확정 추천 전에 나온 첫 공항철도 '연결 확인' = 참고 경로."""
    rows = [r for r in evals if r["path"] in subset and not r["excluded"]]
    rows.sort(key=lambda r: (round(r["actual"], 6), r["perceived"]))
    best = ref = None
    for r in rows:
        if r["status"] != KA.CONNECT_OK:
            continue
        if r["ref"]:
            if ref is None and best is None:
                ref = r
        elif best is None:
            best = r
    st = "none" if best is None and ref is None else ("reference" if best is None else ("ok" if best["korail"] else "no_benefit"))
    return {"best": None if best is None else best["path"], "best_exp": None if best is None else best["actual"],
            "best_lines": "" if best is None else best["lines"], "ref": None if ref is None else ref["path"],
            "status": st, "n_cand": len([r for r in evals if r["path"] in subset])}


def quality_task(args):
    tt_day, date, hm, ods = args
    P = exp_planner(tt_day, date)
    sd = KA.service_day(date, hm)
    m = float(sd.minute)
    rows = []
    for o, d in ods:
        rec = {"date": date, "hhmm": hm, "o": o, "d": d, "tt_day": tt_day}
        try:
            t0 = time.perf_counter()
            paths, sec, maxk = method_paths(P, o, d, m)
            union = list(dict.fromkeys(p for mid in IDS for p in paths[mid]))
            t1 = time.perf_counter()
            evals = evaluate_union(P, union, m)
            ev_sec = {r["path"]: None for r in evals}
            rec.update(paths=json.dumps({k: ["|".join(p) for p in v] for k, v in paths.items()}, ensure_ascii=False),
                       evals=json.dumps(evals, ensure_ascii=False), gen_sec=json.dumps(sec), maxk=json.dumps(maxk),
                       eval_sec=time.perf_counter() - t1, total_sec=time.perf_counter() - t0, error="",
                       v0_sec=sec["V0"] + (time.perf_counter() - t1) * (len(paths["V0"]) / max(1, len(union))))
            del ev_sec
        except Exception as e:                           # noqa: BLE001 — 기록만 (계산 오류로 셈)
            rec.update(paths="{}", evals="[]", gen_sec="{}", maxk="{}", eval_sec=0.0, total_sec=0.0, v0_sec=0.0,
                       error=f"{type(e).__name__}: {e}"[:300])
        rows.append(rec)
    return rows


def _jobs(keys: list, chunk: int) -> list:
    by: dict = {}
    for d, h, o, t in keys:
        by.setdefault((KA.service_day(d, h).timetable_day_type, d, h), []).append((o, t))
    J = []
    for (tt, d, h), ods in sorted(by.items()):
        for i in range(0, len(ods), chunk):
            J.append((tt, d, h, ods[i:i + chunk]))
    return J


def _chunk_name(j) -> str:
    tt, d, h, ods = j
    return hashlib.sha1(json.dumps([tt, d, h, ods], ensure_ascii=False).encode()).hexdigest()[:16]


def run_quality(keys, workers, dry) -> pd.DataFrame:
    d = out_dir(dry) / "quality"
    d.mkdir(parents=True, exist_ok=True)
    J = _jobs(keys, 6)
    todo = [j for j in J if not (d / f"{_chunk_name(j)}.parquet").exists()]
    print(f"  quality: 묶음 {len(J)}개 중 남은 {len(todo)}개", flush=True)
    t0 = time.time()
    groups: dict = {}
    for j in todo:
        groups.setdefault((j[0], j[1]), []).append(j)
    done = 0
    for _, js in groups.items():
        def save(rows, j):
            f = d / f"{_chunk_name(j)}.parquet"
            tmp = f.with_suffix(".tmp")
            pd.DataFrame(rows).to_parquet(tmp)
            tmp.replace(f)                                # 다 쓴 뒤 이름 바꾸기 (중간에 끊겨도 반쪽 파일 없음)
        if workers <= 1:
            for j in js:
                save(quality_task(j), j)
                done += 1
        else:
            with Pool(min(workers, len(js))) as pool:
                for j, rows in zip(js, pool.imap(quality_task, js, chunksize=1)):
                    save(rows, j)
                    done += 1
                    if done % max(1, len(todo) // 20) == 0:
                        print(f"    {done}/{len(todo)} 묶음 ({time.time() - t0:.0f}초)", flush=True)
    frames = [pd.read_parquet(d / f"{_chunk_name(j)}.parquet") for j in J]
    df = pd.concat(frames, ignore_index=True)
    k = df[["date", "hhmm", "o", "d"]].apply(tuple, axis=1)
    if k.duplicated().any() or len(df) != len(keys):
        raise SystemExit(f"[중단] 결과 무결성: 행 {len(df)} / 입력 {len(keys)} / 중복 {int(k.duplicated().sum())}")
    _log(dry, "quality", {"elapsed_sec": round(time.time() - t0), "chunks": len(J), "new_chunks": len(todo)})
    return df


def _log(dry, stage, info):
    f = out_dir(dry) / "runlog.json"
    log = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    log.setdefault(stage, []).append({**info, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    f.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def load_quality(dry) -> pd.DataFrame:
    d = out_dir(dry) / "quality"
    fs = sorted(d.glob("*.parquet"))
    if not fs:
        raise SystemExit("quality 결과가 없습니다 (--stage quality 먼저)")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


# ---------------------------------------------------------------------- 질의별 방법 결과
def per_method(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in df.itertuples():
        if r.error:
            for mid in IDS:
                out.append({"date": r.date, "hhmm": r.hhmm, "o": r.o, "d": r.d, "method": mid, "error": r.error})
            continue
        paths = json.loads(r.paths)
        evals = json.loads(r.evals)
        gen = json.loads(r.gen_sec)
        maxk = json.loads(r.maxk)
        allp = {e["path"] for e in evals}
        ref = select(evals, allp)
        v0 = set(paths["V0"])
        for mid in IDS:
            s = set(paths[mid])
            sel = select(evals, s)
            ev_m = [e for e in evals if e["path"] in s and not e["excluded"]]
            out.append({"date": r.date, "hhmm": r.hhmm, "o": r.o, "d": r.d, "method": mid, "error": "",
                        **sel, "ref_best": ref["best"], "ref_best_exp": ref["best_exp"],
                        "v0_subset": v0 <= s, "extra_verified": sum(1 for e in ev_m if e["path"] not in v0),
                        "chain": sum(1 for e in ev_m if e.get("chain")), "gen_sec": gen.get(mid),
                        "maxk": maxk.get(mid), "has_ref_path": ref["best"] in s if ref["best"] else None})
    return pd.DataFrame(out)


def losses(pm: pd.DataFrame) -> pd.DataFrame:
    x = pm.copy()
    miss = x.best.isna() & x.ref_best.notna()
    x["loss"] = np.where(x.best.notna() & x.ref_best.notna(), x.best_exp - x.ref_best_exp, np.where(miss, np.inf, 0.0))
    x["missing"] = miss
    return x


def degrade(pm: pd.DataFrame) -> pd.DataFrame:
    """5-3: V0 보다 나쁨 (a) 확정 추천 사라짐 (b) 기대 소요 증가 (c) 참고 경로만."""
    key = ["date", "hhmm", "o", "d"]
    v0 = pm[pm.method == "V0"].set_index(key)
    rows = []
    for mid in IDS[1:]:
        x = pm[pm.method == mid].set_index(key)
        j = v0.join(x, lsuffix="_0", rsuffix="_m", how="inner")
        a = j.best_0.notna() & j.best_m.isna()
        b = j.best_0.notna() & j.best_m.notna() & (j.best_exp_m > j.best_exp_0 + 1e-6)
        c = j.best_0.notna() & j.best_m.isna() & j.ref_m.notna()
        rows.append({"method": mid, "a_lost": int(a.sum()), "b_worse": int(b.sum()), "c_ref_only": int(c.sum()),
                     "examples": j[a | b].reset_index()[key].head(8).to_dict("records")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- 교차 검증 (3-2)
def run_xcheck(keys, dry) -> dict:
    rng = random.Random(SEED_XCHECK)
    sample = rng.sample(keys, min(N_XCHECK if not dry else 4, len(keys)))
    by: dict = {}
    for d, h, o, t in sample:
        by.setdefault((KA.service_day(d, h).timetable_day_type, d), []).append((d, h, o, t))
    bad, n = [], 0
    for (tt, date), ks in sorted(by.items()):
        P = _planner(tt, date, {"rep_union": True})
        sep = {mid: _planner(tt, date, {"k": s["k"], "rep_union": s["rep_union"], "group_gc": s["group_gc"]},
                             keep_3a=s["keep_3a"]) for mid, s in METHODS.items()}
        for d, h, o, t in ks:
            sd = KA.service_day(d, h)
            m = float(sd.minute)
            paths, _, _ = method_paths(P, o, t, m)
            union = []
            for mid in IDS:
                union += [p for p in paths[mid] if p not in union]
            evals = evaluate_union(P, union, m)
            for mid in IDS:
                a = select(evals, {"|".join(p) for p in paths[mid]})
                r = sep[mid].plan(o, t, sd)
                b = {"best": None if r.best is None else "|".join(r.best.path),
                     "ref": None if r.reference is None else "|".join(r.reference.path), "status": r.status,
                     "best_exp": None if r.best is None else float(r.best.expected_min)}
                same = (a["best"] == b["best"] and a["ref"] == b["ref"] and a["status"] == b["status"]
                        and (a["best_exp"] is None and b["best_exp"] is None
                             or (a["best_exp"] is not None and b["best_exp"] is not None and abs(a["best_exp"] - b["best_exp"]) < 1e-6)))
                n += 1
                if not same:
                    bad.append({"key": [d, h, o, t], "method": mid, "union": a, "separate": b})
    res = {"queries": len(sample), "comparisons": n, "mismatch": len(bad), "examples": bad[:10]}
    out_dir(dry).mkdir(parents=True, exist_ok=True)
    (out_dir(dry) / "xcheck.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


# ---------------------------------------------------------------------- 감사 (4-2)
def audit_sets(df: pd.DataFrame, pm: pd.DataFrame, keys_b4: set, dry: bool) -> dict:
    key = ["date", "hhmm", "o", "d"]
    v0 = pm[pm.method == "V0"].set_index(key)
    v1 = pm[pm.method == "V1"].set_index(key)
    D1, D2 = [], []
    for r in df.itertuples():
        k = (r.date, r.hhmm, r.o, r.d)
        if r.error or k not in keys_b4:
            continue
        paths = json.loads(r.paths)
        evals = json.loads(r.evals)
        a3 = select(evals, set(paths.get("A3", [])))             # 3-A 켬 계산기 후보로 고른 3-A 확정 추천
        if a3["best"] and a3["best"] not in set(paths["V0"]):
            D1.append(k)
            b0 = v0.loc[k]
            if b0.best_exp is not None and not pd.isna(b0.best_exp) and b0.best_exp > a3["best_exp"] + 1e-6:
                D2.append(k)
    D4 = [k for k in v0.index if k in v1.index and v0.loc[k].best != v1.loc[k].best]
    fixed = set(D1) | set(D2) | set(B8_KEYS) | set(D4) | set(D5_KEYS)
    rest = sorted(set(v0.index) - fixed)
    rng = random.Random(SEED_AUDIT)
    # 층 = 시각 × V0 확정 추천의 3-B 사용 (3-1), 비례 배분
    strata: dict = {}
    for k in rest:
        b = v0.loc[k]
        s3 = "none" if b.best is None or pd.isna(b.best) else ("3b" if b.best_lines else "no3b")
        strata.setdefault((k[0], k[1], s3), []).append(k)
    n_tot = N_AUDIT_RANDOM if not dry else 3
    D6 = []
    for s, ks in sorted(strata.items()):
        n = max(1, round(n_tot * len(ks) / max(1, len(rest))))
        D6 += rng.sample(ks, min(n, len(ks)))
    return {"D1": D1, "D2": D2, "D3": [k for k in B8_KEYS if k in v0.index], "D4": D4,
            "D5": [k for k in D5_KEYS if k in v0.index], "D6": D6[:n_tot] if len(D6) > n_tot else D6}


def audit_task(args):
    tt, date, hm, items = args                 # items: [(o, d, k_list, union_paths_json)]
    P = exp_planner(tt, date)
    R = P.on
    sd = KA.service_day(date, hm)
    m = float(sd.minute)
    rows = []
    for o, d, ks, have in items:
        have = set(json.loads(have))
        rec = {"date": date, "hhmm": hm, "o": o, "d": d}
        for kk in ks:
            _set(R, {"k": kk, "rep_union": False, "group_gc": False})
            R.max_k = kk
            t0 = time.perf_counter()
            try:
                ps = ["|".join(p) for p in R.candidates(o, d, m)]
            except Exception as e:                       # noqa: BLE001
                rec[f"k{kk}_error"] = f"{type(e).__name__}: {e}"[:200]
                continue
            new = [tuple(p.split("|")) for p in ps if p not in have]
            ev = evaluate_union(P, new, m)
            rec[f"k{kk}_new"] = json.dumps(ev, ensure_ascii=False)
            rec[f"k{kk}_sec"] = time.perf_counter() - t0
        _set(R, METHODS["V0"])
        rows.append(rec)
    return rows


def run_audit(df, pm, keys_b4, workers, dry) -> pd.DataFrame:
    sets = audit_sets(df, pm, keys_b4, dry)
    k80 = set(sets["D2"]) | set(sets["D3"]) | set(sets["D5"])
    allk = sorted(set().union(*[set(v) for v in sets.values()]))
    uni = {(r.date, r.hhmm, r.o, r.d): json.dumps([e["path"] for e in json.loads(r.evals)], ensure_ascii=False)
           for r in df.itertuples()}
    by: dict = {}
    for k in allk:
        d, h, o, t = k
        by.setdefault((KA.service_day(d, h).timetable_day_type, d, h), []).append(
            (o, t, [40, 80] if k in k80 else [40], uni.get(k, "[]")))
    J = [(tt, d, h, its[i:i + 4]) for (tt, d, h), its in sorted(by.items()) for i in range(0, len(its), 4)]
    f = out_dir(dry) / "audit.parquet"
    if f.exists():
        print("  audit: 저장본 사용", flush=True)
        a = pd.read_parquet(f)
    else:
        print(f"  audit: 질의 {len(allk)}개 (k=80 {len(k80)})", flush=True)
        rows = []
        if workers <= 1:
            for j in J:
                rows += audit_task(j)
        else:
            with Pool(min(workers, len(J))) as pool:
                for rs in pool.imap_unordered(audit_task, J, chunksize=1):
                    rows += rs
        a = pd.DataFrame(rows)
        a.to_parquet(f)
    (out_dir(dry) / "audit_sets.json").write_text(json.dumps({k: [list(x) for x in v] for k, v in sets.items()},
                                                             ensure_ascii=False, indent=1), encoding="utf-8")
    return a


# ---------------------------------------------------------------------- 성능 (3-3)
def perf_sample(df: pd.DataFrame, pm: pd.DataFrame, dry: bool) -> tuple[list, list]:
    key = ["date", "hhmm", "o", "d"]
    v0 = pm[pm.method == "V0"].set_index(key)
    strata: dict = {}
    for k in sorted(v0.index):
        b = v0.loc[k]
        s3 = "none" if b.best is None or pd.isna(b.best) else ("3b" if b.best_lines else "no3b")
        strata.setdefault((k[0], k[1], s3), []).append(k)
    n_tot = N_PERF if not dry else 6
    rng = random.Random(SEED_PERF)
    S = []
    for s, ks in sorted(strata.items()):
        n = max(1, round(n_tot * len(ks) / len(v0)))
        S += rng.sample(ks, min(n, len(ks)))
    t = df[~df.error.astype(bool)].assign(k=lambda x: list(zip(x.date, x.hhmm, x.o, x.d)))
    stress = list(t.sort_values("v0_sec", ascending=False).k.head(N_STRESS if not dry else 2))
    return S, stress


def run_perf(df, pm, dry) -> dict:
    try:
        import psutil
        proc = psutil.Process()
    except Exception:                                    # noqa: BLE001
        proc = None
    S, stress = perf_sample(df, pm, dry)
    res = {"sample": len(S), "stress": len(stress), "methods": {}, "psutil": proc is not None}
    for mid, spec in METHODS.items():
        t_all = time.time()
        rec = {"plan_sec": [], "gen_sec": [], "build_sec": [], "stress_sec": [], "rss_max_mb": None, "errors": 0}
        for label, ks in (("main", S), ("stress", stress)):
            by: dict = {}
            for k in ks:
                by.setdefault((KA.service_day(k[0], k[1]).timetable_day_type, k[0]), []).append(k)
            for (tt, date), kk in sorted(by.items()):
                t0 = time.perf_counter()
                P = _planner(tt, date, {"k": spec["k"], "rep_union": spec["rep_union"], "group_gc": spec["group_gc"]},
                             keep_3a=spec["keep_3a"])
                if spec["keep_3a"]:
                    P.router_3a()
                rec["build_sec"].append(time.perf_counter() - t0)
                gen = {"t": 0.0}
                orig = P.on.candidates

                def timed(*a, _o=orig, **kw):
                    t1 = time.perf_counter()
                    try:
                        return _o(*a, **kw)
                    finally:
                        gen["t"] += time.perf_counter() - t1
                P.on.candidates = timed
                warm = kk[0]
                P.plan(warm[2], warm[3], KA.service_day(warm[0], warm[1]))           # 예열 (측정 안 함)
                for k in kk:
                    gen["t"] = 0.0
                    t1 = time.perf_counter()
                    try:
                        P.plan(k[2], k[3], KA.service_day(k[0], k[1]))
                    except Exception:                    # noqa: BLE001
                        rec["errors"] += 1
                        continue
                    dt_ = time.perf_counter() - t1
                    if label == "main":
                        rec["plan_sec"].append(dt_)
                        rec["gen_sec"].append(gen["t"])
                    else:
                        rec["stress_sec"].append(dt_)
                    if proc is not None:
                        rss = proc.memory_info().rss / 2 ** 20
                        rec["rss_max_mb"] = max(rec["rss_max_mb"] or 0, rss)
                del P
        ps = np.array(rec["plan_sec"])
        res["methods"][mid] = {
            "n": int(len(ps)), "mean": float(ps.mean()) if len(ps) else None,
            "p95": float(np.quantile(ps, 0.95)) if len(ps) else None, "max": float(ps.max()) if len(ps) else None,
            "gen_mean": float(np.mean(rec["gen_sec"])) if rec["gen_sec"] else None,
            "eval_mean": float(np.mean(ps) - np.mean(rec["gen_sec"])) if rec["gen_sec"] else None,
            "build_mean": float(np.mean(rec["build_sec"])), "stress_p95": float(np.quantile(rec["stress_sec"], 0.95)) if rec["stress_sec"] else None,
            "stress_max": float(max(rec["stress_sec"])) if rec["stress_sec"] else None,
            "rss_max_mb": rec["rss_max_mb"], "errors": rec["errors"], "batch_sec": round(time.time() - t_all),
            "over_budget": (time.time() - t_all) > BUDGET_PERF_SEC}
        print(f"  perf {mid}: p95 {res['methods'][mid]['p95']:.3f}초 · 평균 {res['methods'][mid]['mean']:.3f} "
              f"({res['methods'][mid]['batch_sec']}초)", flush=True)
    (out_dir(dry) / "perf.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------------- 분석·보고
def adopt(M: dict, elig: dict, p95: dict, extra: dict) -> dict:
    """7절 수식 그대로."""
    M0 = M["V0"]
    E = [i for i in IDS[1:] if elig[i]]
    if M0 == 0:
        return {"choice": "V0", "why": "M0 = 0", "E": E}
    Iset = [i for i in E if M[i] < M0]
    if not Iset:
        return {"choice": "V0", "why": "I = ∅ (자격 통과 방법 중 V0 보다 놓침이 적은 방법 없음)", "E": E}
    Mb = min(M[i] for i in Iset)
    D = math.floor(0.10 * M0)
    C = [i for i in Iset if M[i] - Mb <= D]
    pmin = min(p95[i] for i in C)
    T = [i for i in C if p95[i] - pmin <= DELTA_P95]
    for keyf in (lambda i: M[i], lambda i: extra[i], lambda i: SIMPLICITY.index(i), lambda i: IDS.index(i)):
        best = min(keyf(i) for i in T)
        T = [i for i in T if keyf(i) == best]
        if len(T) == 1:
            break
    return {"choice": T[0], "E": E, "I": Iset, "M_best": Mb, "Delta": D, "C": C, "p_min": pmin, "T_final": T}


def v0_paths(df: pd.DataFrame) -> dict:
    """(date, hhmm, o, d) → V0 후보 경로 집합. D-111a: 행마다 df.set_index().loc 하던 것을 한 번에 (보고서 단계 속도)."""
    out = {}
    for k, ps in zip(zip(df.date, df.hhmm, df.o, df.d), df.paths):
        if k in out:
            raise ValueError(f"질의 키 중복: {k}")
        out[k] = set(json.loads(ps)["V0"])
    return out


def report(dry: bool) -> dict:
    ins = inputs()
    df = load_quality(dry)
    pm = per_method(df)
    L = losses(pm)
    dg = degrade(pm).set_index("method")
    key = ["date", "hhmm", "o", "d"]
    S: dict = {"inputs": {"requested": ins["requested"], "duplicate": ins["dup"], "unique": len(ins["unique"])},
               "config": json.loads((out_dir(dry) / "manifest.json").read_text(encoding="utf-8")),
               "runlog": json.loads((out_dir(dry) / "runlog.json").read_text(encoding="utf-8")) if (out_dir(dry) / "runlog.json").exists() else {}}
    xf, pf, af = out_dir(dry) / "xcheck.json", out_dir(dry) / "perf.json", out_dir(dry) / "audit.parquet"
    S["xcheck"] = json.loads(xf.read_text(encoding="utf-8")) if xf.exists() else None
    perf = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else None
    S["perf"] = perf
    b1f = out_dir(dry) / "b1.json"
    S["B1"] = json.loads(b1f.read_text(encoding="utf-8")) if b1f.exists() else None
    elapsed_q = sum(x["elapsed_sec"] for x in S["runlog"].get("quality", []))
    S["quality_elapsed_sec"] = elapsed_q
    keys_k3b, keys_b4 = set(ins["k3b"]), set(ins["b4"])
    v0p = v0_paths(df)
    M, elig, p95, extra, rows = {}, {}, {}, {}, {}
    errors_q = int(df.error.astype(bool).sum())
    for mid in IDS:
        x = L[(L.method == mid) & (L.error == "")]
        fin = x[np.isfinite(x.loss)]
        M[mid] = int(((x.loss > LOSS_TOL) | x.missing).sum())
        extra[mid] = int(x.extra_verified.sum())
        r = {"M": M[mid], "loss_gt0": int((x.loss > 1e-9).sum()), "loss_gt1": int((x.loss > 1.0).sum()),
             "missing": int(x.missing.sum()),
             "loss_median": float(fin.loss[fin.loss > 1e-9].median()) if (fin.loss > 1e-9).any() else 0.0,
             "loss_p95": float(fin.loss.quantile(0.95)), "loss_max": float(fin.loss.max()),
             "match_ref": float((x.best.fillna("∅") == x.ref_best.fillna("∅")).mean()),
             "ref_in_cands": float(x.has_ref_path.dropna().mean()) if x.has_ref_path.notna().any() else None,
             "extra_verified": extra[mid],
             "extra_adopted": int(sum(1 for z in x.itertuples() if z.best and mid != "V0" and z.best not in
                                      v0p[(z.date, z.hhmm, z.o, z.d)])),
             "n_cand_mean": float(x.n_cand.mean()), "v0_subset_fail": int((~x.v0_subset).sum()),
             "chain": int(x.chain.sum()), "best_ar": int(x.best_lines.fillna("").str.contains("AR").sum()),
             "status": x.status.value_counts().to_dict(), "maxk_max": float(x.maxk.max()) if x.maxk.notna().any() else None,
             "gen_sec_max": float(x.gen_sec.max()),
             "timeouts": int((x.gen_sec > QUERY_TIMEOUT_SEC).sum()),
             "M_k3b": int(((x.loss > LOSS_TOL) | x.missing)[[k in keys_k3b for k in zip(x.date, x.hhmm, x.o, x.d)]].sum()),
             "M_b4": int(((x.loss > LOSS_TOL) | x.missing)[[k in keys_b4 for k in zip(x.date, x.hhmm, x.o, x.d)]].sum()),
             "M_by_time": {f"{a} {b}": int(((g.loss > LOSS_TOL) | g.missing).sum()) for (a, b), g in x.groupby(["date", "hhmm"])}}
        if mid != "V0":
            r.update(degrade_a=int(dg.loc[mid].a_lost), degrade_b=int(dg.loc[mid].b_worse),
                     degrade_c=int(dg.loc[mid].c_ref_only), degrade_examples=dg.loc[mid].examples)
        p = (perf or {}).get("methods", {}).get(mid, {})
        p95[mid] = p.get("p95") if p.get("p95") is not None else float("inf")
        r["perf"] = p
        b1ok = bool(S["B1"] and S["B1"]["v23_mismatch"] == 0 and S["B1"]["app_same"] == S["B1"]["app_cases"])
        conds = {"B1": b1ok, "B3": r["chain"] == 0, "B10": True, "B13": r["best_ar"] == 0,
                 "errors": errors_q == 0 and r["timeouts"] == 0 and not p.get("errors"),
                 "no_degrade": mid == "V0" or (r["degrade_a"] + r["degrade_b"] + r["degrade_c"] == 0),
                 "p95": p95[mid] <= P95_LIMIT, "budget": elapsed_q <= BUDGET_QUALITY_SEC and not p.get("over_budget", True),
                 "xcheck": bool(S["xcheck"] and S["xcheck"]["mismatch"] == 0)}
        r["conditions"] = conds
        elig[mid] = all(conds.values())
        rows[mid] = r
    S["methods"] = rows
    S["errors_quality"] = errors_q
    S["error_examples"] = df[df.error.astype(bool)].head(5)[key + ["error"]].to_dict("records")
    # 표적 사례
    if af.exists():
        a = pd.read_parquet(af)
        sets = json.loads((out_dir(dry) / "audit_sets.json").read_text(encoding="utf-8"))
        ev_by = {(r.date, r.hhmm, r.o, r.d): json.loads(r.evals) for r in df.itertuples()}
        aud = {}
        for r in a.itertuples():
            k = (r.date, r.hhmm, r.o, r.d)
            base = ev_by.get(k, [])
            refb = select(base, {e["path"] for e in base})
            for kk in (40, 80):
                col = f"k{kk}_new"
                if col in a.columns and isinstance(getattr(r, col), str):
                    new = json.loads(getattr(r, col))
                    allev = base + new
                    s2 = select(allev, {e["path"] for e in allev})
                    aud.setdefault(kk, []).append({"key": list(k), "ref": refb["best_exp"], "audit": s2["best_exp"],
                                                   "better": (s2["best_exp"] is not None and (refb["best_exp"] is None or s2["best_exp"] < refb["best_exp"] - 1e-6))})
        S["audit"] = {"sets": {k: len(v) for k, v in sets.items()},
                      "k40_better": sum(x["better"] for x in aud.get(40, [])), "k40_n": len(aud.get(40, [])),
                      "k80_better": sum(x["better"] for x in aud.get(80, [])), "k80_n": len(aud.get(80, [])),
                      "k40_examples": [x for x in aud.get(40, []) if x["better"]][:10]}
        tgt = {}
        for name, ks in sets.items():
            ks = {tuple(x) for x in ks}
            tgt[name] = {mid: int(sum(1 for z in L[(L.method == mid)].itertuples()
                                      if (z.date, z.hhmm, z.o, z.d) in ks and z.best == z.ref_best)) for mid in IDS}
            tgt[name]["n"] = len(ks)
        S["targets"] = tgt
    S["named"] = {f"{d} {h} {o}→{t}": {mid: (lambda z: None if z.empty else {"best_exp": z.iloc[0].best_exp, "status": z.iloc[0].status})(
        L[(L.method == mid) & (L.date == d) & (L.hhmm == h) & (L.o == o) & (L.d == t)]) for mid in IDS}
        for d, h, o, t in B8_KEYS + D5_KEYS}
    S["adopt"] = adopt(M, elig, p95, extra)
    S["eligible"] = elig
    write_report(S, dry)
    return S


def write_report(S: dict, dry: bool) -> None:
    R = S["methods"]
    L = ["# D-110 후보 탐색 품질 개선 — 평가 결과", "",
         "- 규칙: [사전 등록](../../docs/v2/preregistration_candidates.md) D-110 (계산 전 고정) · 구현 기록 D-110a",
         "- 재현: `python scripts/v2/54_evaluate_candidates.py --stage xcheck|quality|audit|perf|b1|report`",
         f"- 입력: 요청 {S['inputs']['requested']:,} · 중복 {S['inputs']['duplicate']} · 고유 {S['inputs']['unique']:,}",
         f"- 설정 해시 {S['config']['hash']} (커밋 {S['config']['head'][:7]}, 정차표 {S['config']['k3b_sha256'][:12]})",
         f"- 교차 검증 (3-2): {S['xcheck']}" if S["xcheck"] else "- 교차 검증: **실행 안 함**",
         f"- 품질 평가 계산 시간 {S['quality_elapsed_sec']/3600:.2f}시간 (예산 8시간) · 계산 오류 {S['errors_quality']}",
         "- 참값 = V0~V5 후보 합집합에서 같은 평가기·검증기·선택 규칙으로 고른 최선 (Best-known reference, 전역 최적 아님)", "",
         "| 방법 | M (손실>0.5분·누락) | K3B / B4 | 손실>0 · >1 | 누락 | 손실 P95 / 최대 | 참값 일치 | V0⊆ 실패 | 추가 검증 후보 | 나쁨 a·b·c | p95 (초) | 자격 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mid in IDS:
        r = R[mid]
        p = r["perf"] or {}
        dgs = "—" if mid == "V0" else f"{r['degrade_a']}·{r['degrade_b']}·{r['degrade_c']}"
        L.append(f"| {mid} | **{r['M']}** | {r['M_k3b']} / {r['M_b4']} | {r['loss_gt0']} · {r['loss_gt1']} | {r['missing']} | "
                 f"{r['loss_p95']:.2f} / {r['loss_max']:.2f} | {r['match_ref']:.3f} | {r['v0_subset_fail']} | {r['extra_verified']} | "
                 f"{dgs} | {p.get('p95') if p.get('p95') is None else round(p['p95'], 3)} | "
                 f"{'예' if S['eligible'][mid] else '아니오 ' + str([k for k, v in r['conditions'].items() if not v])} |")
    a = S["adopt"]
    L += ["", "## 7절 채택 수식 적용 (기계적 계산 — 판정 기록은 따로)", "", f"- {a}", "",
          "## 성능 (3-3 독립 측정, 작업자 1·순차, 품질 평가 재사용 없음)", "",
          "| 방법 | n | 평균 | p95 | 최대 | 후보 생성 평균 | 평가·검증 평균 | 계산기 준비 | 스트레스 p95 / 최대 | 최대 메모리 MB | 배치 초 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mid in IDS:
        p = R[mid]["perf"] or {}
        f = (lambda v: "-" if v is None else f"{v:.3f}")
        L.append(f"| {mid} | {p.get('n')} | {f(p.get('mean'))} | {f(p.get('p95'))} | {f(p.get('max'))} | {f(p.get('gen_mean'))} | "
                 f"{f(p.get('eval_mean'))} | {f(p.get('build_mean'))} | {f(p.get('stress_p95'))} / {f(p.get('stress_max'))} | "
                 f"{p.get('rss_max_mb') and round(p['rss_max_mb'])} | {p.get('batch_sec')} |")
    L += ["", "## 필수 조건 (6절)", ""]
    for mid in IDS:
        L.append(f"- {mid}: {R[mid]['conditions']} · B3 이어 갈아타기 {R[mid]['chain']} · 확정 추천 공항철도 {R[mid]['best_ar']} · "
                 f"최대 K {R[mid]['maxk_max']} · 시간 초과 {R[mid]['timeouts']}")
    L += ["", f"- B1: {S['B1']}", "", "## 표적 사례 (4-2 감사 표본에서 참값과 같은 확정 추천 수)", "", f"- {S.get('targets')}",
          f"- 지정 사례: {S['named']}", "", "## k=40·80 심층 감사", "", f"- {S.get('audit')}", "",
          "## 시각별 M", ""] + [f"- {mid}: {R[mid]['M_by_time']}" for mid in IDS] + [
          "", "## 나쁨 사례 (5-3)", ""] + [f"- {mid}: {R[mid].get('degrade_examples')}" for mid in IDS[1:]] + [
          "", "## 한계", "",
          "- 참값은 실험 방법들의 합집합 최선이다. k=40·80 감사는 고정 감사 표본에서만 했다",
          "- 대표 대기(V2·V4)는 출발 30분 구간 하나의 정적 근사이며 최종 평가는 시간 의존 E2·열차 검증 그대로",
          "- 계획 시각표 기준이며 실제 운행은 반영하지 않는다", ""]
    dest = REP if not dry else Path(tempfile.gettempdir())
    (dest / f"{NAME}.md").write_text("\n".join(str(x) for x in L) + "\n", encoding="utf-8", newline="\n")
    (dest / f"{NAME}.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str) + "\n",
                                       encoding="utf-8", newline="\n")
    print(f"저장: {dest / (NAME + '.md')}")


def run_b1(dry) -> dict:
    m53 = _load("m53b", "scripts/v2/53_evaluate_korail_3b.py")
    v = m53.b1_v23(dry)
    app = m53.b1_app(dry)
    res = {"v23_cases": v["cases"], "v23_mismatch": v["mismatch"], "app_cases": app["cases"], "app_same": app["same"],
           "app_diffs": app["diffs"]}
    (out_dir(dry) / "b1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["quality", "xcheck", "audit", "perf", "b1", "report"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if not KE.K3B_MART.exists():
        raise SystemExit(f"3-B 상세 정차표가 없습니다: {KE.K3B_MART}")
    cfg = check_manifest(a.dry, a.stage)
    print(f"설정 해시 {cfg['hash']} (커밋 {cfg['head'][:7]})", flush=True)
    ins = inputs()
    keys = ins["unique"]
    if a.dry:
        rng = random.Random(7)
        keys = sorted(rng.sample(keys, 8))
    print(f"입력: 요청 {ins['requested']:,} · 중복 {ins['dup']} · 고유 {len(ins['unique']):,}" + (" (연습: 8개)" if a.dry else ""), flush=True)
    t0 = time.time()
    if a.stage == "xcheck":
        r = run_xcheck(keys, a.dry)
        print(f"교차 검증: 비교 {r['comparisons']} · 불일치 {r['mismatch']}")
        if r["mismatch"]:
            print("[주의] 불일치가 있으면 정식 평가로 넘어가지 마세요 (출력을 붙여 주세요)")
    elif a.stage == "quality":
        run_quality(keys, 1 if a.dry else a.workers, a.dry)
    elif a.stage == "audit":
        df = load_quality(a.dry)
        run_audit(df, per_method(df), set(ins["b4"]), 1 if a.dry else a.workers, a.dry)
    elif a.stage == "perf":
        df = load_quality(a.dry)
        run_perf(df, per_method(df), a.dry)
    elif a.stage == "b1":
        r = run_b1(a.dry)
        print(f"B1: v2.3 불일치 {r['v23_mismatch']} · 앱 {r['app_same']}/{r['app_cases']}")
    elif a.stage == "report":
        S = report(a.dry)
        print(f"채택 수식 결과: {S['adopt']['choice']}")
    print(f"끝 ({time.time() - t0:.0f}초)")
    if a.dry:
        print("[연습 실행] 결과는 보고에 쓰지 않습니다.")
    return 0


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
