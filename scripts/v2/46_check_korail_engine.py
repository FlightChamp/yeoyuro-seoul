"""
scripts/v2/46_check_korail_engine.py
====================================
Phase 3A-2b — 코레일 엔진 확장의 구조 검증 (사전 등록 D-095 6-1절: V1, T1, T1a, T1b, T2, T3, T5).

V2~V9(경로 변경·역전·E vs H·민감도·충분성)는 3A-2c 에서 따로 계산한다. 이 스크립트는 '엔진이 설계대로 붙었는가'만 본다.

검사
----
V1  기능 끔: v2.3 표본 3,720건을 다시 계산해 커밋된 od_shiftability_mart 의 기준 경로 소요·최대 혼잡·100%+ 노출과 비교 (불일치 0)
T1  경인 ↔ 경부 남쪽 24개 방향쌍 × 평일 6개 출발, 후보 전부:
    구로에서 바로 갈아타는 후보 = 구로 갈아타기 간선 1번·보행 1분 포함·시각표 연결의 두 열차번호가 다름
    모든 후보 = 열차 단위 '불가능한 연결' 0 (경인 ↔ 경부를 열차 변경 없이 잇는 후보 없음)
T1a 구일 → 구로(경인) → 신도림 → 구로(경부) → 가산디지털단지: 재방문 규칙이 빼는지 + 강제로 평가하면 '불가능한 연결'
    + T1·T2 후보 전부 같은 역 재방문 0
T1b 구로 출발·도착: 두 노드 모두 후보, 각 방면 무환승
T2  직통 OD 1순위: 갈아타기 0·보행 0
T3  하차역별 E: 같은 승차역·시각에서 내릴 역이 다르면 다른 열차 집합으로 E
T5  같은 프로세스에서 끔 → 켬 → 끔: 원본 스코어러 간선·v1 상수·표시 역 불변, 끈 라우터 결과 동일

실행
----
    python scripts/v2/46_check_korail_engine.py            (V1 3,720건 포함, 약 3~5분)
    python scripts/v2/46_check_korail_engine.py --quick    (V1 300건만)

산출물
------
    reports/v2/korail_engine_3a2b.md · reports/v2/korail_engine_3a2b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import korail_line1 as K           # noqa: E402
from yeoyuro_v2 import korail_engine as KE         # noqa: E402
from yeoyuro_v2 import time_dependent as td        # noqa: E402
from yeoyuro_v2 import train_timetable as T        # noqa: E402
from yeoyuro_v2 import load_v1                     # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
REPORT = ROOT / "reports" / "v2" / "korail_engine_3a2b.md"
OUT_JSON = ROOT / "reports" / "v2" / "korail_engine_3a2b.json"
BASE_TIMES = ["07:30", "08:00", "08:30", "17:30", "18:00", "18:30"]
DIRECT_OD = [("신도림", "구일"), ("신도림", "가산디지털단지"), ("시청", "남영"), ("종로3가", "회기"),
             ("온수", "서울역"), ("용산", "서울역"), ("청량리", "노량진"), ("도봉산", "광운대")]


def name(code):
    return K.SCOPE[code]


def v1_check(n_limit=None) -> dict:
    mod, sr, disp = load_v1(ROOT)
    m = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "od_shiftability_mart.parquet")
    if n_limit:
        m = m.sample(n=n_limit, random_state=42)
    r = KE.KorailTDRouter(ROOT, mod, sr, disp, "weekday", policy="step", k=5, multi_bin=True, include_korail=False)
    bad, n = [], 0
    for row in m.itertuples():
        tb = td.parse_hhmm(row.base_time)
        res = r.evaluate_od(row.origin, row.destination, tb)
        if not res:
            bad.append(f"{row.origin}→{row.destination} {row.base_time}: 경로 없음")
            continue
        b = r.evaluator(td.bin_index_of(tb)).evaluate(res[0].path, tb, keep_trace=True)
        same = (abs(round(b.actual_time_min, 2) - row.base_actual_min) < 1e-6
                and abs(round(b.max_congestion, 1) - row.base_max_congestion) < 1e-6
                and abs(round(b.exposure[100], 2) - row.base_exposure_100_min) < 0.006)
        if not same:
            bad.append(f"{row.origin}→{row.destination} {row.base_time}: 소요 {b.actual_time_min:.2f}/{row.base_actual_min} "
                       f"최대 {b.max_congestion:.1f}/{row.base_max_congestion} 노출 {b.exposure[100]:.2f}/{row.base_exposure_100_min}")
        n += 1
    return {"cases": int(len(m)), "evaluated": n, "mismatch": len(bad), "examples": bad[:10]}


def _router9(on=True, **kw):
    sys.path[:0] = [str(ROOT / "app" / "streamlit")]
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = sr.load_display_master(DATA9)
    return KE.KorailTDRouter(DATA9, mod, sr, disp, "weekday", policy="step", k=5, multi_bin=True,
                             include_korail=on, **kw), sr


def _has_guro_transfer(p):
    return any({a, b} == set(KE.GURO_NODES) for a, b in zip(p, p[1:]))


_FULL_IDX = {}


def full_index():
    """1~9호선 전체 열차 단위 색인 (평일). 라우터의 kidx 는 1호선만 가지고 있음."""
    if "w" not in _FULL_IDX:
        st = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "train_timetable" / "stop_times.parquet")
        _FULL_IDX["w"] = T.TrainIndex(st[st.day_type == "weekday"])
    return _FULL_IDX["w"]


def t1_t1a_t2(r, sr) -> dict:
    tidx = full_index()
    out = {"T1": {"pairs": 0, "candidates": 0, "direct_guro": 0, "direct_guro_ok": 0, "impossible": 0,
                  "revisit": 0, "top1_guro": 0, "problems": []}}
    pairs = [(a, b) for a in K.GYEONGIN_SOUTH for b in K.GYEONGBU_SOUTH]
    pairs = pairs + [(b, a) for a, b in pairs]
    for a, b in pairs:
        o, d = name(a), name(b)
        out["T1"]["pairs"] += 1
        for bt in BASE_TIMES:
            tb = td.parse_hhmm(bt)
            ev = r.evaluator(td.bin_index_of(tb))
            cands = r.candidates(o, d, tb)
            res = r.evaluate_od(o, d, tb)
            if res and _has_guro_transfer(res[0].path):
                out["T1"]["top1_guro"] += 1
            for p in cands:
                out["T1"]["candidates"] += 1
                if sr.has_station_revisit(p):
                    out["T1"]["revisit"] += 1
                legs = T.legs_from_path(ev, p)
                if T.impossible_legs(tidx, legs, "weekday"):
                    out["T1"]["impossible"] += 1
                    out["T1"]["problems"].append(f"불가능: {o}→{d} {p}")
                cc = T.verify_connection(tidx, legs, tb, "weekday")
                if cc.status == T.CONNECTED and same_train_after_transfer(cc):
                    out["T1"]["problems"].append(f"중복 환승: {o}→{d} {bt} {p}")
                    out["T1"]["duplicate"] = out["T1"].get("duplicate", 0) + 1
                if _has_guro_transfer(p):
                    out["T1"]["direct_guro"] += 1
                    n_g = sum({x, y} == set(KE.GURO_NODES) for x, y in zip(p, p[1:]))
                    rr = ev.evaluate(p, tb)
                    c = T.verify_connection(tidx, legs, tb, "weekday")
                    gi = [i for i, lg in enumerate(legs) if lg.board_node in KE.GURO_NODES and lg.walk_before > 0]
                    ok = (n_g == 1 and rr.transfer_walk_min >= KE.GURO_WALK_MIN - 1e-9 and gi
                          and legs[gi[0]].walk_before == KE.GURO_WALK_MIN)
                    if c.status == T.CONNECTED and gi:
                        # 구로에서 내린 열차와 구로에서 탄 열차 (다른 구간의 이어 타기가 있어도 열차 순서로 찾음)
                        ks = [k for k in range(1, len(c.rides))
                              if c.rides[k - 1].alight_code == K.GURO and c.rides[k].board_code == K.GURO]
                        ok = ok and any(c.rides[k - 1].train != c.rides[k].train
                                        and c.rides[k].dep_min >= c.rides[k - 1].arr_min + KE.GURO_WALK_MIN - 1e-9
                                        for k in ks)
                    if ok:
                        out["T1"]["direct_guro_ok"] += 1
                    else:
                        out["T1"]["problems"].append(f"구로 갈아타기 조건 불충족: {o}→{d} {bt} {p}")
    out["T1"]["dropped_invalid_guro"] = getattr(r, "dropped_guro", 0)
    # T1a: 재방문 우회 경로를 직접 만들어 본다
    detour = ["1_구일", KE.GURO_GI, "1_신도림", KE.GURO_GB, "1_가산디지털단지"]
    ev = r.evaluator(td.bin_index_of(480))
    forced = ev.evaluate(detour, 480)
    legs = T.legs_from_path(ev, detour)
    c = T.verify_connection(full_index(), legs, 480, "weekday")
    out["T1a"] = {"path": detour, "revisit_rule_excludes": bool(sr.has_station_revisit(detour)),
                  "forced_transfer_count": forced.transfer_count, "forced_actual_min": round(forced.actual_time_min, 2),
                  "train_check": c.status, "in_any_candidate": False}
    for bt in BASE_TIMES:
        if any(p == detour for p in r.candidates("구일", "가산디지털단지", td.parse_hhmm(bt))):
            out["T1a"]["in_any_candidate"] = True
    # T1b: 구로 출발·도착
    nodes = sr.candidates_of(r.display, "구로")
    tb_rows = []
    for o, d in [("구로", "구일"), ("구로", "가산디지털단지"), ("구로", "신도림"), ("구일", "구로"),
                 ("가산디지털단지", "구로")]:
        res = r.evaluate_od(o, d, 480)
        q = res[0] if res else None
        tb_rows.append({"od": f"{o}→{d}", "ok": bool(q and q.transfer_count == 0),
                        "path": q.path if q else None, "min": round(q.actual_time_min, 1) if q else None})
    out["T1b"] = {"guro_candidates": nodes, "both_nodes": set(KE.GURO_NODES) <= set(nodes), "cases": tb_rows}
    # T2: 직통
    rows = []
    for o, d in DIRECT_OD:
        res = r.evaluate_od(o, d, 480)
        q = res[0] if res else None
        rows.append({"od": f"{o}→{d}", "transfers": q.transfer_count if q else None,
                     "walk": round(q.transfer_walk_min, 2) if q else None, "path": q.path if q else None,
                     "revisit": bool(q and sr.has_station_revisit(q.path))})
    out["T2"] = rows
    return out


def same_train_after_transfer(c) -> bool:
    """시각표 연결에서 갈아탄 뒤 방금 내린 열차를 다시 탄 경우 (중복 환승, T2)."""
    return any(a.train == b.train and a.group == b.group for a, b in zip(c.rides, c.rides[1:])
               if a.alight_code == b.board_code and b.dep_min > a.arr_min)


def audit_on(r, sr, n=300) -> dict:
    """켠 그래프 감사: 한쪽 끝이 코레일 역인 무작위 OD(시드 42) 후보 전부를 열차 단위로 따라감."""
    names = sorted(r.display.station_key.unique())
    kor = sorted({K.SCOPE[c] for c in K.SEOUL_KORAIL})
    rng = random.Random(42)
    out = {"od": 0, "candidates": 0, "impossible": 0, "revisit": 0, "connected": 0, "not_connected": 0,
           "korail_top1": 0, "excluded_r3": 0, "examples": []}
    tb = 480
    ev = r.evaluator(td.bin_index_of(tb))
    for _ in range(n):
        o, d = rng.choice(kor), rng.choice(names)
        if rng.random() < 0.5:
            o, d = d, o
        if o == d:
            continue
        out["od"] += 1
        res = r.evaluate_od(o, d, tb)
        out["excluded_r3"] += len(r.excluded)
        if res and res[0].korail:
            out["korail_top1"] += 1
        for p in r.candidates(o, d, tb):
            out["candidates"] += 1
            if sr.has_station_revisit(p):
                out["revisit"] += 1
            legs = T.legs_from_path(ev, p)
            if T.impossible_legs(full_index(), legs, "weekday"):
                out["impossible"] += 1
                out["examples"].append(f"{o}→{d}: {p}")
            c = T.verify_connection(full_index(), legs, tb, "weekday")
            out["connected" if c.status == T.CONNECTED else "not_connected"] += 1
            if c.status == T.CONNECTED and same_train_after_transfer(c):
                out["duplicate_transfer"] = out.get("duplicate_transfer", 0) + 1
                out["examples"].append(f"중복 환승 {o}→{d}: {p}")
            if c.status != T.CONNECTED:
                out.setdefault("not_connected_reasons", {}).setdefault(f"{c.status}: {c.reason}", 0)
                out["not_connected_reasons"][f"{c.status}: {c.reason}"] += 1
    return out


def t3(r) -> dict:
    ev = r.evaluator(td.bin_index_of(480))
    rows = []
    for via_a, via_b in [((1018, 1019), (1018, 1019, 1020)),
                         ((1701, 1813), (1701, 1813, 1801, 1802, 1821))]:
        out = {}
        for via in (via_a, via_b):
            lg = {"codes": list(via), "nodes": [f"{via[0]}", f"{via[-1]}"]}
            w = ev.korail_wait(lg, 480.0)
            deps = r.kidx.pair("weekday", "1", via[0], via[-1], via)[0]
            out[name(via[-1])] = {"E": round(float(w.E), 3), "trains_0800_0830": int(sum(480 <= x < 510 for x in deps))}
        rows.append({"board": name(via_a[0]), "by_alight": out})
    return {"rows": rows}


def _sig_adj(adj) -> str:
    h = hashlib.sha256()
    for u in sorted(adj):
        for e in sorted(adj[u], key=lambda x: (x["to"], x["kind"])):
            h.update(f"{u}|{e['to']}|{e['kind']}|{e['time']:.6f}|{e['cost']:.6f};".encode())
    return h.hexdigest()[:16]


def t5() -> dict:
    mod, sr, disp = load_v1(ROOT)
    th0 = dict(mod.THROUGH_JUNCTIONS)
    disp0 = disp.copy()
    od = [("서울역", "강남"), ("신촌", "잠실"), ("응암", "고속터미널"), ("까치산", "성수"), ("방화", "마천")]

    def run(router):
        return [(tuple(x.path), round(x.actual_time_min, 6)) for o, d in od for x in router.evaluate_od(o, d, 480)[:1]]

    a = KE.KorailTDRouter(ROOT, mod, sr, disp, "weekday", include_korail=False)
    ra = run(a)
    sig_a = _sig_adj(a.scorer(td.bin_index_of(480)).adj)
    b = KE.KorailTDRouter(ROOT, mod, sr, disp, "weekday", include_korail=True)
    run(b)
    sig_b_base = _sig_adj(b._base_scorers[td.bin_index_of(480)].adj)
    c = KE.KorailTDRouter(ROOT, mod, sr, disp, "weekday", include_korail=False)
    rc = run(c)
    sig_c = _sig_adj(c.scorer(td.bin_index_of(480)).adj)
    plain = td.TDRouter(ROOT, mod, sr, disp, "weekday")
    rp = run(plain)
    return {"same_results_off_before_after": ra == rc, "same_as_plain_tdrouter": ra == rp,
            "base_adj_unchanged": sig_a == sig_b_base == sig_c,
            "through_junctions_unchanged": dict(mod.THROUGH_JUNCTIONS) == th0,
            "display_unchanged": disp.equals(disp0),
            "augmented_has_korail": "1_남영" in b.scorer(td.bin_index_of(480)).nodes,
            "base_has_no_korail": "1_남영" not in b._base_scorers[td.bin_index_of(480)].nodes}


def write_report(res: dict) -> None:
    v1, t = res["V1"], res["T"]
    T1 = t["T1"]
    L = ["# Phase 3A-2b — 코레일 엔진 확장 구조 검증 (V1·T1·T1a·T1b·T2·T3·T5)", "",
         "- 근거: [사전 등록 6절](../../docs/v2/preregistration_3a2.md) (D-095·D-095 보완). 경로 변경·역전·민감도(V2~V9)는 3A-2c.",
         "- 모듈: `yeoyuro_v2/korail_engine.py` (기존 엔진 파일 변경 없음, `KorailTDRouter(include_korail=False)` 기본)",
         "- 재현: `python scripts/v2/46_check_korail_engine.py`", "",
         "## V1 보존 (기능 끔)", "",
         f"- v2.3 표본 {v1['cases']:,}건 다시 계산 → 커밋된 od_shiftability_mart 와 기준 경로 소요·최대 혼잡·100%+ 노출 **불일치 {v1['mismatch']}**",
         *[f"  - {x}" for x in v1["examples"]],
         "", "## T5 켰다 끈 뒤 원본 불변", "",
         "| 항목 | 결과 |", "|---|---|"]
    for k, v in res["T5"].items():
        L.append(f"| {k} | {'예' if v else '**아니오**'} |")
    L += ["", "## T1 구로 분기 (경인 ↔ 경부 남쪽 24개 방향쌍 × 평일 6개 출발, 후보 전부)", "",
          f"- 후보 경로 {T1['candidates']:,}개 · 구로에서 바로 갈아타는 후보 {T1['direct_guro']:,}개 중 조건 충족 **{T1['direct_guro_ok']:,}** "
          f"(갈아타기 간선 1번, 보행 1분, 시각표 연결 두 열차번호 다름·1분 뒤 출발)",
          f"- 1순위가 구로 갈아타기인 경우 {T1['top1_guro']} / {T1['pairs'] * len(BASE_TIMES)}",
          f"- 열차 단위 '불가능한 연결' **{T1['impossible']}** · 같은 역 재방문 후보 **{T1['revisit']}** · 중복 환승 **{T1.get('duplicate', 0)}**",
          f"- 의미 없는 구로 갈아타기(신도림 쪽 ↔ 다른 방면 노드) 후보에서 제외한 횟수(후보 생성 호출마다 센 누적): {T1['dropped_invalid_guro']} (`valid_guro_transfer`, 아래 '구현 중 발견')",
          *[f"  - {x}" for x in T1["problems"][:10]],
          "- 다른 노선을 거치는 우회 후보(7호선 경유 등)의 환승 횟수는 제한하지 않음 (D-095 보완)", "",
          "## T1a 구로 재방문 우회", ""]
    a = t["T1a"]
    L += [f"- 경로: {' → '.join(a['path'])}",
          f"- 재방문 제외 규칙이 뺌: {'예' if a['revisit_rule_excludes'] else '**아니오**'} · 실제 후보에 나타남: {'**예**' if a['in_any_candidate'] else '아니오'}",
          f"- 규칙을 건너뛰고 강제로 평가하면: 기대 소요 {a['forced_actual_min']}분·갈아타기 {a['forced_transfer_count']}회 "
          f"(방향 전환 열차 변경이 빠진 값) → 열차 단위 검증 **{a['train_check']}** — 두 겹 모두에서 걸러짐", "",
          "## T1b 구로 출발·도착", "",
          f"- 표시 역 '구로' 후보 노드: {', '.join(t['T1b']['guro_candidates'])} (두 노드 모두: {'예' if t['T1b']['both_nodes'] else '**아니오**'})",
          "", "| OD | 무환승 | 기대 소요(분) | 경로 |", "|---|---|---|---|"]
    for c in t["T1b"]["cases"]:
        L.append(f"| {c['od']} | {'예' if c['ok'] else '**아니오**'} | {c['min']} | {' → '.join(c['path'] or [])} |")
    L += ["", "## T2 직통 열차 중복 환승 (평일 08:00 1순위)", "", "| OD | 갈아타기 | 보행(분) | 재방문 |", "|---|---|---|---|"]
    for c in t["T2"]:
        L.append(f"| {c['od']} | {c['transfers']} | {c['walk']} | {'**예**' if c['revisit'] else '아니오'} |")
    L += ["", "## T3 하차역별 목적지 필터 E (평일 08:00~08:30 구간)", "",
          "| 승차역 | 하차역 | E(분) | 그 30분 안 하차역행 열차 |", "|---|---|---|---|"]
    for row in res["T3"]["rows"]:
        for al, v in row["by_alight"].items():
            L.append(f"| {row['board']} | {al} | {v['E']} | {v['trains_0800_0830']} |")
    L += ["", "- 같은 승차역이라도 하차역에 서는 열차만으로 E 를 계산 → 중간 종착 열차가 많은 하차역일수록 E 가 길어짐", ""]
    au = res["audit_on"]
    L += ["## 켠 그래프 감사 (한쪽 끝이 코레일 역인 무작위 OD, 시드 42, 평일 08:00, 후보 전부)", "",
          f"- OD {au['od']} · 후보 경로 {au['candidates']:,} · 열차 단위 '불가능한 연결' **{au['impossible']}** · 같은 역 재방문 **{au['revisit']}**",
          f"- 시각표 연결: 연결 확인 {au['connected']:,} · 그 밖 {au['not_connected']} · R3 로 뺀 후보 {au['excluded_r3']}",
          *[f"  - {k}: {v}" for k, v in au.get("not_connected_reasons", {}).items()],
          f"- 시각표 연결에서 갈아탄 뒤 같은 열차를 다시 탄 후보(중복 환승) **{au.get('duplicate_transfer', 0)}**",
          f"- 1순위가 코레일 구간을 지나는 OD {au['korail_top1']} — 이 경로들은 '혼잡 판정 제외'로 표시 (쾌적 경로 확정 추천 안 함)",
          *[f"  - 불가능: {x}" for x in au["examples"][:10]], ""]
    L += ["## 구현 중 발견 — 의미 없는 구로 갈아타기", "",
          "- 처음 구현에서 '구일 → 구로(경인) → [갈아타기] → 구로(경부) → 신도림 → …' 후보가 나왔다. 구일에서 온 경인선 열차가 그대로 신도림으로 가는데, "
          "구로에서 내려 1분 걷고 같은 열차(예: K30)를 다시 타는 셈이라 열차 연결 검증에서 '같은 열차 재탑승'으로 잡혔다 (1순위로 뽑히지는 않음, 더 느림).",
          "- 구로 갈아타기 간선은 **경인 쪽 역 → 구로 → 경부 쪽 역**(또는 반대)일 때만 후보로 쓰도록 `valid_guro_transfer` 를 넣었다. "
          "신도림 쪽 ↔ 다른 방면 노드 갈아타기(= 구로 종착→시발)는 D-095 E3a 대로 그래프 후보에서 빼고 연결 검증에서만 다룬다.",
          "- 그 뒤 T1·켠 그래프 감사에서 중복 환승 0 을 기준으로 둔다.", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    res = {"V1": v1_check(300 if a.quick else None)}
    print(f"V1 {res['V1']['evaluated']}/{res['V1']['cases']} 불일치 {res['V1']['mismatch']} ({time.time() - t0:.0f}초)")
    res["T5"] = t5()
    print(f"T5 {res['T5']}")
    r, sr = _router9(on=True)
    res["T"] = t1_t1a_t2(r, sr)
    T1 = res["T"]["T1"]
    print(f"T1 후보 {T1['candidates']} · 구로 직접 {T1['direct_guro']} 중 충족 {T1['direct_guro_ok']} · 불가능 {T1['impossible']} · 재방문 {T1['revisit']}")
    print(f"T1a {res['T']['T1a']}")
    res["T3"] = t3(r)
    res["audit_on"] = audit_on(r, sr)
    print(f"켬 감사 {res['audit_on']}")
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    write_report(res)
    print(f"저장: {REPORT.relative_to(ROOT)} ({time.time() - t0:.0f}초)")
    fail = (res["V1"]["mismatch"] or not all(res["T5"].values()) or T1["impossible"] or T1["revisit"]
            or T1["direct_guro_ok"] != T1["direct_guro"] or T1.get("duplicate") or res["T"]["T1a"]["in_any_candidate"]
            or res["T"]["T1a"]["train_check"] != T.IMPOSSIBLE
            or not all(c["ok"] for c in res["T"]["T1b"]["cases"])
            or any(c["transfers"] != 0 for c in res["T"]["T2"])
            or res["audit_on"]["impossible"] or res["audit_on"]["revisit"] or res["audit_on"].get("duplicate_transfer"))
    if fail:
        print("[확인 필요] 위 결과 중 기준 미달 항목이 있습니다")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
