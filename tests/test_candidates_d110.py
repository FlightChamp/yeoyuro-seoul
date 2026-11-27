"""
tests/test_candidates_d110.py — D-110 후보 탐색 비교군 (사전 등록 docs/v2/preregistration_candidates.md 8절, 구현 기록 D-110a).

결과 수치(놓침 건수 등)는 보지 않는다. 계산 방법이 사전 등록대로인지만 본다.
| 8절 | 검사 |
|---|---|
| H | rep_E: 3-A 노드 값 = 예전 계산('1' 묶음), 3-B 노드는 열차가 있으면 0 보다 큼 |
| E | 대표 대기는 탐색 복사본에만 — 같은 경로를 V0·V2 계산기로 평가하면 같은 값 |
| A | V0 후보 ⊆ V1·V2·V4·V5 후보, Yen 앞부분 (K=10 결과 = K=20 결과 앞 10개) |
| D | V3 묶음 키는 GC→GJ 만 바꾸고, 같은 묶음 경로를 지우지 않음 |
| — | 기본값 = V0 (옵션 꺼짐), 방법 실행 순서와 무관 (상태 누수 없음) |
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
from yeoyuro_v2 import korail_app as KA          # noqa: E402
from yeoyuro_v2 import korail_engine as KE       # noqa: E402
from yeoyuro_v2 import korail_line1 as K         # noqa: E402
from yeoyuro_v2 import time_dependent as td      # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
HAS_APP = (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists()
HAS_3B = KA.k3b_ready(ROOT)
need_3b = pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (P4)")
M = 480.0
ODS = [("홍대입구", "망우"), ("수색", "강남구청"), ("증미", "용산"), ("동대문", "김포공항")]


def test_gc_signature():
    a = ("GJ_중랑", "GC_상봉", "GC_망우")
    b = ("GJ_중랑", "GJ_상봉", "GJ_망우")
    assert KE.gc_signature(a) == KE.gc_signature(b) == b
    assert KE.gc_signature(("9L_노량진", "9X_노량진")) == ("9L_노량진", "9X_노량진")     # 다른 노선은 그대로


def _planner(**kw):
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    k3a = kw.pop("keep_3a", False)
    return KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", query_date="2026-10-14", repo_root=ROOT,
                               include_3b=True, router_kw=kw, keep_3a=k3a,
                               cand="v0")          # D-111b 뒤 앱 기본값은 v4 — D-110 검사는 V0 기준으로 고정


@pytest.fixture(scope="module")
def p0():
    return _planner()


@need_3b
def test_defaults_are_v0(p0):
    R = p0.on
    assert (R.rep_union, R.group_gc, R.max_k_v3, R.k, R.max_k) == (False, False, 40, 10, 20)
    assert p0.keep_3a is False


@need_3b
def test_rep_E_3a_unchanged_and_3b_nonzero(p0):
    R = p0.on
    for node in [n for n in R.pieces.korail_nodes if n.startswith("1_")][:26]:
        for b in (2, 5, 25):
            code = R.pieces.node_code[node]
            nxt = sorted({int(r.to_code) for r in R.pieces.ride.itertuples() if r.from_node == node})
            b0 = td.BIN_START_MIN + b * td.BIN_WIDTH_MIN
            es = []
            for c in nxt:
                deps = R.kidx.pair(R.day_type, "1", code, c, (code, c))[0]
                e = K.wait_profile(deps, b0, b0 + 30, KE.MAX_WAIT).E if deps else float("nan")
                if not np.isnan(e):
                    es.append(e)
            old = float(np.median(es)) if es else 0.0
            assert R.rep_E(node, b) == pytest.approx(old)
    for node in ("GJ_왕십리", "SB_왕십리", "AR_공덕", "GJ_회기"):
        assert R.rep_E(node, 5) > 0, node
    assert R._rep_group("GJ_회기") == "GJC" and R._rep_group("GJ_용산") == "GJ" and R._rep_group("GC_상봉") == "GC"


@pytest.fixture(scope="module")
def p2():
    return _planner(rep_union=True)


@need_3b
def test_rep_only_in_search(p0, p2):
    """대표 대기는 탐색 가중치 복사본에만 — 평가값·간선 비용은 그대로 (8-E)."""
    b = td.bin_index_of(M)
    path = "GJ_홍대입구|GJ_서강대|GJ_공덕|GJ_효창공원앞|GJ_용산|GJ_이촌|GJ_서빙고|GJ_한남|GJ_옥수|GJ_응봉|GJ_왕십리|GJ_청량리|GJ_회기|GJ_중랑|GJ_상봉|GJ_망우".split("|")
    a = p0.on.evaluator(b).evaluate(path, M)
    c = p2.on.evaluator(b).evaluate(path, M)
    assert a.actual_time_min == pytest.approx(c.actual_time_min) and a.initial_wait_min == c.initial_wait_min == 0
    s0, s2 = p0.on.scorer(b), p2.on.scorer(b)
    assert {u: [e["time"] for e in v] for u, v in s0.adj.items()} == {u: [e["time"] for e in v] for u, v in s2.adj.items()}
    ex = s2._rep_extra_all
    assert ex and all(v >= 0 for v in ex.values())
    assert not any(u.split("_", 1)[0] == v.split("_", 1)[0] for u, v in ex)                 # 같은 노선 노드끼리 없음
    assert not any({u.split("_", 1)[0], v.split("_", 1)[0]} == {"9L", "9X"} for u, v in ex)


@need_3b
def test_v0_subset_and_order_independent(p0):
    """V0 후보 ⊆ V1·V2·V4·V5 (8-A), 방법을 어떤 순서로 돌려도 같은 후보 (상태 누수 없음)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("m54", ROOT / "scripts" / "v2" / "54_evaluate_candidates.py")
    m54 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m54)
    P = _planner(rep_union=True)
    for o, d in ODS[:3]:
        paths, _, maxk = m54.method_paths(P, o, d, M)
        v0 = set(paths["V0"])
        for mid in ("V1", "V2", "V4", "V5"):
            assert v0 <= set(paths[mid]), (o, d, mid)
        assert maxk["V3"] <= 40 and maxk["V4"] <= 40
    Q = _planner(rep_union=True)
    m54._set(Q.on, m54.METHODS["V4"])
    fresh = {tuple(p) for p in Q.on.candidates("동대문", "김포공항", M)}
    paths, _, _ = m54.method_paths(P, "동대문", "김포공항", M)
    assert set(paths["V4"]) == fresh


@need_3b
def test_yen_prefix(p0):
    """K 를 늘려도 Yen 앞부분이 같다 (V1·V3 단조성 전제, 8-A)."""
    R = p0.on
    sc = R.scorer(td.bin_index_of(M))
    sr = R.sr
    for o, d in ODS:
        vo, vd = sr.ORIGIN_PREFIX + o, sr.DEST_PREFIX + d
        saved = (sc.adj, sc.nodes)
        aug = dict(sc.adj)
        aug[vo] = [{"to": n, "cost": 0.0, "time": 0.0, "cong": float("nan"), "event": 0.0, "kind": "origin_access"}
                   for n in sr.candidates_of(R.display, o) if n in sc.nodes]
        for n in sr.candidates_of(R.display, d):
            if n in sc.nodes:
                aug[n] = list(aug.get(n, [])) + [{"to": vd, "cost": 0.0, "time": 0.0, "cong": float("nan"),
                                                  "event": 0.0, "kind": "destination_exit"}]
        sc.adj, sc.nodes = aug, set(aug) | {e["to"] for lst in aug.values() for e in lst}
        try:
            a, b = sc._yen(vo, vd, K=10, weight="time"), sc._yen(vo, vd, K=20, weight="time")
        finally:
            sc.adj, sc.nodes = saved
        assert b[:len(a)] == a, (o, d)


@need_3b
def test_v3_keeps_gc_and_gj_variants():
    """V3 는 같은 묶음의 GJ·GC 경로를 둘 다 후보에 남긴다 (8-D)."""
    P = _planner(group_gc=True)
    ps = [tuple(p) for p in P.on.candidates("홍대입구", "망우", M)]
    sig: dict = {}
    for p in ps:
        sig.setdefault(KE.gc_signature(p), set()).add(p)
    assert any(len(v) >= 2 for v in sig.values())
    assert any(p[-1] == "GC_망우" for p in ps) and any(p[-1] == "GJ_망우" for p in ps)
