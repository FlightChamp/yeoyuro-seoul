"""
tests/test_d111.py — D-111a 구현 (사전 등록 docs/v2/preregistration_d111.md).

결과 수치(바뀐 건수 등)는 보지 않는다. 설정이 사전 등록대로 전달되는지, 비교 도구의 판정 식이 맞는지만 본다.
| 검사 | 내용 |
|---|---|
| 설정 | v4 = rep_union·group_gc 켬, 원시 후보 상한 40, k 그대로 · 3-B 끔은 언제나 v0 · router_kw 가 이김 |
| 기본값 | D-111b 판정 전까지 DEFAULT_CAND = "v0" (판정 패치에서 이 검사를 함께 바꾼다) |
| 53 | --cand v4 는 --tag 없이 멈춤 · 작업자 시작 함수가 설정을 넣음 |
| 54 | 설정 해시에 HEAD 없음 · V0 후보 색인 = 예전 행마다 찾기 |
| 55 | G0 같음 판정 · 정렬 키 · G1 식(재계산 b 보고서는 통과) · 결과 비교 |
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
from yeoyuro_v2 import korail_app as KA          # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
HAS_APP = (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists()
HAS_3B = KA.k3b_ready(ROOT)
need_3b = pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (P4)")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def m53():
    return _load("m53_t111", "scripts/v2/53_evaluate_korail_3b.py")


@pytest.fixture(scope="module")
def m54():
    return _load("m54_t111", "scripts/v2/54_evaluate_candidates.py")


@pytest.fixture(scope="module")
def m55():
    return _load("m55_t111", "scripts/v2/55_compare_d111.py")


def test_cand_specs():
    assert KA.DEFAULT_CAND == "v0"                      # D-111b 판정 전 (통과하면 판정 패치에서 "v4" 로)
    assert KA.cand_router_kw("v4", True) == ("v4", {"rep_union": True, "group_gc": True, "max_k_v3": 40})
    assert KA.cand_router_kw("v4", False) == ("v0", {})            # 3-B 끔은 언제나 v0
    assert KA.cand_router_kw(None, True) == (KA.DEFAULT_CAND, dict(KA.CAND_SPECS[KA.DEFAULT_CAND]))
    assert KA.cand_router_kw("v0", True) == ("v0", {})
    with pytest.raises(ValueError):
        KA.cand_router_kw("v9", True)
    assert "k" not in KA.CAND_SPECS["v4"]               # k=10 그대로 (D-110 V4 정의)


def test_v4_matches_d110_method(m54):
    s = m54.METHODS["V4"]
    assert (s["rep_union"], s["group_gc"], s["k"], s["keep_3a"]) == (True, True, 10, False)
    assert KA.CAND_SPECS["v4"]["max_k_v3"] == 40


def _planner(**kw):
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    return KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", query_date="2026-10-14", repo_root=ROOT,
                               include_3b=True, **kw)


@need_3b
def test_planner_cand_v4_and_override():
    p = _planner(cand="v4")
    R = p.on
    assert p.cand == "v4"
    assert (R.rep_union, R.group_gc, R.max_k_v3, R.k) == (True, True, 40, 10)
    q = _planner(cand="v4", router_kw={"group_gc": False})       # router_kw 가 같은 키를 이김
    assert (q.on.rep_union, q.on.group_gc) == (True, False)
    r = _planner()                                              # 기본 = DEFAULT_CAND
    assert r.cand == KA.DEFAULT_CAND and r.on.rep_union is (KA.DEFAULT_CAND == "v4")


def test_53_cand_needs_tag(m53, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["53", "--cand", "v4"])
    with pytest.raises(SystemExit) as e:
        m53.main()
    assert "--tag" in str(e.value)
    old = KA.DEFAULT_CAND
    try:
        m53._init_worker("v4")
        assert m53._W["cand"] == "v4" and KA.DEFAULT_CAND == "v4"
    finally:
        m53._init_worker(old)
    assert KA.DEFAULT_CAND == old


def test_54_hash_without_head(m54):
    src = (ROOT / "scripts" / "v2" / "54_evaluate_candidates.py").read_text(encoding="utf-8")
    assert "rev-parse" not in src
    d = m54.config_hash(True)
    assert "head" not in d and {"k3b_sha256", "script", "engine", "app", "methods"} <= set(d)


def test_54_v0_paths_same_as_rowwise(m54):
    key = ["date", "hhmm", "o", "d"]
    df = pd.DataFrame({"date": ["a", "a", "b"], "hhmm": ["1", "2", "1"], "o": ["x", "x", "y"], "d": ["y", "y", "z"],
                       "paths": [json.dumps({"V0": ["p1", "p2"], "V4": ["p3"]}), json.dumps({"V0": []}),
                                 json.dumps({"V0": ["q"]})]})
    fast = m54.v0_paths(df)
    for z in df.itertuples():
        old = set(json.loads(df.set_index(key).loc[(z.date, z.hhmm, z.o, z.d)].paths)["V0"])
        assert fast[(z.date, z.hhmm, z.o, z.d)] == old
    with pytest.raises(ValueError):
        m54.v0_paths(pd.concat([df, df.head(1)]))


def test_55_same_row_and_key(m55):
    a = {"best_path": "1_a|1_b", "ref_path": "", "status": "ok", "best_expected": 12.345}
    assert m55.same_row(a, dict(a))
    assert m55.same_row(a, {**a, "ref_path": None})
    assert not m55.same_row(a, {**a, "best_expected": 12.346})
    assert not m55.same_row(a, {**a, "status": "no_benefit"})
    assert m55.same_row({**a, "best_expected": None}, {**a, "best_expected": float("nan")})
    assert not m55.same_row({**a, "best_expected": None}, a)
    assert m55.sort_key(10.0000001, 3.0) == (10.0, 3.0)
    assert m55.is_worse((10.0, 3.1), (10.0, 3.0)) and not m55.is_worse((9.9, 9.0), (10.0, 3.0))


def test_55_g1_on_b_report(m55):
    f = ROOT / "reports" / "v2" / "korail_3b_eval_b.json"
    J = json.loads(f.read_text(encoding="utf-8"))
    g = m55.gate_g1(J, J)
    assert g["pass"], g                                  # D-105e 에서 통과한 보고서는 같은 식으로도 통과
    bad = json.loads(json.dumps(J))
    bad["base"]["sec_p95"] = 2.61
    assert not m55.gate_g1(bad, J)["items"]["B11"]
    bad = json.loads(json.dumps(J))
    bad["B7"]["hold"] = ["GJ 08:00"]
    assert not m55.gate_g1(bad, J)["pass"]


def test_55_compare_frames(m55):
    base = dict(n_cand=5, sec=1.0, best_lines="", ref_path="", best_expected=30.0)
    rows_b = [dict(base, date="d", hhmm="08:00", o="A", d="B", status="ok", best_path="p"),
              dict(base, date="d", hhmm="08:00", o="A", d="C", status="ok", best_path="q"),
              dict(base, date="d", hhmm="18:00", o="A", d="B", status="ok", best_path="r")]
    rows_n = [dict(rows_b[0], best_path="p2", best_expected=28.0, best_lines="GJ", n_cand=7, sec=1.5),
              dict(rows_b[1]),
              dict(rows_b[2], best_path="", status="none", best_expected=None)]
    C = m55.compare_frames(pd.DataFrame(rows_b), pd.DataFrame(rows_n))
    assert C["compared"] == 3 and C["best_changed"] == 2
    assert C["expected_diff"]["n"] == 1 and C["expected_diff"]["min"] == -2.0
    assert C["vanished"] == [["d", "18:00", "A", "B"]]
    assert C["status_transitions"] == {"ok→none": 1}
    assert C["by_group"]["GJ"] == {"changed": 1, "n": 1}
    assert C["by_slot"]["d 08:00"] == {"changed": 1, "n": 2}
