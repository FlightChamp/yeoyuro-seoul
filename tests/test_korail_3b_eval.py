"""
tests/test_korail_3b_eval.py — Phase 3B-4 평가 도구 (scripts/v2/53_evaluate_korail_3b.py, D-105c).

결과 수치는 검사하지 않는다 (사전 등록 6절 기준은 평가 보고서에서 판정). 여기서는 계산 방법이 D-105c 대로인지만 본다.
"""
from __future__ import annotations

import importlib.util
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
need_app = pytest.mark.skipif(not HAS_APP, reason="app_v26 데이터 없음")
need_3b = pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (P4)")


@pytest.fixture(scope="module")
def m53():
    spec = importlib.util.spec_from_file_location("m53", ROOT / "scripts" / "v2" / "53_evaluate_korail_3b.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_double_transfer_definition(m53):
    s = m53.structure("AR_마곡나루|AR_공덕|GJ_공덕|5_공덕|5_애오개".split("|"))
    assert s["double_transfer"] == ["공덕"] and s["revisit"] == []
    # 9호선 일반↔급행 전환은 같은 노선 (기존 그래프 설계)
    assert m53.structure("SB_선릉|SB_선정릉|9L_선정릉|9X_선정릉|9X_신논현".split("|"))["double_transfer"] == []
    assert m53.structure("4_동대문|4_동대문역사문화공원|5_동대문역사문화공원|5_을지로4가".split("|"))["double_transfer"] == []
    assert m53.structure("2_시청|2_을지로입구|2_시청".split("|"))["revisit"] == ["시청"]


def test_d106_layer_detection(m53):
    assert m53._d106_trace({"line": {"dash": "dot", "color": "#B0B7C3"}})
    assert m53._d106_trace({"hovertemplate": "%{text} · 광역철도 확장 구간 (토글을 켜면 경로 계산)<extra></extra>"})
    assert not m53._d106_trace({"line": {"color": "#0052A4"}, "hovertemplate": None})


def test_variants_cover_prereg(m53):
    assert set(m53.B5_VARIANTS) == {"b1.00", "b1.60", "b2.78", "u-1", "u+1", "m0.5", "m2.0", "x1.0"}
    assert {"base", "noGC", "nodir", "k20", "maxonly"} <= set(m53.VARIANTS)
    assert m53.VARIANTS["base"] == ({}, None, False)
    assert [h for _, h in m53.SLOTS] == ["08:00", "18:00", "23:30", "14:00", "14:00"]
    days = [KA.service_day(d, h).day_type for d, h in m53.SLOTS]
    assert days == ["weekday", "weekday", "weekday", "saturday", "sunday"]
    assert m53.P95_LIMIT == 2.6 and m53.B4_TIMES == ["07:30", "08:00", "08:30", "17:30", "18:00", "18:30"]


@need_app
def test_k3b_sample(m53):
    s = m53.k3b_sample()
    s3 = set(m53.stations_3b())
    assert len(s) == len({(a, b) for a, b, _ in s}) == 312            # 300 + 지정 12 (겹침 없음)
    assert all(a != b and (a in s3 or b in s3) for a, b, _ in s)
    assert [(a, b) for a, b, t in s if t == "designated"] == m53.DESIGNATED
    assert m53.k3b_sample() == s                                       # 시드 고정


@need_3b
def test_router_kw_passthrough_and_default_unchanged():
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    p = KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", repo_root=ROOT, include_3b=True,
                            router_kw={"zero_walk_margin": 2.0, "k": 20, "min_graph": False})
    assert p.on.zero_walk_margin == 2.0 and p.on.k == 20 and p.on.min_graph is False
    q = KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", repo_root=ROOT, include_3b=True)
    assert q.on.zero_walk_margin == 1.0 and q.on.k == 10 and q.on.min_graph is True
