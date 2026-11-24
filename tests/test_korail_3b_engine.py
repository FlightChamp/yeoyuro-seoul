"""
tests/test_korail_3b_engine.py — Phase 3B-2 경로 엔진 (D-105 사전 등록 4절).

3-B 상세 정차표(data/marts/v2/korail_3b, gitignore — P4)는 사용자 PC 에서 52 를 돌려야 생긴다.
정차표가 필요한 검사는 없으면 건너뛴다 (CI·배포 환경). 정차표 없이도 도는 검사(T5 대체 동작·T1 근거 표 노드)는 늘 실행한다.

T1 데이터·그래프 / T2 경춘선 공용 구간 / T3 열차 연결 / T4 후보 탐색 / T5 기존 기능 보존 / T6 시간·환경 / T7 메타데이터
대규모 평가(B1~B13)는 3B-4.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
from yeoyuro_v2 import korail_engine as KE        # noqa: E402
from yeoyuro_v2 import time_dependent as td       # noqa: E402
from yeoyuro_v2 import train_timetable as T       # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
HAS_APP = (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists()
HAS_3B = KE.k3b_available()
need_app = pytest.mark.skipif(not HAS_APP, reason="app_v26 데이터 없음")
need_3b = pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (사용자 PC 에서 52 실행 뒤 검사, P4)")
M = 480.0   # 08:00


def _router(day_type="weekday", **kw):
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    return KE.KorailTDRouter(DATA9, mod, sr, disp, day_type, include_korail=True, policy="step", multi_bin=True, **kw)


@pytest.fixture(scope="module")
def r3b():
    return _router(include_3b=True)


@pytest.fixture(scope="module")
def ev(r3b):
    return r3b.evaluator(td.bin_index_of(M))


# ---------------------------------------------------------------------- 정차표 없이 도는 검사
def test_merge_group_rules():
    assert T.merge_group("GJ", "GJ") == "GJ"
    assert T.merge_group("GJC", "GJ") == "GJ" and T.merge_group("GJ", "GJC") == "GJ"
    assert T.merge_group("GJC", "GC") == "GC"
    assert T.merge_group("GJ", "GC") is None            # 두 승강장 열차를 한 열차로 잇지 않음
    assert T.merge_group("2", "5") is None and T.merge_group("9L", "9X") is None   # 기존 노선은 지금처럼 끊김


def test_node_and_direction_rules():
    assert KE.node_3b("GC", "회기") == "GJ_회기" and KE.node_3b("GC", "상봉") == "GC_상봉"
    assert KE.node_3b("GJ", "상봉") == "GJ_상봉" and KE.node_3b("AR", "서울역") == "AR_서울역"
    # 김포공항: 온 쪽·갈 쪽 이웃으로 진행 방향
    assert KE.train_direction("김포공항", "9L_김포공항", "9L_개화", arriving=True) == "중앙보훈병원"
    assert KE.train_direction("김포공항", "9X_김포공항", "9X_마곡나루", arriving=True) == "개화"
    assert KE.train_direction("김포공항", "AR_김포공항", "AR_마곡나루", arriving=False) == "서울역"
    assert KE.train_direction("김포공항", "AR_김포공항", "AR_마곡나루", arriving=True) == "인천공항"
    assert KE.train_direction("김포공항", "5_김포공항", "5_송정", arriving=True) == "방화"
    assert KE.train_direction("김포공항", "5_김포공항", "5_개화산", arriving=False) == "방화"
    assert KE.train_direction("공덕", "5_공덕", "5_애오개", arriving=True) is None


@need_app
def test_unavailable_mart_falls_back_to_3a(tmp_path):
    """T5·T6: 정차표가 없으면 3-B 만 꺼지고(unavailable) 3-A 결과는 그대로, 추정 열차를 만들지 않는다."""
    r_a = _router()
    r_u = _router(include_3b=True, k3b_mart=tmp_path / "없음.parquet")
    assert r_a.status_3b == "off" and r_u.status_3b == "unavailable"
    assert not any(n.split("_", 1)[0] in KE.GROUPS_3B for n in r_u.pieces.node_code)
    assert not any(g in KE.GROUPS_3B for g in {k[1] for k in r_u.kidx._by_board})
    for o, d in [("금천구청", "월계"), ("신림", "강남"), ("가산디지털단지", "석계")]:
        a = [(q.path, round(q.actual_time_min, 6)) for q in r_a.evaluate_od(o, d, M)]
        u = [(q.path, round(q.actual_time_min, 6)) for q in r_u.evaluate_od(o, d, M)]
        assert a == u


# ---------------------------------------------------------------------- T1 데이터·그래프
@need_3b
def test_t1_graph(r3b):
    p = r3b.pieces
    assert r3b.status_3b == "on" and p.missing_transfer == []
    nodes3 = {n for n in p.node_code if n.split("_", 1)[0] in KE.GROUPS_3B}
    assert {n.split("_", 1)[0] for n in nodes3} == {"GJ", "SB", "AR", "GC"}
    assert {n for n in nodes3 if n.startswith("GC_")} == {"GC_상봉", "GC_망우"}
    assert len([n for n in nodes3 if n.startswith("GJ_")]) == 20 and len([n for n in nodes3 if n.startswith("SB_")]) == 17
    assert len([n for n in nodes3 if n.startswith("AR_")]) == 6
    lines = set(p.ride.line_id)
    assert {"GJ", "GJC", "GC", "SB", "AR"} <= lines
    gjc = p.ride[p.ride.line_id == "GJC"]
    assert set(gjc.from_station) | set(gjc.to_station) == {"청량리", "회기", "중랑"}


@need_3b
def test_t1_transfers(r3b):
    meta = pd.DataFrame(r3b.pieces.transfer_meta)
    assert len(meta.groupby(["station"]).size()) > 0
    assert len(meta.drop_duplicates(["u", "v"])) == len(meta)                      # 중복 환승 간선 없음
    assert len(meta) == 41                                                          # 38연결 (9호선 = 9L·9X 노드)
    assert (meta["max"] > 0).all()                                                  # 그래프 간선에 0분 없음
    assert set(meta[meta["min"] == 0].station) == {"김포공항"}                       # 0분은 김포공항 같은 방향뿐
    # 같은 역번호라고 다른 노선이 이어지지 않음: 모든 환승 간선은 같은 역 이름 안에서만
    assert all(u.split("_", 1)[1].split("@")[0] == v.split("_", 1)[1].split("@")[0] for u, v in zip(meta.u, meta.v))
    b = meta[meta.grade == "B"]
    assert len(b) == 14 and meta.groupby("grade").size().to_dict() == {"A": 11, "B": 14, "U": 16}
    # B 계수 반영 (홍대입구 2↔공항철도 공식 4:56 × 1.81)
    hd = meta[(meta.u == "2_홍대입구") & (meta.v == "AR_홍대입구")]
    assert hd["max"].iloc[0] == pytest.approx((4 + 56 / 60) * 1.81)


@need_3b
def test_t1_b_factor_param():
    r = _router(include_3b=True, b_factor=1.60)
    m = pd.DataFrame(r.pieces.transfer_meta)
    assert m[(m.u == "2_홍대입구") & (m.v == "AR_홍대입구")]["max"].iloc[0] == pytest.approx((4 + 56 / 60) * 1.60)


# ---------------------------------------------------------------------- T2 경춘선 공용 구간
@need_3b
def test_t2_e2_train_sets(r3b):
    k, day = r3b.kidx, "weekday"
    gjc = k.pair(day, "GJC", 1015, 1201, (1015, 1201))[0]
    gj = k.pair(day, "GJ", 1015, 1201, (1015, 1201))[0]
    gc = k.pair(day, "GC", 1015, 1201, (1015, 1201))[0]
    assert len(gjc) == len(gj) + len(gc) and len(gc) > 0 and gjc == sorted(gjc)    # 공용 구간 = 두 노선 열차 출발 합집합
    # 목적지 필터: 청량리→양원(경춘선은 망우 뒤 갈라짐)은 경의중앙 열차만
    via = (1014, 1015, 1201, 1202, 1203, 1204)
    assert k.pair(day, "GJC", 1014, 1204, via)[0] == k.pair(day, "GJ", 1014, 1204, via)[0]
    # 상봉 시발 경춘선 열차는 청량리·회기 출발에 섞이지 않음
    gc_chung = {key[2] for _, _, key in [(0, 0, kk) for kk in k.pair(day, "GC", 1014, 1015, (1014, 1015))[2]]}
    gc_sangbong_start = {key[2] for key, (codes, _, _) in k._trains.items()
                         if key[0] == day and key[1] == "GC" and int(codes[0]) == 1202}
    assert gc_sangbong_start and not (gc_chung & gc_sangbong_start)


@need_3b
def test_t2_same_train_vs_transfer(ev):
    # 회기에서 경춘선을 타고 상봉 경춘 승강장에서 내림 = 한 구간 (경춘선 열차만)
    lg = ev.legs(["GJ_회기", "GJ_중랑", "GC_상봉"])
    assert len(lg) == 1 and lg[0]["group"] == "GC"
    # 용산 → 회기 (경의중앙 → 공용 구간) = 한 구간, 경의중앙 열차만
    lg = ev.legs(["GJ_왕십리", "GJ_청량리", "GJ_회기"])
    assert len(lg) == 1 and lg[0]["group"] == "GJ"
    # 경의중앙 수색 지선 → 용산 → 이촌 직통 = 한 구간 (용산에서 갈아타기 없음)
    lg = ev.legs(["GJ_서강대", "GJ_공덕", "GJ_효창공원앞", "GJ_용산", "GJ_이촌"])
    assert len(lg) == 1
    # 상봉에서 경춘 → 경의중앙 실제 환승 = 두 구간 + 보행 2분
    p = ["GC_망우", "GC_상봉", "GJ_상봉", "GJ_중랑"]
    lg = ev.legs(p)
    assert [x["group"] for x in lg] == ["GC", "GJ"]
    q = ev.evaluate(p, M)
    assert q.transfer_count == 1 and q.transfer_walk_min == pytest.approx(2.0)


@need_3b
def test_t2_single_leg_has_no_transfer(r3b):
    q = r3b.evaluate_od("서강대", "이촌", M)[0]
    assert q.transfer_count == 0 and q.lines_3b == ("GJ",)


# ---------------------------------------------------------------------- T3 열차 연결
@need_3b
def test_t3_connections(r3b, ev):
    for o, d in [("왕십리", "야탑"), ("디지털미디어시티", "왕십리"), ("망우", "회기"), ("야탑", "강남")]:
        q = r3b.evaluate_od(o, d, M)[0]
        c = r3b.verify(q.path, M, ev)
        assert c.status == T.CONNECTED, (o, d, c)
        assert c.arrive_min > M
    # 다른 열차번호 사이 환승이 있는 경로: 두 열차 (야탑 → 선릉 수인분당 → 2호선)
    c = r3b.verify(r3b.evaluate_od("야탑", "강남", M)[0].path, M, ev)
    assert len(c.rides) == 2 and c.rides[0].train != c.rides[1].train


@need_3b
def test_t3_last_train_and_impossible(r3b):
    late = 24 * 60 + 50                                 # 다음 날 00:50
    evl = r3b.evaluator(td.bin_index_of(late))
    c = r3b.verify(["SB_왕십리", "SB_서울숲", "SB_압구정로데오"], late, evl)
    assert c.status == T.NOT_CONNECTED                   # 막차 뒤
    # 그날 한 열차로 갈 수 없는 구간 (경춘선 상봉 승강장 → 양원: 경춘선은 망우 뒤 갈라짐)
    leg = T.Leg("GC", 1202, 1204, "GC_상봉", "GJ_양원", 0.0, (1202, 1203, 1204))
    rs, why = T._ride_leg(r3b.kidx, "weekday", leg, 0.0, 10_000.0)
    assert rs is None and why == T.IMPOSSIBLE


@need_3b
def test_t3_zero_walk_margin(r3b, ev):
    """김포공항 9→공항철도 같은 방향: 보행 0분, 연결 검증에만 여유 1.0분 (기대 소요에는 안 들어감)."""
    p = ["9L_개화", "9L_김포공항", "AR_김포공항", "AR_마곡나루", "AR_디지털미디어시티"]
    q = ev.evaluate(p, M)
    assert q.transfer_walk_min == 0.0
    legs = r3b.verification_legs(ev, p)
    assert legs[1].walk_before == pytest.approx(KE.ZERO_WALK_MARGIN)
    c = r3b.verify(p, M, ev)
    assert c.status == T.CONNECTED and c.rides[1].dep_min >= c.rides[0].arr_min + KE.ZERO_WALK_MARGIN - 1e-9
    r2 = _router(include_3b=True, zero_walk_margin=2.0)
    assert r2.verification_legs(r2.evaluator(td.bin_index_of(M)), p)[1].walk_before == pytest.approx(2.0)
    # 0분 초과 3-B 환승은 3-A 규칙(보행만큼), +여유는 민감도
    p2 = ["9L_공항시장", "9L_김포공항", "AR_김포공항", "AR_마곡나루"]
    assert r3b.verification_legs(ev, p2)[1].walk_before == pytest.approx(1.0)


# ---------------------------------------------------------------------- T4 후보 탐색·방향별 평가
@need_3b
def test_t4_directional_walks(ev):
    def w(p):
        e = next(x for x in ev.s.adj[p[1]] if x["to"] == p[2])
        return e["time"], ev.walk_actual(p, 1, p[1], p[2], e)
    assert w(["9L_개화", "9L_김포공항", "AR_김포공항", "AR_마곡나루"]) == (1.0, 0.0)
    assert w(["9L_공항시장", "9L_김포공항", "AR_김포공항", "AR_마곡나루"]) == (1.0, 1.0)
    assert w(["AR_마곡나루", "AR_김포공항", "9L_김포공항", "9L_개화"]) == (1.0, 0.0)          # 반대로 갈아타도 같은 표
    assert w(["AR_마곡나루", "AR_김포공항", "9L_김포공항", "9L_공항시장"]) == (1.0, 1.0)
    assert w(["5_송정", "5_김포공항", "AR_김포공항", "AR_마곡나루"]) == (4.0, 2.0)            # 방화 방면 ↔ 서울역 방면
    assert w(["AR_마곡나루", "AR_김포공항", "5_김포공항", "5_개화산"]) == (4.0, 3.0)         # 인천공항 방면 ↔ 방화 방면
    assert w(["AR_마곡나루", "AR_김포공항", "5_김포공항", "5_송정"]) == (4.0, 4.0)           # 인천공항 ↔ 마천·하남
    assert w(["AR_김포공항", "AR_김포공항", "5_김포공항", "5_송정"])[1] == 4.0                # 방향을 모르면 최댓값


@need_3b
def test_t4_min_graph_union_and_k(r3b):
    import station_routing as sr
    sc = r3b.scorer(td.bin_index_of(M))
    o, d = "개화", "서울역"
    on = r3b.raw_paths(sc, o, d, 10)
    r_off = _router(include_3b=True, min_graph=False)
    off = r_off.raw_paths(r_off.scorer(td.bin_index_of(M)), o, d, 10)
    assert set(map(tuple, off)) <= set(map(tuple, on))
    # 최댓값 그래프(방향 무시, 김포공항 9↔공항 1분)와 실제 방향 평가 0분의 순위 차이가 반영됨
    best = r3b.evaluate_od(o, d, M)[0]
    assert best.path[:4] == ["9L_개화", "9L_김포공항", "AR_김포공항", "AR_마곡나루"] and best.transfer_walk_min == 0.0
    assert r3b.k == KE.KORAIL_DEFAULT_K
    r20 = _router(include_3b=True, k=20)
    assert r20.k == 20 and len(r20.raw_paths(r20.scorer(td.bin_index_of(M)), o, d, 20)) >= len(on)
    assert sr is not None


@need_3b
def test_t4_platform_variants_not_merged(r3b):
    """같은 역 순서라도 승강장(노드)이 다른 후보는 지우지 않는다 (회기→상봉: 경의중앙 승강장 / 경춘 승강장)."""
    res = r3b.evaluate_od("회기", "상봉", M)
    ends = {q.path[-1] for q in res}
    assert {"GJ_상봉", "GC_상봉"} <= ends


# ---------------------------------------------------------------------- T5 기존 기능 보존
@need_app
def test_t5_3a_only_unchanged_when_3b_off():
    r_a = _router()
    assert r_a.status_3b == "off" and r_a.pieces.dir_transfer == {}
    assert not any(str(x) in KE.GROUPS_3B for x in set(r_a.pieces.ride.line_id))


@need_3b
def test_t5_3a_paths_still_evaluate_same(r3b):
    """3-A 만 지나는 경로는 3-B 를 켜도 같은 값 (같은 경로를 두 라우터로 평가)."""
    r_a = _router()
    for o, d in [("금천구청", "월계"), ("신림", "강남")]:
        for q in r_a.evaluate_od(o, d, M)[:3]:
            q2 = r3b.evaluator(td.bin_index_of(M)).evaluate(q.path, M)
            assert q2.actual_time_min == pytest.approx(q.actual_time_min) and q2.lines_3b == ()


# ---------------------------------------------------------------------- T6 시간·환경
@need_3b
def test_t6_holiday_and_saturday_use_holiday_timetable():
    """D-104b 재발 방지: 평일 공휴일 계산기 = 시각표 요일 sunday, 통계 요일 weekday. 3-B 열차는 휴일 시각표에서 찾음."""
    r = _router("sunday", query_date="2026-05-01", include_3b=True)
    e = r.evaluator(td.bin_index_of(M))
    assert e.tt_day == "sunday" and r.scorer(0).day_type == "weekday"
    q = r.evaluate_od("왕십리", "야탑", M)[0]
    assert "불가능한 연결" not in [x.no_service_reason for x in r.excluded]
    c = r.verify(q.path, M, e)
    assert c.status == T.CONNECTED
    st = pd.read_parquet(KE.K3B_MART)
    hol = set(st[(st.day_type == "sunday") & (st.group == "SB")].train)
    assert all(x.train in hol for x in c.rides)
    rs = _router("saturday", include_3b=True)
    assert set(rs.kidx._trains) and all(k[0] == "saturday" for k in rs.kidx._trains)


@need_3b
def test_t6_after_midnight(r3b):
    t = 24 * 60 + 5                                     # 00:05 (운행일 연속 시각)
    e = r3b.evaluator(td.bin_index_of(t))
    q = r3b.evaluate_od("공덕", "청량리", t)
    if q:
        c = r3b.verify(q[0].path, t, e)
        assert c.status in (T.CONNECTED, T.NOT_CONNECTED)
        if c.status == T.CONNECTED:
            assert c.arrive_min > 24 * 60


# ---------------------------------------------------------------------- T7 메타데이터
@need_3b
def test_t7_airport_reference_status(r3b, ev):
    q = r3b.evaluate_od("김포공항", "서울역", M)[0]
    assert "AR" in q.lines_3b
    assert q.recommendation_status == KE.REFERENCE_ROUTE and q.freshness_status == KE.AR_FRESHNESS
    assert q.timetable_valid_from["AR"] == "2025-12-29"
    c = r3b.verify(q.path, M, ev)
    assert c.status == T.CONNECTED and q.recommendation_status == KE.REFERENCE_ROUTE   # 연결 확인 ≠ 확정 추천
    g = r3b.evaluate_od("왕십리", "야탑", M)[0]
    assert g.recommendation_status == "" and g.freshness_status == "" and g.timetable_valid_from == {"SB": "2026-08-22"}
