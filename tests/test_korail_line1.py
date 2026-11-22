"""
tests/test_korail_line1.py
==========================
Phase 3A-1 (D-085) — 1호선 코레일 서울 시내 구간 시각표 mart 와 '실제 다음 열차' 계산 검사.
원본 없이 커밋된 data/marts/v2/korail_line1 만으로 돈다.

| 검사 | 내용 |
|---|---|
| 무결성 | 범위 36역, 일반열차만, 통과 자리표시·음수 정차 없음, 주행시간 > 0 |
| T1 구로 분기 | 한 열차가 구일·가산디지털단지를 함께 지나지 않음, 구일→가산 직행 열차 없음 |
| T2 광명 셔틀 | 광명행은 서울 안 영등포~금천구청에서만 |
| T3·T4 직통 | 남영→서울역→시청, 회기→청량리→제기동이 한 열차, 제기동↔회기 바로 연결 없음 |
| T5 목적지 정차 필터 | next_train = 전수 탐색 결과, 중간 종착·다른 분기 열차는 후보 아님 |
| T6 자정 이후 | 24:xx 운행, 막차 뒤 None, 00:30 입력 = 운행일 24:30 |
| T12 원천 | 한글 손실 사본·일반열차 통과 표시·음수 정차는 멈춤 |
| 격리 | 앱·엔진 코드가 이 모듈을 import 하지 않음 (기존 v2.3·v2.6 결과 불변) |
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MART = ROOT / "data" / "marts" / "v2" / "korail_line1"

from yeoyuro_v2 import korail_line1 as K                              # noqa: E402

pytestmark = pytest.mark.skipif(not (MART / "stop_times.parquet").exists(), reason="3A-1 mart 없음")


@pytest.fixture(scope="module")
def st():
    return pd.read_parquet(MART / "stop_times.parquet")


@pytest.fixture(scope="module")
def idx(st):
    return K.NextTrainIndex(st)


@pytest.fixture(scope="module")
def seqs(st):
    return {k: list(g.sort_values("stop_seq").code) for k, g in st.groupby(["day_type", "train", "direction"])}


# ------------------------------------------------------------------ 무결성
def test_scope_and_integrity(st):
    assert set(st.code) <= set(K.SCOPE)
    for day in ("weekday", "saturday", "sunday"):
        assert set(K.SEOUL_KORAIL) <= set(st[st.day_type == day].code), day       # 26역 모두 정차
    assert not st.express.any()                                                    # 일반열차만 (D-085)
    assert not st.train.isin(K.ANOMALOUS_EXPRESS).any()
    assert (st.dwell_min.dropna() >= 0).all()                                      # 음수 정차 없음
    assert not (st.arr_min.isna() & st.dep_min.isna()).any()
    t = st[["arr_min", "dep_min"]].stack()
    assert t.min() >= 240 and t.max() <= 27 * 60                                   # 00:00:00 자리표시 없음
    seg = K.segments(st)
    assert (seg.run_min > 0).all() and seg.run_min.max() <= 10
    assert (st.groupby(["day_type", "train", "direction"]).code.apply(lambda s: s.is_unique)).all()


def test_station_names_match_constants():
    s = pd.read_csv(MART / "stations.csv")
    k = s[s.operator == "코레일"]
    assert len(k) == 26 and dict(zip(k.code, k.station)) == {c: n for c, (n, _) in K.SEOUL_KORAIL.items()}
    assert k[["lat", "lon"]].notna().all().all()


# ------------------------------------------------------------------ T1·T2 분기
def test_guro_branch(seqs, idx):
    assert not any(1813 in c and 1702 in c for c in seqs.values())
    for t in range(300, 1500, 7):
        assert idx.next_train("weekday", 1813, 1702, t) is None                   # 구일→가산 직행 없음
    follow = {c[c.index(1701) + 1] for c in seqs.values() if 1701 in c and c.index(1701) + 1 < len(c)}
    assert {1813, 1702} <= follow                                                  # 구로에서 두 갈래


def test_gwangmyeong_shuttle(st):
    gm = st[st.dest == "광명"]
    assert len(gm) and set(gm.code) <= {1006, 1007, 1701, 1702, 1714, 1703}


# ------------------------------------------------------------------ T3·T4 직통
def test_seoul_station_through(seqs, idx):
    d = idx.next_train("weekday", 1002, 151, 8 * 60)
    assert d is not None and d.ride_min < 10
    c = seqs[("weekday", d.train, d.direction)]
    assert c[c.index(1002):c.index(151) + 1] == [1002, 150, 151]                   # 같은 열차, 환승 아님


def test_cheongnyangni_through(seqs, idx, st):
    d = idx.next_train("weekday", 1015, 157, 8 * 60)
    assert d is not None
    c = seqs[("weekday", d.train, d.direction)]
    assert c[c.index(1015):c.index(157) + 1] == [1015, 158, 157]
    seg = K.segments(st)
    assert not (((seg.from_code == 157) & (seg.to_code == 1015)) | ((seg.from_code == 1015) & (seg.to_code == 157))).any()


# ------------------------------------------------------------------ T5 목적지 정차 필터
def _brute(st, day, o, t, q):
    best = None
    for (dd, train, direction), g in st[st.day_type == day].groupby(["day_type", "train", "direction"]):
        g = g.sort_values("stop_seq")
        codes = list(g.code)
        if o not in codes or t not in codes or codes.index(t) <= codes.index(o):
            continue
        dep = g.dep_min.iloc[codes.index(o)]
        if np.isnan(dep) or dep < q - 1e-9:
            continue
        if best is None or dep < best:
            best = dep
    return best


def test_next_train_equals_brute_force(st, idx):
    rnd = random.Random(85)
    korail = list(K.SEOUL_KORAIL)
    for _ in range(60):
        day = rnd.choice(["weekday", "saturday", "sunday"])
        o, t = rnd.sample(korail + [150, 158], 2)
        q = rnd.uniform(300, 1500)
        got = idx.next_train(day, o, t, q)
        exp = _brute(st, day, o, t, q)
        assert (got is None) == (exp is None), (day, o, t, q)
        if got:
            assert got.dep_min == exp and got.wait_min == pytest.approx(exp - q)


def test_short_turn_trains_are_skipped(st, idx, seqs):
    """석계(1018)→월계(1020): 광운대 종착 열차는 후보가 아님 → 대기가 '모든 열차' 기준보다 길어지는 시각이 있다."""
    skipped = 0
    for q in range(420, 600, 1):
        d = idx.next_train("weekday", 1018, 1020, q)
        assert d is not None and 1020 in seqs[("weekday", d.train, d.direction)]
        a = idx.next_train("weekday", 1018, 1019, q)                              # 광운대까지는 모든 열차
        if a.dep_min < d.dep_min:
            assert 1020 not in seqs[("weekday", a.train, a.direction)]             # 앞 열차 = 광운대 종착
            skipped += 1
    assert skipped > 0


# ------------------------------------------------------------------ T6 자정 이후
def test_after_midnight(st, idx):
    assert K.to_service_minutes("00:30") == 1470 and K.to_service_minutes("23:59") == 1439
    assert K.to_minutes("24:10:00") == 1450 and K.to_minutes("05:20:00") == 320
    assert (st.dep_min >= 1440).any()
    deps = idx.departures("weekday", 1004, 1002)                                   # 노량진→남영
    last = max(deps)
    assert last >= 1440                                                            # 자정 넘어 운행
    assert idx.next_train("weekday", 1004, 1002, last + 0.01) is None              # 막차 뒤 = 경로 없음
    d = idx.next_train("weekday", 1004, 1002, 1439)
    assert d is not None and d.dep_min >= 1439


# ------------------------------------------------------------------ 기대 대기
def test_expected_wait():
    e, u = K.expected_wait([0, 6, 12, 18, 24, 30], 0, 30)
    assert e == pytest.approx(3.0) and u == 0
    e, _ = K.expected_wait([0, 2, 12], 0, 12)                                      # Σh²/(2Σh) = (4+100)/24
    assert e == pytest.approx(104 / 24)
    e, u = K.expected_wait([100], 0, 30, max_wait=60)                              # 60분 넘게 기다리는 구간 제외
    assert u == pytest.approx(1.0) and np.isnan(e)
    assert K.headway_half([0, 10, 20], 0, 30) == pytest.approx(5.0)


# ------------------------------------------------------------------ D-093 대기 분해 (검증 3)
def test_wait_profile_uses_trains_after_bin_end():
    """구간 끝 직전 도착 승객은 다음 30분의 열차를 탄다 — 구간 안 열차만 쓰면 대기가 과소추정된다."""
    w = K.wait_profile([25, 40], 0, 30)
    # 0~25분 도착: 25분 열차(대기 25→0), 25~30분 도착: 40분 열차(대기 15→10)
    assert w.E == pytest.approx((25 ** 2 / 2 + (15 ** 2 - 10 ** 2) / 2) / 30)
    assert w.frac_served == pytest.approx(1.0) and w.n_in_bin == 1


def test_wait_profile_ignores_trains_before_bin():
    assert K.wait_profile([-5, 0, 10, 20, 30], 0, 30).E == pytest.approx(5.0)


def test_wait_profile_single_train_and_last_train():
    w = K.wait_profile([10], 0, 30)                      # 막차가 구간 가운데: 10분 뒤 도착 승객은 '운행 종료'
    assert w.E == pytest.approx(5.0) and w.frac_served == pytest.approx(1 / 3)
    assert w.frac_after_last == pytest.approx(2 / 3) and w.reason == K.NO_SERVICE_AFTER_LAST
    w = K.wait_profile([], 0, 30)                        # 그날 운행 없음
    assert np.isnan(w.E) and w.frac_after_last == pytest.approx(1.0)


def test_wait_profile_distinguishes_no_service_reasons():
    w = K.wait_profile([100, 110], 0, 30)                # 첫차까지 60분 초과 → '운행 시작 전'
    assert np.isnan(w.E) and w.reason == K.NO_SERVICE_BEFORE_FIRST and w.frac_before_first == pytest.approx(1.0)
    w = K.wait_profile([5, 100], 30, 60)                 # 앞 열차는 이미 떠남, 다음은 60분 넘게 뒤 → '장시간 대기 제외'
    assert w.frac_long_gap == pytest.approx(1 / 3) and w.reason == K.NO_SERVICE_LONG_GAP
    assert w.E == pytest.approx(50.0)
    w = K.wait_profile([0, 6, 12, 18, 24, 30], 0, 30)
    assert w.reason is None


def test_wait_profile_matches_expected_wait_on_real_data(st, idx):
    """3A-1 의 expected_wait 와 같은 E·같은 '운행 없음' 합계 (기존 3A-1 보고 수치 불변)."""
    rng = random.Random(42)
    codes = sorted(K.SEOUL_KORAIL)
    checked = 0
    while checked < 300:
        o, d = rng.sample(codes, 2)
        deps = idx.departures("weekday", o, d)
        if not deps:
            continue
        b = rng.randrange(270, 1500, 30)
        w = K.wait_profile(deps, b, b + 30)
        e, u = K.expected_wait(deps, b, b + 30)
        assert (np.isnan(e) and np.isnan(w.E)) or w.E == pytest.approx(e)
        assert 1 - w.frac_served == pytest.approx(u, abs=1e-9)
        assert w.frac_served + w.frac_before_first + w.frac_after_last + w.frac_long_gap == pytest.approx(1.0)
        assert np.isnan(w.E) or 0 <= w.E <= 60
        checked += 1


# ------------------------------------------------------------------ D-093 구로 갈아타기 전제
def test_guro_transfer_needed_only_between_branches(idx):
    """경인(구일~온수) ↔ 경부(가산디지털단지~금천구청) 직통 열차는 0 → 구로에서 갈아타야 한다.
    서울 방향 ↔ 각 지선은 같은 열차번호로 직통 → 갈아타기 없음."""
    for day in ("weekday", "saturday", "sunday"):
        for a in K.GYEONGIN_SOUTH:
            for b in K.GYEONGBU_SOUTH:
                assert not K.direct_exists(idx, day, a, b) and not K.direct_exists(idx, day, b, a)
        for far in K.GYEONGIN_SOUTH + K.GYEONGBU_SOUTH:
            assert K.direct_exists(idx, day, 1007, far) and K.direct_exists(idx, day, far, 1007)   # 신도림 ↔ 지선
    assert K.SAME_LINE_TRANSFER_WALK == {K.GURO: 1.0}
    assert K.GURO_WALK_SENSITIVITY == (0.0, 1.0, 3.0, 5.0)


# ------------------------------------------------------------------ T12 원천 방어
def _write(tmp_path, rows):
    p = tmp_path / "tt.csv"
    cols = "고유번호,호선,역사코드,역사명,주중주말,방향,급행여부,열차코드,열차도착시간,열차출발시간,출발역,도착역"
    p.write_bytes(("\n".join([cols] + rows) + "\n").encode("cp949"))
    return p


def test_corrupted_copy_rejected(tmp_path):
    p = _write(tmp_path, ["1,1,1003,???,DAY,UP,0,K1,05:00:00,05:00:30,???,???"])
    with pytest.raises(K.SourceError, match="한글 손실"):
        K.read_timetable(p)


def test_normal_train_with_pass_mark_rejected(tmp_path):
    p = _write(tmp_path, ["1,1,1003,용산,DAY,UP,0,K1,00:00:00,05:00:30,천안,청량리"])
    with pytest.raises(K.SourceError, match="통과 표시"):
        K.line1_rows(K.read_timetable(p))


def test_negative_dwell_rejected(tmp_path):
    p = _write(tmp_path, ["1,1,1003,용산,DAY,UP,0,K1,05:10:00,05:09:00,천안,청량리",
                          "2,1,1002,남영,DAY,UP,0,K1,05:12:00,05:12:30,천안,청량리"])
    with pytest.raises(K.SourceError, match="음수 정차"):
        K.stop_times(K.line1_rows(K.read_timetable(p)))


def test_express_pass_rows_never_become_stops(tmp_path):
    p = _write(tmp_path, ["1,1,1003,용산,DAY,DOWN,1,K9,,05:00:00,용산,동인천",
                          "2,1,1813,구일,DAY,DOWN,1,K9,00:00:00,05:20:00,용산,동인천",
                          "3,1,1801,개봉,DAY,DOWN,1,K9,05:22:00,05:22:30,용산,동인천"])
    rows = K.line1_rows(K.read_timetable(p))
    assert rows.is_pass.sum() == 1 and rows[rows.is_pass].arr_min.isna().all()
    s = K.stop_times(rows, include_express=True)
    assert list(s.code) == [1003, 1801]                                            # 통과역 구일은 정차 아님
    assert K.stop_times(rows).empty                                                # 기본은 일반열차만


# ------------------------------------------------------------------ 격리 (기존 결과 불변)
# 검증 전용 모듈 (엔진·앱이 import 하지 않음): 3A-2a 시각표 연결 검증은 korail_line1 을 쓴다
ISOLATED = {"korail_line1.py", "train_timetable.py"}


def test_engine_and_app_do_not_import_korail_line1():
    """P1(D-093): 기능을 켜기 전까지 엔진·앱은 코레일·열차 단위 검증 모듈을 쓰지 않는다."""
    pat = re.compile(r"korail_line1|train_timetable")
    files = [*ROOT.glob("app/**/*.py"), *ROOT.glob("yeoyuro_v2/*.py"), ROOT / "station_routing.py"]
    users = [str(p.relative_to(ROOT)) for p in files
             if p.name not in ISOLATED and p.exists() and pat.search(p.read_text(encoding="utf-8"))]
    assert users == []


# ------------------------------------------------------------------ 환승 후보·노선도 초안
def test_transfer_candidates():
    tc = pd.read_csv(MART / "transfer_candidates.csv")
    assert len(tc) == 8 and tc.equals(K.transfer_candidates().astype(tc.dtypes.to_dict()))
    nr = tc.set_index("station")
    for st_name in ("노량진", "석계"):                                             # D-088: 3분 확정
        assert nr.loc[st_name, "walk_min"] == 3.0 and nr.loc[st_name, "status"] == "confirmed"
    assert nr.loc["석계", "dataset_mean_min"] == pytest.approx(3.467)                # 민감도 값으로 보존
    ds = tc[tc.status == "dataset+official_distance"]
    assert len(ds) == 6 and (ds.walk_min == ds.dataset_mean_min).all() and ds.distance_m.notna().all()


def test_map_draft():
    d = pd.read_csv(MART / "map_draft.csv")
    assert len(d) == 26 and set(d.code) == set(K.SEOUL_KORAIL)
    assert (d.placement == "anchor").sum() == 8
    assert d.x_px.between(0, 5120).all() and d.y_px.between(0, 2880).all()
