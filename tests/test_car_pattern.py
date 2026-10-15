"""
공모전 P0-2 '여유 칸 안내(시범)' 테스트.

- 순수 로직(build_pattern / segment_hint / route_hints)은 합성 데이터로 항상 검사한다.
- 앱 표시 검사는 패턴 파일(data/app_v26/data/marts/car_pattern_sample.parquet)이 있을 때만 돈다.
  (공모전 제공 샘플에서 만든 파일이라 git 에 없다 → 없으면 skip, 그리고 '없으면 안내가 숨는다' 를 검사)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from yeoyuro_v2.car_pattern import BUSY, build_pattern, route_hints, segment_hint   # noqa: E402

MART = ROOT / "data" / "app_v26" / "data" / "marts" / "car_pattern_sample.parquet"


def _raw(cars_by_row, stations=("수유", "미아"), dates=("2026-06-17", "2026-06-18"), forms=("404", "406"),
         hhmm="08:05", blank_last=False):
    """합성 원자료. 날짜 × 편성마다 같은 칸 값 10초 행 3개."""
    rows = []
    for d in dates:
        for f in forms:
            for s in range(3):
                r = {"발생일자": f"{d} {hhmm}:{s * 10:02d}", "편성번호": f,
                     "현재역 명": stations[0], "다음역 명": stations[1]}
                for k, v in enumerate(cars_by_row):
                    r[f"{k + 1}번째 칸 혼잡도(%)"] = v
                    r[f"{k + 1}번째 칸 혼잡도(등급)"] = 1
                if blank_last:
                    r["10번째 칸 혼잡도(%)"] = None
                rows.append(r)
    return pd.DataFrame(rows)


CARS = [90, 85, 50, 80, 95, 100, 88, 55, 92, 99]          # 평균 83.4, 가장 여유 3번째(50), 다음 8번째(55)


def test_build_pattern_basic():
    p = build_pattern(_raw(CARS))
    assert len(p) == 1
    r = p.iloc[0]
    assert (r.from_station, r.to_station, r.time_bin) == ("수유", "미아", "08:00")
    assert r.n_passages == 4 and r.n_days == 2
    assert r.train_mean == pytest.approx(sum(CARS) / 10)


def test_build_pattern_filters():
    assert build_pattern(_raw(CARS, blank_last=True)).empty            # 빈 칸 있는 행 제외
    assert build_pattern(_raw(CARS, dates=("2026-06-17",))).empty      # 하루뿐이면 제외 (MIN_DAYS)
    assert build_pattern(_raw(CARS, dates=("2026-06-20", "2026-06-21"))).empty   # 주말 제외
    p = build_pattern(_raw(CARS, stations=("당고개", "상계")))         # 옛 역명 → 앱 역명
    assert p.iloc[0].from_station == "불암산"


def test_segment_hint_picks_least_crowded_cars():
    p = build_pattern(_raw(CARS))
    h = segment_hint(p, "4", ["수유", "미아"], 8 * 60 + 10)
    assert h["cars"] == [3, 8]
    assert h["gain"] == pytest.approx(sum(CARS) / 10 - 50, abs=0.05)
    assert h["n_edges"] == 1 and h["time_bin"] == "08:00"


def test_segment_hint_none_cases():
    p = build_pattern(_raw(CARS))
    assert segment_hint(p, "4", ["수유", "미아"], 9 * 60) is None          # 다른 시간대
    assert segment_hint(p, "2", ["수유", "미아"], 8 * 60) is None          # 다른 호선
    assert segment_hint(p, "4", ["미아", "수유"], 8 * 60) is None          # 반대 방향
    assert segment_hint(p, "4", ["수유"], 8 * 60) is None
    assert segment_hint(None, "4", ["수유", "미아"], 8 * 60) is None
    quiet = build_pattern(_raw([BUSY - 10] * 9 + [BUSY - 30]))
    assert segment_hint(quiet, "4", ["수유", "미아"], 8 * 60) is None      # 붐비지 않으면 안내 안 함


def test_route_hints_boarding_time_accumulates():
    p = build_pattern(_raw(CARS, hhmm="08:35"))
    segs = [{"kind": "ride", "line": "2", "stations": ["가", "나"], "minutes": 20},
            {"kind": "transfer", "minutes": 4, "wait_min": 3},
            {"kind": "ride", "line": "4", "stations": ["수유", "미아"], "minutes": 2}]
    # 08:10 출발 → 4호선 승차 08:37 (20분 승차 + 4분 환승 + 3분 대기) → 08:30 시간대 패턴 사용
    h = route_hints(p, segs, 8 * 60 + 10)
    assert list(h) == [2] and h[2]["time_bin"] == "08:30" and h[2]["cars"] == [3, 8]
    assert route_hints(p, segs, 8 * 60) == {}                    # 승차 08:27 → 08:00 시간대엔 근거 없음
    assert route_hints(None, segs, 8 * 60) == {}


# ---------------------------------------------------------------- 앱 표시
HINT_TEXT = "여유 칸(시범)"


def _markdown_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.mark.skipif(not MART.exists(), reason="패턴 파일 없음 (scripts/v2/33_build_car_pattern.py 로 생성)")
def test_app_shows_hint_weekday_line4():
    import app_harness as H
    at = H.search(("수유", "서울역", "2026-10-14", "08:00", "calm"))
    assert not at.exception
    txt = _markdown_text(at)
    assert HINT_TEXT in txt and "실시간 아님" in txt


@pytest.mark.skipif(not MART.exists(), reason="패턴 파일 없음")
def test_app_no_hint_on_weekend():
    import app_harness as H
    at = H.search(("수유", "서울역", "2026-10-17", "08:00", "calm"))      # 토요일
    assert not at.exception
    assert HINT_TEXT not in _markdown_text(at)


def test_app_hides_hint_without_mart(tmp_path, monkeypatch):
    """패턴 파일이 없으면 안내가 숨고 화면은 정상."""
    import app_harness as H
    if MART.exists():
        backup = tmp_path / MART.name
        shutil.move(str(MART), backup)
        try:
            at = H.search(("수유", "서울역", "2026-10-14", "08:00", "calm"))
        finally:
            shutil.move(str(backup), MART)
    else:
        at = H.search(("수유", "서울역", "2026-10-14", "08:00", "calm"))
    assert not at.exception
    assert HINT_TEXT not in _markdown_text(at)


def test_find_sample_from_downloads(tmp_path, monkeypatch):
    """대상 폴더가 비어 있으면 다운로드 폴더에서 '편성_날짜.csv' 가 든 zip 을 찾아 복사한다."""
    import zipfile
    from yeoyuro_v2.car_pattern import find_sample
    dl = tmp_path / "Downloads"
    dl.mkdir()
    with zipfile.ZipFile(dl / "샘플.zip", "w") as z:
        for n in ("404_20260617.csv", "404_20260618.csv", "406_20260617.csv"):
            z.writestr(n, _raw(CARS).to_csv(index=False))
    (dl / "다른파일.zip").write_bytes(b"not a zip")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    tgt = tmp_path / "data" / "raw" / "seoulmetro_car_sample"
    assert find_sample(tgt) == tgt and (tgt / "샘플.zip").exists()
    from yeoyuro_v2.car_pattern import read_sample
    assert len(read_sample(tgt)) == 3 * len(_raw(CARS))
    with pytest.raises(FileNotFoundError):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
        find_sample(tmp_path / "other")
