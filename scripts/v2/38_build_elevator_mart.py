"""
scripts/v2/38_build_elevator_mart.py
====================================
공모전판 — 빠른하차정보(1~8호선)에서 앱용 '엘리베이터 하차 칸' 표를 만든다.

    python scripts/v2/38_build_elevator_mart.py
    입력: data/raw/fast_exit/fast_exit_all.csv  (scripts/v2/34_fetch_fast_exit.py 로 받은 원본)
    출력: data/app_v26/data/marts/fast_exit_elevator.parquet

원본은 공공데이터포털에 공개된 서울교통공사 데이터다. 앱 표에는 출처와 기준일자를 함께 둔다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.facility_match import alight_elevators, build_elevator_mart   # noqa: E402
from yeoyuro_v2.proc import utf8_stdout                                        # noqa: E402

OUT = ROOT / "data" / "app_v26" / "data" / "marts" / "fast_exit_elevator.parquet"


def coverage(mart: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """앱 노선망의 '이전역 → 하차역' 도착 조합 가운데 엘리베이터 칸을 안내할 수 있는 비율."""
    e = edges[(edges.edge_type == "ride") & edges.line_id.astype(str).isin(list("12345678"))]
    e = e.drop_duplicates(["line_id", "from_station", "to_station"])
    ok = sum(bool(alight_elevators(mart, str(r.line_id), r.to_station, r.from_station)) for r in e.itertuples())
    return {"arrivals": len(e), "with_elevator": ok}


def main() -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "fast_exit" / "fast_exit_all.csv"))
    a = ap.parse_args()
    fe = pd.read_csv(a.src, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    mart = build_elevator_mart(fe)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mart.to_parquet(OUT, index=False)
    print("=" * 64)
    print(f"저장: {OUT}")
    print(f"엘리베이터 칸-문 {len(mart)}행 · 노선별 {mart.line.value_counts().sort_index().to_dict()}")
    print(f"기준일자 {sorted(mart.crtr_ymd.unique())}")
    edges = ROOT / "data" / "app_v26" / "data" / "marts" / "route_edge_mart.parquet"
    if edges.exists():
        c = coverage(mart, pd.read_parquet(edges))
        print(f"안내 가능한 도착(이전역→하차역) {c['with_elevator']}/{c['arrivals']} "
              f"({100 * c['with_elevator'] / c['arrivals']:.1f}%) · 나머지는 방향을 하나로 정할 수 없거나 자료 없음 → 안내하지 않음")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
