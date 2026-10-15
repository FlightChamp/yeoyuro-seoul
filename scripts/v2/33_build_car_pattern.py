"""
scripts/v2/33_build_car_pattern.py
==================================
공모전 P0-2 — 칸별 혼잡도 샘플로 '여유 칸 안내(시범)'용 패턴 표를 만든다.

    python scripts/v2/33_build_car_pattern.py --src <샘플 zip 또는 CSV 폴더>
    → data/app_v26/data/marts/car_pattern_sample.parquet   (앱이 있으면 읽고, 없으면 안내를 숨긴다)

이 파일은 공모전 제공 샘플에서 파생된 집계이므로 기본적으로 git 에 올리지 않는다 (.gitignore).
공개 배포 앱에 포함할지는 데이터 이용 조건 확인 후 결정한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.car_pattern import build_pattern, find_sample, read_sample   # noqa: E402
from yeoyuro_v2.proc import utf8_stdout                         # noqa: E402

OUT = ROOT / "data" / "app_v26" / "data" / "marts" / "car_pattern_sample.parquet"


def main() -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "seoulmetro_car_sample"))
    a = ap.parse_args()
    pat = build_pattern(read_sample(find_sample(Path(a.src))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pat.to_parquet(OUT, index=False)
    busy = pat[pat.train_mean >= 60]
    print("=" * 64)
    print(f"저장: {OUT}")
    print(f"구간×30분 조합 {len(pat)}개 (그중 평소 열차 평균 60% 이상 {len(busy)}개 → 앱에서 안내 대상)")
    print(f"시간대 {pat.time_bin.min()} ~ {pat.time_bin.max()}, 역간 {pat[['from_station','to_station']].drop_duplicates().shape[0]}개")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
