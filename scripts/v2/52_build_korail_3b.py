"""
scripts/v2/52_build_korail_3b.py — Phase 3B-1 원본 시각표 → 열차 단위 정차표 + 데이터 검사 + 환승 계수 C1 (D-105)
==============================================================================================================

    python scripts/v2/52_build_korail_3b.py                       (원본 폴더 기본값 data/raw/korail_3b)
    python scripts/v2/52_build_korail_3b.py --src "<원본 폴더>"

원본 폴더에 둘 파일 (원본 이름 그대로, 해시 확인 — D-105 2절):
    202608281a045c7905b490.xlsx                  코레일 광역전철 평일 (경의중앙·경춘)
    202608281a045c7cc64640.xlsx                  코레일 광역전철 휴일
    수인분당선 시각표 2026.8.22 적용.xlsx
    공항철도 일반열차 운행시각표 2025.12.29.적용.xlsx
    서울교통공사_환승역거리 소요시간 정보_20260831.csv     (4-3 C1 검증 자료 ②)

산출:
    data/marts/v2/korail_3b/stop_times.parquet, stations.csv     상세 정차표 (gitignore — D-105 P4)
    reports/v2/korail_3b_build.{md,json}                         검사 결과·요약 통계·C1 (커밋)

검사에서 문제가 나오면 산출물을 쓰지 않고 멈춘다 (종료 코드 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2 import korail_3b as K     # noqa: E402

MART = ROOT / "data" / "marts" / "v2" / "korail_3b"
REP = ROOT / "reports" / "v2"
EVIDENCE = ROOT / "data" / "master" / "transfer_3b_evidence.csv"
V1_TRANSFER = ROOT / "data" / "marts" / "transfer_edge_mart.parquet"


def run_c1(src: Path) -> dict:
    off_path = src / K.OFFICIAL_TRANSFER_FILE
    if not off_path.exists():
        raise SystemExit(f"[멈춤] C1 검증 자료 ② 가 없습니다: {off_path.name} 를 원본 폴더에 두세요")
    ev = pd.read_csv(EVIDENCE, dtype={"line_a": str, "line_b": str})
    v1 = pd.read_parquet(V1_TRANSFER)
    v1 = v1[v1.edge_type == "transfer"]
    off = pd.read_csv(off_path, encoding="cp949", dtype=str)
    val = pd.concat([K.validation_3b(ev), K.validation_18(v1, off)], ignore_index=True)
    c = K.calibrate(val)
    c["official_file_sha256"] = K.sha256(off_path)
    c["validation_rows"] = val.round(3).to_dict("records")
    return c


def write_report(check: dict, c1: dict, hashes: dict) -> None:
    e = c1["errors"]
    L = ["# Phase 3B-1 경의중앙·경춘·수인분당·공항철도 정차표 — D-105", "",
         "- 규칙: [사전 등록](../../docs/v2/preregistration_3b.md) 3절(T1~T5)·4-3절(C1)",
         "- 재현: `python scripts/v2/52_build_korail_3b.py` (원본은 저장소에 없음, 상세 정차표도 gitignore — P4)", "",
         "## 원본", "", "| 파일 | SHA-256 |", "|---|---|"]
    L += [f"| {n} | `{h}` |" for n, h in hashes.items()]
    L += ["", "## 데이터 검사 (T4·T5, 운행일 경계)", "",
          "| 항목 | 결과 |", "|---|---|",
          f"| T4 범위 안 열차 수 = 3B-0 감사 | 불일치 **{len(check['t4_mismatch'])}** |",
          f"| T5 경의중앙 평일 > 휴일 | {check['t5_gj_weekday_gt_holiday'][0]} > {check['t5_gj_weekday_gt_holiday'][1]} "
          f"→ {'예' if check['t5_gj_weekday_gt_holiday'][2] else '아니오'} |",
          f"| 시각 역행·같은 역 중복·정차 순서 뒤섞임 | {len(check['time_reversal'])}·{len(check['dup_station'])}·{len(check['non_monotonic'])} |",
          f"| 열차번호 중복 | {check['dup_train']} |",
          f"| 운행일 04:30~다음 날 04:29 밖 시각 | {len(check['outside_service_day'])} |",
          f"| 범위 안 두 역 미만이라 뺀 열차 | {check['skipped_out_of_scope_trains']} |",
          f"| 정차표 행 수 | {check['rows']:,} |", "",
          "| 구간·요일·방향 | 열차 수 |", "|---|---|"] + [f"| {k.replace('|', ' · ')} | {v} |" for k, v in check["t4"].items()]
    L += ["", "그룹별 열차 수 (휴일 시각표 = 토·일 두 벌):", "", "| 요일·그룹 | 열차 |", "|---|---|"] + \
         [f"| {k.replace('|', ' · ')} | {v} |" for k, v in check["trains"].items()]
    L += ["", "## B 등급 환승 계수 C1 (사전 등록 4-3)", "",
          f"- 규칙: {c1['rule']}",
          "", "| 검증 자료 | n | ×1.00 | ×1.60 | ×1.81 | ×2.78 |", "|---|---|---|---|---|---|"]
    for name, lab in (("all", "①∪② (판정)"), ("3B", "① 3-B"), ("1-8", "② 1~8호선")):
        r = e[name]
        L.append(f"| {lab} | {r['n']} | {r['1.00']:.3f} | {r['1.60']:.3f} | {r['1.81']:.3f} | {r['2.78']:.3f} |")
    L += ["", f"- 값 = 절대 오차 중앙값(분). 오차 최소 계수 ×{c1['best_by_error']:.2f} → **기본값 ×{c1['chosen']:.2f}**",
          f"- 관측 ÷ 공식 시간 비 중앙값: {c1['ratio_median']}",
          "- 이 기본값은 3B-4 결과를 본 뒤 바꾸지 않는다 (C4). 경로 영향은 3B-4 B5 에서 네 계수로 보고",
          "- 한계: 관측값 자체도 실측이 아님 (① 수도권 환승 데이터·지도 앱, ② v1 환승시간)", ""]
    (REP / "korail_3b_build.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    slim = {k: v for k, v in c1.items() if k != "validation_rows"}
    (REP / "korail_3b_build.json").write_text(
        json.dumps({"check": check, "c1": slim, "c1_validation": c1["validation_rows"], "sources": hashes},
                   ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "korail_3b"))
    a = ap.parse_args()
    src = Path(a.src)
    try:
        paths = K.locate(src)
    except ValueError as e:
        print(f"[멈춤] 원본 확인 실패: {e}\n  원본 폴더: {src}")
        return 1
    print("[OK] 원본 4개 해시 일치")
    stop_times, stations, check = K.build(paths)
    bad = K.problems(check)
    if bad:
        print("[멈춤] 데이터 검사 문제 — 산출물을 쓰지 않았습니다:")
        for b in bad:
            print("   ", b)
        return 1
    print(f"[OK] 데이터 검사 통과 (T4 불일치 0, T5 {check['t5_gj_weekday_gt_holiday'][:2]}, 행 {check['rows']:,})")
    c1 = run_c1(src)
    print(f"[OK] C1: 오차 최소 ×{c1['best_by_error']:.2f} → 기본값 ×{c1['chosen']:.2f}")
    MART.mkdir(parents=True, exist_ok=True)
    stop_times.to_parquet(MART / "stop_times.parquet", index=False)
    stations.to_csv(MART / "stations.csv", index=False, encoding="utf-8", lineterminator="\n")
    hashes = {n: K.sha256(p) for n, p in ((p.name, p) for p in paths.values())}
    hashes[K.OFFICIAL_TRANSFER_FILE] = c1["official_file_sha256"]
    REP.mkdir(parents=True, exist_ok=True)
    write_report(check, c1, hashes)
    print(f"저장: {MART / 'stop_times.parquet'} (gitignore), {REP / 'korail_3b_build.md'}")
    return 0


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
