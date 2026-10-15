"""
scripts/v2/35_facility_poc.py
=============================
공모전 Facility PoC — 엘리베이터 칸-문(빠른하차정보) × 칸별 혼잡도 패턴(4호선 샘플) 실데이터 결합.

    python scripts/v2/35_facility_poc.py
    입력: data/raw/fast_exit/fast_exit_all.csv            (scripts/v2/34 로 수집, 공공데이터포털)
          data/app_v26/data/marts/car_pattern_sample.parquet (scripts/v2/33 로 생성)
    출력: reports/v2/facility_poc.md · facility_poc.json · facility_poc_cases.csv

판정 규칙 (분석 전에 고정)
- 후보 = 엘리베이터 칸 ± 1칸, 열차 평균 60% 이상인 구간·시간만
- 옆 칸이 엘리베이터 칸보다 5%p 이상 덜 붐빌 때만 옆 칸을 권한다 (아니면 엘리베이터 칸 그대로)
- 칸 번호 기준(진행 방향)이 두 자료에서 같은지 확인되지 않아, 반대 기준일 때의 결과를 함께 계산한다
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.facility_match import load_fast_exit, match, parse_position, survey   # noqa: E402
from yeoyuro_v2.proc import utf8_stdout                                               # noqa: E402

FE = ROOT / "data" / "raw" / "fast_exit" / "fast_exit_all.csv"
PAT = ROOT / "data" / "app_v26" / "data" / "marts" / "car_pattern_sample.parquet"
OUT = ROOT / "reports" / "v2"
EXAMPLE = ("동대문", "동대문역사문화공원", "07:30")


def main() -> int:
    utf8_stdout()
    for p, how in ((FE, "python scripts/v2/34_fetch_fast_exit.py --key <인증키>"),
                   (PAT, "python scripts/v2/33_build_car_pattern.py")):
        if not p.exists():
            print(f"[중단] {p} 가 없습니다 → {how}")
            return 1
    fe_all = load_fast_exit(FE, line=None)
    fe = fe_all[fe_all.lineNm == "4호선"]
    pat = pd.read_parquet(PAT)
    s = survey(fe, pat)
    s["moved"] = s.pick != s.nearest
    s["mirror_moved"] = s.mirror_pick != s.nearest
    s["orientation_robust"] = s.pick == s.mirror_pick

    # 같은 엘리베이터의 양방향 표기: 칸 번호 합 = 편성 칸 수 + 1 이면 '진행 방향 기준' 번호
    cars_per_line = {"1호선": 10, "2호선": 10, "3호선": 10, "4호선": 10, "5호선": 8, "6호선": 8, "7호선": 8}
    both, mirrored, both4, mirrored4 = 0, 0, 0, 0
    for ln, g in fe_all[fe_all.plfmCmgFac == "엘리베이터"].groupby("lineNm"):
        n = cars_per_line.get(ln)
        for t in g.facPstnNm.unique():
            q = parse_position(t)
            if n and len(q) == 2:
                ok = q[0][1] + q[1][1] == n + 1
                both, mirrored = both + 1, mirrored + ok
                if ln == "4호선":
                    both4, mirrored4 = both4 + 1, mirrored4 + ok

    ex = s[(s["from"] == EXAMPLE[0]) & (s["to"] == EXAMPLE[1]) & (s.time_bin == EXAMPLE[2])]
    ex_row = ex.iloc[0].to_dict() if len(ex) else None
    ex_detail = None
    if ex_row:
        r = pat[(pat.from_station == EXAMPLE[0]) & (pat.to_station == EXAMPLE[1]) & (pat.time_bin == EXAMPLE[2])].iloc[0]
        vec = [float(r[f"car_{i}"]) for i in range(1, 11)]
        ex_detail = {"car_values": [round(x, 1) for x in vec], "match": match(vec, int(ex_row["nearest"])),
                     "match_mirrored": match(vec, int(ex_row["nearest"]), mirrored=True)}

    res = {
        "source": {"fast_exit_rows": int(len(fe_all)), "fast_exit_line4_rows": int(len(fe)),
                   "line4_elevator_rows": int((fe.plfmCmgFac == "엘리베이터").sum()),
                   "line4_stations": int(fe.stnNm.nunique()), "crtrYmd": sorted(fe_all.crtrYmd.unique().tolist()),
                   "elevator_positions_both_directions": both, "of_which_mirrored_numbering": mirrored,
                   "line4_both_directions": both4, "line4_mirrored": mirrored4},
        "rule": {"window": 1, "meaningful_pp": 5.0, "busy_train_mean": 60.0},
        "survey": {"combos": int(len(s)), "edges": int(s[["from", "to"]].drop_duplicates().shape[0]),
                   "neighbor_better_share": round(float(s.moved.mean()), 3),
                   "gain_when_moved_mean": round(float(s[s.moved].gain.mean()), 1),
                   "gain_when_moved_median": round(float(s[s.moved].gain.median()), 1),
                   "all_candidates_100plus": int(s.all_crowded.sum()),
                   "orientation_same_pick_share": round(float(s.orientation_robust.mean()), 3),
                   "mirror_neighbor_better_share": round(float(s.mirror_moved.mean()), 3)},
        "example": ex_row, "example_detail": ex_detail,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "facility_poc_cases.csv", index=False, encoding="utf-8-sig")
    (OUT / "facility_poc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(res)

    v = res["survey"]
    print("=" * 72)
    print(f"빠른하차정보 {res['source']['fast_exit_rows']}행 (4호선 {res['source']['fast_exit_line4_rows']}행, "
          f"엘리베이터 {res['source']['line4_elevator_rows']}행, 기준일 {','.join(res['source']['crtrYmd'])})")
    print(f"양방향 표기 엘리베이터(1~7호선) {both}곳 중 {mirrored}곳이 '칸 번호 합 = 칸 수 + 1' "
          f"(4호선 {mirrored4}/{both4}) → 칸 번호 = 진행 방향 기준")
    print(f"매칭 조합 {v['combos']}개 (역간 {v['edges']}개, 열차 평균 60% 이상)")
    print(f"옆 칸이 5%p 이상 덜 붐빔: {v['neighbor_better_share']:.0%} · 그때 평균 {v['gain_when_moved_mean']}%p "
          f"(중앙 {v['gain_when_moved_median']}) · 한 칸 더 걷기")
    print(f"후보 칸이 모두 100% 이상: {v['all_candidates_100plus']}건 → 출발 시각 대안 권유")
    print(f"칸 번호 기준을 반대로 가정해도 같은 칸: {v['orientation_same_pick_share']:.0%}")
    if ex_detail:
        m, mm = ex_detail["match"], ex_detail["match_mirrored"]
        print(f"예시 {EXAMPLE[0]}→{EXAMPLE[1]} 평일 {EXAMPLE[2]}: 엘리베이터 {ex_row['elevator']} → "
              f"{m['nearest']}번째 칸 {m['nearest_value']}% vs {m['pick']}번째 칸 {m['pick_value']}% "
              f"(−{m['gain']}%p, {m['extra_cars']}칸 더) · 반대 기준 가정 시 {mm['pick']}번째 칸 (−{mm['gain']}%p)")
    print("저장:", OUT / "facility_poc.md")
    print("=" * 72)
    return 0


def write_md(res: dict) -> None:
    v, src = res["survey"], res["source"]
    L = ["# Facility PoC — 엘리베이터 칸-문 × 칸별 혼잡도 (4호선)", "",
         f"- 빠른하차정보(공공데이터포털, 기준일 {','.join(src['crtrYmd'])}): {src['fast_exit_rows']}행, "
         f"4호선 {src['fast_exit_line4_rows']}행·엘리베이터 {src['line4_elevator_rows']}행·{src['line4_stations']}개 역",
         f"- 같은 엘리베이터의 양방향 표기(1~7호선) {src['elevator_positions_both_directions']}곳 중 "
         f"{src['of_which_mirrored_numbering']}곳이 '칸 번호 합 = 칸 수 + 1' (4호선 {src['line4_mirrored']}/{src['line4_both_directions']})"
         " → 빠른하차정보의 칸 번호는 진행 방향 기준",
         "- 칸별 혼잡도 샘플의 칸 번호 기준은 공식 정의 없음 → 반대 기준 가정 결과를 함께 보고", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 매칭 조합 (엘리베이터 역 도착 역간 × 30분, 열차 평균 60%+) | {v['combos']} (역간 {v['edges']}) |",
         f"| 옆 칸이 5%p 이상 덜 붐빈 비율 | {v['neighbor_better_share']:.0%} |",
         f"| 그때 감소 (평균 / 중앙) | {v['gain_when_moved_mean']} / {v['gain_when_moved_median']}%p, 한 칸 더 걷기 |",
         f"| 후보 칸이 모두 100% 이상 | {v['all_candidates_100plus']}건 (출발 시각 대안 권유) |",
         f"| 칸 번호 기준을 반대로 가정해도 같은 칸 | {v['orientation_same_pick_share']:.0%} |"]
    if res.get("example_detail"):
        e, m, mm = res["example"], res["example_detail"]["match"], res["example_detail"]["match_mirrored"]
        L += ["", f"예시: {e['from']}→{e['to']} ({e['toward']} 방면) 평일 {e['time_bin']}, 통과 {e['n_passages']}회 평균",
              f"- 엘리베이터 {e['elevator']} → 가장 가까운 칸 {m['nearest']}번째 ({m['nearest_value']}%)",
              f"- 후보 {m['candidates']} → {m['pick']}번째 칸 ({m['pick_value']}%, −{m['gain']}%p, {m['extra_cars']}칸 더)",
              f"- 칸 번호 기준을 반대로 가정해도 {mm['pick']}번째 칸 (−{mm['gain']}%p)",
              f"- 후보 칸이 모두 100% 이상: {m['all_crowded']} → 출발 시각 대안을 함께 권함"]
    L += ["", "한계: 4호선 샘플(편성 6개, 2026년 6월 평일)의 과거 패턴이며 실시간이 아니다. 통과 수가 적은 조합이 있다."]
    (OUT / "facility_poc.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
