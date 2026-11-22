"""
scripts/v2/44_sb_extension_ridership.py
=======================================
D-091 근거 — 수인분당선 연장 종점 후보(복정 이후) 역별 이용 규모 비교.

복정–야탑 연장을 '개인 이스터에그'가 아니라 데이터로도 설명할 수 있는지 확인한다.
노선도·경로 엔진에는 영향이 없다 (보고서만 만든다).

하는 일
-------
1. 서울시 '지하철 호선별 역별 시간대별 승하차 인원 정보'(월별 합계, cp949)를 읽는다
2. 가장 최근 12개월(자료의 마지막 달 기준)을 골라 역별 **하루 평균 승하차**(승차+하차, 월 합계 ÷ 일수)와
   **아침 7~9시 하루 평균 승차**를 계산
3. 분당선 전 역 순위 + 연장 후보(복정 ~ 오리)와 같은 역의 다른 노선(8호선 모란·복정, 경강선 이매) 표

실행
----
    python scripts/v2/44_sb_extension_ridership.py --src "<원본 폴더>"
    (원본 폴더 = '서울시 지하철 호선별 역별 시간대별 승하차 인원 정보.csv' 가 있는 폴더. 하위 폴더까지 찾는다.
     없으면 [건너뜀] — 커밋된 보고서를 그대로 쓴다)

산출물
------
    reports/v2/sb_extension_ridership.csv   역별 수치 (커밋, 테스트가 읽음)
    reports/v2/sb_extension_ridership.md    요약 보고서

해석 주의
---------
- 환승역 인원은 개찰구가 속한 노선에 잡힌다. 예: 복정의 분당선 인원은 거의 0이고 8호선 복정에 잡힌다.
- 신분당선(정자·미금 환승)은 이 자료에 없다. 정자·미금 수치는 분당선 개찰구분만이다.
- '이용 규모'이지 '혼잡도'가 아니다. 혼잡도는 3-B 에서도 결측으로 둔다 (Phase 3 원칙).
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2.korail_line1 import locate_source  # noqa: E402

SRC_FILE = "서울시 지하철 호선별 역별 시간대별 승하차 인원 정보.csv"
OUT_CSV = ROOT / "reports" / "v2" / "sb_extension_ridership.csv"
OUT_MD = ROOT / "reports" / "v2" / "sb_extension_ridership.md"

# 복정 이후 분당선 순서 (표준데이터 역번호 순, D-091)
EXT_ORDER = ["복정", "가천대", "태평", "모란", "야탑", "이매", "서현", "수내", "정자", "미금", "오리"]
# 같은 역의 다른 노선 (해석용)
OTHER_LINES = [("8호선", "복정"), ("8호선", "모란"), ("경강선", "이매")]
# 연장 끝을 어디서 끊을지 볼 때 같이 보는 '범위 밖 환승' (D-091)
OUT_OF_SCOPE_TRANSFER = {"이매": "경강선", "정자": "신분당선", "미금": "신분당선"}


KEY = ["사용월", "호선명", "지하철역"]


def read_raw(path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="cp949", dtype={"사용월": str})
    d.columns = [c.strip() for c in d.columns]
    return d


def dedupe(d: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """같은 (월·노선·역) 행이 인원까지 똑같이 두 번 실린 경우 하나만 남긴다 (원본에 202603·202607 이 통째로 중복).
    인원이 다른 같은 키 행은 그대로 두고 합산한다 (개찰구 묶음이 나뉜 옛 자료) — 개수만 보고한다."""
    num = [c for c in d.columns if c.endswith("인원")]
    before = len(d)
    d = d.drop_duplicates(subset=KEY + num)
    exact = before - len(d)
    split = int(d.duplicated(KEY, keep=False).sum())
    return d, exact, split


def summarize(raw: pd.DataFrame, months: int = 12) -> tuple[pd.DataFrame, list[str]]:
    """최근 months 개월 역·노선별 하루 평균 승하차, 7~9시 하루 평균 승차."""
    ms = sorted(raw["사용월"].unique())[-months:]
    d, exact, split = dedupe(raw[raw["사용월"].isin(ms)])
    d = d.copy()
    summarize.dups = (exact, split)
    ride = [c for c in d.columns if c.endswith("승차인원")]
    alight = [c for c in d.columns if c.endswith("하차인원")]
    am = [c for c in ride if c.startswith(("07시", "08시"))]
    d["total"] = d[ride + alight].sum(axis=1)
    d["am"] = d[am].sum(axis=1)
    days = sum(calendar.monthrange(int(m[:4]), int(m[4:]))[1] for m in ms)
    g = (d.groupby(["호선명", "지하철역"])[["total", "am"]].sum() / days).round(0).astype(int).reset_index()
    g["n_months"] = d.groupby(["호선명", "지하철역"])["사용월"].nunique().values
    g = g.rename(columns={"호선명": "line", "지하철역": "station", "total": "daily_total", "am": "am_board"})
    return g, ms


def build_table(g: pd.DataFrame) -> pd.DataFrame:
    bd = g[g.line == "분당선"].sort_values("daily_total", ascending=False).reset_index(drop=True)
    bd["rank_bundang"] = bd.index + 1
    bd["n_bundang"] = len(bd)
    rows = []
    for s in EXT_ORDER:
        r = bd[bd.station == s]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(dict(line="분당선", station=s, order=EXT_ORDER.index(s), daily_total=int(r.daily_total),
                         am_board=int(r.am_board), rank_bundang=int(r.rank_bundang), n_bundang=int(r.n_bundang),
                         n_months=int(r.n_months), out_of_scope_transfer=OUT_OF_SCOPE_TRANSFER.get(s, "")))
    for ln, s in OTHER_LINES:
        r = g[(g.line == ln) & (g.station == s)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(dict(line=ln, station=s, order=-1, daily_total=int(r.daily_total), am_board=int(r.am_board),
                         rank_bundang=-1, n_bundang=-1, n_months=int(r.n_months), out_of_scope_transfer=""))
    return pd.DataFrame(rows)


def write_report(t: pd.DataFrame, ms: list[str], src_name: str) -> None:
    bd = t[t.line == "분당선"].set_index("station")
    L = ["# D-091 근거 — 수인분당선 복정 이후 연장 후보 역별 이용 규모", "",
         f"- 원본: `{src_name}` (서울시 열린데이터광장, 월별 합계)",
         f"- 기간: {ms[0]} ~ {ms[-1]} ({len(ms)}개월), 하루 평균 = 기간 합계 ÷ 기간 일수",
         f"- 원본의 완전 중복 행 {summarize.dups[0]:,}개 제거 (같은 월·노선·역·인원이 두 번 실림), "
         f"인원이 다른 같은 키 행 {summarize.dups[1]:,}개는 합산",
         "- 재현: `python scripts/v2/44_sb_extension_ridership.py --src \"<원본 폴더>\"`", "",
         "## 1. 분당선 개찰구 기준 (복정 → 오리 순서)", "",
         "| 순서 | 역 | 하루 평균 승하차 | 분당선 순위 | 7~9시 하루 평균 승차 | 범위 밖 환승 |",
         "|---|---|---|---|---|---|"]
    for r in t[t.line == "분당선"].itertuples():
        L.append(f"| {r.order} | {r.station} | {r.daily_total:,} | {r.rank_bundang} / {r.n_bundang} | "
                 f"{r.am_board:,} | {r.out_of_scope_transfer or '-'} |")
    L += ["", "## 2. 같은 역의 다른 노선 (해석용)", "", "| 노선 | 역 | 하루 평균 승하차 | 7~9시 하루 평균 승차 |",
          "|---|---|---|---|"]
    for r in t[t.line != "분당선"].itertuples():
        L.append(f"| {r.line} | {r.station} | {r.daily_total:,} | {r.am_board:,} |")
    top = t[t.line == "분당선"].sort_values("rank_bundang").iloc[0]
    L += ["", "## 3. 읽는 법", "",
          f"- 분당선 {int(top.n_bundang)}개 역 중 하루 평균 승하차 1위는 **{top.station} ({int(top.daily_total):,}명)**.",
          f"- 야탑은 복정 이후 '범위 밖 환승'(이매 경강선, 정자·미금 신분당선)이 나오기 전 마지막 역이다. "
          f"서현({int(bd.loc['서현', 'daily_total']):,})은 야탑보다 적고 이매(경강선)를 넘어야 한다.",
          "- 환승역 인원은 개찰구가 속한 노선에 잡힌다: 복정은 8호선 쪽에, 모란은 분당선·8호선에 나뉘어 잡힌다.",
          "- 신분당선은 이 자료에 없다. 정자·미금은 분당선 개찰구분만이다.",
          "- 7~9시 승차는 주말·공휴일을 포함한 전체 일수 평균이다.",
          "- 이 표는 **이용 규모**다. 혼잡도가 아니며, 3-B 에서도 코레일 구간 혼잡도는 결측으로 둔다.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--months", type=int, default=12)
    a = ap.parse_args()
    try:
        p = locate_source(a.src, SRC_FILE)
    except FileNotFoundError as e:
        print(f"[건너뜀] {e} — 커밋된 보고서를 그대로 쓸 수 있습니다")
        return 0
    print(f"원본: {p}")
    g, ms = summarize(read_raw(p), a.months)
    t = build_table(g)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")
    write_report(t, ms, p.name)
    print(t.to_string(index=False))
    print(f"\n완전 중복 제거 {summarize.dups[0]:,}행 · 같은 키 합산 {summarize.dups[1]:,}행")
    print(f"\n기간 {ms[0]}~{ms[-1]} · 저장: {OUT_CSV.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
