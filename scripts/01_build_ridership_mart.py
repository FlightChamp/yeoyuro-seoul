"""
01_build_ridership_mart.py
==========================
여유로 서울 Phase 1 — 48개월 역별 시간대별 이용인원 통합 파이프라인.

이 스크립트는 '실제 원본 48개 파일을 전수 감사한 결과'를 그대로 코드로 옮긴 것이다.
추측으로 만든 파서가 아니라, 아래 이상 징후를 이미 확인하고 대응한 파서다.

전수 감사에서 확인된 원본 이상 징후
-----------------------------------
A. 컬럼 드리프트
   - '06시 이전' vs '6시 이전'  : 2023-08 ~ 2023-11 4개 파일만 '6시 이전'
   - 2023-12 파일에만 '00 ~ 01','01 ~ 02','02 ~ 03','03 ~ 04' 4개 컬럼 추가 존재
     (= '24시 이후'를 시간대별로 쪼갠 것. 합산하면 '24시 이후'와 일치하므로
        그대로 더하면 심야 인원이 2배가 된다 -> 반드시 제외)
B. 역명 표기
   - 부기명 괄호 표기가 섞임: '교대(법원.검찰청)', '동대문역사문화공원(DDP)' 등 57개
   - 개명 5건: 당고개->불암산, 삼각지->삼각지(전쟁기념관)(4·6호선),
               상봉(시외버스터미널)->상봉, 뚝섬유원지->자양(뚝섬한강공원)
   - 4호선 '총신대입구(이수)' 와 7호선 '이수' 는 같은 역, 다른 표기
C. 범위 밖 역 혼입
   - 7호선 부천/인천 구간(역번호 2753~2760)이 일부 월에만 유령 행으로 등장.
     전부 값이 0이므로 실데이터가 아니라 잡음 -> 제거
D. 승하차 호선 귀속 문제 (가장 중요)
   - 신내(6호선), 연신내(6호선) : 전 기간 0
   - 충무로(3호선) : 총합 136명 (사실상 0, 4호선에 전량 귀속)
   - 까치산 : 5호선에만 존재. 2호선 신정지선 승하차 없음
   -> 환승역 승하차를 '호선별 수요'로 해석하면 안 된다. 역 단위로만 해석해야 한다.

출력
----
    data/marts/ridership_hourly_mart.parquet   (long format)
    data/master/station_master.csv
    data/master/station_alias_master.csv
    reports/data_quality/ridership_quality_report.md

사용법
------
    python scripts/01_build_ridership_mart.py --raw data/raw/ridership_monthly --root .
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 상수 — 감사 결과 기반
# ---------------------------------------------------------------------
HOUR_BINS = (["06시 이전"]
             + [f"{h:02d} ~ {h+1:02d}" for h in range(6, 24)]
             + ["24시 이후"])

# 2023-12 파일에만 존재. '24시 이후'의 세부 분해이므로 합산 대상에서 제외한다.
DUPLICATE_MIDNIGHT_COLS = ["00 ~ 01", "01 ~ 02", "02 ~ 03", "03 ~ 04"]

HOUR_BIN_ALIASES = {"6시 이전": "06시 이전"}

# 범위 밖(7호선 부천/인천 구간) — 값이 전부 0인 유령 행
OUT_OF_SCOPE_CODES = {("7호선", c) for c in range(2753, 2761)}

# 괄호 부기명 제거만으로는 해결되지 않는 강제 매핑
FORCED_CANONICAL = {
    "총신대입구(이수)": "이수",
    "청량리(서울시립대입구)": "청량리",
}

# 승하차가 해당 호선에 귀속되지 않는 역 (감사 결과 D)
NON_ATTRIBUTED_UIDS = {
    "6_2649": "신내 — 전 기간 0 (경춘선 공용역, 코레일 집계 추정)",
    "6_2615": "연신내 — 전 기간 0, 승하차는 3호선에 전량 귀속",
    "3_321": "충무로 — 총합 136명, 승하차는 4호선에 전량 귀속",
}

FILENAME_RE = re.compile(r"(\d{4})년[_ ]?(\d{1,2})월")


def canonical_station_name(name: str) -> str:
    """'교대(법원.검찰청)' -> '교대', '총신대입구(이수)' -> '이수'"""
    name = str(name).strip()
    if name in FORCED_CANONICAL:
        return FORCED_CANONICAL[name]
    base = re.sub(r"\(.*\)$", "", name).strip()
    return base if base else name


# ---------------------------------------------------------------------
# 1. 적재
# ---------------------------------------------------------------------
class RidershipLoader:
    """월별 xlsx 를 읽어 컬럼 드리프트를 흡수한 뒤 하나로 합친다."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.file_log: list[dict] = []

    def _find_files(self) -> list[Path]:
        files = sorted(self.raw_dir.glob("*역별*시간대별*이용인원*.xlsx"))
        if not files:
            files = sorted(self.raw_dir.glob("*.xlsx"))
        if not files:
            raise FileNotFoundError(f"원본 파일을 찾지 못했습니다: {self.raw_dir}")
        return files

    def load(self) -> pd.DataFrame:
        frames = []
        for path in self._find_files():
            m = FILENAME_RE.search(path.name)
            if not m:
                self.file_log.append({"file": path.name, "status": "SKIP(파일명 파싱 실패)"})
                continue
            ym = f"{m.group(1)}-{int(m.group(2)):02d}"

            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
            df = df.rename(columns={k: v for k, v in HOUR_BIN_ALIASES.items() if k in df.columns})

            dropped = [c for c in DUPLICATE_MIDNIGHT_COLS if c in df.columns]
            if dropped:
                df = df.drop(columns=dropped)

            missing = [c for c in HOUR_BINS if c not in df.columns]
            if missing:
                raise ValueError(f"{path.name}: 시간대 컬럼 누락 {missing}")

            df["source_ym"] = ym
            df["source_file"] = path.name
            frames.append(df)
            self.file_log.append({
                "file": path.name, "ym": ym, "rows": len(df),
                "status": "OK", "dropped_cols": ";".join(dropped),
            })
            print(f"  loaded {ym}  rows={len(df):>6}"
                  f"{'  [심야 분해컬럼 제거]' if dropped else ''}", flush=True)

        raw = pd.concat(frames, ignore_index=True)
        raw["날짜"] = pd.to_datetime(raw["날짜"])
        return raw


# ---------------------------------------------------------------------
# 2. 마스터 생성
# ---------------------------------------------------------------------
class MasterBuilder:
    def __init__(self, raw: pd.DataFrame):
        self.raw = raw

    def build_station_master(self) -> pd.DataFrame:
        raw = self.raw.copy()
        raw["ym"] = raw["source_ym"]
        # 최신 월 기준 표기를 '현재 역명'으로 채택한다.
        # (사전순 last 를 쓰면 '상봉(시외버스터미널)->상봉' 같은 축약 개명을 거꾸로 잡는다)
        latest = (raw.sort_values("ym")
                     .groupby(["호선", "역번호"])["역명"].last().rename("station_name_latest"))
        agg = (raw.groupby(["호선", "역번호"])
                  .agg(names=("역명", lambda s: sorted(set(s))),
                       first_ym=("ym", "min"), last_ym=("ym", "max"),
                       n_months=("ym", "nunique"), total=("합 계", "sum"))
                  .join(latest, on=["호선", "역번호"])
                  .reset_index())

        agg["line_id"] = agg["호선"].str.replace("호선", "", regex=False)
        agg["station_code"] = agg["역번호"]
        agg["station_uid"] = agg["line_id"] + "_" + agg["station_code"].astype(str)
        agg["station_name_raw"] = agg["station_name_latest"]
        agg["station_name"] = agg["station_name_raw"].apply(canonical_station_name)
        agg["renamed_from"] = [
            "|".join([n for n in names if n != cur])
            for names, cur in zip(agg["names"], agg["station_name_raw"])]
        agg["in_project_scope"] = [
            0 if (h, c) in OUT_OF_SCOPE_CODES else 1
            for h, c in zip(agg["호선"], agg["station_code"])
        ]

        # 같은 역명이 여러 호선에 존재할 때, 이 호선에 승하차가 얼마나 귀속되는가
        station_total = agg.groupby("station_name")["total"].transform("sum")
        agg["share_in_station"] = np.where(station_total > 0,
                                           (agg["total"] / station_total).round(4), 0.0)
        agg["attribution_flag"] = np.select(
            [agg["total"] <= 0, agg["share_in_station"] < 0.01],
            ["none", "negligible"], default="ok")
        agg["is_transfer_station"] = (
            agg.groupby("station_name")["station_uid"].transform("nunique") > 1).astype(int)
        agg["coverage_full"] = (agg["n_months"] == agg["n_months"].max()).astype(int)
        agg["attribution_note"] = agg["station_uid"].map(NON_ATTRIBUTED_UIDS).fillna("")

        cols = ["station_uid", "line_id", "station_code", "station_name", "station_name_raw",
                "renamed_from", "in_project_scope", "is_transfer_station",
                "attribution_flag", "share_in_station", "attribution_note",
                "coverage_full", "n_months", "first_ym", "last_ym", "total"]
        return agg[cols].sort_values(["line_id", "station_code"]).reset_index(drop=True)

    @staticmethod
    def build_alias_master(raw: pd.DataFrame, station_master: pd.DataFrame) -> pd.DataFrame:
        rows = []
        seen = set()
        pair = raw[["호선", "역번호", "역명"]].drop_duplicates()
        pair["line_id"] = pair["호선"].str.replace("호선", "", regex=False)
        pair["station_uid"] = pair["line_id"] + "_" + pair["역번호"].astype(str)
        canon_map = dict(zip(station_master["station_uid"], station_master["station_name"]))
        latest = dict(zip(station_master["station_uid"], station_master["station_name_raw"]))
        for _, r in pair.iterrows():
            canon = canon_map.get(r["station_uid"])
            if canon is None or r["역명"] == canon:
                continue
            key = (r["역명"], canon, r["line_id"])
            if key in seen:
                continue
            seen.add(key)
            src = "parenthetical" if r["역명"] == latest.get(r["station_uid"]) else "renamed"
            rows.append({"alias_name": r["역명"], "canonical_name": canon,
                         "line_id": r["line_id"], "station_uid": r["station_uid"],
                         "alias_source": src})
        return pd.DataFrame(rows).sort_values(["canonical_name", "alias_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 3. Mart 변환
# ---------------------------------------------------------------------
class MartBuilder:
    def __init__(self, raw: pd.DataFrame, station_master: pd.DataFrame):
        self.raw = raw
        self.sm = station_master

    def build(self) -> pd.DataFrame:
        raw = self.raw.copy()
        raw["line_id"] = raw["호선"].str.replace("호선", "", regex=False)
        raw["station_uid"] = raw["line_id"] + "_" + raw["역번호"].astype(str)

        keep = set(self.sm.loc[self.sm["in_project_scope"] == 1, "station_uid"])
        before = len(raw)
        raw = raw[raw["station_uid"].isin(keep)]
        self.dropped_rows = before - len(raw)

        long = raw.melt(
            id_vars=["날짜", "line_id", "station_uid", "역번호", "역명", "구분", "source_ym"],
            value_vars=HOUR_BINS, var_name="hour_bin_raw", value_name="passenger_count")

        long["hour_bin"] = long["hour_bin_raw"].map(self._hour_bin_code)
        long["hour_start"] = long["hour_bin_raw"].map(self._hour_start)
        long["direction"] = long["구분"].map({"승차": "boarding", "하차": "alighting"})
        long["station_name"] = long["station_uid"].map(
            dict(zip(self.sm["station_uid"], self.sm["station_name"])))
        long["date"] = long["날짜"].dt.date
        long["dow"] = long["날짜"].dt.day_of_week
        long["day_type"] = np.select(
            [long["dow"] == 5, long["dow"] == 6], ["saturday", "sunday"], default="weekday")
        long["year"] = long["날짜"].dt.year
        long["month"] = long["날짜"].dt.month

        out = long[["date", "year", "month", "dow", "day_type", "line_id", "station_uid",
                    "station_name", "direction", "hour_bin", "hour_start",
                    "passenger_count", "source_ym"]]
        return out.reset_index(drop=True)

    @staticmethod
    def _hour_bin_code(label: str) -> str:
        if label == "06시 이전":
            return "before_06"
        if label == "24시 이후":
            return "after_24"
        return label.split(" ~ ")[0]

    @staticmethod
    def _hour_start(label: str) -> int:
        if label == "06시 이전":
            return 5
        if label == "24시 이후":
            return 24
        return int(label.split(" ~ ")[0])


# ---------------------------------------------------------------------
# 4. 품질 리포트
# ---------------------------------------------------------------------
def quality_report(raw: pd.DataFrame, mart: pd.DataFrame,
                   sm: pd.DataFrame, alias: pd.DataFrame,
                   loader: RidershipLoader, dropped_rows: int) -> str:
    L = []
    A = L.append
    A("# ridership_hourly_mart 품질 리포트\n")
    A(f"- 입력 파일 수: **{len(loader.file_log)}**")
    A(f"- 원본 행: **{len(raw):,}** → mart 행: **{len(mart):,}** (long format)")
    A(f"- 기간: **{raw['날짜'].min().date()} ~ {raw['날짜'].max().date()}**")

    months = raw["날짜"].dt.to_period("M")
    rng = pd.period_range(months.min(), months.max(), freq="M")
    missing_m = [str(p) for p in rng if p not in set(months.unique())]
    A(f"- 월 커버리지: {len(set(months.unique()))}/{len(rng)}개월, 누락 {missing_m or '없음'}")

    dc = raw.groupby(months)["날짜"].nunique()
    bad_days = [(str(p), int(dc.get(p, 0)), int(p.days_in_month))
                for p in rng if dc.get(p, 0) != p.days_in_month]
    A(f"- 일자 커버리지 불일치 월: {bad_days or '없음'}")

    total_sum = raw[HOUR_BINS].sum(axis=1)
    mismatch = int((total_sum - raw["합 계"]).abs().gt(0.5).sum())
    A(f"- 시간대 합 vs '합 계' 불일치 행: **{mismatch}**")
    A(f"- 결측 셀: {int(raw[HOUR_BINS].isna().sum().sum())} / 음수 셀: {int((raw[HOUR_BINS] < 0).sum().sum())}")
    A(f"- 키(날짜·호선·역번호·구분) 중복 행: {int(raw.duplicated(['날짜','호선','역번호','구분']).sum())}")
    A(f"- 범위 밖 제거 행: **{dropped_rows:,}**\n")

    A("## 호선별 역 수 (범위 내)\n")
    tbl = sm[sm.in_project_scope == 1].groupby("line_id").size()
    A("| 호선 | 역 수 |")
    A("|---|---|")
    for k, v in tbl.items():
        A(f"| {k}호선 | {v} |")

    flagged = sm[sm.attribution_flag != "ok"]
    A("\n## 승하차 호선 귀속 이상 (해석 주의)\n")
    if len(flagged):
        A("| station_uid | 역명 | 총합 | 역내 비중 | 플래그 | 비고 |")
        A("|---|---|---|---|---|---|")
        for _, r in flagged.iterrows():
            A(f"| {r.station_uid} | {r.station_name} | {int(r.total):,} | "
              f"{r.share_in_station:.4f} | {r.attribution_flag} | {r.attribution_note} |")
        A("\n> 위 역들의 승하차는 **다른 호선에 전량 귀속**된다. "
          "혼잡도 모델에서 '호선별 수요' 피처로 쓰면 결측·왜곡이 발생한다.")
    else:
        A("이상 없음")

    renamed = sm[sm.renamed_from != ""]
    A("\n## 기간 중 개명된 역\n")
    A("| station_uid | 이전 표기 | 현재 표기 |")
    A("|---|---|---|")
    for _, r in renamed.iterrows():
        A(f"| {r.station_uid} | {r.renamed_from} | {r.station_name_raw} |")

    A(f"\n## alias 매핑\n\n- 총 {len(alias)}건 생성 "
      f"(괄호 부기명 {int((alias.alias_source=='parenthetical').sum())}건, "
      f"개명 {int((alias.alias_source=='renamed').sum())}건)")
    return "\n".join(L)


# ---------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="여유로 서울 Phase 1 승하차 통합 파이프라인")
    ap.add_argument("--raw", default="data/raw/ridership_monthly", help="원본 xlsx 폴더")
    ap.add_argument("--root", default=".", help="프로젝트 루트")
    args = ap.parse_args(argv)

    root = Path(args.root)
    raw_dir = Path(args.raw)
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir

    print(f"[1/4] 원본 적재: {raw_dir}")
    loader = RidershipLoader(raw_dir)
    raw = loader.load()
    print(f"      총 {len(raw):,} 행\n")

    print("[2/4] 마스터 생성")
    mb = MasterBuilder(raw)
    sm = mb.build_station_master()
    alias = mb.build_alias_master(raw, sm)
    (root / "data/master").mkdir(parents=True, exist_ok=True)
    sm.to_csv(root / "data/master/station_master.csv", index=False, encoding="utf-8-sig")
    alias.to_csv(root / "data/master/station_alias_master.csv", index=False, encoding="utf-8-sig")
    print(f"      station_master {len(sm)}행 / alias {len(alias)}행\n")

    print("[3/4] mart 생성")
    mart_builder = MartBuilder(raw, sm)
    mart = mart_builder.build()
    (root / "data/marts").mkdir(parents=True, exist_ok=True)
    out_path = root / "data/marts/ridership_hourly_mart.parquet"
    try:
        mart.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_path.with_suffix(".csv.gz")
        mart.to_csv(out_path, index=False, encoding="utf-8-sig", compression="gzip")
        print("      (pyarrow 미설치 → csv.gz 로 저장. pip install pyarrow 권장)")
    print(f"      {len(mart):,} 행 → {out_path}\n")

    print("[4/4] 품질 리포트")
    rep = quality_report(raw, mart, sm, alias, loader, mart_builder.dropped_rows)
    rep_path = root / "reports/data_quality/ridership_quality_report.md"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(rep, encoding="utf-8")
    print(f"      → {rep_path}\n")
    print(rep[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
