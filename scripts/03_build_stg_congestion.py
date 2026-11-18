"""
03_build_stg_congestion.py
==========================
서울교통공사 지하철혼잡도정보 11개 파일을 단일 long-format staging 테이블로 통합한다.

이 스크립트가 처리하는 실제 스키마 드리프트 (원본 11개 파일 직접 확인 결과)
-------------------------------------------------------------------------
[schema A] 2022-12-31 ~ 2025-03-31  (csv 5개 + xlsx 1개, 45 columns)
    연번, 요일구분, 호선(int), 역번호, 출발역, 상하구분, 5시30분 ... 00시30분
    - 2022-12-31 파일만 요일구분이 '공휴일' (이후 파일은 '일요일')
    - 2024-12-31 파일은 요일구분 값에 좌우 공백(' 토요일 ')
    - 20250331.xlsx 는 빈 'Sheet1' 이 함께 들어있음 → 스킵 필요

[schema B] 2025-06-30 ~ 2025-11-30  (csv 3개, 44 columns)
    연번 컬럼 삭제, 호선이 '1호선' 문자열로 변경

[schema C] 2026-03-31 ~ 2026-06-30  (xlsx 2개, 44 columns, 시트 3개)
    시트가 평일/토요일/일요일로 분리, 요일구분 → '구분', 출발역 → '역명',
    시간 컬럼이 '05:30~06:00' ... '24:30~25:00' 구간 표기로 변경,
    값이 반올림되지 않은 원시 float

출력
----
    data/staging/stg_congestion.parquet   (pyarrow 없으면 .csv.gz 로 자동 대체)
    reports/data_quality/congestion_file_profile.csv
    reports/data_quality/congestion_station_drift.csv
    reports/data_quality/congestion_alias_candidates.csv
    reports/data_quality/congestion_scope_check.csv

사용법
------
    python 03_build_stg_congestion.py --raw-dir data/raw/congestion_quarterly --out-root .
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ENCODING_OUT = "utf-8-sig"
CSV_ENCODINGS = ["cp949", "utf-8-sig", "utf-8"]

# --------------------------------------------------------------------------
# 1. 정규화 사전
# --------------------------------------------------------------------------
DAY_TYPE_MAP = {
    "평일": "weekday",
    "토요일": "saturday",
    "일요일": "sunday",
    "공휴일": "sunday",   # 2022 파일은 '공휴일' 표기. 의미상 휴일 패턴으로 통합하되 별도 플래그 유지
}
DAY_TYPE_RAW_HOLIDAY = {"공휴일"}

DIRECTION_MAP = {
    "상선": "up",
    "하선": "down",
    "내선": "inner",
    "외선": "outer",
}

# 원본에서 실제로 관측된 개명/표기 변경 (파일 간 차집합으로 확인)
STATION_ALIAS = {
    ("2", 217): "잠실새내",          # 신천 → 잠실새내
    ("4", 409): "불암산",            # 당고개 → 불암산
    ("4", 416): "미아사거리",        # 미아삼거리 → 미아사거리
    ("7", 2730): "자양(뚝섬한강공원)",  # 뚝섬유원지 → 자양
    ("4", 432): "이수",              # 총신대입구(이수) → 이수 (7호선 표기와 통일)
    ("7", 2738): "이수",             # 총신대입구 → 이수
    ("8", 2810): "암사역사공원",
}

# 괄호 부기명 제거 시 canonical 로 삼을 이름 (그래프 노드 키 용도)
def canonical_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", "", s)
    # '올림픽공원(한국체대)' -> '올림픽공원'  /  '강동(마천)' -> '강동'
    s = re.sub(r"\(.*?\)$", "", s)
    # '신촌(지하)' 같이 괄호가 노선 구분인 경우도 위에서 제거됨
    if s.endswith("S") and len(s) > 1:      # '응암S'
        s = s[:-1]
    if s.endswith("E") and len(s) > 1:      # '성수E'
        s = s[:-1]
    return s


# 분기·순환 계통을 나타내는 특수 역번호 (원본에서 직접 확인)
BRANCH_CODE_MAP = {
    ("2", 9001): "seongsu_branch_east",   # 성수E : 성수지선 계통
    ("2", 9002): "seongsu_branch",        # 성수   : 본선/지선 분기
    ("2", 9003): "sinjeong_branch",       # 신도림 : 신정지선 분기
    ("5", 9005): "macheon_branch",        # 강동(마천) : 마천 방면
    ("6", 9006): "eungam_loop",           # 응암S : 응암순환 진입
}
# 본선 쪽 분기역 (동일 역명이 일반 코드로도 존재)
MAINLINE_BRANCH_STATIONS = {("2", 211), ("2", 234), ("5", 2549)}

# 프로젝트 범위: 원본 혼잡도 데이터는 서울교통공사 관할만 담고 있어
# 아래 역번호 구간 필터만으로 §4 범위와 일치한다.
SCOPE_CODE_RANGES = {
    "1": [(150, 159)],                    # 서울역~청량리
    "2": [(201, 260), (9001, 9003)],      # 본선+성수지선+신정지선
    "3": [(309, 342)],                    # 지축~오금
    "4": [(409, 434)],                    # 불암산~남태령 (진접선 405/406/408 제외)
    "5": [(2511, 2566), (9005, 9005)],    # 방화~하남검단산(2562~2566 하남구간 포함) + 마천지선
    "6": [(2611, 2653), (9006, 9006)],    # 응암~신내
    "7": [(2711, 2752)],                  # 장암~온수
    "8": [(2810, 2828)],                  # 암사역사공원~모란
}

# 39개 30분 bin (05:30 시작 ~ 25:00 종료)
def build_time_bins() -> pd.DataFrame:
    rows = []
    start = 5 * 60 + 30
    for i in range(39):
        s = start + i * 30
        e = s + 30
        rows.append({
            "time_bin_index": i,
            "time_bin": f"{s // 60:02d}:{s % 60:02d}~{e // 60:02d}:{e % 60:02d}",
            "bin_start_min": s,
            "is_am_peak": 1 if 7 * 60 <= s < 9 * 60 + 30 else 0,
            "is_pm_peak": 1 if 17 * 60 <= s < 19 * 60 + 30 else 0,
        })
    return pd.DataFrame(rows)


TIME_BINS = build_time_bins()


def parse_time_column(col: str) -> int | None:
    """'5시30분' 또는 '05:30~06:00' → bin_start_min"""
    c = str(col).strip()
    m = re.match(r"^(\d{1,2})시\s*(\d{1,2})분$", c)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if h == 0:                      # '00시00분' 은 익일 24:00 을 의미
            h = 24
        return h * 60 + mi
    m = re.match(r"^(\d{1,2}):(\d{2})\s*~", c)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


# --------------------------------------------------------------------------
# 2. 로더
# --------------------------------------------------------------------------
@dataclass
class FileProfile:
    file_name: str
    snapshot_date: str
    schema: str
    n_rows_raw: int
    n_rows_long: int
    day_types_raw: str
    n_station_keys: int
    n_time_cols: int
    pct_ge_100: float
    pct_ge_130: float
    max_value: float
    n_out_of_scope_rows: int


class CongestionStagingBuilder:

    def __init__(self, raw_dir: Path, out_root: Path):
        self.raw_dir = raw_dir
        self.out_root = out_root
        self.profiles: list[FileProfile] = []
        self.frames: list[pd.DataFrame] = []

    # ---------- 파일 탐색 ----------
    def find_files(self) -> list[Path]:
        pats = ["*지하철혼잡도정보*.csv", "*지하철혼잡도정보*.xlsx"]
        files: list[Path] = []
        for p in pats:
            files.extend(sorted(self.raw_dir.glob(p)))
        if not files:
            raise FileNotFoundError(
                f"혼잡도 파일을 찾지 못했습니다: {self.raw_dir}\n"
                f"--raw-dir 경로를 확인하세요."
            )
        return files

    @staticmethod
    def _snapshot_date(path: Path) -> str:
        m = re.search(r"(20\d{6})", path.name)
        if not m:
            return ""
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        last = None
        for enc in CSV_ENCODINGS:
            try:
                return pd.read_csv(path, encoding=enc)
            except (UnicodeDecodeError, LookupError) as e:
                last = e
        raise RuntimeError(f"인코딩 판별 실패: {path.name} ({last})")

    def _read_any(self, path: Path) -> tuple[pd.DataFrame, str]:
        """파일을 읽어 (raw df, schema tag) 반환. xlsx 다중시트/빈시트 처리 포함."""
        if path.suffix.lower() == ".csv":
            df = self._read_csv(path)
            schema = "A" if "연번" in df.columns else "B"
            return df, schema

        xl = pd.ExcelFile(path)
        sheets = [s for s in xl.sheet_names]
        parts = []
        day_sheets = [s for s in sheets if str(s).strip() in DAY_TYPE_MAP]
        if day_sheets:                                   # schema C
            for s in day_sheets:
                d = pd.read_excel(xl, s)
                if d.empty:
                    continue
                d["__sheet_day__"] = str(s).strip()
                parts.append(d)
            return pd.concat(parts, ignore_index=True), "C"

        for s in sheets:                                  # schema A(xlsx)
            d = pd.read_excel(xl, s)
            if d.empty or d.shape[1] < 5:
                continue                                  # 빈 'Sheet1' 스킵
            parts.append(d)
        return pd.concat(parts, ignore_index=True), "A"

    # ---------- 정규화 ----------
    def normalize(self, df: pd.DataFrame, schema: str, path: Path) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # 요일구분
        if "__sheet_day__" in df.columns:
            day_raw = df["__sheet_day__"]
        elif "요일구분" in df.columns:
            day_raw = df["요일구분"]
        elif "구분" in df.columns:
            day_raw = df["구분"]
        else:
            raise KeyError(f"요일 컬럼을 찾을 수 없습니다: {path.name}")
        day_raw = day_raw.astype(str).str.strip()

        # 역명 컬럼
        st_col = "역명" if "역명" in df.columns else "출발역"

        out = pd.DataFrame({
            "day_type_raw": day_raw,
            "day_type": day_raw.map(DAY_TYPE_MAP),
            "is_holiday_labeled": day_raw.isin(DAY_TYPE_RAW_HOLIDAY).astype(int),
            "line_id": (df["호선"].astype(str).str.strip()
                        .str.replace("호선", "", regex=False)),
            "station_code": pd.to_numeric(df["역번호"], errors="coerce").astype("Int64"),
            "station_name_raw": df[st_col].astype(str).str.strip(),
            "direction_raw": df["상하구분"].astype(str).str.strip(),
        })
        out["direction"] = out["direction_raw"].map(DIRECTION_MAP)

        # 호선 오기입 교정: 역번호 체계는 노선별로 배타적이므로 code 기준으로 검증한다.
        # (2024-06-30 파일에서 5호선 둔촌동/올림픽공원이 '2호선'으로 기재된 오류 확인)
        out["line_id_raw"] = out["line_id"]
        out["line_id"] = [
            self._line_from_code(c, l) for c, l in zip(out["station_code"], out["line_id"])
        ]
        out["is_line_corrected"] = (out["line_id"] != out["line_id_raw"]).astype(int)

        # 시간 컬럼 → long
        time_cols = {c: parse_time_column(c) for c in df.columns}
        time_cols = {c: v for c, v in time_cols.items() if v is not None}
        if len(time_cols) != 39:
            raise ValueError(f"{path.name}: 시간 컬럼 {len(time_cols)}개 (기대 39개)")

        vals = df[list(time_cols)].apply(
            lambda s: pd.to_numeric(
                s.astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"))
        vals.columns = [time_cols[c] for c in time_cols]

        long = pd.concat([out.reset_index(drop=True), vals.reset_index(drop=True)], axis=1)
        long = long.melt(
            id_vars=list(out.columns),
            var_name="bin_start_min", value_name="congestion_rate")
        long = long.merge(TIME_BINS, on="bin_start_min", how="left")

        # 역명/분기 정규화
        keys = list(zip(long["line_id"], long["station_code"].astype("float").astype("Int64")))
        long["station_name"] = [
            STATION_ALIAS.get((l, int(c)) if pd.notna(c) else (l, -1), None)
            for l, c in keys
        ]
        long["station_name"] = long["station_name"].fillna(
            long["station_name_raw"].map(canonical_name))
        long["station_name"] = long["station_name"].map(canonical_name)

        long["branch_code"] = [
            BRANCH_CODE_MAP.get((l, int(c)) if pd.notna(c) else (l, -1), "main")
            for l, c in keys
        ]
        long["is_branch_row"] = (long["branch_code"] != "main").astype(int)

        # 메타
        long["snapshot_date"] = self._snapshot_date(path)
        long["snapshot_id"] = "CG_" + long["snapshot_date"].str.replace("-", "", regex=False)
        long["source_file"] = path.name
        long["schema_version"] = schema
        long["station_uid"] = (long["line_id"] + "_" + long["station_name"]
                               + np.where(long["is_branch_row"] == 1,
                                          "@" + long["branch_code"], ""))
        long["in_project_scope"] = self._scope_flag(long)
        return long

    @staticmethod
    def _line_from_code(code, line_raw: str) -> str:
        """역번호가 어떤 노선 구간에 속하는지로 호선을 재판정한다."""
        if pd.isna(code):
            return line_raw
        c = int(code)
        for line, ranges in SCOPE_CODE_RANGES.items():
            for lo, hi in ranges:
                if lo <= c <= hi:
                    return line
        return line_raw

    @staticmethod
    def _scope_flag(df: pd.DataFrame) -> pd.Series:
        flag = pd.Series(0, index=df.index, dtype=int)
        for line, ranges in SCOPE_CODE_RANGES.items():
            m_line = df["line_id"] == line
            for lo, hi in ranges:
                flag |= (m_line & df["station_code"].between(lo, hi)).astype(int)
        return flag

    # ---------- 실행 ----------
    def run(self) -> pd.DataFrame:
        for path in self.find_files():
            raw, schema = self._read_any(path)
            long = self.normalize(raw, schema, path)

            v = long["congestion_rate"]
            keys = long[["line_id", "station_code", "station_name"]].drop_duplicates()
            self.profiles.append(FileProfile(
                file_name=path.name,
                snapshot_date=self._snapshot_date(path),
                schema=schema,
                n_rows_raw=len(raw),
                n_rows_long=len(long),
                day_types_raw="|".join(sorted(long["day_type_raw"].unique())),
                n_station_keys=len(keys),
                n_time_cols=39,
                pct_ge_100=round(float((v >= 100).mean() * 100), 3),
                pct_ge_130=round(float((v >= 130).mean() * 100), 3),
                max_value=round(float(v.max()), 1),
                n_out_of_scope_rows=int((long["in_project_scope"] == 0).sum()),
            ))
            self.frames.append(long)

        stg = pd.concat(self.frames, ignore_index=True)
        cols = ["snapshot_id", "snapshot_date", "day_type", "day_type_raw", "is_holiday_labeled",
                "line_id", "station_code", "station_uid", "station_name", "station_name_raw",
                "branch_code", "is_branch_row", "direction", "direction_raw", "line_id_raw", "is_line_corrected",
                "time_bin_index", "time_bin", "bin_start_min", "is_am_peak", "is_pm_peak",
                "congestion_rate", "in_project_scope", "schema_version", "source_file"]
        return stg[cols]

    # ---------- 리포트 ----------
    def write_outputs(self, stg: pd.DataFrame) -> dict:
        stg_dir = self.out_root / "data" / "staging"
        rep_dir = self.out_root / "reports" / "data_quality"
        stg_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)
        written = {}

        try:
            path = stg_dir / "stg_congestion.parquet"
            stg.to_parquet(path, index=False)
        except Exception:
            path = stg_dir / "stg_congestion.csv.gz"
            stg.to_csv(path, index=False, encoding=ENCODING_OUT, compression="gzip")
            print("  [알림] pyarrow 미설치 → parquet 대신 csv.gz 로 저장했습니다. "
                  "(pip install pyarrow 권장)")
        written["staging"] = path

        prof = pd.DataFrame([p.__dict__ for p in self.profiles])
        p1 = rep_dir / "congestion_file_profile.csv"
        prof.to_csv(p1, index=False, encoding=ENCODING_OUT)
        written["profile"] = p1

        # 역 집합 드리프트
        latest = stg["snapshot_date"].max()
        base = set(map(tuple, stg.loc[stg["snapshot_date"] == latest,
                                      ["line_id", "station_code"]].drop_duplicates().values))
        drift = []
        for snap, g in stg.groupby("snapshot_date"):
            cur = set(map(tuple, g[["line_id", "station_code"]].drop_duplicates().values))
            for l, c in sorted(base - cur):
                drift.append({"snapshot_date": snap, "line_id": l, "station_code": c,
                              "status": "missing_vs_latest"})
            for l, c in sorted(cur - base):
                drift.append({"snapshot_date": snap, "line_id": l, "station_code": c,
                              "status": "extra_vs_latest"})
        p2 = rep_dir / "congestion_station_drift.csv"
        pd.DataFrame(drift).to_csv(p2, index=False, encoding=ENCODING_OUT)
        written["drift"] = p2

        # alias 후보: 같은 (호선,역번호)에 서로 다른 원본 역명
        al = (stg.groupby(["line_id", "station_code"])["station_name_raw"]
              .nunique().reset_index(name="n_names"))
        al = al[al["n_names"] > 1]
        cand = (stg.merge(al[["line_id", "station_code"]], on=["line_id", "station_code"])
                .groupby(["line_id", "station_code", "station_name_raw"])["snapshot_date"]
                .agg(["min", "max", "count"]).reset_index()
                .rename(columns={"min": "first_seen", "max": "last_seen", "count": "n_rows"}))
        p3 = rep_dir / "congestion_alias_candidates.csv"
        cand.to_csv(p3, index=False, encoding=ENCODING_OUT)
        written["alias"] = p3

        # 노선별 범위 점검
        ins = stg[stg["in_project_scope"] == 1]
        scope = (ins.groupby("line_id")
                 .agg(n_stations=("station_code", "nunique"),
                      min_code=("station_code", "min"),
                      max_code=("station_code", "max"),
                      directions=("direction", lambda s: "|".join(sorted(set(s)))))
                 .reset_index())
        p4 = rep_dir / "congestion_scope_check.csv"
        scope.to_csv(p4, index=False, encoding=ENCODING_OUT)
        written["scope"] = p4
        return written


# --------------------------------------------------------------------------
# 3. 콘솔 리포트
# --------------------------------------------------------------------------
def print_report(stg: pd.DataFrame, builder: CongestionStagingBuilder, written: dict) -> None:
    line = "=" * 78
    print(line)
    print(" 03_build_stg_congestion — 통합 결과")
    print(line)
    prof = pd.DataFrame([p.__dict__ for p in builder.profiles])
    print(prof[["snapshot_date", "schema", "n_rows_raw", "n_station_keys",
                "pct_ge_100", "pct_ge_130", "max_value",
                "n_out_of_scope_rows"]].to_string(index=False))

    print("\n[통합 결과]")
    print(f"  파일 수            : {len(builder.profiles)} (기대 11)")
    print(f"  long rows          : {len(stg):,}")
    print(f"  범위 내 rows       : {int((stg['in_project_scope'] == 1).sum()):,}"
          f"  ({(stg['in_project_scope'] == 1).mean() * 100:.2f}%)")
    print(f"  요일유형           : {sorted(stg['day_type'].dropna().unique())}")
    print(f"  방향               : {sorted(stg['direction'].dropna().unique())}")
    print(f"  결측 congestion    : {int(stg['congestion_rate'].isna().sum()):,}")
    print(f"  음수 congestion    : {int((stg['congestion_rate'] < 0).sum()):,}")
    print(f"  0 값               : {int((stg['congestion_rate'] == 0).sum()):,}")

    ins = stg[stg["in_project_scope"] == 1]
    print("\n[노선별 범위 점검]  (기대: 1호선 10역 / 2호선 54 / 3호선 34 / 4호선 26 /"
          " 5호선 57 / 6호선 40 / 7호선 42 / 8호선 19)")
    chk = (ins.groupby("line_id")
           .agg(n_stations=("station_code", "nunique"),
                directions=("direction", lambda s: "|".join(sorted(set(s))))))
    print(chk.to_string())

    print(f"\n[호선 오기입 교정] {int(stg[chr(39)+chr(39)] if False else stg['is_line_corrected'].sum()):,} rows")
    if stg["is_line_corrected"].sum():
        print(stg[stg["is_line_corrected"] == 1]
              .groupby(["source_file", "line_id_raw", "line_id", "station_code", "station_name_raw"])
              .size().reset_index(name="rows").to_string(index=False))

    print("\n[분기·순환 계통 row]")
    br = (ins[ins["is_branch_row"] == 1]
          .groupby(["line_id", "station_code", "station_name_raw", "branch_code"])
          .size().reset_index(name="rows"))
    print(br.to_string(index=False))

    print("\n[단방향 역 = 그래프에서 비대칭 edge 로 만들어야 하는 역]")
    d = (ins[ins["snapshot_date"] == ins["snapshot_date"].max()]
         .groupby(["line_id", "station_code", "station_name_raw"])["direction"].nunique())
    uni = d[d == 1].reset_index()
    print(uni.to_string(index=False) if len(uni) else "  없음")

    print("\n[고혼잡 임계 재검토]")
    v = ins["congestion_rate"]
    for th in (80, 100, 130, 150):
        print(f"  >= {th:3d}% : {(v >= th).mean() * 100:6.3f}%  ({int((v >= th).sum()):,} cells)")
    print(f"  상위 1% 분위수 = {v.quantile(0.99):.1f}%,  상위 0.1% = {v.quantile(0.999):.1f}%")

    print("\n[생성 파일]")
    for k, p in written.items():
        print(f"  {k:9s} : {p}")
    print(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/congestion_quarterly")
    ap.add_argument("--out-root", default=".")
    args = ap.parse_args(argv)

    builder = CongestionStagingBuilder(Path(args.raw_dir), Path(args.out_root))
    stg = builder.run()
    written = builder.write_outputs(stg)
    print_report(stg, builder, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
