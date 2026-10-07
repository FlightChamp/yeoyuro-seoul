"""
scripts/v2/23a_build_demand_proxy.py
====================================
v2.3 수요 proxy 용 역 × 요일유형 × 시간 평균 승하차 (일평균).

실제 OD 행렬이 없으므로 OD 가중치는 이 표로 만든 proxy 다 (preregistration_v23.md §5).
v1 의 01_build_ridership_mart.py 는 48개월 전체를 long format 으로 펼쳐 메모리를 많이 쓴다.
여기서는 파일을 하나씩 읽어 바로 집계하므로 가볍다. 역명 정규화 규칙은 v1 과 같다.

입력  : 서울교통공사 '역별 시간대별 이용인원' 월별 xlsx (기본: 최근 12개월만 사용)
출력  : data/marts/v2/demand_station_hour.parquet  (commit 대상, 수 KB)
        columns = station_key, day_type, hour, boardings, alightings, n_days, months

실행
----
    python scripts/v2/23a_build_demand_proxy.py --raw data/raw/ridership_monthly
    (출력 파일은 저장소에 포함되어 있으므로, 원본이 없으면 실행하지 않아도 된다)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

FILENAME_RE = re.compile(r"(\d{4})년[_ ]?(\d{1,2})월")
HOUR_COLS = {"06시 이전": 5, "6시 이전": 5, **{f"{h:02d} ~ {h + 1:02d}": h for h in range(6, 24)},
             "24시 이후": 24}
FORCED = {"총신대입구(이수)": "이수", "청량리(서울시립대입구)": "청량리"}
RENAMES = {"당고개": "불암산", "뚝섬유원지": "자양"}


def canon(name: str) -> str:
    name = str(name).strip()
    if name in FORCED:
        return FORCED[name]
    base = re.sub(r"\(.*\)$", "", name).strip() or name
    return RENAMES.get(base, base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "ridership_monthly"))
    ap.add_argument("--months", type=int, default=12, help="최근 N개월만 사용")
    args = ap.parse_args(argv)

    files = []
    for p in Path(args.raw).glob("*.xlsx"):
        m = FILENAME_RE.search(p.name)
        if m:
            files.append((f"{m.group(1)}-{int(m.group(2)):02d}", p))
    if not files:
        print(f"[중단] 원본 xlsx 가 없습니다: {args.raw}")
        return 1
    files = sorted(files)[-args.months:]
    print("사용 월:", ", ".join(ym for ym, _ in files))

    disp = pd.read_csv(ROOT / "data" / "master" / "station_display_master.csv")
    keys = set(disp.station_key)
    parts, unmatched = [], set()
    for ym, p in files:
        df = pd.read_excel(p, sheet_name=0, engine="openpyxl")
        df = df.drop(columns=[c for c in ("00 ~ 01", "01 ~ 02", "02 ~ 03", "03 ~ 04") if c in df.columns])
        hc = [c for c in df.columns if c in HOUR_COLS]
        df["station_key"] = df["역명"].map(canon)
        unmatched |= set(df.loc[~df.station_key.isin(keys), "역명"].astype(str))
        df = df[df.station_key.isin(keys)]
        df["날짜"] = pd.to_datetime(df["날짜"])
        dow = df["날짜"].dt.dayofweek
        df["day_type"] = dow.map(lambda x: "saturday" if x == 5 else ("sunday" if x == 6 else "weekday"))
        long = df.melt(id_vars=["날짜", "day_type", "station_key", "구분"], value_vars=hc,
                       var_name="hc", value_name="n")
        long["hour"] = long.hc.map(HOUR_COLS)
        # 환승역은 여러 호선 행에 나뉘어 있으므로 역 단위로 합산
        g = long.groupby(["날짜", "day_type", "station_key", "hour", "구분"], as_index=False)["n"].sum()
        parts.append(g)
        print(f"  {ym}: {len(df):,}행")
    allg = pd.concat(parts, ignore_index=True)
    ndays = allg.groupby("day_type")["날짜"].nunique()
    piv = (allg.pivot_table(index=["day_type", "station_key", "hour"], columns="구분",
                            values="n", aggfunc="sum", fill_value=0).reset_index())
    piv.columns.name = None
    piv = piv.rename(columns={"승차": "boardings", "하차": "alightings"})
    piv["n_days"] = piv.day_type.map(ndays)
    piv["boardings"] = (piv.boardings / piv.n_days).round(1)
    piv["alightings"] = (piv.alightings / piv.n_days).round(1)
    piv["months"] = f"{files[0][0]}~{files[-1][0]}"

    out = ROOT / "data" / "marts" / "v2" / "demand_station_hour.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    piv[["station_key", "day_type", "hour", "boardings", "alightings", "n_days", "months"]].to_parquet(out, index=False)
    missing = sorted(keys - set(piv.station_key))
    print(f"\n저장: {out}  ({len(piv):,}행, 역 {piv.station_key.nunique()}개)")
    print(f"원본 역명 중 display master 와 매칭 안 된 이름 {len(unmatched)}개: {sorted(unmatched)[:15]}")
    print(f"display master 역 중 승하차 없음 {len(missing)}개: {missing}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
