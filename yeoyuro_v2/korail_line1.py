"""
yeoyuro_v2/korail_line1.py
==========================
Phase 3A-1 — 1호선 코레일 서울 시내 구간(경부·경인·경원) 시각표 정제와 '실제 다음 열차' 계산.

**별도 검증 환경.** 이 모듈은 앱·경로 엔진(time_dependent, app_bridge, station_routing)에서 import 하지 않는다.
기존 v2.3·v2.6 결과에는 영향이 없다(tests/test_korail_line1.py 가 import 여부를 검사). 엔진 연결은 3A-2.

출처 (3A-0 감사, reports/v2/line1k_data_audit.md)
-------------------------------------------------
- 시각표 : 서울교통공사_서울 도시철도 열차운행시각표_20260901.csv (cp949, 한글 정상 원본)
           공공데이터포털 15098251. 주중주말 DAY/SAT/END, 방향 UP/DOWN, 급행여부 1/0
- 역코드 : 전국도시철도역사정보표준데이터(20260630) 역번호 = 시각표 역사코드 (코레일 역 일치)

규칙 (decision_log D-085)
-------------------------
- 범위: 서울 시내 코레일 26역 + 1호선 서울교통공사 구간 10역(서울역~청량리·동묘앞, 직통 확인용)
- 일반열차(급행여부 0)만. 급행은 3A-3 에서 공식 정차역 대조 후 결정
- 시각: '자정부터 분'. 24:xx·25:xx 는 1,440 분 이상 그대로 (운행일 기준). 실제 00~02시 표기는 +1,440
- 급행 행의 열차도착시간 '00:00:00' 은 통과 자리표시 → 정차로 쓰지 않음. 일반열차에는 없음(있으면 오류)
- 정차시간(dwell) = 출발 − 도착. 음수가 나오면 오류로 멈춘다(계산에 들어가지 않게)
- 다음 열차: **목적지 정차 필터** — 출발역에서 떠난 뒤 같은 열차가 목적지에 도착하는 열차만 후보.
  중간 종착(용산·구로·광운대 종착 등)과 다른 분기(구로에서 경인/경부) 열차는 자동으로 빠진다
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TT_FILE = "서울교통공사_서울 도시철도 열차운행시각표_20260901.csv"
STD_FILE_PATTERN = "전국도시철도역사정보표준데이터"
TT_COLUMNS = ["id", "line", "code", "name", "day", "dir", "exp", "train", "arr", "dep", "origin", "dest"]
DAY_MAP = {"DAY": "weekday", "SAT": "saturday", "END": "sunday"}
PASS_MARK = "00:00:00"                       # 급행 통과역 자리표시 (3A-0 §6.1, 3A-1 §2)

# 서울 시내 코레일 26역 (표준데이터 주소 '서울', 3A-0 §4) — 코드: (역명, 계통)
SEOUL_KORAIL = {
    1002: ("남영", "경부"), 1003: ("용산", "경부"), 1004: ("노량진", "경부"), 1005: ("대방", "경부"),
    1032: ("신길", "경부"), 1006: ("영등포", "경부"), 1007: ("신도림", "경부"), 1701: ("구로", "경부"),
    1702: ("가산디지털단지", "경부"), 1714: ("독산", "경부"), 1703: ("금천구청", "경부"),
    1813: ("구일", "경인"), 1801: ("개봉", "경인"), 1802: ("오류동", "경인"), 1821: ("온수", "경인"),
    1015: ("회기", "경원"), 1016: ("외대앞", "경원"), 1017: ("신이문", "경원"), 1018: ("석계", "경원"),
    1019: ("광운대", "경원"), 1020: ("월계", "경원"), 1021: ("녹천", "경원"), 1022: ("창동", "경원"),
    1901: ("방학", "경원"), 1902: ("도봉", "경원"), 1903: ("도봉산", "경원"),
}
# 1호선 서울교통공사 구간 (이미 v2 그래프에 있음 — 직통 운행 확인용으로만 같이 정제)
METRO_L1 = {150: "서울역", 151: "시청", 152: "종각", 153: "종로3가", 154: "종로5가", 155: "동대문",
            156: "신설동", 157: "제기동", 158: "청량리", 159: "동묘앞"}
SCOPE = {**{c: n for c, (n, _) in SEOUL_KORAIL.items()}, **METRO_L1}

# 계통별 역 순서 (지도 배치 초안·검사용). 실제 운행 순서는 시각표에서 다시 확인한다
CORRIDORS = {
    "경부": [150, 1002, 1003, 1004, 1005, 1032, 1006, 1007, 1701, 1702, 1714, 1703],
    "경인": [1701, 1813, 1801, 1802, 1821],
    "경원": [158, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1901, 1902, 1903],
}
# 3A-0 §5.2: 지하 서울역 코드(150)로 기록됐지만 남영을 거치지 않는 급행 3편성 (지상 서울역 추정, 미확인)
ANOMALOUS_EXPRESS = {"K1902", "K1904", "K1945"}


class SourceError(ValueError):
    """원천 파일이 감사 기준(인코딩·형식·값)을 만족하지 않음."""


# ---------------------------------------------------------------------- 1. 원천 읽기
def locate_source(src, pattern: str) -> Path:
    """src 폴더(여러 개 가능, 하위 폴더 포함)에서 이름에 pattern 을 포함한 파일. 공백·밑줄 차이는 무시."""
    norm = lambda s: re.sub(r"[\s_()\-]", "", str(s)).lower()       # noqa: E731
    target = norm(Path(pattern).stem)
    for base in [Path(x) for x in (src if isinstance(src, (list, tuple)) else [src])]:
        if (base / pattern).is_file():
            return base / pattern
        hits = sorted(p for p in base.rglob("*") if p.is_file() and target in norm(p.stem))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"원본 파일을 찾지 못했습니다: {pattern}")


def read_timetable(path) -> pd.DataFrame:
    """시각표 원본을 문자열 그대로 읽는다 (앞자리 0·시각 형식 보존). 한글 손실 사본이면 멈춘다 (T12)."""
    d = pd.read_csv(path, encoding="cp949", dtype=str, keep_default_na=False, na_values=[""])
    if len(d.columns) != len(TT_COLUMNS):
        raise SourceError(f"열 개수 {len(d.columns)} ≠ {len(TT_COLUMNS)}")
    d.columns = TT_COLUMNS
    for col in ("name", "origin", "dest"):
        bad = d[col].fillna("").str.contains(r"\?{2,}")
        if bad.any():
            raise SourceError(f"한글 손실 사본입니다 ({col} 열 {int(bad.sum())}행에 '??'). "
                              "공공데이터포털 원본을 다시 받으세요 (3A-0 §1)")
    return d


def to_minutes(s) -> float:
    """'HH:MM:SS' → 자정부터 분. 24:xx 이상은 그대로, 실제 00~02시는 운행일 기준 +1,440."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or pd.isna(s):
        return np.nan
    h, m, x = (int(v) for v in str(s).split(":"))
    v = h * 60 + m + x / 60
    return v + 1440 if h < 3 else v


def to_service_minutes(hhmm: str) -> float:
    """사용자 입력 시각 → 운행일 기준 분. 00:00~02:59 는 전날 운행일의 24:00~26:59 로 본다."""
    h, m = (int(v) for v in hhmm.split(":")[:2])
    return h * 60 + m + (1440 if h < 3 else 0)


# ---------------------------------------------------------------------- 2. 정제
def line1_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """1호선 행 + 분 단위 시각 + 통과 표시. 급행 통과 행은 도착을 비우고 pass=True."""
    d = raw[raw.line == "1"].copy()
    d["code"] = d.code.astype(int)
    d["day_type"] = d.day.map(DAY_MAP)
    if d.day_type.isna().any():
        raise SourceError(f"알 수 없는 주중주말 값: {sorted(set(d.day[d.day_type.isna()]))}")
    d["express"] = d.exp.astype(int).astype(bool)
    d["is_pass"] = d.arr.eq(PASS_MARK)
    if (d.is_pass & ~d.express).any():
        raise SourceError("일반열차에 통과 표시(도착 00:00:00)가 있음 — 정의 재확인 필요")
    d["arr_min"] = d.arr.where(~d.is_pass).map(to_minutes)
    d["dep_min"] = d.dep.map(to_minutes)
    d["t_min"] = d.dep_min.fillna(d.arr_min)                 # 정렬용 (통과 행은 통과 시각 = 출발 칸)
    return d


def stop_times(rows: pd.DataFrame, scope=SCOPE, include_express: bool = False) -> pd.DataFrame:
    """열차별 정차 순서표 (범위 안 역만). 일반열차 기본. 통과 행은 항상 제외한다.

    반환 열: day_type, train, direction, express, stop_seq, code, station, arr_min, dep_min, dwell_min,
             origin, dest, first_in_scope, last_in_scope
    """
    d = rows if include_express else rows[~rows.express]
    d = d[~d.is_pass & d.code.isin(list(scope))].copy()
    d = d.sort_values(["day_type", "train", "dir", "t_min"], kind="mergesort")
    d["stop_seq"] = d.groupby(["day_type", "train", "dir"]).cumcount()
    d["dwell_min"] = d.dep_min - d.arr_min
    neg = d.dwell_min < -1e-9
    if neg.any():
        ex = d.loc[neg, ["day_type", "train", "code", "arr", "dep"]].head(5).to_dict("records")
        raise SourceError(f"음수 정차시간 {int(neg.sum())}행 — 계산에 넣지 않고 멈춤: {ex}")
    g = d.groupby(["day_type", "train", "dir"]).stop_seq
    d["first_in_scope"] = d.stop_seq.eq(0)
    d["last_in_scope"] = d.stop_seq.eq(g.transform("max"))
    d["station"] = d.code.map(scope)
    out = d.rename(columns={"dir": "direction"})[
        ["day_type", "train", "direction", "express", "stop_seq", "code", "station",
         "arr_min", "dep_min", "dwell_min", "origin", "dest", "first_in_scope", "last_in_scope"]]
    return out.reset_index(drop=True)


def segments(st: pd.DataFrame) -> pd.DataFrame:
    """같은 열차의 연속 정차역 쌍 (주행 구간). run_min = 다음 역 도착 − 이번 역 출발."""
    d = st.sort_values(["day_type", "train", "direction", "stop_seq"])
    nxt = d.groupby(["day_type", "train", "direction"]).shift(-1)
    s = pd.DataFrame({"day_type": d.day_type, "train": d.train, "direction": d.direction,
                      "from_code": d.code, "to_code": nxt.code, "dep_min": d.dep_min,
                      "arr_min": nxt.arr_min})
    s = s.dropna(subset=["to_code"]).astype({"to_code": int})
    s["run_min"] = s.arr_min - s.dep_min
    return s.reset_index(drop=True)


# ---------------------------------------------------------------------- 3. 실제 다음 열차
@dataclass(frozen=True)
class Departure:
    train: str
    direction: str
    dep_min: float          # 출발역 출발
    arr_min: float          # 목적지 도착
    wait_min: float         # 질의 시각부터 출발까지
    ride_min: float         # 출발 → 도착


class NextTrainIndex:
    """(요일, 출발역, 목적지) → 목적지에 정차하는 열차의 출발·도착 시각 (출발 시각 순).

    목적지 정차 필터: 같은 열차 안에서 출발역 다음 순서에 목적지가 있어야 후보가 된다.
    → 중간 종착 열차, 다른 분기로 가는 열차, 반대 방향 열차는 자동으로 제외된다.
    """

    def __init__(self, st: pd.DataFrame):
        self._pairs: dict[tuple, tuple[list, list, list, list]] = {}
        cols = ["day_type", "train", "direction", "stop_seq", "code", "arr_min", "dep_min"]
        rows = {}
        for k, g in st[cols].sort_values(cols[:4]).groupby(["day_type", "train", "direction"], sort=False):
            day, train, direction = k
            seq = list(g.itertuples(index=False))
            for i, a in enumerate(seq):
                if np.isnan(a.dep_min):
                    continue
                for b in seq[i + 1:]:
                    if np.isnan(b.arr_min):
                        continue
                    rows.setdefault((day, a.code, b.code), []).append((a.dep_min, b.arr_min, train, direction))
        for key, lst in rows.items():
            lst.sort()
            self._pairs[key] = tuple(list(x) for x in zip(*lst))

    def departures(self, day_type: str, origin: int, dest: int) -> list[float]:
        return list(self._pairs.get((day_type, origin, dest), ([],))[0])

    def next_train(self, day_type: str, origin: int, dest: int, t_min: float) -> Departure | None:
        """질의 시각 t_min(운행일 기준 분) 이후 처음 출발하는, 목적지에 정차하는 열차. 없으면 None(막차 이후)."""
        v = self._pairs.get((day_type, origin, dest))
        if not v:
            return None
        deps, arrs, trains, dirs = v
        i = bisect_left(deps, t_min - 1e-9)
        if i >= len(deps):
            return None
        return Departure(trains[i], dirs[i], deps[i], arrs[i], deps[i] - t_min, arrs[i] - deps[i])


def expected_wait(deps, start: float, end: float, max_wait: float = 60.0) -> tuple[float, float]:
    """[start, end) 에 균등하게 도착한 승객의 평균 대기(분)와 '다음 열차까지 max_wait 초과' 비율.

    배차가 고르지 않으면 평균 대기는 (평균 배차)/2 보다 크다: Σh² / (2Σh).
    max_wait 를 넘는 구간은 평균에서 빼고 비율로 따로 돌려준다 (운행 없음으로 처리, 3A-1 §7 R3).
    """
    d = sorted(x for x in deps if x >= start)
    if end <= start:
        raise ValueError("end > start 여야 함")
    total, covered, prev = 0.0, 0.0, start
    for x in d:
        if prev >= end:
            break
        a, b = prev, min(x, end)                 # 이 구간 도착 승객은 열차 x 를 탄다
        if b > a:
            # 대기 = x − τ. max_wait 를 넘는 τ(τ < x − max_wait) 는 제외
            lo = max(a, x - max_wait)
            if b > lo:
                total += ((x - lo) ** 2 - (x - b) ** 2) / 2
                covered += b - lo
        prev = max(prev, x)
    span = end - start
    unserved = (span - covered) / span
    return (total / covered if covered > 0 else np.nan), unserved


def headway_half(deps_all, start: float, end: float) -> float:
    """기존 1~8호선 방식 비교값: 그 30분에 그 역·방향을 떠나는 모든 열차 수 n → (30/n)/2."""
    n = sum(1 for x in deps_all if start <= x < end)
    return (end - start) / n / 2 if n else np.nan


# ---------------------------------------------------------------------- 3b. D-093 대기 분해·구로 갈아타기
# 운행 없음 사유 (D-093 R3): 화면·보고서에서 서로 다르게 표시한다
NO_SERVICE_BEFORE_FIRST = "운행 시작 전"      # 그날 첫 목적지행 열차까지 max_wait 초과
NO_SERVICE_AFTER_LAST = "운행 종료"           # 그날 더 이상 목적지행 열차가 없음 (막차 뒤)
NO_SERVICE_LONG_GAP = "장시간 대기 제외"      # 다음 열차는 있지만 max_wait 초과


@dataclass(frozen=True)
class WaitProfile:
    """[start, end) 에 균등하게 도착한 승객 기준 대기 분해 (D-093 R1·R3·검증 3).

    E: 다음 목적지행 열차까지 max_wait 이하로 탈 수 있는 승객만의 평균 대기 (없으면 NaN)
    frac_served + frac_before_first + frac_after_last + frac_long_gap = 1
    n_in_bin: 구간 안에서 출발하는 목적지행 열차 수 (0·1대인 구간 검증용)
    """
    E: float
    frac_served: float
    frac_before_first: float
    frac_after_last: float
    frac_long_gap: float
    n_in_bin: int

    @property
    def reason(self) -> str | None:
        """대기 사유 하나로 요약: 모두 탈 수 있으면 None, 아니면 가장 큰 비중의 '운행 없음' 사유."""
        parts = {NO_SERVICE_BEFORE_FIRST: self.frac_before_first, NO_SERVICE_AFTER_LAST: self.frac_after_last,
                 NO_SERVICE_LONG_GAP: self.frac_long_gap}
        k, v = max(parts.items(), key=lambda kv: kv[1])
        return k if v > 1e-12 else None


def wait_profile(deps, start: float, end: float, max_wait: float = 60.0) -> WaitProfile:
    """그날 목적지행 전체 출발 시각 deps(운행일 기준 분)로 [start, end) 대기를 정확히 적분한다.

    - 구간 끝 뒤 첫 열차까지 포함: 구간 끝 직전 도착 승객은 구간 밖(다음 30분) 열차를 탄다 (검증 3)
    - 구간 앞 열차는 쓰지 않음: 도착 시점 이전에 떠난 열차는 탈 수 없다
    - 0·1대 구간, 막차 뒤, 첫차 전을 각각 구분해 E 가 잘못 계산되지 않게 한다
    expected_wait() 와 E·운행 없음 합계가 같다 (테스트로 고정).
    """
    if end <= start:
        raise ValueError("end > start 여야 함")
    d = sorted(float(x) for x in deps)
    first = d[0] if d else None
    span = end - start
    served = before = after = gap = total = 0.0
    prev = start
    for x in d:
        if x < start:
            continue
        if prev >= end:
            break
        a, b = prev, min(x, end)
        if b > a:
            lo = max(a, x - max_wait)               # τ ≥ x − max_wait 이면 탈 수 있음
            if b > lo:
                total += ((x - lo) ** 2 - (x - b) ** 2) / 2
                served += b - lo
            miss = max(0.0, min(b, lo) - a)          # 대기 max_wait 초과 승객
            if x == first:
                before += miss
            else:
                gap += miss
        prev = max(prev, x)
    if prev < end:                                    # 그 뒤로 목적지행 열차가 없음 = 막차 뒤 (그날 운행이 아예 없어도 여기)
        after += end - prev
    n_in = sum(1 for x in d if start <= x < end)
    E = total / served if served > 0 else float("nan")
    return WaitProfile(E, served / span, before / span, after / span, gap / span, n_in)


# 같은 1호선 안 갈아타기 (D-093 1): 구로에서만, 열차번호가 바뀔 때만
GURO = 1701
SAME_LINE_TRANSFER_WALK = {GURO: 1.0}             # 분. 사용자 교차확인(네이버지도·카카오지하철·공식 안내) 2026-11-22
GURO_WALK_SENSITIVITY = (0.0, 1.0, 3.0, 5.0)      # 민감도 (D-093)
GYEONGIN_SOUTH = (1813, 1801, 1802, 1821)        # 구일·개봉·오류동·온수
GYEONGBU_SOUTH = (1702, 1714, 1703)               # 가산디지털단지·독산·금천구청


def direct_exists(idx: "NextTrainIndex", day_type: str, origin: int, dest: int) -> bool:
    """같은 열차번호로 origin → dest 를 가는 열차가 그날 한 대라도 있는가 (직통이면 구로 갈아타기 불필요)."""
    return len(idx.departures(day_type, origin, dest)) > 0


# ---------------------------------------------------------------------- 4. 표준데이터 교차검증
def read_standard(path) -> pd.DataFrame:
    x = pd.read_excel(path, sheet_name="표준데이터 역사", dtype=str)
    x["code"] = pd.to_numeric(x["역번호"], errors="coerce")
    x["name_std"] = x["역사명"].str.replace(r"역$", "", regex=True)
    x["lat"] = pd.to_numeric(x["역위도"], errors="coerce")
    x["lon"] = pd.to_numeric(x["역경도"], errors="coerce")
    return x


def cross_check(rows: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    """26역: 시각표 역명 = 표준데이터 역명(역 접미사 제거) = 모듈 상수, 한 코드 한 이름."""
    tt = rows[rows.code.isin(list(SEOUL_KORAIL))].groupby("code").name.agg(lambda s: sorted(set(s)))
    out = []
    for code, (name, corridor) in SEOUL_KORAIL.items():
        s = std[std.code == code]
        out.append({"code": code, "station": name, "corridor": corridor,
                    "tt_names": "|".join(tt.get(code, [])),
                    "std_name": "|".join(sorted(set(s.name_std))),
                    "lat": s.lat.iloc[0] if len(s) else np.nan, "lon": s.lon.iloc[0] if len(s) else np.nan})
    df = pd.DataFrame(out)
    df["match"] = (df.tt_names == df.station) & (df.std_name == df.station)
    return df


# ---------------------------------------------------------------------- 5. 환승 후보 (3A-2 에서 사용)
# 1~8호선 v1 환승시간과 같은 원천(서울교통공사_수도권 도시철도 환승 데이터, 패턴 평균)을 우선 쓴다.
# 그 원천의 반올림 값(10:00·07:00·05:00 — 같은 값이 수십 쌍에 반복, 같은 승강장 환승에도 07:00)은 자리표시로 본다.
SUDO = "서울교통공사_수도권 도시철도 환승 데이터 (패턴 평균)"
OFFICIAL_DIST = "서울교통공사 환승역거리 소요시간 정보 20250331"
TRANSFER_CANDIDATES = [
    # station, korail_code, other_line, distance_m, dataset_mean_min, walk_min, status, source
    ("신도림", 1007, "2", 81, 2.911, 2.911, "dataset+official_distance", f"{SUDO} · 거리 {OFFICIAL_DIST}"),
    ("신길", 1032, "5", 242, 6.517, 6.517, "dataset+official_distance", f"{SUDO} · 거리 {OFFICIAL_DIST}"),
    ("가산디지털단지", 1702, "7", 110, 3.267, 3.267, "dataset+official_distance", f"{SUDO} · 거리 {OFFICIAL_DIST}"),
    ("온수", 1821, "7", 102, 4.233, 4.233, "dataset+official_distance", f"{SUDO} · 거리 {OFFICIAL_DIST}"),
    ("창동", 1022, "4", 84, 2.217, 2.217, "dataset+official_distance", f"{SUDO} · 거리 {OFFICIAL_DIST}"),
    ("도봉산", 1903, "7", 44, 1.783, 1.783, "dataset+official_distance", f"{SUDO} · 거리 {OFFICIAL_DIST}"),
    ("노량진", 1004, "9", 150, np.nan, 3.0, "confirmed",
     "국가철도공단 보도자료 2015-10-30·서울시 2015-11-02 (환승통로 개통: 300→150m, 5→3분) + "
     "네이버지도·카카오지하철 3분 (사용자 교차확인 2026-11-22, D-088). 수도권 환승 데이터 값 07:00 은 자리표시"),
    ("석계", 1018, "6", np.nan, 3.467, 3.0, "confirmed",
     "네이버지도·카카오지하철 3분 (사용자 교차확인 2026-11-22, D-088). 공식 환승거리표에 없음. "
     f"{SUDO} 03:28(3.47분)은 민감도 값"),
]


def transfer_candidates() -> pd.DataFrame:
    return pd.DataFrame(TRANSFER_CANDIDATES, columns=["station", "korail_code", "other_line", "distance_m",
                                                      "dataset_mean_min", "walk_min", "status", "source"])


# ---------------------------------------------------------------------- 6. 노선도 배치 초안
def map_draft(nodes: pd.DataFrame, std_xy: pd.DataFrame, spacing: float = 120.0) -> pd.DataFrame:
    """26역 노선도 좌표 초안 (앱에서 쓰지 않음, 사용자 배치 검토용).

    - 같은 이름의 다른 노선 역이 노선도에 있으면 그 좌표를 그대로 쓴다 (anchor)
    - anchor 사이 역은 계통 순서대로 같은 간격으로 나눈다 (interp)
    - 마지막 anchor 뒤(독산·금천구청)는 위·경도 방향으로 spacing px 씩 늘린다 (extrap)
    - 기존 역 점과 40px 안으로 겹치면 conflict 에 그 역 이름을 적는다
    """
    name_of = {**{c: n for c, (n, _) in SEOUL_KORAIL.items()}, **METRO_L1}
    pos = {}
    for code, name in name_of.items():
        r = nodes[nodes.station_name == name]
        if len(r):
            pos[code] = (float(r.x_px.iloc[0]), float(r.y_px.iloc[0]), "anchor")
    ll = {int(c): (lat, lon) for c, lat, lon in std_xy[["code", "lat", "lon"]].itertuples(index=False)}
    for corridor, seq in CORRIDORS.items():
        anchors = [i for i, c in enumerate(seq) if c in pos]
        for a, b in zip(anchors, anchors[1:]):
            (x0, y0, _), (x1, y1, _) = pos[seq[a]], pos[seq[b]]
            for k in range(a + 1, b):
                f = (k - a) / (b - a)
                pos[seq[k]] = (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, "interp")
        last = anchors[-1] if anchors else None
        for k in range(last + 1, len(seq)) if last is not None else []:
            prev, cur = seq[k - 1], seq[k]
            (la0, lo0), (la1, lo1) = ll[prev], ll[cur]
            v = np.array([lo1 - lo0, -(la1 - la0)])           # 화면 y 는 아래로 증가
            v = v / (np.linalg.norm(v) or 1)
            px, py, _ = pos[prev]
            pos[cur] = (px + v[0] * spacing, py + v[1] * spacing, "extrap")
    rows = []
    for code, (name, corridor) in SEOUL_KORAIL.items():
        x, y, how = pos[code]
        near = nodes[((nodes.x_px - x) ** 2 + (nodes.y_px - y) ** 2) ** 0.5 < 40]
        near = near[near.station_name != name]
        rows.append({"code": code, "station": name, "corridor": corridor, "x_px": round(x, 1),
                     "y_px": round(y, 1), "placement": how,
                     "conflict": "|".join(sorted(set(near.station_name)))})
    return pd.DataFrame(rows)
