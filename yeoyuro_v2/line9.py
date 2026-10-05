"""
yeoyuro_v2/line9.py
===================
v2.6 — 9호선(일반 9L / 급행 9X) 데이터 정제와 v1 호환 mart 생성.

출처 (프로젝트 원본, 직접 검증)
------------------------------
- 열차 시각표 : 서울교통공사_서울 도시철도 열차운행시각표_20260901.csv
                업로드 사본은 한글이 '?' 로 손실 → 역명은 역코드(4101~4138)로 복구
- 역코드↔역명 : 서울시메트로9호선_도시철도_운행정보_20260228.xlsx ("4138-중앙보훈병원" 형식)
- 역 정보     : 서울메트로9호선_역사정보_20260531.xlsx (역번호 901~938, 거리, 승강장 유형)
- 혼잡도      : 2024·2025·2026 9호선 역별 시간별 혼잡도 (상/하선 × 일반/급행 × 평일/휴일)
- 환승 거리   : 서울교통공사_환승역거리_소요시간_정보_20250331.csv (거리 ÷ 1.2m/s)

규칙 (decision_log D-039~D-047)
------------------------------
- 노드: 9L_역명 (38개), 9X_역명 (급행 정차 16개). 방향 up = 코드 증가 = 중앙보훈병원행 = 혼잡도 '상선'
- 역간 운행시간: 변동성 측정 후 결정 (RUN_VAR_*). 정차는 역·시간대별 표
- 혼잡: 2024~2026 중앙값 = baseline, 3개년 최댓값 = max_3y 시나리오 (p90 이라 부르지 않음)
- 환승 보행: 거리시간 × factor (기본 1.81 = 1~8호선 실측 ÷ 거리시간 중앙값). 실측 아님
- 일반↔급행 같은 역 갈아타기: 도보 0분 근사 (방향별 승강장 1개). 대기는 시각표 연계 대기
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LINES9 = ("9L", "9X")
TT_FILE = "서울교통공사_서울_도시철도_열차운행시각표_20260901.csv"
KRIC_FILE = "서울시메트로9호선_도시철도_운행정보_20260228.xlsx"
STN_FILE = "서울메트로9호선_역사정보_20260531.xlsx"
TRANSFER_FILE = "서울교통공사_환승역거리_소요시간_정보_20250331.csv"
CONG_FILES = {2024: "2024년_9호선_역별_시간별_혼잡도_자료.xlsx",
              2025: "2025년_9호선_역별_시간별_혼잡도_자료.xlsx",
              2026: "2026년_9호선종합_역별_시간별_혼잡도_자료.xlsx"}
DAY_MAP = {"DAY": "weekday", "SAT": "saturday", "END": "sunday"}

# 역간 운행시간 bin 변동성 판정 기준 (계산 전에 고정, D-040)
RUN_VAR_SECTION_TOL_MIN = 0.25      # 구간별 bin 중앙값 − 전체 중앙값 의 절대값 허용치
RUN_VAR_SECTION_SHARE = 0.95        # 이 비율 이상의 구간이 허용치 안이어야 함
RUN_VAR_TOTAL_TOL = 0.02            # 전 구간 합의 bin 별 편차 허용 비율
MIN_TRAINS_PER_BIN = 3

TRANSFER_FACTOR_BASE = 1.81         # D-042
TRANSFER_FACTORS = {"tf100": 1.00, "tf160": 1.60, "tf181": 1.81, "tf278": 2.78}
LINE9_TRANSFER_STATIONS = {          # 9호선 역 → 범위 안 연결 노선 (노량진은 1호선 범위 밖)
    "김포공항": ["5"], "당산": ["2"], "여의도": ["5"], "동작": ["4"],
    "고속터미널": ["3", "7"], "종합운동장": ["2"], "석촌": ["8"], "올림픽공원": ["5"]}


def _norm(name: str) -> str:
    return re.sub(r"[\s_()\-]", "", str(name)).lower()


_FOUND: dict = {}


def locate(src: Path, name: str) -> Path:
    """원본 파일 찾기. 공백·밑줄·괄호 차이를 무시하고 src 하위 폴더까지 찾는다.

    (공공데이터 원본 이름은 공백을 쓰고, 일부 업로드 사본은 공백이 '_' 로 바뀐다)
    """
    srcs = [Path(x) for x in (src if isinstance(src, (list, tuple)) else [src])]
    key = (tuple(map(str, srcs)), name)
    if key in _FOUND:
        return _FOUND[key]
    target = _norm(name)
    for base in srcs:
        direct = base / name
        if direct.exists():
            _FOUND[key] = direct
            return direct
        for p in base.rglob("*"):
            if p.is_file() and _norm(p.name) == target:
                _FOUND[key] = p
                return p
    raise FileNotFoundError(f"원본 파일을 찾지 못했습니다: {name}  (찾은 폴더: {', '.join(map(str, srcs))})")


def missing_sources(src: Path) -> list[str]:
    names = [TT_FILE, KRIC_FILE, STN_FILE, TRANSFER_FILE, *CONG_FILES.values(),
             "서울교통공사_환승역환승인원정보_20251130.csv", "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv"]
    out = []
    for n in names:
        try:
            locate(src, n)
        except FileNotFoundError:
            out.append(n)
    return out


def canon(name: str) -> str:
    return re.sub(r"\(.*\)$", "", str(name)).strip()


def bin_label_of(minute: float) -> str:
    s = int((minute - 300) // 30) * 30 + 300          # 05:00 기준 30분
    e = s + 30
    return f"{(s // 60) % 24:02d}:{s % 60:02d}~{(e // 60) % 24:02d}:{e % 60:02d}"


def _tmin(s):
    if pd.isna(s):
        return np.nan
    h, m, x = (int(v) for v in str(s).split(":"))
    return h * 60 + m + x / 60


# ---------------------------------------------------------------------- 1. 역 마스터
def station_codes(src: Path) -> dict[int, str]:
    """운영사 운행정보의 '4138-중앙보훈병원' 목록에서 역코드→역명. 38개여야 한다."""
    k = pd.read_excel(locate(src, KRIC_FILE), header=3)
    codes = {}
    for v in k["운행구간정거장"].dropna().astype(str):
        for m in re.finditer(r"(\d{4})-([^+]+)", v):
            codes[int(m.group(1))] = canon(m.group(2))
    return dict(sorted(codes.items()))


def station_master(src: Path) -> pd.DataFrame:
    s = pd.read_excel(locate(src, STN_FILE), header=3)
    s = s[["역 번호", "역명(한글)", "승강장 유형", "역 위치(경도)", "역 위치(위도)", "상행거리", "하행거리"]].copy()
    s.columns = ["stn_no", "name_raw", "platform_type", "lon", "lat", "dist_prev_km", "dist_next_km"]
    s["station"] = s.name_raw.map(canon)
    s["code"] = s.stn_no.astype(int) + 3200                       # 901 → 4101
    s["dist_next_km"] = pd.to_numeric(s.dist_next_km, errors="coerce")
    return s


# ---------------------------------------------------------------------- 2. 시각표
def timetable(src: Path, codes: dict[int, str]) -> pd.DataFrame:
    d = pd.read_csv(locate(src, TT_FILE), dtype=str)
    d.columns = ["id", "line", "code", "name", "day", "dir", "exp", "train", "arr", "dep", "o", "dst"]
    n = d[d.line == "9"].copy()
    n["name_corrupted"] = n.name.str.fullmatch(r"\?+").fillna(False)
    n["code"] = n.code.astype(int)
    n["station"] = n.code.map(codes)
    n["pattern"] = np.where(n.exp == "1", "9X", "9L")
    n["direction"] = np.where(n.dir == "UP", "up", "down")
    n["day_type"] = n.day.map(DAY_MAP)
    n["a"] = n.arr.map(_tmin)
    n["p"] = n.dep.map(_tmin)
    return n


def segments(tt: pd.DataFrame) -> pd.DataFrame:
    """열차별 연속 정차역 쌍. run = 도착(v) − 출발(u), dwell_u = 출발(u) − 도착(u)."""
    rows = []
    for (day, tr, dr, pat), g in tt.groupby(["day_type", "train", "direction", "pattern"]):
        g = g.sort_values("code", ascending=(dr == "up"))
        c, st, a, p = g.code.values, g.station.values, g.a.values, g.p.values
        for i in range(len(c) - 1):
            rows.append((day, dr, pat, tr, c[i], c[i + 1], st[i], st[i + 1], p[i], a[i + 1] - p[i],
                         (p[i] - a[i]) if not np.isnan(a[i]) else np.nan))
    s = pd.DataFrame(rows, columns=["day_type", "direction", "pattern", "train", "from_code", "to_code",
                                    "from_station", "to_station", "dep", "run", "dwell_from"])
    s["time_bin"] = s.dep.map(bin_label_of)
    return s


def run_variability(seg: pd.DataFrame) -> dict:
    """역간 운행시간이 시간대(30분 bin)에 따라 얼마나 변하는가. 평일 기준."""
    w = seg[seg.day_type == "weekday"]
    key = ["pattern", "direction", "from_code", "to_code"]
    overall = w.groupby(key).run.median().rename("overall")
    b = w.groupby(key + ["time_bin"]).run.agg(["median", "size"]).reset_index()
    b = b[b["size"] >= MIN_TRAINS_PER_BIN].merge(overall.reset_index(), on=key)
    b["dev"] = (b["median"] - b["overall"]).abs()
    sec = b.groupby(key).dev.max().reset_index()
    share_ok = float((sec.dev <= RUN_VAR_SECTION_TOL_MIN).mean())
    tot = b.groupby(["pattern", "direction", "time_bin"]).agg(m=("median", "sum"), o=("overall", "sum"),
                                                             n=("median", "size"))
    full = tot[tot.n == tot.n.groupby(level=[0, 1]).transform("max")]
    total_dev = float(((full.m - full.o).abs() / full.o).max())
    small = share_ok >= RUN_VAR_SECTION_SHARE and total_dev <= RUN_VAR_TOTAL_TOL
    day_cmp = (seg.groupby(key + ["day_type"]).run.median().unstack("day_type"))
    day_dev = float((day_cmp.sub(day_cmp["weekday"], axis=0)).abs().max().max())
    return {"section_share_within_tol": share_ok, "max_section_dev": float(sec.dev.max()),
            "total_max_rel_dev": total_dev, "small": small, "day_type_max_dev": day_dev,
            "per_section": sec}


# ---------------------------------------------------------------------- 3. mart
def route_edges(seg: pd.DataFrame, sm: pd.DataFrame, by_day: bool = False) -> pd.DataFrame:
    """v1 route_edge_mart 와 같은 스키마. travel_time_min = 평일 운행시간 중앙값 (정차 제외)."""
    w = seg[seg.day_type == "weekday"]
    med = w.groupby(["pattern", "direction", "from_code", "to_code", "from_station", "to_station"]) \
           .run.median().reset_index()
    dist = {}
    for r in sm.itertuples():
        dist[r.code] = r.dist_next_km
    rows = []
    for r in med.itertuples():
        lo = min(r.from_code, r.to_code)
        km = float(np.nansum([dist.get(c, np.nan) for c in range(lo, max(r.from_code, r.to_code))]))
        rows.append({"from_node": f"{r.pattern}_{r.from_station}", "to_node": f"{r.pattern}_{r.to_station}",
                     "line_id": r.pattern, "from_station": r.from_station, "to_station": r.to_station,
                     "from_code": int(r.from_code), "to_code": int(r.to_code),
                     "travel_time_min": round(float(r.run), 3), "distance_km": round(km, 2),
                     "edge_type": "ride", "branch_code": "main", "from_branch": "main", "to_branch": "main",
                     "is_bidirectional": 1,
                     "direction_of_edge": "forward" if r.direction == "up" else "reverse"})
    return pd.DataFrame(rows)


def run_table(seg: pd.DataFrame) -> pd.DataFrame:
    """구간·요일·시간대별 운행시간 중앙값 (변동성 기준 미달 시 엔진이 사용, D-040).

    bin 에 열차가 MIN_TRAINS_PER_BIN 미만이면 행을 만들지 않는다 → 엔진은 패턴 평일 중앙값으로 돌아간다.
    """
    t = seg.groupby(["pattern", "direction", "from_station", "to_station", "day_type", "time_bin"]).run \
           .agg(["median", "size"]).reset_index()
    t = t[t["size"] >= MIN_TRAINS_PER_BIN]
    t["from_node"] = t.pattern + "_" + t.from_station
    t["to_node"] = t.pattern + "_" + t.to_station
    return t.rename(columns={"median": "run_min", "size": "n_trains"})[
        ["from_node", "to_node", "pattern", "direction", "day_type", "time_bin", "run_min", "n_trains"]]


def dwell_table(seg: pd.DataFrame) -> pd.DataFrame:
    """역·방향·요일·시간대별 정차시간 중앙값. 엔진이 9호선 중간역 정차에 쓴다."""
    d = seg.dropna(subset=["dwell_from"])
    t = d.groupby(["pattern", "from_station", "direction", "day_type", "time_bin"]).dwell_from \
         .agg(["median", "size"]).reset_index()
    t["station_uid"] = t.pattern + "_" + t.from_station
    return t.rename(columns={"median": "dwell_min", "size": "n_trains", "from_station": "station"})


def headway_tables(tt: pd.DataFrame, bins: list[str]):
    """v1 headway_station_30min / headway_line_30min 과 같은 스키마 (9L, 9X 따로)."""
    t = tt.dropna(subset=["p"]).copy()
    t["time_bin"] = t.p.map(bin_label_of)
    cnt = t.groupby(["pattern", "station", "code", "direction", "day_type", "time_bin"]).size() \
           .rename("n_departures").reset_index()
    base = (t[["pattern", "station", "code"]].drop_duplicates()
            .merge(pd.DataFrame({"direction": ["up", "down"]}), how="cross")
            .merge(pd.DataFrame({"day_type": ["weekday", "saturday", "sunday"]}), how="cross")
            .merge(pd.DataFrame({"time_bin": bins}), how="cross"))
    st = base.merge(cnt, how="left", on=["pattern", "station", "code", "direction", "day_type", "time_bin"])
    st["n_departures"] = st.n_departures.fillna(0).astype(int)
    st["is_operating"] = (st.n_departures > 0).astype(int)
    st["headway_freq_min"] = np.where(st.n_departures > 0, 30.0 / st.n_departures.clip(lower=1), np.nan)
    st["expected_wait_min"] = st.headway_freq_min / 2
    st = st.rename(columns={"pattern": "line_id", "code": "station_code"})
    st["station_uid"] = st.line_id + "_" + st.station
    st["station_name"] = st.station
    st["station_name_raw"] = st.station
    st["section_name"] = "9호선 " + np.where(st.line_id == "9X", "급행", "일반")
    st = st[["line_id", "station_uid", "station_code", "station_name", "station_name_raw", "section_name",
             "day_type", "direction", "time_bin", "n_departures", "headway_freq_min", "expected_wait_min",
             "is_operating"]]
    ln = (st[st.is_operating == 1].groupby(["line_id", "day_type", "direction", "time_bin"])
          .agg(avg_headway_freq_min=("headway_freq_min", "mean"),
               avg_expected_wait_min=("expected_wait_min", "mean")).reset_index().round(2))
    return st, ln


def switch_wait_table(tt: pd.DataFrame, cap_min: float = 30.0) -> pd.DataFrame:
    """같은 역·같은 방향에서 일반↔급행 갈아탈 때 실제 대기 (도착 후 다음 상대 패턴 출발까지).

    시간대(도착 시각 기준 30분 bin) 평균. 배차/2 대신 시각표 연계를 반영한다 (D-044).
    """
    rows = []
    for (day, dr, stn), g in tt.groupby(["day_type", "direction", "station"]):
        if set(g.pattern) != {"9L", "9X"}:
            continue
        for fr, to in (("9L", "9X"), ("9X", "9L")):
            arr = g[(g.pattern == fr)].a.dropna().values
            dep = np.sort(g[(g.pattern == to)].p.dropna().values)
            for x in arr:
                k = np.searchsorted(dep, x, side="left")
                if k < len(dep) and dep[k] - x <= cap_min:
                    rows.append((fr, to, stn, dr, day, bin_label_of(x), dep[k] - x))
    w = pd.DataFrame(rows, columns=["from_line", "to_line", "station", "direction", "day_type", "time_bin", "wait"])
    return (w.groupby(["from_line", "to_line", "station", "direction", "day_type", "time_bin"])
             .wait.agg(wait_min="mean", n="size").reset_index())


def congestion_lookup(src: Path, C0: float, KAPPA: float, express_stations: set[str]):
    """3개년 9호선 혼잡도 → v1 congestion_edge_lookup 스키마. baseline(중앙값)과 max_3y 를 같이 반환."""
    frames = []
    for year, f in CONG_FILES.items():
        x = pd.read_excel(locate(src, f), sheet_name=None, header=1)
        for sheet, d in x.items():
            m = re.match(r"(상선|하선)(일반|급행)\((평일|휴일)\)", sheet)
            if not m:
                continue
            d = d.rename(columns={"구분": "station"}).dropna(subset=["station"])
            long = d.melt(id_vars="station", var_name="col", value_name="v")
            long["v"] = pd.to_numeric(long.v, errors="coerce")
            long["start"] = long.col.astype(str).str[:5]
            long = long[long.start.str.match(r"\d\d:\d\d")]
            long["year"] = year
            long["direction"] = "up" if m.group(1) == "상선" else "down"
            long["pattern"] = "9X" if m.group(2) == "급행" else "9L"
            long["day_src"] = m.group(3)
            frames.append(long)
    c = pd.concat(frames, ignore_index=True)
    c["station"] = c.station.map(canon)
    c.loc[c.v <= 0, "v"] = np.nan                       # 0 = 종착 또는 미운행 → 결측 (D-043)
    hm = c.start.str.split(":", expand=True).astype(int)
    c["time_bin_index"] = ((hm[0] * 60 + hm[1] - 330) % 1440) // 30
    c = c[c.time_bin_index.between(0, 38)]
    g = c.groupby(["pattern", "station", "direction", "day_src", "time_bin_index"]).v \
         .agg(median="median", max_3y="max", n="count").reset_index()
    out = []
    for dt_src, dts in (("평일", ["weekday"]), ("휴일", ["saturday", "sunday"])):
        part = g[g.day_src == dt_src]
        for dt in dts:
            p = part.copy()
            p["day_type"] = dt
            out.append(p)
    lk = pd.concat(out, ignore_index=True)
    lk = lk[(lk.pattern == "9L") | lk.station.isin(express_stations)]
    lk["line_id"] = lk.pattern
    lk["station_uid"] = lk.pattern + "_" + lk.station
    lk["station_name"] = lk.station
    lk["branch_code"] = "main"
    s = 330 + lk.time_bin_index * 30
    lk["time_bin"] = [f"{(a // 60) % 24:02d}:{a % 60:02d}~{((a + 30) // 60) % 24:02d}:{(a + 30) % 60:02d}" for a in s]
    lk["n_snapshots"] = lk.n.astype(int)

    def finish(val):
        t = lk.copy()
        t["congestion_median"] = t[val].round(2)
        t["congestion_p90"] = np.nan          # 9호선은 스냅샷이 3개뿐 → p90 정의하지 않음 (D-045)
        t["perceived_multiplier"] = 1.0 + KAPPA * np.clip(t.congestion_median.fillna(0) - C0, 0, None) / 100
        return t[["line_id", "station_uid", "station_name", "branch_code", "direction", "day_type",
                  "time_bin_index", "time_bin", "congestion_median", "congestion_p90", "n_snapshots",
                  "perceived_multiplier"]]
    return finish("median"), finish("max_3y"), c


def transfer_edges(src: Path, factor: float, v1_transfer: pd.DataFrame, volumes: pd.DataFrame,
                   express_stations: set[str], penalty_max: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1~8호선 ↔ 9L/9X 환승 + 같은 역 9L↔9X 갈아타기 edge. v1 transfer_edge_mart 스키마."""
    d = pd.read_csv(locate(src, TRANSFER_FILE), encoding="cp949")
    d["sec"] = d["환승소요시간"].map(lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))
    d["from_line"] = d["호선"].astype(str)
    d["to_line"] = d["환승노선"].str.extract(r"(\d)")[0]
    d["station"] = d["환승역명"].map(canon)
    raw = d[(d.to_line == "9") & d.from_line.isin(list("2345678"))][["station", "from_line", "sec", "환승거리"]]
    # 환승 혼잡 패널티: v1 과 같은 방식 (전체 환승 edge 의 인원 백분위 × 3분)
    ref = np.sort(v1_transfer.weekday_volume.dropna().values)
    vol = dict(zip(volumes.station_name, volumes.weekday_volume))
    rows = []
    for r in raw.itertuples():
        if r.station not in LINE9_TRANSFER_STATIONS or r.from_line not in LINE9_TRANSFER_STATIONS[r.station]:
            continue
        t = r.sec / 60 * factor
        v = vol.get(r.station, np.nan)
        pen = round(float(np.searchsorted(ref, v, side="right") / len(ref)) * penalty_max, 2) if not np.isnan(v) else 0.0
        for p9 in (["9L", "9X"] if r.station in express_stations else ["9L"]):
            for a, b, fl, tl in ((f"{r.from_line}_{r.station}", f"{p9}_{r.station}", r.from_line, p9),
                                 (f"{p9}_{r.station}", f"{r.from_line}_{r.station}", p9, r.from_line)):
                rows.append({"from_node": a, "to_node": b, "station_name": r.station, "from_line": fl,
                             "to_line": tl, "transfer_time_min": round(t, 3), "transfer_time_min_best": round(t, 3),
                             "n_patterns": 0, "is_branch_transfer": 0, "weekday_volume": v,
                             "volume_penalty_min": pen, "transfer_penalty_min": round(t + pen, 3),
                             "edge_type": "transfer", "is_bidirectional": 1})
    for stn in sorted(express_stations):        # 일반↔급행: 도보 0분 근사 (D-046), 대기는 엔진이 연계표로
        for a, b in (("9L", "9X"), ("9X", "9L")):
            rows.append({"from_node": f"{a}_{stn}", "to_node": f"{b}_{stn}", "station_name": stn, "from_line": a,
                         "to_line": b, "transfer_time_min": 0.0, "transfer_time_min_best": 0.0, "n_patterns": 0,
                         "is_branch_transfer": 0, "weekday_volume": np.nan, "volume_penalty_min": 0.0,
                         "transfer_penalty_min": 0.0, "edge_type": "transfer", "is_bidirectional": 1})
    return pd.DataFrame(rows), raw


def display_rows(display: pd.DataFrame, sm: pd.DataFrame, express_stations: set[str]) -> pd.DataFrame:
    """station_display_master 에 9호선을 합친다. 기존 역은 노선·후보 uid 만 덧붙인다."""
    d = display.copy()
    nxt = int(d.station_no.max()) + 1
    add = []
    for r in sm.itertuples():
        uids = [f"9L_{r.station}"] + ([f"9X_{r.station}"] if r.station in express_stations else [])
        hit = d.station_key == r.station
        if hit.any():
            i = d.index[hit][0]
            d.at[i, "available_lines"] = f"{d.at[i, 'available_lines']},9"
            d.at[i, "candidate_station_uids"] = f"{d.at[i, 'candidate_station_uids']};" + ";".join(uids)
            d.at[i, "project_branches"] = f"{d.at[i, 'project_branches']};" + ";".join(f"{u.split('_')[0]}:main" for u in uids)
            d.at[i, "is_transfer_station"] = True
            d.at[i, "n_lines"] = int(d.at[i, "n_lines"]) + 1
        else:
            add.append({"station_no": nxt, "station_key": r.station, "station_name": r.station,
                        "display_name": r.station + "역", "available_lines": "9",
                        "candidate_station_uids": ";".join(uids),
                        "project_branches": ";".join(f"{u.split('_')[0]}:main" for u in uids),
                        "is_transfer_station": False, "is_event_station": False, "alias_or_note": "",
                        "memo": "v2.6 9호선", "n_lines": 1, "is_branch_node": False})
            nxt += 1
    return pd.concat([d, pd.DataFrame(add)], ignore_index=True)


def register_v1_directions(mod) -> None:
    """v1 엔진의 방향 표에 9L/9X 를 등록한다 (원본 파일 수정 없음, 실행 시점 확장)."""
    for ln in LINES9:
        mod.CODE_ASC_DIRECTION.setdefault(ln, "up")
        mod.CODE_DESC_DIRECTION.setdefault(ln, "down")


SCENARIOS = {"base": ("tf181", "median"), "tf100": ("tf100", "median"), "tf160": ("tf160", "median"),
             "tf278": ("tf278", "median"), "max3y": ("tf181", "max_3y"), "cap160": ("tf181", "median_cap160")}
CAPACITY_9_PER_CAR = 922 / 6          # 9호선 혼잡 100% = 6칸 922명
CAPACITY_18_PER_CAR = 160             # 1~8호선 혼잡 100% = 1칸 160명
CAP160_FACTOR = CAPACITY_9_PER_CAR / CAPACITY_18_PER_CAR     # 0.9604 (v2.6c, preregistration_v26c_capacity)


def build_shadow_root(root: Path, out: Path, scenario: str = "base") -> Path:
    """커밋된 9호선 mart(data/marts/v2/line9)와 1~8호선 mart 를 합쳐 엔진용 root 를 만든다.

    원본 xlsx/csv 없이 재현 가능하다. 반환: out (RouteScorer/TDRouter 의 root 로 사용).
    """
    import shutil
    tf, cong = SCENARIOS[scenario]
    m9 = root / "data" / "marts" / "v2" / "line9"
    base = root / "data" / "marts"
    (out / "data" / "marts").mkdir(parents=True, exist_ok=True)
    (out / "data" / "master").mkdir(parents=True, exist_ok=True)
    for f in (root / "data" / "master").glob("*.csv"):
        shutil.copy(f, out / "data" / "master" / f.name)
    shutil.copy(m9 / "station_display_master_with9.csv", out / "data" / "master" / "station_display_master.csv")

    def cat(name, f9):
        return pd.concat([pd.read_parquet(base / f"{name}.parquet"), pd.read_parquet(m9 / f9)], ignore_index=True)
    cat("route_edge_mart", "route_edges.parquet").to_parquet(out / "data/marts/route_edge_mart.parquet", index=False)
    cat("transfer_edge_mart", f"transfer_{tf}.parquet").to_parquet(out / "data/marts/transfer_edge_mart.parquet", index=False)
    lk9 = pd.read_parquet(m9 / ("congestion_max3y.parquet" if cong == "max_3y" else "congestion_median.parquet"))
    if cong == "median_cap160":
        # 9호선 % 를 1~8호선 정원(1칸 160명) 기준으로 환산. 체감 배수도 같은 규칙으로 다시 계산
        lk9["congestion_median"] = (lk9.congestion_median * CAP160_FACTOR).round(2)
        lk9["perceived_multiplier"] = 1.0 + 0.5 * np.clip(lk9.congestion_median.fillna(0) - 80.0, 0, None) / 100
    pd.concat([pd.read_parquet(base / "congestion_edge_lookup.parquet"), lk9], ignore_index=True) \
        .to_parquet(out / "data/marts/congestion_edge_lookup.parquet", index=False)
    cat("headway_station_30min", "headway_station.parquet").to_parquet(out / "data/marts/headway_station_30min.parquet", index=False)
    cat("headway_line_30min", "headway_line.parquet").to_parquet(out / "data/marts/headway_line_30min.parquet", index=False)
    for f9, dst in (("dwell.parquet", "v2_line9_dwell.parquet"), ("switch_wait.parquet", "v2_line9_switch_wait.parquet")):
        shutil.copy(m9 / f9, out / "data" / "marts" / dst)
    run = m9 / "run.parquet"
    if (m9 / "RUN_BY_BIN").exists() and run.exists():           # 변동성 기준 미달 시에만 생성되는 표식
        shutil.copy(run, out / "data" / "marts" / "v2_line9_run.parquet")
    return out
