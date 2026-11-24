"""
yeoyuro_v2/korail_3b.py — Phase 3B-1 경의중앙·경춘·수인분당·공항철도 열차 단위 시각표 (D-105 사전 등록 3절)
=========================================================================================================

원본 → 열차 단위 정차표 (3-A 열차 단위 시각표와 같은 열) + 데이터 검사 + B 등급 환승 계수 보정 오차 (4-3 C1).

원본 (D-105 2절, 저장소에 넣지 않음 — P4)
-----------------------------------------
코레일 광역전철 평일·휴일 xlsx (경의중앙선·경춘선 시트를 씀), 수인분당선 xlsx, 공항철도 일반열차 xlsx.
형식: 열마다 열차 1개, 위에 열차번호·시발역·종착역, 역마다 도착 행 + 출발 행, 빈칸 = 안 섬.
공항철도는 한 시트에 하행·상행 두 블록, 도착 칸 '---' + 출발 칸 시각 = 통과(직통).

산출 (P4: 상세 정차표는 재배포 권한 확인 전까지 gitignore)
-----------------------------------------------------------
data/marts/v2/korail_3b/stop_times.parquet   day_type·group·line·train·direction·stop_seq·code·arr_min·dep_min
data/marts/v2/korail_3b/stations.csv         그룹별 역 (역번호 = 전국도시철도역사정보표준데이터 20260630)

요일 (D-105 2절): 평일 시각표 → weekday, 휴일 시각표 → saturday·sunday 두 벌 (토·일·공휴일 = 휴일).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------- 원본 (D-105 2절)
SOURCES = {
    "gj_weekday": ("202608281a045c7905b490.xlsx", "5cee7c8d3409d02bcde2c412b6ab6bc5d5fa5335d07e9331187d89bed6473a70"),
    "gj_holiday": ("202608281a045c7cc64640.xlsx", "a6e95ad13a77fd6a39ee42eb869b7a2238bed6e4a1050514b27f871fd0d3527a"),
    "sb": ("수인분당선 시각표 2026.8.22 적용.xlsx", "44543f0280afb17acbcdb12aefcd40ec79ff28fc28b6c1e1f80d9743ee3d14d0"),
    "ar": ("공항철도 일반열차 운행시각표 2025.12.29.적용.xlsx", "2fd72bee9b1709e0c7a88c3a09f3363ea1955a29695a2a51f984f3c0821ef638"),
}
OFFICIAL_TRANSFER_FILE = "서울교통공사_환승역거리 소요시간 정보_20260831.csv"   # 4-3 C1 검증 자료 ② (선택)
VALID_FROM = {"GJ": "2026-09-01", "GC": "2026-09-01", "SB": "2026-08-22", "AR": "2025-12-29"}

# ---------------------------------------------------------------------- 범위·역번호 (D-105 1절, 표준데이터 20260630)
# 그룹 안 역 순서. direction: 순서가 커지는 쪽 = DOWN, 작아지는 쪽 = UP
ORDER = {
    "GJ": ["수색", "디지털미디어시티", "가좌", "홍대입구", "서강대", "공덕", "효창공원앞", "용산", "이촌", "서빙고", "한남",
           "옥수", "응봉", "왕십리", "청량리", "회기", "중랑", "상봉", "망우", "양원"],
    "GC": ["청량리", "회기", "중랑", "상봉", "망우"],
    "SB": ["왕십리", "서울숲", "압구정로데오", "강남구청", "선정릉", "선릉", "한티", "도곡", "구룡", "개포동", "대모산입구", "수서",
           "복정", "가천대", "태평", "모란", "야탑"],
    "AR": ["서울역", "공덕", "홍대입구", "디지털미디어시티", "마곡나루", "김포공항"],
}
CODE = {
    "용산": 1003, "이촌": 1008, "서빙고": 1009, "한남": 1010, "옥수": 1011, "응봉": 1012, "왕십리": 1013, "청량리": 1014,
    "회기": 1015, "중랑": 1201, "상봉": 1202, "망우": 1203, "양원": 1204, "효창공원앞": 1261, "공덕": 1262, "서강대": 1263,
    "홍대입구": 1264, "가좌": 1265, "디지털미디어시티": 1266, "수색": 1267,
    "서울숲": 1847, "압구정로데오": 1848, "강남구청": 1849, "선정릉": 1850, "선릉": 1023, "한티": 1024, "도곡": 1025,
    "구룡": 1026, "개포동": 1027, "대모산입구": 1028, "수서": 1030, "복정": 1031, "가천대": 1851, "태평": 1852, "모란": 1853,
    "야탑": 1854,
}
# 공항철도 역번호는 문자(A01…)라 정수로 옮긴다: A01 → 20100, A042 → 20420 (원래 번호는 stations.csv 에 남김)
AR_CODE = {"서울역": ("A01", 20100), "공덕": ("A02", 20200), "홍대입구": ("A03", 20300),
           "디지털미디어시티": ("A04", 20400), "마곡나루": ("A042", 20420), "김포공항": ("A05", 20500)}
# 감사 구간 (T4 개수): (그룹, 구간 역 목록)
SEGMENTS = {"GJ_A": ("GJ", ORDER["GJ"][7:]), "GJ_B": ("GJ", ORDER["GJ"][:8][::-1]), "GC": ("GC", ORDER["GC"]),
            "SB": ("SB", ORDER["SB"]), "AR": ("AR", ORDER["AR"])}
# 3B-0 감사 수치 (D-105 3절 T4) — 구간·요일·방향별 '범위 안 두 역 이상 정차' 열차 수
EXPECTED_T4 = {
    ("GJ_A", "weekday", "UP"): 89, ("GJ_A", "weekday", "DOWN"): 90, ("GJ_A", "holiday", "UP"): 72, ("GJ_A", "holiday", "DOWN"): 73,
    ("GJ_B", "weekday", "UP"): 86, ("GJ_B", "weekday", "DOWN"): 85, ("GJ_B", "holiday", "UP"): 72, ("GJ_B", "holiday", "DOWN"): 72,
    ("GC", "weekday", "UP"): 60, ("GC", "weekday", "DOWN"): 61, ("GC", "holiday", "UP"): 43, ("GC", "holiday", "DOWN"): 43,
    ("SB", "weekday", "UP"): 168, ("SB", "weekday", "DOWN"): 168, ("SB", "holiday", "UP"): 130, ("SB", "holiday", "DOWN"): 130,
    ("AR", "weekday", "*"): 366, ("AR", "holiday", "*"): 318,
}
# 시각표 약칭 → 역명 (3B-0 감사)
ALIAS = {"디엠시": "디지털미디어시티", "홍대입": "홍대입구", "효창공": "효창공원앞", "1양원": "양원", "로데오": "압구정로데오",
         "강남구": "강남구청", "구룡역": "구룡", "대모산": "대모산입구", "서울": "서울역"}
GJ_SHEETS = {"경의중앙선 상행": ("GJ", "UP"), "경의중앙선 하행": ("GJ", "DOWN"),
             "경춘선 상행": ("GC", "UP"), "경춘선 하행": ("GC", "DOWN")}   # 코레일 시트 이름 → (그룹, 원본 방향)
SERVICE_START, SERVICE_END = 270.0, 1709.0          # 운행일 04:30 ~ 다음 날 04:29 (D-103)
HDR = {"조정안", "개정안", "시발역", "종착역", "열차번호", "시발시간", "종착시간", "비 고", "비고"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def locate(src: Path) -> dict:
    """원본 이름으로 찾고 해시를 확인. {key: Path}. 없거나 해시가 다르면 ValueError."""
    out, bad = {}, []
    for key, (name, digest) in SOURCES.items():
        p = Path(src) / name
        if not p.exists():
            bad.append(f"{name} 없음")
            continue
        got = sha256(p)
        if got != digest:
            bad.append(f"{name} 해시 다름 ({got[:12]} ≠ {digest[:12]})")
        out[key] = p
    if bad:
        raise ValueError("; ".join(bad))
    return out


# ---------------------------------------------------------------------- 파서 (3B-0 감사 파서를 옮김)
def _tmin(v):
    """셀 → (분 또는 None, 통과 표시 '---' 여부)."""
    if v is None:
        return None, False
    if isinstance(v, (dt.time, dt.datetime)):
        return v.hour * 60 + v.minute + v.second / 60, False
    s = str(v).strip()
    if not s:
        return None, False
    if s == "---":
        return None, True
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        return int(m[1]) * 60 + int(m[2]) + int(m[3] or 0) / 60, False
    raise ValueError(f"시각으로 읽을 수 없는 칸: {s!r}")


def _blocks(rows):
    hs = [i for i, r in enumerate(rows) if r and isinstance(r[0], str) and r[0].strip() == "열차번호"]
    for j, h in enumerate(hs):
        yield h, (hs[j + 1] - 1 if j + 1 < len(hs) else len(rows))


def parse_workbook(path: Path, sheets=None) -> pd.DataFrame:
    """xlsx → 긴 표 (sheet, block, train, origin, dest, note, station, arr, dep, passing). sheets 가 있으면 그 시트만."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        if sheets is not None and ws.title not in sheets:
            continue
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        for b, (h, end) in enumerate(_blocks(rows)):
            meta = {}
            for k in range(max(0, h - 4), min(h + 6, end)):
                lab = str(rows[k][0]).strip() if rows[k] and rows[k][0] else ""
                if lab in ("열차번호", "시발역", "종착역"):
                    meta[lab] = rows[k]
            note_row = next((rows[k] for k in range(h, end)
                             if rows[k] and str(rows[k][0] or "").strip() in ("비 고", "비고")), None)
            trains = meta["열차번호"]
            cols = [c for c in range(1, len(trains)) if trains[c] not in (None, "") and str(trains[c]).strip()]
            st_rows, k = [], h + 1
            while k < end:
                lab = rows[k][0] if rows[k] else None
                if isinstance(lab, str) and lab.strip() and lab.strip() not in HDR and not lab.strip().startswith("["):
                    nxt = rows[k + 1] if k + 1 < end else None
                    dep_row = nxt if nxt is not None and not (isinstance(nxt[0], str) and nxt[0].strip()) else None
                    st_rows.append((lab.strip(), rows[k], dep_row))
                    k += 2 if dep_row is not None else 1
                else:
                    k += 1

            def cell(r, c):
                return r[c] if r is not None and c < len(r) else None
            for c in cols:
                base = dict(sheet=ws.title, block=b, train=str(trains[c]).strip(),
                            origin=str(cell(meta.get("시발역"), c) or "").strip(),
                            dest=str(cell(meta.get("종착역"), c) or "").strip(),
                            note=str(cell(note_row, c) or "").strip())
                for name, ar, dr in st_rows:
                    a, ap = _tmin(cell(ar, c))
                    d, dp = _tmin(cell(dr, c))
                    if ap and d is not None:                       # 공항철도 통과
                        out.append({**base, "station": name, "arr": None, "dep": d, "passing": True})
                    elif a is None and d is None:
                        if ap or dp:
                            out.append({**base, "station": name, "arr": None, "dep": None, "passing": True})
                    else:
                        out.append({**base, "station": name, "arr": a, "dep": d, "passing": False})
    return pd.DataFrame(out)


def _canon(s: str) -> str:
    s = re.sub(r"\s+", "", str(s))
    return ALIAS.get(s, s)


# ---------------------------------------------------------------------- 정차표 만들기
def _unfold(times: list) -> list:
    """정차 순서대로 시각을 펼친다: 04:30 전 시작 = 전날 운행일 +1440, 크게 줄면 자정 넘김 +1440."""
    out, off, prev = [], 0.0, None
    first = next((v for v in times if v is not None), None)
    if first is not None and first < SERVICE_START:
        off = 1440.0
    for v in times:
        if v is None:
            out.append(None)
            continue
        x = v + off
        if prev is not None and x < prev - 1e-9 and prev - x > 600:
            off += 1440.0
            x = v + off
        out.append(x)
        prev = x
    return out


def build(paths: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """원본 경로 {key: Path} → (stop_times, stations, 검사 dict)."""
    raw = []
    for key, day in (("gj_weekday", "weekday"), ("gj_holiday", "holiday")):
        df = parse_workbook(paths[key], sheets=set(GJ_SHEETS))
        df["group"] = df.sheet.map(lambda s: GJ_SHEETS[s][0])
        df["day"] = day
        raw.append(df)
    sb = parse_workbook(paths["sb"])
    sb["group"], sb["day"] = "SB", sb.sheet.map(lambda s: "weekday" if "평일" in s else "holiday")
    ar = parse_workbook(paths["ar"])
    ar["group"], ar["day"] = "AR", ar.sheet.map(lambda s: "weekday" if "평일" in s else "holiday")
    raw += [sb, ar]
    R = pd.concat(raw, ignore_index=True)
    R["st"] = R.station.map(_canon)

    rows, check = [], {"skipped_out_of_scope_trains": 0, "time_reversal": [], "dup_station": [], "non_monotonic": [],
                       "outside_service_day": [], "dup_train": [], "passing_in_scope": {}}
    for (day, group, sheet, block, train), t in R.groupby(["day", "group", "sheet", "block", "train"], sort=False):
        order = ORDER[group]
        stops = t[~t.passing]
        times = []
        for r in stops.itertuples():
            times += [None if pd.isna(r.arr) else float(r.arr), None if pd.isna(r.dep) else float(r.dep)]
        ut = _unfold(times)
        st = []
        for i, r in enumerate(stops.itertuples()):
            a, d = ut[2 * i], ut[2 * i + 1]
            if r.st in order:
                st.append((r.st, a, d))
        if len(st) < 2:
            check["skipped_out_of_scope_trains"] += 1
            continue
        if group == "AR":
            check["passing_in_scope"][day] = check["passing_in_scope"].get(day, 0) + int(
                t[t.passing & t.st.isin(order)].shape[0] > 0)
        idx = [order.index(s) for s, _, _ in st]
        names = [s for s, _, _ in st]
        if len(set(names)) != len(names):
            check["dup_station"].append(f"{day} {train}")
        if not (all(a < b for a, b in zip(idx, idx[1:])) or all(a > b for a, b in zip(idx, idx[1:]))):
            check["non_monotonic"].append(f"{day} {train} {names}")
        direction = "DOWN" if idx[-1] > idx[0] else "UP"
        flat = [x for _, a, d in st for x in (a, d) if x is not None]
        if any(b < a - 1e-9 for a, b in zip(flat, flat[1:])):
            check["time_reversal"].append(f"{day} {train}")
        if flat and (min(flat) < SERVICE_START or max(flat) > SERVICE_END):
            check["outside_service_day"].append(f"{day} {train} {min(flat):.1f}~{max(flat):.1f}")
        for n, (s, a, d) in enumerate(st):
            code = AR_CODE[s][1] if group == "AR" else CODE[s]
            rows.append(dict(day=day, group=group, line=group, train=train, direction=direction, stop_seq=n, code=code,
                             arr_min=np.nan if (n == 0 or a is None) else a,
                             dep_min=np.nan if (n == len(st) - 1 or d is None) else d,
                             src_sheet=sheet))
    S = pd.DataFrame(rows)
    # 출발 시각이 없는 정차(종착 아님)는 도착 시각으로, 도착이 없는 정차(시발 아님)는 출발 시각으로 메운다 (3-A 와 같음)
    mid = (S.stop_seq > 0)
    last = S.groupby(["day", "group", "train"]).stop_seq.transform("max")
    S.loc[mid & S.arr_min.isna(), "arr_min"] = S.loc[mid & S.arr_min.isna(), "dep_min"]
    keep_dep = S.stop_seq < last
    S.loc[keep_dep & S.dep_min.isna(), "dep_min"] = S.loc[keep_dep & S.dep_min.isna(), "arr_min"]
    dup = S.drop_duplicates(["day", "group", "train", "src_sheet"]).duplicated(["day", "group", "train"])
    check["dup_train"] = int(dup.sum())

    # 요일: 휴일 → saturday·sunday 두 벌 (D-105 2절)
    wk = S[S.day == "weekday"].assign(day_type="weekday")
    ho = S[S.day == "holiday"]
    out = pd.concat([wk, ho.assign(day_type="saturday"), ho.assign(day_type="sunday")], ignore_index=True)
    out = out[["day_type", "group", "line", "train", "direction", "stop_seq", "code", "arr_min", "dep_min"]]
    out = out.astype({"stop_seq": "int64", "code": "int64"}).sort_values(
        ["day_type", "group", "train", "stop_seq"]).reset_index(drop=True)

    # T4: 구간별 '범위 안 두 역 이상 정차' 열차 수 (원본 방향 = 시트 상행/하행, 공항철도는 합)
    t4 = {}
    code_name = {**{v: k for k, v in CODE.items()}, **{v[1]: k for k, v in AR_CODE.items()}}
    for seg, (group, sts) in SEGMENTS.items():
        g = S[S.group == group].assign(st=lambda x: x.code.map(code_name))
        g = g[g.st.isin(sts)]
        n = g.groupby(["day", "src_sheet", "train"]).size()
        n = n[n >= 2].reset_index()
        if group == "AR":
            for day, c in n.groupby("day").train.nunique().items():
                t4[(seg, day, "*")] = int(c)
        else:
            src_dir = n.src_sheet.map(lambda s: "UP" if "상행" in s else "DOWN")
            for (day, d), c in n.assign(d=src_dir).groupby(["day", "d"]).train.nunique().items():
                t4[(seg, day, d)] = int(c)
    check["t4"] = {"|".join(k): v for k, v in sorted(t4.items())}
    check["t4_mismatch"] = {"|".join(k): (t4.get(k), v) for k, v in EXPECTED_T4.items() if t4.get(k) != v}
    wk_n = S[(S.group == "GJ") & (S.day == "weekday")].train.nunique()
    ho_n = S[(S.group == "GJ") & (S.day == "holiday")].train.nunique()
    check["t5_gj_weekday_gt_holiday"] = (int(wk_n), int(ho_n), bool(wk_n > ho_n))
    check["rows"] = int(len(out))
    check["trains"] = {f"{d}|{g}": int(c) for (d, g), c in out.groupby(["day_type", "group"]).train.nunique().items()}

    st_rows = []
    for g, names in ORDER.items():
        for i, s in enumerate(names):
            if g == "AR":
                st_rows.append(dict(group=g, order=i, station=s, code=AR_CODE[s][1], std_code=AR_CODE[s][0]))
            else:
                st_rows.append(dict(group=g, order=i, station=s, code=CODE[s], std_code=str(CODE[s])))
    return out, pd.DataFrame(st_rows), check


def problems(check: dict) -> list:
    """검사에서 멈춰야 할 문제 (D-105 3절 T4·T5, 2절 운행일 경계)."""
    p = []
    for k in ("time_reversal", "dup_station", "non_monotonic", "outside_service_day"):
        if check[k]:
            p.append(f"{k} {len(check[k])}: {check[k][:3]}")
    if check["dup_train"]:
        p.append(f"dup_train {check['dup_train']}")
    if check["t4_mismatch"]:
        p.append(f"T4 불일치 {check['t4_mismatch']}")
    if not check["t5_gj_weekday_gt_holiday"][2]:
        p.append(f"T5 평일 ≤ 휴일 {check['t5_gj_weekday_gt_holiday']}")
    return p


# ---------------------------------------------------------------------- 4-3 C1 B 등급 계수 보정 오차
FACTORS = (1.00, 1.60, 1.81, 2.78)
KEEP_MARGIN = 0.10           # 1.81 과 오차 차이가 이보다 작으면 1.81 유지


def _mmss(s) -> float:
    a, b = str(s).split(":")
    return int(a) + int(b) / 60


def validation_3b(evidence: pd.DataFrame) -> pd.DataFrame:
    """검증 자료 ①: 3-B A 등급 중 공식 시간이 있는 연결 + U 3쌍 (왕십리 2·5↔수인분당, 김포공항 5↔공항철도 방향 평균)."""
    e = evidence.copy()
    a = e[(e.grade == "A") & e.official_time.notna()]
    a = a[~((a.station == "왕십리") & (a.line_b == "SB"))]
    rows = [dict(set="3B", station=r.station, pair=f"{r.line_a}↔{r.line_b}", observed=float(r.minutes),
                 official=_mmss(r.official_time)) for r in a.itertuples()]
    u = {("왕십리", "2", "SB"): 99 / 1.2 / 60, ("왕십리", "5", "SB"): _mmss("03:23"), ("김포공항", "5", "AR"): 122 / 1.2 / 60}
    for (st, la, lb), off in u.items():
        g = e[(e.station == st) & (e.line_a == la) & (e.line_b == lb) & (e.grade == "U")]
        rows.append(dict(set="3B", station=st, pair=f"{la}↔{lb}", observed=float(g.minutes.mean()), official=off))
    return pd.DataFrame(rows)


def validation_18(v1_transfer: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """검증 자료 ②: 1~8호선 v1 환승 ↔ 공식 거리시간 (D-042 와 같은 짝짓기). v1_transfer: station_name·from_line·to_line·transfer_time_min."""
    d = official.copy()
    d["from_line"] = d["호선"].astype(str)
    d["to_line"] = d["환승노선"].astype(str).str.extract(r"^(\d)호선$")[0]
    d = d.dropna(subset=["to_line"])
    d["stn"] = d["환승역명"].astype(str).str.replace(r"역$", "", regex=True).map(lambda s: "서울역" if s == "서울" else s)
    m = v1_transfer.astype({"from_line": str, "to_line": str}).merge(
        d, left_on=["station_name", "from_line", "to_line"], right_on=["stn", "from_line", "to_line"])
    return pd.DataFrame(dict(set="1-8", station=m.station_name, pair=m.from_line + "↔" + m.to_line,
                             observed=m.transfer_time_min.astype(float), official=m["환승소요시간"].map(_mmss)))


def calibrate(val: pd.DataFrame) -> dict:
    """후보 계수별 절대 오차 중앙값. ①∪② 최소 → 기본값 (1.81 과 차이 < 0.10분이면 1.81)."""
    res = {}
    for name, g in [("all", val)] + list(val.groupby("set")):
        res[name] = {f"{f:.2f}": round(float((g.observed - g.official * f).abs().median()), 4) for f in FACTORS}
        res[name]["n"] = int(len(g))
    allv = {float(k): v for k, v in res["all"].items() if k != "n"}
    best = min(allv, key=lambda f: (allv[f], abs(f - 1.81)))
    chosen = 1.81 if allv[1.81] - allv[best] < KEEP_MARGIN else best
    ratio = (val.observed / val.official).replace([np.inf, -np.inf], np.nan)
    return {"errors": res, "best_by_error": best, "chosen": chosen,
            "rule": "①∪② 절대 오차 중앙값 최소, 1.81 과 차이 0.10분 미만이면 1.81 (D-105 4-3 C1)",
            "ratio_median": {k: round(float(r.median()), 3) for k, r in ratio.groupby(val.set)}}
