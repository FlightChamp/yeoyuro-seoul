"""
yeoyuro_v2/car_pattern.py
=========================
공모전 P0-2 — '여유 칸 안내' 시범(PoC) 로직.

서울교통공사 칸별 혼잡도 샘플(4호선, 2026.6, 평일)에서
  구간(현재역→다음역, 방향 포함) × 30분 × 칸  평균을 만들고,
경로 결과의 승차 구간마다 "평소 가장 여유로운 칸"을 고른다.

검증: scripts/v2/32_car_sample_validation.py (하루 빼기 75.8% / 10.6%p, 배포형 72.4% / 8.3%p).
원칙
- 과거 패턴 기반 시범 기능이다. 그날 그 열차의 실시간 칸 혼잡이 아니다 (화면에 항상 표기).
- 평가와 같은 정의를 쓴다: 원본 빈 칸이 있는 행 제외, 평일, 열차 평균 60% 이상 구간·시간에서만 안내.
- 이 모듈은 Streamlit 을 모른다. 라우팅·혼잡 계산에는 관여하지 않는다.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ALIAS = {"당고개": "불암산", "총신대입구": "이수"}      # 샘플 역명 → 앱 역명
BUSY = 60.0             # 안내 대상: 이 구간·시간의 평소 열차 평균이 60% 이상
MIN_PASSAGES = 3        # 칸 평균에 쓴 통과 수 하한
MIN_DAYS = 2            # 서로 다른 날짜 수 하한
N_CARS = 10
CROWDED = 100.0         # 추천 칸조차 평소 정원 100% 이상이면 출발 시각 대안을 함께 권한다


def read_sample(src: Path) -> pd.DataFrame:
    """샘플 읽기. src = zip 파일, 또는 폴더(안의 CSV — 하위 폴더 포함 — 를 읽고, CSV 가 없으면 폴더 안 zip 을 읽는다)."""
    src = Path(src)

    def from_zip(zp: Path) -> list[pd.DataFrame]:
        with zipfile.ZipFile(zp) as z:
            return [pd.read_csv(io.BytesIO(z.read(n)), encoding="utf-8-sig")
                    for n in sorted(z.namelist()) if n.lower().endswith(".csv")]

    frames: list[pd.DataFrame] = []
    if src.is_file() and src.suffix.lower() == ".zip":
        frames = from_zip(src)
    elif src.is_dir():
        csvs = sorted(src.rglob("*.csv"))
        if csvs:
            frames = [pd.read_csv(p, encoding="utf-8-sig") for p in csvs]
        else:
            for zp in sorted(src.glob("*.zip")):
                frames += from_zip(zp)
    if not frames:
        raise FileNotFoundError(f"샘플 CSV 를 찾지 못했습니다: {src}  (zip 파일을 이 폴더에 넣어 주세요)")
    return pd.concat(frames, ignore_index=True)


_SAMPLE_ENTRY = __import__("re").compile(r"(^|/)\d{3}_2026\d{4}\.csv$")


def find_sample(folder: Path) -> Path:
    """샘플 위치 찾기. folder 에 CSV/zip 이 있으면 folder, 없으면 다운로드 폴더에서
    '편성번호_날짜.csv'(예: 404_20260617.csv) 들이 든 zip 을 찾아 folder 로 복사한다."""
    import shutil
    folder = Path(folder)
    if folder.is_file() or (folder.is_dir() and (any(folder.rglob("*.csv")) or any(folder.glob("*.zip")))):
        return folder
    dl = Path.home() / "Downloads"
    cands = []
    for zp in sorted(dl.glob("*.zip"), key=lambda q: q.stat().st_mtime, reverse=True) if dl.is_dir() else []:
        try:
            with zipfile.ZipFile(zp) as z:
                if sum(bool(_SAMPLE_ENTRY.search(n)) for n in z.namelist()) >= 3:
                    cands.append(zp)
        except (zipfile.BadZipFile, OSError):
            continue
    if not cands:
        raise FileNotFoundError(f"샘플 zip 을 찾지 못했습니다. 공모전 칸별 혼잡도 샘플 zip 을 {folder} 에 넣어 주세요")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cands[0], folder / cands[0].name)
    print(f"[샘플] 다운로드 폴더의 {cands[0].name} 을(를) {folder} 로 복사했습니다")
    return folder


def build_pattern(raw: pd.DataFrame, line: str = "4") -> pd.DataFrame:
    """칸별 평균 패턴 표. 열: line, from_station, to_station, time_bin, car_1..car_10, train_mean, n_passages, n_days."""
    pc = [c for c in raw.columns if "혼잡도" in c and "등급" not in c][:N_CARS]
    d = raw.copy()
    vals = pd.DataFrame({c: pd.to_numeric(d[c].astype(str).str.rstrip("%"), errors="coerce") for c in pc})
    d = d[vals.notna().all(axis=1)].copy()                 # 원본 빈 칸 있는 행 제외 (검증과 같은 정의)
    d[pc] = vals.loc[d.index]
    d["t"] = pd.to_datetime(d["발생일자"])
    d = d[d.t.dt.dayofweek < 5]                            # 평일
    d["date"] = d.t.dt.date
    d["time_bin"] = d.t.dt.floor("30min").dt.strftime("%H:%M")
    d["from_station"] = d["현재역 명"].replace(ALIAS)
    d["to_station"] = d["다음역 명"].replace(ALIAS)
    p = d.groupby(["편성번호", "date", "from_station", "to_station", "time_bin"])[pc].median().reset_index()
    g = p.groupby(["from_station", "to_station", "time_bin"])
    out = g[pc].mean()
    out.columns = [f"car_{i + 1}" for i in range(len(pc))]
    out["n_passages"] = g.size()
    out["n_days"] = g["date"].nunique()
    out["train_mean"] = out[[f"car_{i + 1}" for i in range(len(pc))]].mean(axis=1)
    out = out.reset_index()
    out.insert(0, "line", str(line))
    return out[(out.n_passages >= MIN_PASSAGES) & (out.n_days >= MIN_DAYS)].reset_index(drop=True)


def _bin(minute_of_day: float) -> str:
    m = int(minute_of_day // 30 * 30) % (24 * 60)
    return "%02d:%02d" % (m // 60, m % 60)


def segment_hint(pattern: pd.DataFrame | None, line: str, stations: list[str], board_min: float) -> dict | None:
    """승차 구간 하나의 여유 칸 안내. 안내할 근거가 없으면 None.

    stations : 그 구간의 역 순서 (승차역 … 하차역)
    board_min: 승차 시각(자정 기준 분). 구간 안의 모든 역간을 같은 30분 시간대로 본다 (시범 단순화).
    반환: {"cars": [추천 칸 번호 2개], "gain": 열차 평균 대비 %p, "train_mean": %, "n_edges": 근거 역간 수, "time_bin": "08:00"}
    """
    if pattern is None or pattern.empty or len(stations) < 2:
        return None
    sub = pattern[(pattern.line.astype(str) == str(line)) & (pattern.time_bin == _bin(board_min))]
    if sub.empty:
        return None
    idx = sub.set_index(["from_station", "to_station"])
    cols = [c for c in pattern.columns if c.startswith("car_")]
    rows = [idx.loc[(a, b), cols].values.astype(float) for a, b in zip(stations, stations[1:]) if (a, b) in idx.index]
    if not rows:
        return None
    m = np.vstack(rows).mean(axis=0)                     # 구간 전체에 걸친 칸별 평균
    train_mean = float(m.mean())
    if train_mean < BUSY:
        return None                                      # 붐비지 않는 구간·시간은 안내하지 않는다 (검증 대상과 같은 조건)
    order = np.argsort(m)
    return {"cars": [int(order[0]) + 1, int(order[1]) + 1], "gain": round(train_mean - float(m[order[0]]), 1),
            "train_mean": round(train_mean, 1), "best": round(float(m[order[0]]), 1),
            "all_crowded": bool(float(m[order[0]]) >= CROWDED),
            "n_edges": len(rows), "time_bin": _bin(board_min)}


def route_hints(pattern: pd.DataFrame | None, segments: list[dict], depart_min: float) -> dict[int, dict]:
    """경로의 승차 구간 번호 → 안내. 승차 시각 = 출발 + 앞선 구간의 승차·환승·대기 시간 (최초 대기는 포함하지 않음)."""
    out, t = {}, float(depart_min)
    for i, sg in enumerate(segments or []):
        if sg.get("kind") == "ride":
            h = segment_hint(pattern, str(sg.get("line")), list(sg.get("stations") or []), t)
            if h:
                out[i] = h
            t += float(sg.get("minutes") or 0)
        else:
            t += float(sg.get("minutes") or 0) + float(sg.get("wait_min") or 0)
    return out


# ---------------------------------------------------------------- 등급만 개방될 때 (설계 + 샘플 시험)
def build_grade_pattern(raw: pd.DataFrame, line: str = "4") -> pd.DataFrame:
    """칸별 '등급'(1~4)만 있을 때의 패턴 표: 구간 × 30분 × 칸의 과거 평균 등급. 열 구조는 build_pattern 과 같다."""
    gc = [c for c in raw.columns if "등급" in c][:N_CARS]
    d = raw.copy()
    g = pd.DataFrame({c: pd.to_numeric(d[c], errors="coerce") for c in gc})
    d = d[g.notna().all(axis=1)].copy()
    d[gc] = g.loc[d.index]
    d["t"] = pd.to_datetime(d["발생일자"])
    d = d[d.t.dt.dayofweek < 5]
    d["date"] = d.t.dt.date
    d["time_bin"] = d.t.dt.floor("30min").dt.strftime("%H:%M")
    d["from_station"] = d["현재역 명"].replace(ALIAS)
    d["to_station"] = d["다음역 명"].replace(ALIAS)
    p = d.groupby(["편성번호", "date", "from_station", "to_station", "time_bin"])[gc].median().reset_index()
    gr = p.groupby(["from_station", "to_station", "time_bin"])
    out = gr[gc].mean()
    out.columns = [f"car_{i + 1}" for i in range(len(gc))]
    out["n_passages"] = gr.size()
    out["n_days"] = gr["date"].nunique()
    out = out.reset_index()
    out.insert(0, "line", str(line))
    return out[(out.n_passages >= MIN_PASSAGES) & (out.n_days >= MIN_DAYS)].reset_index(drop=True)


def car_advice(pct_hint: dict | None, grade_vec=None, tol: float = 1e-9) -> dict:
    """칸 안내 단계 (정직한 fallback).

    1) % 패턴 안내가 있으면 그대로                         → level 'pct'
    2) 등급만 있고 칸 사이 등급 차이가 있으면 가장 낮은 칸  → level 'grade' (제한적 안내)
    3) 등급이 모두 같으면 억지로 고르지 않는다              → level 'none' + 문구
    grade_vec: 그 구간·시간의 칸별 (과거 평균) 등급 10개, 없으면 None
    """
    if pct_hint:
        return {"level": "pct", **pct_hint}
    if grade_vec is not None:
        v = np.asarray(grade_vec, dtype=float)
        if v.size and np.isfinite(v).all() and (v.max() - v.min()) > tol:
            best = np.flatnonzero(np.isclose(v, v.min())) + 1
            return {"level": "grade", "cars": [int(c) for c in best[:2]]}
    return {"level": "none", "message": "현재 데이터로는 칸 사이의 의미 있는 차이를 구분하기 어렵습니다"}
