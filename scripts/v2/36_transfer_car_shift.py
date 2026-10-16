"""
scripts/v2/36_transfer_car_shift.py
===================================
P1 보조분석 — 환승동선 인접 칸의 역 통과 전후 승하차 변화.
사전 등록: docs/v2/preregistration_transfer_car.md (이 코드 실행 전에 커밋)

    python scripts/v2/36_transfer_car_shift.py
    → reports/v2/transfer_car_report.md · transfer_car.json

주 가설: 빠른 환승 '하차 칸 ±1' 에서 역 통과 전후 혼잡 변화(중심화 Δ)가 다른 칸보다 더 음수다.
인과 주장 없음. 검증 대상은 환승동선이며, 엘리베이터 Facility PoC 와 별개의 보조 근거다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.car_pattern import ALIAS, find_sample, read_sample   # noqa: E402

TIP = ROOT / "data" / "marts" / "transfer_tip_mart.parquet"
OUT = ROOT / "reports" / "v2"
N_CARS = 10
WINDOW = 1
SEG_GAP_S = 120          # 같은 역간의 10초 행이 이보다 오래 끊기면 다른 통과로 본다
LINK_GAP_S = 600         # 도착 전 역간 끝 ~ 출발 후 역간 시작 허용 간격 (사전 등록: 10분)
MIN_COMBO = 5            # 역×방향 조합 집계 최소 통과 수
B, SEED = 2000, 20261016
PEAK = (("07:00", "09:00"), ("18:00", "19:30"))


def norm(name: str) -> str:
    s = str(name).replace("방면", "").strip()
    return ALIAS.get(s, s)


# ---------------------------------------------------------------- 1. 열차 통과
def segments(d: pd.DataFrame, pc: list[str]) -> pd.DataFrame:
    """10초 행 → 역간 1회 통과(세그먼트)별 칸 중앙값. 10초 행은 독립 표본으로 쓰지 않는다."""
    d = d.sort_values(["편성번호", "t"]).copy()
    key_change = (d["a"] != d["a"].shift()) | (d["b"] != d["b"].shift()) | (d["편성번호"] != d["편성번호"].shift())
    gap = d["t"].diff().dt.total_seconds().fillna(1e9) > SEG_GAP_S
    d["seg"] = (key_change | gap).cumsum()
    g = d.groupby("seg")
    s = g[pc].median()
    s["train"] = g["편성번호"].first()
    s["a"], s["b"] = g["a"].first(), g["b"].first()
    s["start"], s["end"] = g["t"].min(), g["t"].max()
    return s.reset_index(drop=True)


def station_passages(seg: pd.DataFrame, pc: list[str]) -> pd.DataFrame:
    """연속한 두 세그먼트 (이전역→S, S→다음역) = 역 S 통과 1회. Δ = 출발 후 − 도착 전 (칸별)."""
    seg = seg.sort_values(["train", "start"]).reset_index(drop=True)
    x, y = seg.iloc[:-1].reset_index(drop=True), seg.iloc[1:].reset_index(drop=True)
    ok = ((x.train == y.train) & (x.b == y.a) & (y.b != x.a)
          & ((y.start - x.end).dt.total_seconds().between(0, LINK_GAP_S)))
    x, y = x[ok].reset_index(drop=True), y[ok].reset_index(drop=True)
    delta = y[pc].values - x[pc].values
    out = pd.DataFrame(delta, columns=[f"d{i}" for i in range(1, N_CARS + 1)])
    out["train"], out["station"], out["prev"], out["next"] = x.train, x.b, x.a, y.b
    out["date"] = x.end.dt.date
    out["time"] = x.end.dt.strftime("%H:%M")
    return out


# ---------------------------------------------------------------- 2. 대상 칸
def target_cars(tip: pd.DataFrame, kind: str = "alight") -> dict[tuple[str, str], set[int]]:
    """(환승역, 다음역) → 칸 번호 집합 (±WINDOW 합집합). kind='alight' 는 primary, 'board' 는 탐색."""
    if kind == "alight":
        t = tip[tip.from_line.astype(str) == "4"]
        car, toward = "alight_car", "arrive_toward"
    else:
        t = tip[tip.to_line.astype(str) == "4"]
        car, toward = "board_car", "depart_toward"
    out: dict[tuple[str, str], set[int]] = {}
    for r in t.itertuples():
        c = pd.to_numeric(getattr(r, car), errors="coerce")
        if pd.isna(c):
            continue
        k = (norm(r.station_name), norm(getattr(r, toward)))
        out.setdefault(k, set()).update(x for x in range(int(c) - WINDOW, int(c) + WINDOW + 1) if 1 <= x <= N_CARS)
    return out


def effects(ps: pd.DataFrame, targets: dict, mirrored: bool = False) -> pd.DataFrame:
    """통과별 e = mean(중심화 Δ, 대상 칸) − mean(중심화 Δ, 나머지 칸)."""
    D = ps[[f"d{i}" for i in range(1, N_CARS + 1)]].values.astype(float)
    Dc = D - D.mean(axis=1, keepdims=True)
    rows = []
    for i, r in enumerate(ps[["station", "next", "date", "time", "train"]].itertuples(index=False)):
        tg = targets.get((r.station, r.next))
        if not tg:
            continue
        cars = {N_CARS + 1 - c for c in tg} if mirrored else set(tg)
        m = np.zeros(N_CARS, bool)
        m[[c - 1 for c in cars]] = True
        if m.all() or not m.any():
            continue
        rows.append({"station": r.station, "next": r.next, "date": r.date, "time": r.time, "train": r.train,
                     "n_target": int(m.sum()), "e": float(Dc[i, m].mean() - Dc[i, ~m].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. 통계
def hier_boot(e: pd.DataFrame, b: int = B, seed: int = SEED) -> tuple[float, float]:
    """날짜 → 열차 통과 2단계 복원추출로 pooled 평균의 95% CI."""
    rng = np.random.default_rng(seed)
    by_date = [g.e.values for _, g in e.groupby("date")]
    stats = np.empty(b)
    for k in range(b):
        picks = rng.integers(0, len(by_date), len(by_date))
        vals = [by_date[j][rng.integers(0, len(by_date[j]), len(by_date[j]))] for j in picks]
        stats[k] = np.concatenate(vals).mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(e: pd.DataFrame) -> dict:
    if e.empty:
        return {"n_passages": 0}
    lo, hi = hier_boot(e)
    c = e.groupby(["station", "next"]).e.agg(["mean", "size"]).reset_index()
    c = c[c["size"] >= MIN_COMBO]
    neg = int((c["mean"] < 0).sum())
    return {"n_passages": int(len(e)), "n_dates": int(e.date.nunique()), "pooled": round(float(e.e.mean()), 2),
            "ci95": [round(lo, 2), round(hi, 2)], "combos": int(len(c)), "combos_negative": neg,
            "pass": bool(hi < 0 and len(c) > 0 and neg >= len(c) / 2),
            "by_combo": [{"station": r.station, "next": r.next, "mean": round(float(r.mean), 2), "n": int(r.size)}
                         for r in c.itertuples()]}


def in_peak(t: str) -> bool:
    return any(a <= t < b for a, b in PEAK)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raw = read_sample(find_sample(ROOT / "data" / "raw" / "seoulmetro_car_sample"))
    pc = [c for c in raw.columns if "혼잡도" in c and "등급" not in c][:N_CARS]
    v = pd.DataFrame({c: pd.to_numeric(raw[c].astype(str).str.rstrip("%"), errors="coerce") for c in pc})
    d = raw[v.notna().all(axis=1)].copy()                     # 원본 빈 칸 행 제외
    d[pc] = v.loc[d.index]
    d["t"] = pd.to_datetime(d["발생일자"])
    d = d[d.t.dt.dayofweek < 5]                               # 평일
    d["a"], d["b"] = d["현재역 명"].map(norm), d["다음역 명"].map(norm)

    seg = segments(d, pc)
    ps = station_passages(seg, pc)
    tip = pd.read_parquet(TIP)
    alight, board = target_cars(tip, "alight"), target_cars(tip, "board")

    res = {"prereg": "docs/v2/preregistration_transfer_car.md",
           "data": {"segments": int(len(seg)), "station_passages_all": int(len(ps)),
                    "transfer_station_directions": sorted(f"{s}→{n}" for s, n in alight)}}
    for name, tg, mir in (("primary_alight", alight, False), ("mirror_alight", alight, True),
                          ("explore_board", board, False), ("explore_board_mirror", board, True)):
        e = effects(ps, tg, mir)
        res[name] = summarize(e)
        if name == "primary_alight":
            pk = e[e.time.map(in_peak)]
            res["explore_primary_peak_only"] = summarize(pk)
    p, m = res["primary_alight"], res["mirror_alight"]
    res["verdict"] = {"primary_pass": p.get("pass", False), "mirror_also_passes": m.get("pass", False)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transfer_car.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(res)
    print("=" * 72)
    for k in ("primary_alight", "mirror_alight", "explore_primary_peak_only", "explore_board", "explore_board_mirror"):
        s = res[k]
        if s.get("n_passages"):
            print(f"[{k:26s}] 통과 {s['n_passages']:4d} · pooled {s['pooled']:+.2f}%p · 95% CI [{s['ci95'][0]:+.2f}, "
                  f"{s['ci95'][1]:+.2f}] · 음수 조합 {s['combos_negative']}/{s['combos']}")
    v = res["verdict"]
    print(f"판정: primary {'통과' if v['primary_pass'] else '미통과'} · mirror 도 통과 여부 {v['mirror_also_passes']}")
    print("저장:", OUT / "transfer_car_report.md")
    print("=" * 72)
    return 0


def write_md(res: dict) -> None:
    p, m, v = res["primary_alight"], res["mirror_alight"], res["verdict"]
    L = ["# 환승동선 인접 칸의 역 통과 전후 승하차 변화 (P1 보조분석)", "",
         f"- 사전 등록: `{res['prereg']}` (실행 전 커밋). 결과와 무관하게 이 보고서를 남긴다.",
         "- 단위: 열차 통과 1회 = (편성, 날짜, 환승역, 이전역→다음역). 10초 행은 역간마다 칸별 중앙값으로 압축.",
         "- 지표: 중심화 Δ(출발 후 − 도착 전, 열차 평균 Δ 제거)의 '하차 칸 ±1' 평균 − 나머지 칸 평균. 하차 집중이면 음수.",
         "- CI: 날짜 → 열차 통과 2단계 hierarchical bootstrap (B=2000).", "",
         f"## 판정: {'경향 관측 (기획서 한 문장 추가 가능)' if v['primary_pass'] else '기준 미달 (기획서에 추가하지 않음)'}", "",
         "| 분석 | 통과 수 | pooled (%p) | 95% CI | 음수 조합 / 집계 조합 | 판정 조건 충족 |", "|---|---|---|---|---|---|"]
    for k, label in (("primary_alight", "Primary · 하차 칸 · 진행 방향 기준"), ("mirror_alight", "대조 · 하차 칸 · 반전 기준"),
                     ("explore_primary_peak_only", "탐색 · Primary 피크만"), ("explore_board", "탐색 · 승차 칸 · 진행 방향"),
                     ("explore_board_mirror", "탐색 · 승차 칸 · 반전")):
        s = res[k]
        if s.get("n_passages"):
            L.append(f"| {label} | {s['n_passages']} | {s['pooled']:+.2f} | [{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}] | "
                     f"{s['combos_negative']} / {s['combos']} | {'예' if s['pass'] else '아니오'} |")
    L += ["", "역×방향별 (Primary, 통과 5회 이상):", "", "| 환승역 | 다음역 | 평균 (%p) | 통과 수 |", "|---|---|---|---|"]
    L += [f"| {c['station']} | {c['next']} | {c['mean']:+.2f} | {c['n']} |" for c in p.get("by_combo", [])]
    pk = res.get("explore_primary_peak_only", {})
    if pk.get("n_passages"):
        L += ["", f"탐색(피크 한정) 결과 pooled {pk['pooled']:+.2f}%p, 95% CI [{pk['ci95'][0]:+.2f}, {pk['ci95'][1]:+.2f}] 는 "
              "사전 등록상 판정에 쓰지 않는다. 탐색 분석 4개 중 하나이므로 우연일 가능성을 배제할 수 없다."]
    L += ["", "해석 원칙: 인과 주장 없음. 칸 번호 공식 기준을 이 결과로 확정하지 않는다. "
          "탐색 분석(피크·승차 칸)은 판정에 쓰지 않는다. 승차 칸은 빠른 환승 표에서 방향과 무관하게 같은 값이 반복돼 품질 문제가 있다."]
    (OUT / "transfer_car_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
