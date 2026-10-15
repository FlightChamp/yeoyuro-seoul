"""
scripts/v2/32_car_sample_validation.py
======================================
공모전 P0-1 — 서울교통공사 칸별 혼잡도 샘플(4호선)로 두 가지를 검증한다.

  A. 외부 정합성 : 앱의 기대 혼잡도(과거 분기 자료 중앙값) vs 샘플 실측 (구간×방향×요일×30분)
                   + 정원 100% 초과 판정의 혼동행렬, 오분류(FP/FN)의 100% 기준선과의 거리 분포
  B. 칸 추천     : 과거 자료로 '평균이 가장 낮은 칸'을 고르면 보지 않은 날·편성에서도 덜 붐비는가
                   - 하루 빼기(LODO)      : 평가 대상 = 그날 실제 열차 평균 60% 이상 피크 통과 (기존 결과 유지)
                   - 편성 하나 빼기(LOFO) : 같은 대상, 편성 단위 holdout
                   - 배포형 holdout        : 학습 기간의 패턴만으로 '혼잡 구간·시간'을 먼저 고른 뒤, 그날 통과 전부를 평가
                                            (오늘 데이터를 보기 전에 추천 대상을 정하는 실제 서비스 상황)

음수 혼잡도는 공식 의미가 확인되지 않아 세 가지 처리로 민감도를 본다: raw / clip0(0 미만→0) / missing(음수→결측).
계산·라우팅 로직은 바꾸지 않는다 (앱 RouteScorer 의 lookup 을 읽기만 한다).

실행
----
    python scripts/v2/32_car_sample_validation.py --src <샘플 zip 또는 CSV 폴더>
    (기본: data/raw/seoulmetro_car_sample)

출력
----
    reports/v2/car_sample_validation.json   전체 수치
    reports/v2/car_sample_validation.md     요약 표
    reports/v2/fig_car_external.png         기대 vs 실측 산점도 (평일, raw)
    reports/v2/fig_car_choice.png           칸 선택 효과 (검증 방식별)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app" / "streamlit"))

OUT = ROOT / "reports" / "v2"
ALIAS = {"당고개": "불암산", "총신대입구": "이수"}          # 샘플 역명 → 앱 그래프 역명 (개명·병기)
OUT_OF_SCOPE = {"오남", "별내별가람", "진접", "풍양"}       # 진접선: 서울교통공사 관할 밖, 앱 그래프에 없음
PEAK_BINS = ["07:00", "07:30", "08:00", "08:30", "18:00", "18:30"]
BUSY = 60.0                                                 # '혼잡 통과' 기준 (열차 평균 %)
TREATMENTS = ["raw", "clip0", "missing"]


# --------------------------------------------------------------------------- 데이터
def load_sample(src: Path) -> pd.DataFrame:
    from yeoyuro_v2.car_pattern import find_sample, read_sample   # 앱 패턴 생성과 같은 읽기 규칙
    try:
        d = read_sample(find_sample(src))
    except FileNotFoundError as e:
        raise SystemExit(f"[중단] {e}")
    d["t"] = pd.to_datetime(d["발생일자"])
    d["date"] = d.t.dt.date
    d["bin"] = d.t.dt.floor("30min").dt.strftime("%H:%M")
    dow = d.t.dt.dayofweek
    d["day_type"] = np.where(dow == 5, "saturday", np.where(dow == 6, "sunday", "weekday"))
    return d


def car_cols(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if "혼잡도" in c and "등급" not in c]


def treat(d: pd.DataFrame, pc: list[str], how: str) -> pd.DataFrame:
    x = d.copy()
    blank = pd.DataFrame({c: pd.to_numeric(x[c].astype(str).str.rstrip("%"), errors="coerce") for c in pc}).isna()
    x["_blank"] = blank.any(axis=1)          # 원본 파일에서 비어 있던 칸 (음수 처리와 무관)
    for c in pc:
        v = pd.to_numeric(x[c].astype(str).str.rstrip("%"), errors="coerce")
        if how == "clip0":
            v = v.clip(lower=0)
        elif how == "missing":
            v = v.where(v >= 0)
        x[c] = v
    x["mean"] = x[pc].mean(axis=1, skipna=True)
    return x


# --------------------------------------------------------------------------- A. 외부 정합성
def app_lookup():
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    rs = mod.RouteScorer(ROOT / "data" / "app_v26", "weekday", "08:00", "2026-06-17")
    r4 = rs.ride[rs.ride.line_id.astype(str) == "4"][["from_node", "to_node", "direction"]].copy()
    emap = {(sr.station_of(a), sr.station_of(b)): (a, dr) for a, b, dr in zip(r4.from_node, r4.to_node, r4.direction)}
    lk = rs.lookup.copy()
    lk["bin"] = lk.time_bin.astype(str).str[:5]
    return emap, lk[["station_uid", "direction", "day_type", "bin", "congestion_median"]]


def coverage(d: pd.DataFrame, emap: dict) -> dict:
    n = len(d)
    oos = d["현재역 명"].isin(OUT_OF_SCOPE) | d["다음역 명"].isin(OUT_OF_SCOPE)
    raw_key = pd.Series(list(zip(d["현재역 명"], d["다음역 명"])), index=d.index)
    ali_key = pd.Series(list(zip(d["현재역 명"].replace(ALIAS), d["다음역 명"].replace(ALIAS))), index=d.index)
    hit_raw = raw_key.map(lambda k: k in emap)
    hit_ali = ali_key.map(lambda k: k in emap)
    return {"rows": int(n), "compared": round(float(hit_ali.mean()), 4), "out_of_scope_jinjeop": round(float(oos.mean()), 4),
            "name_alias_recovered": round(float((hit_ali & ~hit_raw).mean()), 4),
            "other_unmatched": round(float((~hit_ali & ~oos).mean()), 4)}


def external(d: pd.DataFrame, emap: dict, lk: pd.DataFrame) -> dict:
    key = pd.Series(list(zip(d["현재역 명"].replace(ALIAS), d["다음역 명"].replace(ALIAS))), index=d.index)
    m = key.map(emap)
    x = d[m.notna()].copy()
    x["uid"] = m[m.notna()].str[0]
    x["dir"] = m[m.notna()].str[1]
    p = x.groupby(["편성번호", "date", "uid", "dir", "day_type", "bin"]).agg(obs=("mean", "median")).reset_index()
    p = p.merge(lk, left_on=["uid", "dir", "day_type", "bin"],
                right_on=["station_uid", "direction", "day_type", "bin"], how="inner")
    e = p.groupby(["uid", "dir", "day_type", "bin"]).agg(obs=("obs", "mean"), exp=("congestion_median", "first")).reset_index()
    out = {"by_day": {}}
    for dt in ["weekday", "saturday", "sunday"]:
        z = e[e.day_type == dt].dropna()
        out["by_day"][dt] = {"cells": len(z), "pearson": round(z.obs.corr(z.exp), 3),
                             "spearman": round(z.obs.corr(z.exp, method="spearman"), 3),
                             "mae": round((z.obs - z.exp).abs().mean(), 2), "bias": round((z.obs - z.exp).mean(), 2)}

    def cm(df, ecol, ocol):
        df = df.dropna(subset=[ecol, ocol])
        tp = int(((df[ecol] >= 100) & (df[ocol] >= 100)).sum()); fp = int(((df[ecol] >= 100) & (df[ocol] < 100)).sum())
        fn = int(((df[ecol] < 100) & (df[ocol] >= 100)).sum()); tn = int(((df[ecol] < 100) & (df[ocol] < 100)).sum())
        fpd = (100 - df.loc[(df[ecol] >= 100) & (df[ocol] < 100), ocol])          # 실측이 100 아래로 얼마나
        fnd = (100 - df.loc[(df[ecol] < 100) & (df[ocol] >= 100), ecol])          # 기대가 100 아래로 얼마나
        fnd_obs = (df.loc[(df[ecol] < 100) & (df[ocol] >= 100), ocol] - 100)      # 실측이 100 위로 얼마나

        def dist(s):
            if len(s) == 0:
                return None
            return {"n": len(s), "median": round(s.median(), 1), "p75": round(s.quantile(.75), 1),
                    "max": round(s.max(), 1), "share_within_10": round((s <= 10).mean(), 3),
                    "share_within_20": round((s <= 20).mean(), 3)}
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                "fp_obs_gap_below_100": dist(fpd), "fn_exp_gap_below_100": dist(fnd), "fn_obs_gap_above_100": dist(fnd_obs)}
    w = e[e.day_type == "weekday"]
    wp = p[p.day_type == "weekday"]
    out["threshold100_weekday_cell"] = cm(w, "exp", "obs")
    out["threshold100_weekday_passage"] = cm(wp, "congestion_median", "obs")
    out["_cells"] = e
    return out


# --------------------------------------------------------------------------- B. 칸 추천
KEY = ["현재역 명", "다음역 명", "bin"]          # 구간(방향 포함) × 30분


def passages(d: pd.DataFrame, pc: list[str], complete: bool = True) -> pd.DataFrame:
    """통과 = 편성 × 날짜 × 구간 × 30분, 칸별 중앙값.
    complete=True: 원본에서 빈 칸이 하나라도 있던 행은 쓰지 않는다 (10칸 비교의 공정성. 기본값, 최초 분석과 같은 정의)."""
    w = d[d.day_type == "weekday"]
    if complete:
        w = w[~w["_blank"]]
    p = w.groupby(["편성번호", "date"] + KEY)[pc].median().reset_index()
    p["mean"] = p[pc].mean(axis=1, skipna=True)
    need = 10 if complete else 5
    return p[p[pc].notna().sum(axis=1) >= min(need, len(pc)) if not complete else p[pc].notna().any(axis=1)].reset_index(drop=True)


def recommend(train: pd.DataFrame, pc: list[str]) -> dict:
    prof = train.groupby(KEY)[pc].mean()
    return {k: int(np.nanargmin(v)) for k, v in zip(prof.index, prof.values) if np.isfinite(v).any()}


def score(te: pd.DataFrame, rec: dict, pc: list[str], rng) -> list[dict]:
    rows = []
    for _, x in te.iterrows():
        k = tuple(x[KEY])
        if k not in rec:
            continue
        b = rec[k]
        v = x[pc].values.astype(float)
        if not np.isfinite(v[b]):
            continue
        ok = np.isfinite(v)
        vv = v[ok]
        r = rng.choice(np.where(ok)[0])
        rows.append({"train": x["편성번호"], "key": k, "rec": b, "gain": np.nanmean(v) - v[b],
                     "hit": v[b] <= np.median(vv), "rand_gain": np.nanmean(v) - v[r], "range": vv.max() - vv.min()})
    return rows


def evaluate(p: pd.DataFrame, pc: list[str], mode: str, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    peak = p[p.bin.isin(PEAK_BINS)]
    res = []
    if mode in ("lodo", "lofo"):
        col = "date" if mode == "lodo" else "편성번호"
        q = peak[peak["mean"] >= BUSY]                         # 기존 정의: 그날 실제 60% 이상 피크 통과
        for h in sorted(q[col].unique()):
            res += score(q[q[col] == h], recommend(q[q[col] != h], pc), pc, rng)
    elif mode == "deploy":
        # 학습 기간 자료만으로 '혼잡 구간·시간'을 먼저 정한다 → 그날 그 구간·시간 통과 전부를 평가
        for h in sorted(p.date.unique()):
            tr, te = p[p.date != h], p[p.date == h]
            busy_keys = set(tr.groupby(KEY)["mean"].mean().loc[lambda s: s >= BUSY].index)
            rec = {k: v for k, v in recommend(tr, pc).items() if k in busy_keys}
            res += score(te[[tuple(r) in busy_keys for r in te[KEY].values]], rec, pc, rng)
    r = pd.DataFrame(res)
    if r.empty:
        return r, {"n": 0}
    s = {"n": len(r), "hit_rate": round(r.hit.mean(), 3), "gain_mean": round(r.gain.mean(), 2),
         "gain_median": round(r.gain.median(), 2), "random_gain_mean": round(r.rand_gain.mean(), 2),
         "car_range_mean": round(r.range.mean(), 1), "keys": int(r.key.nunique()),
         "by_train": r.groupby("train").agg(n=("gain", "size"), hit=("hit", "mean"), gain=("gain", "mean"))
                      .round(3).reset_index().to_dict("records")}
    return r, s


def grade_fallback(d: pd.DataFrame, pc: list[str]) -> dict:
    """등급(1~4)만 개방된다고 가정한 하루 빼기 검증. 평가 대상은 LODO 와 같다 (그날 실제 60% 이상 피크 통과).
    학습 기간의 칸별 평균 '등급'이 가장 낮은 칸을 고르고, 모든 칸의 평균 등급이 같으면 추천하지 않는다.
    채점은 % 값으로 한다 (등급의 실제 효과를 보기 위해)."""
    gc = [c for c in d.columns if "등급" in c][:len(pc)]
    w = d[(d.day_type == "weekday") & (~d["_blank"])].copy()
    for c in gc:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[w[gc].notna().all(axis=1)]
    p = w.groupby(["편성번호", "date"] + KEY)[pc + gc].median().reset_index()
    p["mean"] = p[pc].mean(axis=1)
    q = p[p.bin.isin(PEAK_BINS) & (p["mean"] >= BUSY)]
    n_eval, rows = 0, []
    for h in sorted(q.date.unique()):
        tr, te = q[q.date != h], q[q.date == h]
        prof_g = tr.groupby(KEY)[gc].mean()
        prof_p = tr.groupby(KEY)[pc].mean()
        for _, x in te.iterrows():
            k = tuple(x[KEY])
            if k not in prof_g.index:
                continue
            n_eval += 1
            g = prof_g.loc[k].values.astype(float)
            if np.isclose(g.max(), g.min()):
                continue                                   # 등급으로는 구분 불가 → 추천하지 않음
            b = int(np.argmin(g)); bp = int(np.argmin(prof_p.loc[k].values.astype(float)))
            v = x[pc].values.astype(float)
            rows.append({"gain": v.mean() - v[b], "hit": v[b] <= np.median(v),
                         "gain_pct_same": v.mean() - v[bp], "hit_pct_same": v[bp] <= np.median(v)})
    r = pd.DataFrame(rows)
    if r.empty:
        return {"n_eval": n_eval, "n_recommended": 0}
    return {"n_eval": n_eval, "n_recommended": len(r), "coverage": round(len(r) / n_eval, 3),
            "hit_rate": round(r.hit.mean(), 3), "gain_mean": round(r.gain.mean(), 2),
            "pct_on_same_cases": {"hit_rate": round(r.hit_pct_same.mean(), 3), "gain_mean": round(r.gain_pct_same.mean(), 2)}}


# --------------------------------------------------------------------------- 실행
def main() -> int:
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "raw" / "seoulmetro_car_sample"))
    a = ap.parse_args()
    raw = load_sample(Path(a.src))
    pc = car_cols(raw)
    emap, lk = app_lookup()
    res = {"sample": {"rows": len(raw), "trains": int(raw["편성번호"].nunique()),
                      "dates": [str(raw.date.min()), str(raw.date.max())]},
           "coverage": coverage(raw, emap), "treatments": {}}
    vals = treat(raw, pc, "raw")[pc]
    res["sample"]["negative_share_of_car_values"] = round(float((vals < 0).values.mean()), 4)
    res["sample"]["le_minus20_share"] = round(float((vals <= -20).values.mean()), 5)
    recs = {}
    for how in TREATMENTS:
        d = treat(raw, pc, how)
        ext = external(d, emap, lk)
        cells = ext.pop("_cells")
        p = passages(d, pc)
        tr = {"external": ext}
        for mode in ["lodo", "lofo", "deploy"]:
            r, s = evaluate(p, pc, mode)
            tr[mode] = s
            if mode == "lodo":
                recs[how] = recommend(p[(p.bin.isin(PEAK_BINS)) & (p["mean"] >= BUSY)], pc)
        res["treatments"][how] = tr
        if how == "raw":
            pp = passages(d, pc, complete=False)
            res["raw_including_partial_car_rows"] = {m: evaluate(pp, pc, m)[1] for m in ["lodo", "lofo", "deploy"]}
            raw_cells, raw_lodo = cells, evaluate(p, pc, "lodo")[0]
            raw_dep = evaluate(p, pc, "deploy")[0]
            raw_lofo = evaluate(p, pc, "lofo")[0]
            res["grade_only_fallback_lodo"] = grade_fallback(d, pc)
    # 추천 칸 일치율 (전체 기간 프로필 기준, raw 대비)
    base = recs["raw"]
    res["agreement_vs_raw"] = {h: round(np.mean([recs[h].get(k) == v for k, v in base.items()]), 3) for h in TREATMENTS}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "car_sample_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    write_md(res)
    try:
        figures(raw_cells, raw_lodo, raw_lofo, raw_dep)
    except Exception as e:                                     # 그림 실패는 검증 통과와 무관
        print("[주의] 그림 생성 실패:", e)
    print_summary(res)
    return 0


def write_md(res: dict) -> None:
    L = ["# 칸별 혼잡도 샘플 검증 (P0-1)", "",
         f"샘플: {res['sample']['rows']:,}행, 편성 {res['sample']['trains']}개, {res['sample']['dates'][0]}~{res['sample']['dates'][1]}",
         "비교 범위: 앱 구간과 맞춘 행 {:.1%} · 진접선(앱 범위 밖) {:.1%} · 옛 역명 보정으로 맞춘 행 {:.1%} · 그 외 불일치 {:.1%}".format(
             res["coverage"]["compared"], res["coverage"]["out_of_scope_jinjeop"],
             res["coverage"]["name_alias_recovered"], res["coverage"]["other_unmatched"]), "",
         "| 처리 | 평일 Pearson | 평일 Spearman | 100% 정밀도(셀) | 100% 재현율(셀) | LODO 성공률 | LODO 감소 | LOFO 성공률 | LOFO 감소 | 배포형 성공률 | 배포형 감소 | 추천 칸 일치율 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h, t in res["treatments"].items():
        e = t["external"]
        L.append(f"| {h} | {e['by_day']['weekday']['pearson']} | {e['by_day']['weekday']['spearman']} | "
                 f"{e['threshold100_weekday_cell']['precision']} | {e['threshold100_weekday_cell']['recall']} | "
                 f"{t['lodo']['hit_rate']} | {t['lodo']['gain_mean']} | {t['lofo']['hit_rate']} | {t['lofo']['gain_mean']} | "
                 f"{t['deploy']['hit_rate']} | {t['deploy']['gain_mean']} | {res['agreement_vs_raw'][h]} |")
    g = res.get("grade_only_fallback_lodo") or {}
    if g.get("n_recommended"):
        L += ["", "등급만 개방될 때 (하루 빼기, raw): 평가 {n_eval}건 중 등급으로 칸을 구분할 수 있었던 {n_recommended}건({coverage:.0%})만 안내 · "
              "성공률 {hit_rate} · 감소 {gain_mean}%p (같은 건을 % 값으로 고르면 {p_hit} · {p_gain}%p). 나머지는 '구분 어려움'으로 표시."
              .format(p_hit=g["pct_on_same_cases"]["hit_rate"], p_gain=g["pct_on_same_cases"]["gain_mean"], **g)]
    (OUT / "car_sample_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def figures(cells, lodo, lofo, dep) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}      # 설치된 글꼴만 지정 → findfont 경고 없음
    ko = [n for n in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR") if n in have]
    plt.rcParams["font.family"] = ko[:1] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    w = cells[cells.day_type == "weekday"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(w.exp, w.obs, s=6, alpha=.35, color="#1F4E79")
    lim = [min(w.exp.min(), w.obs.min()) - 5, max(w.exp.max(), w.obs.max()) + 5]
    ax.plot(lim, lim, color="#9CA3AF", lw=1)
    ax.axhline(100, color="#D9534F", lw=.8, ls=":"); ax.axvline(100, color="#D9534F", lw=.8, ls=":")
    ax.set_xlabel("앱 기대 혼잡도 (과거 분기 중앙값, %)"); ax.set_ylabel("샘플 실측 열차 평균 (%)")
    ax.set_title("평일 · 구간×방향×30분 (4호선 샘플)")
    fig.tight_layout(); fig.savefig(OUT / "fig_car_external.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labs = ["하루 빼기", "편성 하나 빼기", "배포형"]
    g = [lodo.gain.mean(), lofo.gain.mean(), dep.gain.mean()]
    rg = [lodo.rand_gain.mean(), lofo.rand_gain.mean(), dep.rand_gain.mean()]
    xs = np.arange(3)
    ax.bar(xs - .18, g, .36, label="추천 칸", color="#1F4E79"); ax.bar(xs + .18, rg, .36, label="아무 칸", color="#9DB4CC")
    ax.set_xticks(xs, labs); ax.set_ylabel("열차 평균 대비 감소 (%p)"); ax.legend(frameon=False)
    ax.set_title("칸 선택 효과 (평일 피크, raw)")
    fig.tight_layout(); fig.savefig(OUT / "fig_car_choice.png", dpi=160); plt.close(fig)


def print_summary(res: dict) -> None:
    print("=" * 72)
    print("칸별 혼잡도 샘플 검증 (P0-1)  →", OUT / "car_sample_validation.md")
    print("비교 범위:", res["coverage"])
    print("음수 칸 값 비율:", res["sample"]["negative_share_of_car_values"])
    for h, t in res["treatments"].items():
        e = t["external"]; c = e["threshold100_weekday_cell"]
        print(f"[{h:7s}] 평일 r={e['by_day']['weekday']['pearson']} ρ={e['by_day']['weekday']['spearman']} "
              f"| 100% P={c['precision']} R={c['recall']} "
              f"| LODO {t['lodo']['hit_rate']}/{t['lodo']['gain_mean']}p "
              f"| LOFO {t['lofo']['hit_rate']}/{t['lofo']['gain_mean']}p "
              f"| 배포형 {t['deploy']['hit_rate']}/{t['deploy']['gain_mean']}p (n={t['deploy']['n']}) "
              f"| 일치 {res['agreement_vs_raw'][h]}")
    g = res.get("grade_only_fallback_lodo") or {}
    if g.get("n_recommended"):
        print(f"[등급만  ] 하루 빼기: 평가 {g['n_eval']}건 중 {g['n_recommended']}건({g['coverage']:.0%})만 칸 구분 가능 "
              f"| 성공률 {g['hit_rate']} · 감소 {g['gain_mean']}p (같은 건을 %로 고르면 "
              f"{g['pct_on_same_cases']['hit_rate']} · {g['pct_on_same_cases']['gain_mean']}p)")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
