"""
scripts/v2/50_check_late_night.py — 심야 운행일 일관화 검증 T3·T4·T5·T7·T8 (D-103, 사전 등록 10-3절)
================================================================================================

| T | 이 스크립트가 하는 일 |
|---|---|
| T8 | v2.3 OD 표본 412쌍 × 심야 입력 12개(토·일·월 00:00·00:30·01:00·01:30) — 기존 1~9호선 카드를 **전(달력 요일·입력 시각)** 과 **후(운행일 요일·연속 시각·운행일 날짜)** 로 계산해 요일·시간대·경로·소요·최대 혼잡 변경 건수와 원인 |
| T3 | 같은 412쌍 × 비심야 입력 6개(평일·토·일 08:00·18:00)에서 전 = 후 (인자가 같으므로 결과도 같아야 함) |
| T4 | 확장 카드: K 표본 412쌍 × 5개 시각(05:30·08:00·18:00·23:30·24:30)을 다시 계산해 D-101 검증 저장본(W2.parquet)과 상태·확정 경로·열차번호 비교 (저장본 없으면 건너뜀) |
| T5 | 심야 입력에서 두 카드가 쓰는 운행일·요일 유형이 같은지 |
| T7 | 앱 실행: 구현 전 앱(D-103 사전 등록 커밋)과 지금 앱을 끔 상태로 비교 — 비심야 사례는 같아야, 심야 사례는 차이를 보고. 켬/끔에서 기존 카드가 같은지 |

실행:  python scripts/v2/50_check_late_night.py --workers 8     (사용자 PC 예상 10~20분, 미확인)
       python scripts/v2/50_check_late_night.py --dry          (작은 표본 연습, 저장 안 함)
산출:  reports/v2/late_night_3a2e.{md,json}, data/marts/v2/late_night_3a2e/*.parquet (git 무시)
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit"), str(ROOT / "tests")]
from yeoyuro_v2 import service_day as SD          # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"
OUT = ROOT / "data" / "marts" / "v2" / "late_night_3a2e"
REP = ROOT / "reports" / "v2"
PREREG_SUBJECT = "Phase 3A-2e D-103 사전 등록"
LATE_INPUTS = [(d, t) for d in ("2026-11-28", "2026-11-29", "2026-11-30") for t in ("00:00", "00:30", "01:00", "01:30")]
DAY_INPUTS = [(d, t) for d in ("2026-11-25", "2026-11-28", "2026-11-29") for t in ("08:00", "18:00")]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def old_args(date: str, hhmm: str):
    """D-103 전 앱: 요일 = 입력 달력 날짜, 시각·날짜 = 입력 그대로."""
    return SD.day_type_of(dt.date.fromisoformat(date)), hhmm, date


def new_args(date: str, hhmm: str):
    c = SD.service_clock(date, hhmm)
    return c.day_type, c.hhmm, str(c.service_date)


# ---------------------------------------------------------------------- 기존 카드 계산 (앱과 같은 경로)
_C: dict = {}


def _mods():
    if "m" not in _C:
        import station_routing as sr
        from yeoyuro_v2.line9 import register_v1_directions
        mod = sr.load_scorer_class(ROOT)
        register_v1_directions(mod)
        _C["m"] = (mod, sr, sr.load_display_master(DATA9), sr.load_headway(DATA9))
    return _C["m"]


def _scorer(args):
    if args not in _C:
        from yeoyuro_v2.time_dependent import TimeDependentEvaluator
        mod, sr, disp, hw = _mods()
        rs = mod.RouteScorer(DATA9, *args)
        _C[args] = (rs, TimeDependentEvaluator(rs, hw, "step"))
    return _C[args]


def card(o, d, args, mode="calm") -> dict:
    from yeoyuro_v2.app_bridge import retime_result
    from yeoyuro_v2.time_dependent import parse_hhmm
    mod, sr, disp, hw = _mods()
    rs, ev = _scorer(args)
    res = sr.find_route_by_station(rs, disp, o, d, mode)
    if not res.get("ok"):
        return {"ok": False, "day_type": rs.day_type, "time_bin": rs.time_bin}
    res = retime_result(res, lambda b: ev, parse_hhmm(args[1]), mode)
    r = res["recommended"]
    return {"ok": True, "day_type": rs.day_type, "time_bin": rs.time_bin, "path": "|".join(r["path"]),
            "actual": round(float(r["actual_time_min"]), 3), "maxc": round(float(r.get("max_congestion") or 0), 3),
            "exp100": round(float(r.get("exposure_100_min") or 0), 3)}


def t8_task(args):
    o, d, inputs = args
    rows = []
    for date, hm in inputs:
        a, b = old_args(date, hm), new_args(date, hm)
        try:
            ca, cb = card(o, d, a), card(o, d, b)
        except Exception as e:                      # noqa: BLE001
            rows.append({"o": o, "d": d, "date": date, "hhmm": hm, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        row = {"o": o, "d": d, "date": date, "hhmm": hm, "same_args": a == b}
        row.update({f"old_{k}": v for k, v in ca.items()})
        row.update({f"new_{k}": v for k, v in cb.items()})
        rows.append(row)
    return rows


def run_pool(fn, jobs, workers, label):
    rows, t0 = [], time.time()
    if workers <= 1:
        for j in jobs:
            rows += fn(j)
        return pd.DataFrame(rows)
    with Pool(workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(fn, jobs, chunksize=1), 1):
            rows += rs
            if i % max(1, len(jobs) // 10) == 0:
                print(f"    {label} {i}/{len(jobs)} ({time.time() - t0:.0f}초)", flush=True)
    return pd.DataFrame(rows)


def analyze_cards(df: pd.DataFrame) -> dict:
    ok = df[df.get("error").isna()] if "error" in df else df
    out = {"rows": int(len(df)), "errors": int(len(df) - len(ok))}
    if not len(ok):
        return out
    both = ok[ok.old_ok.fillna(False) & ok.new_ok.fillna(False)]
    def ch(c):
        return (both[f"old_{c}"] != both[f"new_{c}"])
    out.update({
        "both_ok": int(len(both)), "ok_changed": int((ok.old_ok != ok.new_ok).sum()),
        "day_type_changed": int(ch("day_type").sum()), "time_bin_changed": int(ch("time_bin").sum()),
        "path_changed": int(ch("path").sum()), "actual_changed": int(ch("actual").sum()),
        "maxc_changed": int(ch("maxc").sum()), "exp100_changed": int(ch("exp100").sum()),
        "any_changed": int((ch("path") | ch("actual") | ch("maxc") | ch("exp100")).sum()),
        "old_maxc_zero": int((both.old_maxc == 0).sum()), "new_maxc_zero": int((both.new_maxc == 0).sum()),
        "maxc_old_mean": float(both.old_maxc.mean()), "maxc_new_mean": float(both.new_maxc.mean()),
        "actual_diff_median": float((both.new_actual - both.old_actual).median()),
        "actual_diff_absmax": float((both.new_actual - both.old_actual).abs().max()),
        "by_input": {f"{k[0]} {k[1]}": {"n": int(len(g)), "path_changed": int((g.old_path != g.new_path).sum()),
                                        "old": f"{g.old_day_type.iloc[0]} {g.old_time_bin.iloc[0]}",
                                        "new": f"{g.new_day_type.iloc[0]} {g.new_time_bin.iloc[0]}",
                                        "maxc_old_mean": round(float(g.old_maxc.mean()), 2),
                                        "maxc_new_mean": round(float(g.new_maxc.mean()), 2)}
                     for k, g in both.groupby(["date", "hhmm"])},
        "examples": both[ch("path")].head(5)[["o", "d", "date", "hhmm", "old_path", "new_path", "old_actual", "new_actual"]]
        .to_dict("records")})
    return out


# ---------------------------------------------------------------------- T4 확장 카드
def run_t4(workers, dry) -> dict:
    f = ROOT / "data" / "marts" / "v2" / "korail_app_3a2d" / "W2.parquet"
    if not f.exists():
        return {"skipped": "D-101 검증 저장본 W2.parquet 없음"}
    m49 = _load("m49", "scripts/v2/49_check_korail_app.py")
    e47 = m49._e47()
    ods = [(o, d) for o, d, _ in e47.k_sample(e47.router(e47.spec(on=True)).display.station_key.unique())]
    times = [330, 480, 1080, 1410, 1470]
    if dry:
        ods = ods[:5]
    new = run_pool(m49.w2_task, [(o, d, times) for o, d in ods], 1 if dry else workers, "T4")
    old = pd.read_parquet(f)
    k = ["o", "d", "t"]
    cols = ["status", "best_status", "best_path", "best_trains"]
    mm = new[k + cols].merge(old[k + cols], on=k, suffixes=("_new", "_old"))
    diff = mm[(mm[[c + "_new" for c in cols]].astype(str).values != mm[[c + "_old" for c in cols]].astype(str).values).any(axis=1)]
    return {"compared": int(len(mm)), "different": int(len(diff)), "examples": diff.head(5).to_dict("records"),
            "best_not_connected": int((new.best_status.notna() & (new.best_status != "연결 확인")).sum())}


# ---------------------------------------------------------------------- T5
def run_t5() -> dict:
    from yeoyuro_v2 import korail_app as KA
    rows = []
    for date, hm in LATE_INPUTS + DAY_INPUTS:
        a = new_args(date, hm)
        b = KA.service_day(date, hm)
        rows.append({"input": f"{date} {hm}", "base_card": f"{a[2]} {a[0]} {a[1]}",
                     "extension": f"{b.date} {b.day_type} {int(b.minute)}", "same": a[0] == b.day_type and a[2] == str(b.date)})
    return {"cases": len(rows), "same": sum(r["same"] for r in rows), "rows": rows}


# ---------------------------------------------------------------------- T7 앱
def _before_app() -> Path:
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-F", f"--grep={PREREG_SUBJECT}"],
                         capture_output=True, text=True, encoding="utf-8").stdout.split()
    if not sha:
        raise SystemExit(f"'{PREREG_SUBJECT}' 커밋을 찾지 못했습니다")
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha[0]}:app/streamlit/yeoyuro_seoul_app.py"],
                         capture_output=True, encoding="utf-8").stdout
    p = APP.parent / "_w1_before_app.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return p


def run_t7(dry) -> dict:
    m49 = _load("m49b", "scripts/v2/49_check_korail_app.py")
    day = [c for c in m49.W1_CASES if c[3] not in ("00:30",)]
    late = [("신림", "강남", "2026-11-28", "00:30", "calm"), ("노량진", "온수", "2026-11-28", "00:30", "calm"),
            ("상봉", "건대입구", "2026-11-30", "00:10", "calm")]
    if dry:
        day, late = day[:2], late[:1]
    before = _before_app()
    out = {"day_cases": len(day), "day_same": 0, "day_diffs": [], "late": []}
    try:
        for c in day:
            a, b = m49._at(before), m49._at(APP)
            m49._search(a, *c)
            m49._search(b, *c)
            if m49._screen(a) == m49._screen(b):
                out["day_same"] += 1
            else:
                sa, sb = m49._screen(a), m49._screen(b)
                out["day_diffs"].append({"case": " ".join(c), "keys": [k for k in sa if sa[k] != sb[k]]})
        for c in late:
            a, b = m49._at(before), m49._at(APP)
            m49._search(a, *c)
            m49._search(b, *c)
            on = m49._at(APP)
            on.toggle(key="ext_on").set_value(True)
            on.run()
            m49._search(on, *c)
            ext = {"기대 소요", "시각표 확인 도착", "1~9호선만 이용 시"}
            mb = [(m.label, m.value, m.delta or "") for m in b.metric]
            mo = [(m.label, m.value, m.delta or "") for m in on.metric if m.label not in ext]
            out["late"].append({"case": " ".join(c),
                                "before_metrics": [(m.label, m.value) for m in a.metric][:4],
                                "after_metrics": mb[:4],
                                "label_shown": any("운행일 기준" in x.value for x in b.caption),
                                "on_off_base_same": mb == mo,
                                "exceptions": [e.value[:200] for e in a.exception] + [e.value[:200] for e in b.exception]
                                + [e.value[:200] for e in on.exception]})
    finally:
        before.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------- 보고서
def write_report(S: dict, dest: Path) -> None:
    t8, t3, t4, t5, t7 = S["T8"], S["T3"], S["T4"], S["T5"], S["T7"]
    L = ["# Phase 3A-2e 심야 운행일 일관화 검증 — D-103", "",
         "- 규칙: [사전 등록 문서 10절](../../docs/v2/preregistration_3a2.md) D-103 (구현 전 고정) · 구현 기록 D-103a",
         "- 재현: `python scripts/v2/50_check_late_night.py --workers 8`", "",
         "| T | 검증 | 결과 | 기준 |", "|---|---|---|---|",
         f"| T3 | 비심야 불변 (412쌍 × {len(DAY_INPUTS)}개 입력) | 인자 같음 {S['T3_same_args']}/{t3.get('rows')} · 결과 바뀐 건 **{t3.get('any_changed')}** · 오류 {t3.get('errors')} | 0 |",
         f"| T4 | 확장 카드 = D-101 저장본 | {('비교 ' + str(t4.get('compared')) + ' · 다름 **' + str(t4.get('different')) + '** · 확정 추천 연결 불가 ' + str(t4.get('best_not_connected'))) if 'compared' in t4 else t4.get('skipped')} | 다름 0 · 연결 불가 0 |",
         f"| T5 | 두 카드 운행일 | 같음 {t5['same']}/{t5['cases']} | 모두 같음 |",
         f"| T7 | 앱 비심야 화면 = 구현 전 | 같음 **{t7['day_same']}/{t7['day_cases']}** · 심야 사례 {len(t7['late'])}개는 아래 | 모두 같음 |",
         f"| T8 | 심야 전후 (412쌍 × {len(LATE_INPUTS)}개 입력 = {t8.get('rows')}건) | 경로 바뀜 {t8.get('path_changed')} · 소요 바뀜 {t8.get('actual_changed')} · 최대 혼잡 바뀜 {t8.get('maxc_changed')} · 결과 있음 여부 바뀜 {t8.get('ok_changed')} · 오류 {t8.get('errors')} | 보고 |",
         "", "## T8 입력별 (전 → 후)", "", "| 입력 | 건수 | 전 (요일·시간대) | 후 | 경로 바뀜 | 최대 혼잡 평균 전→후 |", "|---|---|---|---|---|---|"]
    for k, v in (t8.get("by_input") or {}).items():
        L.append(f"| {k} | {v['n']} | {v['old']} | {v['new']} | {v['path_changed']} | {v['maxc_old_mean']} → {v['maxc_new_mean']} |")
    L += ["", f"- 최대 혼잡 0% 인 건: 전 {t8.get('old_maxc_zero')} → 후 {t8.get('new_maxc_zero')} (전은 주말 심야에 운행·혼잡 자료가 없는 시각을 봄)",
          f"- 소요 차이 중앙 {t8.get('actual_diff_median')} · 최대 {t8.get('actual_diff_absmax')}분",
          f"- 경로 바뀐 예: {t8.get('examples')}", "",
          "## T7 심야 화면 사례 (전·후, 켬/끔)", "", "| 사례 | 전 metric | 후 metric | 운행일 문구 | 켬=끔 기존 카드 | 오류 |", "|---|---|---|---|---|---|"]
    for x in t7["late"]:
        L.append(f"| {x['case']} | {x['before_metrics']} | {x['after_metrics']} | {x['label_shown']} | {x['on_off_base_same']} | {len(x['exceptions'])} |")
    L += ["", "## T7 비심야 차이", ""] + ([f"- {x['case']}: {x['keys']}" for x in t7["day_diffs"]] or ["- 없음"])
    L += ["", "## T5", "", "| 입력 | 기존 카드 (운행일 요일 시각) | 확장 카드 (운행일 요일 분) | 같음 |", "|---|---|---|---|"]
    L += [f"| {r['input']} | {r['base_card']} | {r['extension']} | {r['same']} |" for r in t5["rows"]]
    L += ["", "## 한계", "", "- 혼잡도의 운행일 귀속은 원본 자료 모양(2026-06 평일 24:30 열 값 있음·주말 0)으로 판단, 공식 정의 문서 미확인",
          "- 기존 카드는 열차별 막차 연결을 확인하지 않는다 (확장 카드만 확인)", "- 공휴일 미구분 (D-104 에서 처리)", ""]
    (dest / "late_night_3a2e.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    (dest / "late_night_3a2e.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str) + "\n",
                                               encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    e47 = _load("e47", "scripts/v2/47_evaluate_korail_3a2c.py")
    pairs = [(o, d) for o, d, _ in e47.v23_pairs(412)]
    late, day = LATE_INPUTS, DAY_INPUTS
    if a.dry:
        pairs, late, day = pairs[:4], LATE_INPUTS[:3], DAY_INPUTS[:2]
    w = 1 if a.dry else a.workers
    OUT.mkdir(parents=True, exist_ok=True)
    S = {"pairs": len(pairs)}
    t0 = time.time()
    df8 = run_pool(t8_task, [(o, d, late) for o, d in pairs], w, "T8")
    S["T8"] = analyze_cards(df8)
    print(f"  → T8 끝 ({time.time() - t0:.0f}초)", flush=True)
    t0 = time.time()
    df3 = run_pool(t8_task, [(o, d, day) for o, d in pairs], w, "T3")
    S["T3"] = analyze_cards(df3)
    S["T3_same_args"] = int(df3.same_args.sum()) if "same_args" in df3 else 0
    print(f"  → T3 끝 ({time.time() - t0:.0f}초)", flush=True)
    if not a.dry:
        df8.to_parquet(OUT / "T8.parquet")
        df3.to_parquet(OUT / "T3.parquet")
    t0 = time.time()
    S["T4"] = run_t4(a.workers, a.dry)
    print(f"  → T4 끝 ({time.time() - t0:.0f}초)", flush=True)
    S["T5"] = run_t5()
    t0 = time.time()
    S["T7"] = run_t7(a.dry)
    print(f"  → T7 끝 ({time.time() - t0:.0f}초)", flush=True)
    if a.dry:
        d = Path(tempfile.gettempdir()) / "late_night_3a2e_dry"
        d.mkdir(exist_ok=True)
        write_report(S, d)
        print(f"연습 보고서: {d / 'late_night_3a2e.md'}")
        print("[연습 실행] 결과는 저장하지 않았습니다 (보고 금지).")
    else:
        write_report(S, REP)
        print(f"저장: {REP / 'late_night_3a2e.md'}, {REP / 'late_night_3a2e.json'}")
    return 0


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
