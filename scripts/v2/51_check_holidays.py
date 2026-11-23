"""
scripts/v2/51_check_holidays.py — 공휴일 시각표 요일 검증 H6·H7·H9 (D-104, 사전 등록 11-4절)
==========================================================================================

| H | 이 스크립트가 하는 일 |
|---|---|
| H9 | 확장 카드 전·후: K 표본 412쌍 × 평일 공휴일 2026-05-01(금, 노동절) × 3개 시각(08:00·18:00·23:30). 전 = D-103 (시각표 요일 = 평일), 후 = D-104 (휴일 시각표). 상태·확정 경로·열차번호·기대 소요 변경 건수. 2025~2027 평일 공휴일은 모두 같은 '평일 → 휴일 시각표' 변경이라 대표 1일로 계산 (날짜마다 다른 것은 이벤트 배수뿐) |
| H6 | 앱: 구현 전 앱(D-104 사전 등록 커밋)과 지금 앱을 끔 상태로 비공휴일 사례에서 비교 → 같아야 함 |
| H7 | 앱: 평일 공휴일 사례에서 전·후 차이(안내 문구) 보고 |
| — | 2025~2027 공휴일 수·평일 공휴일 수 (자료에서 셈) |

실행:  python scripts/v2/51_check_holidays.py --workers 8      (사용자 PC 예상 5~10분, 미확인)
       python scripts/v2/51_check_holidays.py --dry
산출:  reports/v2/holidays_3a2f.{md,json}
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
from yeoyuro_v2 import holidays as H              # noqa: E402
from yeoyuro_v2 import korail_app as KA           # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"
REP = ROOT / "reports" / "v2"
PREREG_SUBJECT = "Phase 3A-2f D-104 사전 등록"
H9_DATE = "2026-05-01"
H9_TIMES = ["08:00", "18:00", "23:30"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------- H9 작업자 (Windows spawn 에서도 피클되도록 모듈 함수만)
_P: dict = {}


def _planner(day_type: str):
    if day_type not in _P:
        import station_routing as sr
        from yeoyuro_v2.line9 import register_v1_directions
        mod = sr.load_scorer_class(ROOT)
        register_v1_directions(mod)
        disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
        _P[day_type] = KA.ExtensionPlanner(DATA9, mod, sr, disp, day_type, query_date=H9_DATE, repo_root=ROOT)
    return _P[day_type]


def h9_task(args):
    mode, o, d = args
    sd = KA.service_day(H9_DATE, "08:00")
    p = _planner("weekday" if mode == "before" else sd.timetable_day_type)
    rows = []
    for hm in H9_TIMES:
        s = KA.service_day(H9_DATE, hm)
        try:
            r = p.plan(o, d, s)
        except Exception as e:                    # noqa: BLE001
            rows.append({"mode": mode, "o": o, "d": d, "hhmm": hm, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        b = r.best
        rows.append({"mode": mode, "o": o, "d": d, "hhmm": hm, "status": r.status,
                     "best_status": "" if b is None else b.status, "best_path": "" if b is None else "|".join(b.path),
                     "best_trains": "" if b is None else "|".join(x.train for x in b.rides),
                     "expected": None if b is None else round(b.expected_min, 3),
                     "train_total": None if b is None or b.train_total_min is None else round(b.train_total_min, 3),
                     "excl": json.dumps(dict(r.excluded_train), ensure_ascii=False)})
    return rows


def run_pool(jobs, workers):
    rows, t0 = [], time.time()
    if workers <= 1:
        for j in jobs:
            rows += h9_task(j)
        return pd.DataFrame(rows)
    with Pool(workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(h9_task, jobs, chunksize=1), 1):
            rows += rs
            if i % max(1, len(jobs) // 10) == 0:
                print(f"    H9 {i}/{len(jobs)} ({time.time() - t0:.0f}초)", flush=True)
    return pd.DataFrame(rows)


def analyze_h9(df: pd.DataFrame) -> dict:
    k = ["o", "d", "hhmm"]
    b, a = df[df["mode"] == "before"], df[df["mode"] == "after"]
    m = b.merge(a, on=k, suffixes=("_b", "_a"))
    ch = {c: int((m[f"{c}_b"].fillna("").astype(str) != m[f"{c}_a"].fillna("").astype(str)).sum())
          for c in ("status", "best_path", "best_trains", "expected")}
    both = m[(m.best_status_b == "연결 확인") & (m.best_status_a == "연결 확인")]
    d = (both.train_total_a - both.train_total_b).dropna()
    return {"compared": int(len(m)), "errors": int(df.get("error", pd.Series(dtype=str)).notna().sum()) if "error" in df else 0,
            "changed": ch, "status_before": b.status.value_counts().to_dict(), "status_after": a.status.value_counts().to_dict(),
            "best_not_connected_after": int(((a.best_status != "") & (a.best_status != "연결 확인")).sum()),
            "timetable_total_diff_median": float(d.median()) if len(d) else None,
            "timetable_total_diff_max_abs": float(d.abs().max()) if len(d) else None,
            "by_time": {hm: {"status_changed": int((g.status_b != g.status_a).sum()),
                             "trains_changed": int((g.best_trains_b.fillna("") != g.best_trains_a.fillna("")).sum())}
                        for hm, g in m.groupby("hhmm")},
            "examples": m[m.best_trains_b.fillna("") != m.best_trains_a.fillna("")].head(5)[
                k + ["status_b", "status_a", "best_trains_b", "best_trains_a", "train_total_b", "train_total_a"]].to_dict("records")}


# ---------------------------------------------------------------------- H6·H7 앱
def _before_app() -> Path:
    sha = []
    for g in (PREREG_SUBJECT, PREREG_SUBJECT.split(" ", 2)[-1]):     # 전체 제목, 없으면 'D-104 사전 등록'
        sha = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-F", f"--grep={g}"],
                             capture_output=True, text=True, encoding="utf-8").stdout.split()
        if sha:
            break
    if not sha:
        raise SystemExit(f"'{PREREG_SUBJECT}' 커밋을 찾지 못했습니다")
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha[0]}:app/streamlit/yeoyuro_seoul_app.py"],
                         capture_output=True, encoding="utf-8").stdout
    p = APP.parent / "_w1_before_app.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return p


def run_app(dry) -> dict:
    m49 = _load("m49c", "scripts/v2/49_check_korail_app.py")
    day = [c for c in m49.W1_CASES if not H.is_holiday(dt.date.fromisoformat(c[2]))]
    hol = [("신림", "강남", "2026-05-01", "08:00", "calm"), ("홍대입구", "잠실", "2025-06-03", "18:00", "calm"),
           ("상봉", "건대입구", "2026-05-02", "00:30", "calm")]
    if dry:
        day, hol = day[:2], hol[:1]
    before = _before_app()
    out = {"day_cases": len(day), "day_same": 0, "day_diffs": [], "holiday": []}
    try:
        for c in day:
            a, b = m49._at(before), m49._at(APP)
            m49._search(a, *c)
            m49._search(b, *c)
            sa, sb = m49._screen(a), m49._screen(b)
            if sa == sb:
                out["day_same"] += 1
            else:
                out["day_diffs"].append({"case": " ".join(c), "keys": [k for k in sa if sa[k] != sb[k]]})
        for c in hol:
            a, b = m49._at(before), m49._at(APP)
            m49._search(a, *c)
            m49._search(b, *c)
            sa, sb = m49._screen(a), m49._screen(b)
            out["holiday"].append({"case": " ".join(c), "changed_keys": [k for k in sa if sa[k] != sb[k]],
                                   "metrics_same": sa["metrics"] == sb["metrics"],
                                   "notice": KA.HOLIDAY_BASE in " ".join(sb["captions"]),
                                   "exceptions": sa["exceptions"] + sb["exceptions"]})
    finally:
        before.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------- 보고서
def counts() -> dict:
    out = {}
    for y in (2025, 2026, 2027):
        ds = [d for d in H._table() if d.year == y]
        out[y] = {"dates": len(ds), "weekday_holidays": sum(d.weekday() < 5 for d in ds),
                  "saturday_holidays": sum(d.weekday() == 5 for d in ds), "rows": sum(len(H._table()[d]) for d in ds)}
    return out


def write_report(S: dict, dest: Path) -> None:
    h9, app = S["H9"], S["APP"]
    L = ["# Phase 3A-2f 공휴일 시각표 요일 검증 — D-104", "",
         "- 규칙: [사전 등록 문서 11절](../../docs/v2/preregistration_3a2.md) D-104 (구현 전 고정) · 구현 기록 D-104a",
         "- 재현: `python scripts/v2/51_check_holidays.py --workers 8`", "",
         "| H | 검증 | 결과 | 기준 |", "|---|---|---|---|",
         f"| 자료 | 2025~2027 공휴일 날짜 수 (평일·토요일 공휴일) | {', '.join(f'{y}: {v['dates']}일 (평일 {v['weekday_holidays']}·토 {v['saturday_holidays']})' for y, v in S['counts'].items())} | 사전 등록 표와 같음 (테스트) |",
         f"| H6 | 비공휴일 앱 화면 = 구현 전 | 같음 **{app['day_same']}/{app['day_cases']}** | 모두 같음 |",
         f"| H7 | 공휴일 앱 화면 | {[(x['case'], x['changed_keys'], '안내' if x['notice'] else '안내 없음') for x in app['holiday']]} | 기존 카드는 안내 문구만 다름 |",
         f"| H9 | 확장 카드 전·후 ({H9_DATE} × 412쌍 × {len(H9_TIMES)}) | 비교 {h9['compared']} · 상태 바뀜 {h9['changed']['status']} · 열차 바뀜 {h9['changed']['best_trains']} · 경로 바뀜 {h9['changed']['best_path']} · 기대 소요 바뀜 {h9['changed']['expected']} · 확정 추천 연결 불가 {h9['best_not_connected_after']} · 오류 {h9['errors']} | 보고 · 연결 불가 0 |",
         "", "## H9 상세", "",
         f"- 상태 전 {h9['status_before']} → 후 {h9['status_after']}",
         f"- 시각별: {h9['by_time']}",
         f"- 시각표 확인 소요 차이(둘 다 연결 확인): 중앙 {h9['timetable_total_diff_median']} · 최대 {h9['timetable_total_diff_max_abs']}분",
         f"- 열차가 바뀐 예: {h9['examples']}",
         "", "## H6 차이", ""] + ([f"- {x['case']}: {x['keys']}" for x in app["day_diffs"]] or ["- 없음"]) + [
         "", "## 한계", "",
         "- 혼잡·배차 통계는 공휴일 관측값이 아니라 요일 기준 (P3·P4) — 기존 카드의 배차 기반 대기도 평일 공휴일에 평일 배차",
         "- 토요일과 겹친 공휴일 = 휴일 시각표는 설계 가정 (미확인), 선거일 특별 운행 미반영",
         "- 공휴일 판정 범위(2025~2027)와 시각표 유효기간(2026-09-01 시각표)은 다름", ""]
    (dest / "holidays_3a2f.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    (dest / "holidays_3a2f.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str) + "\n",
                                             encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    e47 = _load("e47", "scripts/v2/47_evaluate_korail_3a2c.py")
    ods = [(o, d) for o, d, _ in e47.k_sample(e47.router(e47.spec(on=True)).display.station_key.unique())]
    if a.dry:
        ods = ods[:3]
    S = {"counts": counts()}
    t0 = time.time()
    jobs = [("before", o, d) for o, d in ods]
    df_b = run_pool(jobs, 1 if a.dry else a.workers)            # 요일별로 작업자 풀을 따로 (작업자 하나 = 계산기 하나, 메모리)
    jobs = [("after", o, d) for o, d in ods]
    df_a = run_pool(jobs, 1 if a.dry else a.workers)
    S["H9"] = analyze_h9(pd.concat([df_b, df_a], ignore_index=True))
    print(f"  → H9 끝 ({time.time() - t0:.0f}초)", flush=True)
    t0 = time.time()
    S["APP"] = run_app(a.dry)
    print(f"  → H6·H7 끝 ({time.time() - t0:.0f}초)", flush=True)
    if a.dry:
        d = Path(tempfile.gettempdir()) / "holidays_3a2f_dry"
        d.mkdir(exist_ok=True)
        write_report(S, d)
        print(f"연습 보고서: {d / 'holidays_3a2f.md'}")
        print("[연습 실행] 결과는 저장하지 않았습니다 (보고 금지).")
    else:
        write_report(S, REP)
        print(f"저장: {REP / 'holidays_3a2f.md'}, {REP / 'holidays_3a2f.json'}")
    return 0


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
