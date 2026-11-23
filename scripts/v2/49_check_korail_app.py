"""
scripts/v2/49_check_korail_app.py — 광역철도 확장 앱 통합 검증 W1~W8 (D-101, 사전 등록 9-5절)
==========================================================================================

| W | 검증 | 이 스크립트가 하는 일 |
|---|---|---|
| W1 | 끔 불변 | 구현 전 앱(D-101 사전 등록 커밋의 앱 파일)과 지금 앱을 **끔** 상태로 같은 조건에서 실행해 화면(metric·요약·안내·역 목록)과 노선도 그림 데이터를 비교 |
| W2 | 켬 확정 추천에 연결 불가 0 | K 표본 412쌍 × 평일 05:30~25:30 30분 간격(41개) → ExtensionPlanner.plan → 확정 추천의 연결 상태 |
| W3 | 역 선택 | 끔·켬 역 목록, 이름 중복, 공통 역 배지, 켬→끔 전환 시 비움 (앱 실행) |
| W4 | 노선도 | 켬 그림: 3-A 점·선·클릭 영역, 3-B 예정 점선·클릭 없음 |
| W5 | 자정 이후 | 토 00:30 → 평일 시각표, 자정 넘어 도착하는 연결 확인 사례 수, 막차 뒤 안내 |
| W6 | 문구 | 켬 화면에 두 기준·혼잡도 미확인·시각표 고정·운행일 문구 |
| W7 | 쾌적 분리 | 켬·끔에서 기존 결과 카드 metric 이 같고, 확장 카드는 그 아래 따로 |
| W8 | 성능 | 계산기 준비 시간, 검색 1회 시간 (보고만) |

실행:  python scripts/v2/49_check_korail_app.py --workers 8      (사용자 PC 예상 15~30분, 미확인)
       python scripts/v2/49_check_korail_app.py --dry             (작은 표본 연습, 결과 저장 안 함)
산출:  reports/v2/korail_app_3a2d.{md,json}, data/marts/v2/korail_app_3a2d/W2.parquet (git 무시)
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
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit"), str(ROOT / "tests")]
from yeoyuro_v2 import korail_app as KA          # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"
OUT = ROOT / "data" / "marts" / "v2" / "korail_app_3a2d"
REP = ROOT / "reports" / "v2"
PREREG_SUBJECT = "Phase 3A-2d D-101 사전 등록"
W2_DATE = "2026-11-25"                     # 평일 (수)
W2_TIMES = [330 + 30 * i for i in range(41)]  # 05:30 ~ 25:30


def _e47():
    spec = importlib.util.spec_from_file_location("e47", ROOT / "scripts" / "v2" / "47_evaluate_korail_3a2c.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hhmm(m: int) -> str:
    return "%02d:%02d" % ((m // 60) % 24, m % 60)


# ---------------------------------------------------------------------- W2 작업자
_P: dict = {}


def _planner():
    if "p" not in _P:
        import station_routing as sr
        from yeoyuro_v2.line9 import register_v1_directions
        mod = sr.load_scorer_class(ROOT)
        register_v1_directions(mod)
        disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
        t0 = time.perf_counter()
        _P["p"] = KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", repo_root=ROOT)
        _P["build"] = time.perf_counter() - t0
    return _P["p"]


def w2_task(args):
    o, d, times = args
    p = _planner()
    rows = []
    for m in times:
        sd = KA.service_day(W2_DATE, hhmm(m))
        t0 = time.perf_counter()
        try:
            r = p.plan(o, d, sd)
        except Exception as e:                    # noqa: BLE001 — 기록만
            rows.append({"o": o, "d": d, "t": m, "status": "error", "err": f"{type(e).__name__}: {e}"[:200]})
            continue
        b = r.best
        rows.append({
            "o": o, "d": d, "t": m, "sec": time.perf_counter() - t0, "status": r.status,
            "n_cand": len(r.candidates), "best_status": None if b is None else b.status,
            "best_korail": None if b is None else b.korail, "best_expected": None if b is None else b.expected_min,
            "best_train_total": None if b is None else b.train_total_min,
            "best_crosses_midnight": None if b is None or b.train_total_min is None else (m + b.train_total_min) >= 1440,
            "best_trains": None if b is None else "|".join(x.train for x in b.rides),
            "best_path": None if b is None else "|".join(b.path),
            "top_status": r.candidates[0].status if r.candidates else None,
            "top_is_best": bool(r.candidates) and b is not None and r.candidates[0].path == b.path,
            "excl_train": json.dumps(dict(r.excluded_train), ensure_ascii=False),
            "excl_engine": json.dumps(dict(r.excluded_engine), ensure_ascii=False),
            "baseline": r.baseline_min, "build": _P.get("build")})
    return rows


def run_w2(ods, times, workers) -> pd.DataFrame:
    jobs = [(o, d, times) for o, d in ods]
    rows, t0 = [], time.time()
    if workers <= 1:
        for j in jobs:
            rows += w2_task(j)
    else:
        with Pool(workers) as pool:
            for i, rs in enumerate(pool.imap_unordered(w2_task, jobs, chunksize=1), 1):
                rows += rs
                if i % max(1, len(jobs) // 10) == 0:
                    print(f"    W2 {i}/{len(jobs)} 쌍 ({time.time() - t0:.0f}초)", flush=True)
    return pd.DataFrame(rows)


def analyze_w2(df: pd.DataFrame) -> dict:
    ok = df[df.status != "error"]
    has = ok[ok.best_status.notna()]
    ex = Counter()
    for s in ok.excl_train:
        ex.update(json.loads(s))
    return {
        "plans": int(len(df)), "errors": int((df.status == "error").sum()),
        "error_examples": df[df.status == "error"].head(5).to_dict("records"),
        "status": ok.status.value_counts().to_dict(),
        "best_not_connected": int((has.best_status != KA.CONNECT_OK).sum()),          # W2 기준: 0
        "top_not_connected": int((ok.top_status.notna() & (ok.top_status != KA.CONNECT_OK)).sum()),
        "best_not_top": int((has.top_is_best == False).sum()),                          # noqa: E712 — 다음 후보로 넘어간 수
        "excluded_train_reasons": dict(ex.most_common()),
        "midnight_connected": int(has.best_crosses_midnight.fillna(False).sum()),
        "korail_best": int(has.best_korail.fillna(False).sum()),
        "sec_mean": float(ok.sec.mean()), "sec_p95": float(ok.sec.quantile(0.95)), "sec_max": float(ok.sec.max()),
        "build_sec": float(ok.build.dropna().max()) if ok.build.notna().any() else None,
    }


# ---------------------------------------------------------------------- 앱 실행 (W1·W3·W4·W6·W7)
def _at(app_path: Path):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(app_path), default_timeout=900)
    at.session_state["nav_page"] = "경로 추천"
    at.session_state["view_route"] = "경로"
    at.run()
    return at


def _search(at, o, d, date, hm, mode="calm"):
    at.selectbox(key="sel_origin").set_value(o)
    at.selectbox(key="sel_dest").set_value(d)
    at.date_input[0].set_value(dt.date.fromisoformat(date))
    h, m = map(int, hm.split(":"))
    at.time_input[0].set_value(dt.time(h, m))
    [s for s in at.selectbox if s.label == "선호 모드"][0].set_value(mode)
    at.run()
    [b for b in at.button if b.label == "경로 검색"][0].click()
    t0 = time.perf_counter()
    at.run()
    return time.perf_counter() - t0


def _screen(at) -> dict:
    charts = at.get("plotly_chart")
    spec = charts[0].proto.spec if charts else ""
    return {"metrics": [(m.label, m.value, m.delta or "") for m in at.metric],
            "captions": [c.value for c in at.caption],
            "info": [x.value for x in at.info], "warning": [x.value for x in at.warning],
            "options": list(at.selectbox(key="sel_origin").options),
            "map_spec_len": len(spec), "map_spec_hash": hash(spec),
            "exceptions": [e.value[:300] for e in at.exception]}


W1_CASES = [
    ("상봉", "건대입구", "2026-10-14", "08:00", "calm"), ("신림", "강남", "2026-10-14", "08:15", "calm"),
    ("강변", "여의도", "2026-10-14", "08:30", "calm"), ("노원", "수유", "2026-10-14", "08:00", "calm"),
    ("가양", "고속터미널", "2026-10-14", "08:00", "calm"), ("김포공항", "종합운동장", "2026-10-14", "08:00", "calm"),
    ("석촌", "잠실", "2026-10-14", "18:00", "calm"), ("서울역", "잠실", "2026-10-14", "18:00", "calm"),
    ("신림", "강남", "2026-10-14", "08:15", "fast"), ("서울역", "잠실", "2026-10-14", "18:00", "min_transfer"),
    ("공덕", "여의나루", "2025-09-27", "18:00", "calm"),
    ("신도림", "석계", "2026-11-25", "08:00", "calm"), ("가산디지털단지", "창동", "2026-11-25", "23:30", "fast"),
    ("노량진", "온수", "2026-11-28", "00:30", "calm"),
]


def _before_app() -> Path:
    """D-101 사전 등록 커밋의 앱 파일 (= 구현 직전 앱). 같은 폴더에 임시 파일로 둔다 (import 경로가 같도록)."""
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "--format=%H", "-F", f"--grep={PREREG_SUBJECT}"],
                         capture_output=True, text=True, encoding="utf-8").stdout.split()
    if not sha:
        raise SystemExit(f"'{PREREG_SUBJECT}' 커밋을 찾지 못했습니다")
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha[0]}:app/streamlit/yeoyuro_seoul_app.py"],
                         capture_output=True, encoding="utf-8").stdout
    p = APP.parent / "_w1_before_app.py"
    p.write_text(src, encoding="utf-8", newline="\n")
    return p


def run_w1(cases) -> dict:
    before = _before_app()
    out = {"cases": len(cases), "same": 0, "diffs": []}
    try:
        for c in cases:
            a, b = _at(before), _at(APP)
            _search(a, *c)
            _search(b, *c)
            sa, sb = _screen(a), _screen(b)
            if sa == sb:
                out["same"] += 1
            else:
                out["diffs"].append({"case": " ".join(c), "keys": [k for k in sa if sa[k] != sb[k]]})
    finally:
        before.unlink(missing_ok=True)
    return out


def run_app_checks() -> dict:
    R = {}
    # W3: 목록·전환
    at = _at(APP)
    off = list(at.selectbox(key="sel_origin").options)
    at.toggle(key="ext_on").set_value(True)
    at.run()
    on = list(at.selectbox(key="sel_origin").options)
    at.selectbox(key="sel_origin").set_value("월계")
    at.selectbox(key="sel_dest").set_value("석계")
    at.run()
    at.toggle(key="ext_on").set_value(False)
    at.run()
    # 선택 상자 options 는 화면에 보이는 이름(배지 포함) — 공통 역 8곳은 배지가 바뀌어 이름이 달라진다
    R["W3"] = {"off": len(off), "on": len(on), "added_n": len(on) - len(off),
               "label_changed": sorted(set(on) - set(off)), "dup_on": len(on) - len(set(on)),
               "cleared_on_off": at.session_state["origin_station_key"] is None,
               "dest_kept": at.session_state["destination_station_key"] == "석계",
               "flash": any(KA.OFF_CLEARED in w.value for w in at.warning),
               "exceptions": [e.value[:200] for e in at.exception]}
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    x = KA.extended_display(disp)
    R["W3"]["common_badge"] = {n: x.loc[x.station_key == n, "available_lines"].iloc[0] for n in KA.common_stations(disp)}
    # W4: 노선도 그림 (켬)
    at = _at(APP)
    at.toggle(key="ext_on").set_value(True)
    at.run()
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    traces = spec.get("data", [])
    dotted = [t for t in traces if (t.get("line") or {}).get("dash") == "dot"]
    clicks = [t for t in traces if t.get("customdata")]
    keys = {row[0] for t in clicks for row in t["customdata"]}
    planned_nodes = [t for t in traces if "3-B 예정" in str(t.get("hovertemplate", ""))]
    R["W4"] = {"traces": len(traces), "planned_b_dotted": len(dotted), "planned_b_nodes_trace": len(planned_nodes),
               "planned_b_clickable": any(t.get("customdata") for t in planned_nodes + dotted),
               "click_keys": len(keys), "new_stations_clickable": sorted(set(KA.korail_only_stations(disp)) & keys)}
    at2 = _at(APP)
    spec_off = json.loads(at2.get("plotly_chart")[0].proto.spec)
    R["W4"]["off_dotted"] = sum(1 for t in spec_off.get("data", []) if (t.get("line") or {}).get("dash") == "dot")
    # W6·W7: 켬 화면 문구, 기존 결과 카드 metric 이 끔과 같은지
    for o, d, date, hm, mode in (("신도림", "석계", "2026-11-25", "08:00", "calm"),
                                 ("가산디지털단지", "월계", "2026-11-25", "08:00", "calm"),
                                 ("금천구청", "월계", "2026-11-28", "00:30", "calm")):
        a = _at(APP)
        a.toggle(key="ext_on").set_value(True)
        a.run()
        sec = _search(a, o, d, date, hm, mode)
        b = _at(APP)
        _search(b, o, d, date, hm, mode) if not KA.is_korail_only(o, disp) and not KA.is_korail_only(d, disp) else None
        caps = " ".join(c.value for c in a.caption)
        mds = " ".join(m.value for m in a.markdown)
        ext_labels = {"기대 소요", "시각표 확인 도착", "1~9호선만 이용 시"}
        base_on = [m for m in [(m.label, m.value, m.delta or "") for m in a.metric] if m[0] not in ext_labels]
        base_off = [(m.label, m.value, m.delta or "") for m in b.metric]
        R.setdefault("W6_W7", []).append({
            "case": f"{o}→{d} {date} {hm}", "search_sec": round(sec, 2),
            "texts": {"기대 소요 기준": KA.BASIS_EXPECTED in caps, "시각표 기준": KA.BASIS_TIMETABLE in caps,
                      "시각표 고정": KA.BASIS_STATIC in caps, "운행일": "운행일 기준:" in caps,
                      "혼잡도 미확인": KA.BADGE_UNKNOWN in mds, "쾌적 판정 안 함": KA.NO_COMFORT in caps,
                      "확인 못함": any(KA.NONE_FOUND in w.value for w in a.warning),
                      "1~9호선 대상 아님": any(KA.NOT_IN_BASE in i.value for i in a.info)},
            "base_card_same_as_off": base_on == base_off,
            "exceptions": [e.value[:200] for e in a.exception] + [e.value[:200] for e in b.exception]})
    return R


# ---------------------------------------------------------------------- W5
def run_w5() -> dict:
    p = _planner()
    s = KA.service_day("2026-11-28", "00:30")
    r = p.plan("금천구청", "월계", s)
    s2 = KA.service_day("2026-11-25", "23:00")
    r2 = p.plan("금천구청", "월계", s2)
    return {"sat_0030": {"service_date": str(s.date), "day_type": s.day_type, "minute": s.minute, "label": s.label,
                         "status": r.status, "reason": r.main_reason},
            "gumcheon_2300": {"status": r2.status, "best_trains": [x.train for x in r2.best.rides] if r2.best else None,
                              "best_arrive": r2.best.train_arrive if r2.best else None,
                              "top_status": r2.candidates[0].status if r2.candidates else None,
                              "excluded_train": dict(r2.excluded_train)}}


# ---------------------------------------------------------------------- 보고서
def write_report(S: dict, dest: Path) -> None:
    w2, w1 = S["W2"], S["W1"]
    A = S["APP"]
    L = ["# Phase 3A-2d 앱 통합 검증 — D-101 (광역철도 확장 토글)", "",
         "- 규칙: [사전 등록 문서 9절](../../docs/v2/preregistration_3a2.md) D-101 (구현 전 고정) · 구현 기록 D-102",
         "- 재현: `python scripts/v2/49_check_korail_app.py --workers 8`", "",
         "| W | 검증 | 결과 | 기준 |", "|---|---|---|---|",
         f"| W1 | 끔 불변 (구현 전 앱과 같은 조건 {w1['cases']}개 비교) | 같음 **{w1['same']}/{w1['cases']}** | 모두 같음 |",
         f"| W2 | 켬 확정 추천 (K 표본 {S['n_od']}쌍 × {S['n_times']}개 시각 = {w2['plans']:,}건) | 확정 추천이 연결 확인이 아닌 경우 **{w2['best_not_connected']}** · 오류 {w2['errors']} | 0 |",
         f"| W3 | 역 선택 | 끔 {A['W3']['off']} · 켬 {A['W3']['on']} (+{A['W3']['added_n']}) · 중복 {A['W3']['dup_on']} · 켬→끔 비움 {A['W3']['cleared_on_off']} · 안내 {A['W3']['flash']} | +18 · 중복 0 · 비움 · 안내 |",
         f"| W4 | 노선도 | 켬: 3-B 점선 {A['W4']['planned_b_dotted']} · 3-B 클릭 {A['W4']['planned_b_clickable']} · 새 역 클릭 {len(A['W4']['new_stations_clickable'])}/18 · 끔 점선 {A['W4']['off_dotted']} | 점선 있음 · 클릭 안 됨 · 18 · 끔 0 |",
         f"| W5 | 자정 이후 | 토 00:30 → {S['W5']['sat_0030']['service_date']} {S['W5']['sat_0030']['day_type']} {S['W5']['sat_0030']['minute']:.0f}분 → {S['W5']['sat_0030']['status']} ({S['W5']['sat_0030']['reason']}) · 자정 넘어 도착하는 연결 확인 {w2['midnight_connected']}건 | 평일 시각표 · 1건 이상 · 안내 |",
         f"| W6 | 문구 | {', '.join(k for c in A['W6_W7'] for k, v in c['texts'].items() if v) and '아래 표'} | 아래 표 |",
         f"| W7 | 쾌적 분리 | 기존 결과 카드 = 끔: {[c['base_card_same_as_off'] for c in A['W6_W7']]} | 모두 True |",
         f"| W8 | 성능 (보고만) | 계산기 준비 {w2['build_sec'] or 0:.1f}초 · plan 1회 평균 {w2['sec_mean']:.2f}초 (p95 {w2['sec_p95']:.2f}, 최대 {w2['sec_max']:.2f}) · 앱 검색 {[c['search_sec'] for c in A['W6_W7']]}초 | — |",
         "", "## W2 상세", "",
         f"- 상태: {w2['status']}",
         f"- 기대 소요 1순위가 연결 확인이 아니어서 **다음 후보로 넘어간** 건: {w2['best_not_top']} · 1순위가 연결 확인 아님(확정 추천 없음 포함): {w2['top_not_connected']}",
         f"- 열차 단위로 뺀 후보 사유: {w2['excluded_train_reasons']}",
         f"- 확정 추천에 코레일 구간이 있는 건: {w2['korail_best']}",
         f"- 금천구청→월계 23:00 (D-099 S3 예외): {S['W5']['gumcheon_2300']}",
         "", "## W1 차이", ""]
    L += [f"- {x['case']}: {x['keys']}" for x in w1["diffs"]] or ["- 없음"]
    L += ["", "## W6·W7 화면 사례", "", "| 사례 | 문구 | 기존 카드 = 끔 | 검색 초 | 오류 |", "|---|---|---|---|---|"]
    for c in A["W6_W7"]:
        L.append(f"| {c['case']} | {', '.join(k for k, v in c['texts'].items() if v)} | {c['base_card_same_as_off']} | {c['search_sec']} | {len(c['exceptions'])} |")
    L += ["", "## W3 공통 역 배지", "", f"- {A['W3']['common_badge']}",
          f"- 켬에서 화면 이름이 새로 생기거나 바뀐 항목(새 역 18 + 배지 바뀐 공통 역 8): {', '.join(A['W3']['label_changed'])}", ""]
    (dest / "korail_app_3a2d.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    (dest / "korail_app_3a2d.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str) + "\n",
                                               encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true", help="W2 저장본이 있어도 다시 계산")
    a = ap.parse_args()
    e47 = _e47()
    # K 표본 = 3A-2c 평가와 같은 412쌍 (47 과 같은 방식: 켬 라우터의 표시 역 목록으로 뽑음)
    ods = [(o, d) for o, d, _ in e47.k_sample(e47.router(e47.spec(on=True)).display.station_key.unique())]
    times = W2_TIMES
    if a.dry:
        ods, times = ods[:6], W2_TIMES[::10]
    t0 = time.time()
    print(f"  W2: {len(ods)}쌍 × {len(times)}개 시각", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / ("W2_dry.parquet" if a.dry else "W2.parquet")
    if f.exists() and not a.force and not a.dry:
        w2df = pd.read_parquet(f)
    else:
        w2df = run_w2(ods, times, 1 if a.dry else a.workers)
        if not a.dry:
            w2df.to_parquet(f)
    print(f"  → W2 끝 ({time.time() - t0:.0f}초)", flush=True)
    S = {"n_od": len(ods), "n_times": len(times), "W2": analyze_w2(w2df)}
    t0 = time.time()
    S["W5"] = run_w5()
    print(f"  → W5 끝 ({time.time() - t0:.0f}초)", flush=True)
    t0 = time.time()
    S["W1"] = run_w1(W1_CASES[:3] if a.dry else W1_CASES)
    print(f"  → W1 끝 ({time.time() - t0:.0f}초)", flush=True)
    t0 = time.time()
    S["APP"] = run_app_checks()
    print(f"  → W3·W4·W6·W7 끝 ({time.time() - t0:.0f}초)", flush=True)
    if a.dry:
        d = Path(tempfile.gettempdir()) / "korail_app_3a2d_dry"
        d.mkdir(exist_ok=True)
        write_report(S, d)
        print(f"연습 보고서: {d / 'korail_app_3a2d.md'}")
        print("[연습 실행] 결과는 저장하지 않았습니다 (보고 금지).")
    else:
        write_report(S, REP)
        print(f"저장: {REP / 'korail_app_3a2d.md'}, {REP / 'korail_app_3a2d.json'}")
    return 0


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
