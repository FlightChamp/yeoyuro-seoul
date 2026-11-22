"""
scripts/v2/45_build_train_timetable.py
======================================
Phase 3A-2a — 1~9호선 열차 단위 시각표 mart + 시각표 연결 검증 감사 (D-093 사전 등록 2-1절, D-094).

하는 일
-------
1. 원천(서울 도시철도 열차운행시각표 20260901, cp949·한글 정상본)을 1~9호선 열차별 정차 행으로 정제
   - 급행 통과 행 제외, 1호선 급행 제외(D-085), 9호선 = 일반 9L·급행 9X
   - 일반열차 도착 00:00:00(도착 누락)·음수 정차(도착 > 출발)는 도착을 비우고 감사에 기록
2. 1호선은 3A-1 mart(코레일 서울 구간 + 서울역~청량리)와 행 단위로 같은지 대조
3. 9호선은 운영사 운행정보(20260228, 있으면)와 열차번호·역별 출발 시각 대조 (참고 — 날짜가 다름)
4. 경로 엔진(앱 데이터 data/app_v26) 후보 경로를 열차 단위로 따라가 '불가능한 연결'이 0 인지 확인
   (평일 08:00·토 14:00·일 22:30, 무작위 OD 400쌍씩, 시드 42 — 후보 경로 전부)

실행
----
    python scripts/v2/45_build_train_timetable.py --src "<원본 폴더>"
    (원본 폴더 = 열차운행시각표 20260901 원본이 있는 폴더, 하위 폴더까지 찾는다.
     운행정보 xlsx 가 같이 있으면 9호선 대조도 한다. 시각표가 없으면 [건너뜀] — 커밋된 mart 사용)
    --skip-engine-audit : 4번(약 1~2분) 건너뜀

산출물
------
    data/marts/v2/train_timetable/stop_times.parquet   1~9호선 열차별 정차 (커밋, 테스트가 읽음)
    data/marts/v2/train_timetable/audit.json           정제 감사 수치
    reports/v2/train_timetable_3a2a.md                 감사 보고서 (숫자는 여기서만 인용)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import korail_line1 as K          # noqa: E402
from yeoyuro_v2 import train_timetable as T       # noqa: E402

MART = ROOT / "data" / "marts" / "v2" / "train_timetable"
REPORT = ROOT / "reports" / "v2" / "train_timetable_3a2a.md"
L9_INFO = "도시철도_운행정보"
AUDIT_CASES = [("weekday", "08:00"), ("saturday", "14:00"), ("sunday", "22:30")]
AUDIT_N = 400


def write_parquet(df: pd.DataFrame, path: Path):
    """pandas 2·3 어디서나 같게 읽히도록 (40_build_korail_line1 과 같은 방식)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([(c, pa.string() if (df[c].dtype == object or str(df[c].dtype) in ("str", "string"))
                         else pa.from_numpy_dtype(df[c].dtype)) for c in df.columns])
    t = pa.Table.from_pandas(df, schema=schema, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(t, path)


def compare_line1(st: pd.DataFrame) -> dict:
    km = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "korail_line1" / "stop_times.parquet")
    s1 = st[(st.group == "1") & st.code.isin(list(K.SCOPE))]
    key = ["day_type", "train", "direction", "code"]
    m = s1[key + ["arr_min", "dep_min"]].merge(km[key + ["arr_min", "dep_min"]], on=key, how="outer",
                                                indicator=True, suffixes=("", "_k"))
    both = m[m._merge == "both"]
    diff = int(((both.arr_min.fillna(-1) - both.arr_min_k.fillna(-1)).abs() > 1e-6).sum()
               + ((both.dep_min.fillna(-1) - both.dep_min_k.fillna(-1)).abs() > 1e-6).sum())
    return {"rows_both": int(len(both)), "only_new": int((m._merge == "left_only").sum()),
            "only_3a1": int((m._merge == "right_only").sum()), "time_diff": diff}


def compare_line9(st: pd.DataFrame, xlsx: Path) -> dict:
    x = pd.read_excel(xlsx, header=3)
    dm = {"평일": "weekday", "토요일": "saturday", "일요일": "sunday", "공휴일": "sunday", "휴일": "sunday"}
    rows = []
    for r in x.itertuples(index=False):
        day = dm.get(r[6])
        if day is None or not isinstance(r[9], str):
            continue
        for it in r[9].split("+"):
            c, t = it.split("-", 1)
            h, mm = map(int, t.split(":"))
            rows.append((day, str(r[0]), int(c), h * 60 + mm + (1440 if h < 3 else 0)))
    X = pd.DataFrame(rows, columns=["day_type", "train", "code", "dep_x"])
    S = st[st.group.isin(["9L", "9X"])][["day_type", "train", "code", "dep_min"]].dropna()
    m = X.merge(S, on=["day_type", "train", "code"], how="left")
    d = (m.dep_min - m.dep_x).dropna()
    return {"file": xlsx.name, "rows": int(len(X)), "trains": int(X.train.nunique()),
            "matched_rate": round(float(m.dep_min.notna().mean()), 4),
            "diff_median": round(float(d.median()), 2), "diff_p90_abs": round(float(d.abs().quantile(0.9)), 2),
            "within_1min": round(float((d.abs() <= 1).mean()), 4), "diff_max_abs": round(float(d.abs().max()), 2)}


def engine_audit(st: pd.DataFrame) -> list[dict]:
    """앱 데이터의 경로 후보 전부를 열차 단위로 따라가 본다. 불가능한 연결 = 엔진 오류 신호."""
    data = ROOT / "data" / "app_v26"
    sys.path[:0] = [str(ROOT / "app" / "streamlit")]
    import station_routing as sr
    from yeoyuro_v2 import time_dependent as td
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = sr.load_display_master(data)
    names = sorted(disp.station_name.unique())
    idx = T.TrainIndex(st)
    rng = random.Random(42)
    out = []
    for day, hhmm in AUDIT_CASES:
        t_min = K.to_service_minutes(hhmm)
        r = td.TDRouter(data, mod, sr, disp, day_type=day)
        ev = r.evaluator(td.bin_index_of(t_min))
        cnt, bad, n_paths, n_od = collections.Counter(), [], 0, 0
        for _ in range(AUDIT_N):
            o, d = rng.sample(names, 2)
            paths = r.candidates(o, d, t_min)
            n_od += 1
            for p in paths:
                legs = T.legs_from_path(ev, p)
                n_paths += 1
                imp = T.impossible_legs(idx, legs, day)
                if imp:
                    bad.append(f"{o}→{d}: " + ", ".join(f"{lg.board_node}→{lg.alight_node}" for lg in imp))
                c = T.verify_connection(idx, legs, t_min, day)
                cnt[c.status] += 1
                if c.status == T.NOT_CONNECTED:
                    cnt["사유:" + c.reason] += 1
        out.append({"day": day, "time": hhmm, "od": n_od, "paths": n_paths, "counts": dict(cnt),
                    "impossible_examples": bad[:10], "impossible": len(bad)})
        print(f"  엔진 감사 {day} {hhmm}: 후보 경로 {n_paths} · {dict(cnt)} · 불가능한 연결 {len(bad)}")
    return out


def write_report(audit: dict) -> None:
    a = audit["clean"]
    L = ["# Phase 3A-2a — 1~9호선 열차 단위 시각표 · 시각표 연결 감사", "",
         "- 근거: [사전 등록 2-1절](../../docs/v2/preregistration_3a2.md), D-094. 이 보고서는 경로 점수·순위를 다루지 않는다.",
         "- 재현: `python scripts/v2/45_build_train_timetable.py --src \"<원본 폴더>\"`", "",
         "## 1. 정제", "",
         "| 항목 | 값 |", "|---|---|",
         f"| 원천 행 | {a['rows_raw']:,} |",
         f"| 급행 통과 행(도착 00:00:00, 정차 아님) 제외 | {a['pass_rows_express']:,} |",
         f"| 1호선 급행 행 제외 (D-085) | {a['line1_express_rows_dropped']:,} |",
         f"| 일반열차 도착 00:00:00 → 도착 누락으로 처리 | {a['missing_arrival_normal']} |",
         f"| 음수 정차(도착 > 출발) → 도착을 비움 | {len(a['negative_dwell_rows'])} |",
         f"| 남은 정차 행 · 열차 | {a['rows_kept']:,} · {a['trains']:,} |", ""]
    for r in a["missing_arrival_examples"]:
        L.append(f"- 도착 누락: {r['line']}호선 {r['day']} {r['train']} 열차 {r['code']}역 (도착 {r['arr']}, 출발 {r['dep']})")
    for r in a["negative_dwell_rows"]:
        L.append(f"- 음수 정차: {r['line']}호선 {r['day_type']} {r['train']} 열차 {r['code']}역 (도착 {r['arr']}, 출발 {r['dep']})")
    c1 = audit["line1_vs_3a1"]
    L += ["", "## 2. 1호선 — 3A-1 mart 대조", "",
          f"- 같은 (요일·열차·방향·역) 행 {c1['rows_both']:,} · 새 mart 에만 {c1['only_new']} · 3A-1 에만 {c1['only_3a1']} · 시각 차이 {c1['time_diff']}",
          "- 0 이면 3A-1 의 코레일 서울 구간 계산과 같은 원천·같은 규칙", ""]
    l9 = audit.get("line9_vs_info")
    L += ["## 3. 9호선 — 운영사 운행정보 대조 (참고)", ""]
    if l9:
        L += [f"- 운행정보 `{l9['file']}` 열차 {l9['trains']} · 역별 출발 {l9['rows']:,}행",
              f"- 열차번호·역 일치 {l9['matched_rate'] * 100:.1f}% · 출발 시각 차이(시각표 − 운행정보) 중앙 {l9['diff_median']}분, "
              f"|차이| 90% {l9['diff_p90_abs']}분, 1분 이내 {l9['within_1min'] * 100:.1f}%, 최대 {l9['diff_max_abs']}분",
              "- 운행정보는 2026-02-28 기준·분 단위, 시각표는 2026-09-01 기준·초 단위 → 시각 차이는 기준일 차이일 가능성이 크지만 확인하지 않음(미확인). "
              "엔진·연결 검증은 시각표(2026-09-01)를 쓴다 (v2.6 과 같은 원천)", ""]
    else:
        L += ["- 운행정보 파일 없음 — 대조 건너뜀", ""]
    L += ["## 4. 경로 엔진 후보의 시각표 연결 (앱 데이터, 기능 끔 = 1~9호선)", "",
          "| 요일·시각 | OD | 후보 경로 | 연결 확인 | 연결 불가 (사유) | 불가능한 연결 |", "|---|---|---|---|---|---|"]
    for e in audit.get("engine", []):
        c = e["counts"]
        why = ", ".join(f"{k[3:]} {v}" for k, v in c.items() if k.startswith("사유:")) or "-"
        L.append(f"| {e['day']} {e['time']} | {e['od']} | {e['paths']:,} | {c.get(T.CONNECTED, 0):,} | "
                 f"{c.get(T.NOT_CONNECTED, 0)} ({why}) | {e['impossible']} |")
    for e in audit.get("engine", []):
        for x in e["impossible_examples"]:
            L.append(f"- 불가능한 연결 ({e['day']}): {x}")
    L += ["", "- 불가능한 연결 = 그날 어떤 시각에도 그 구간을 열차로 이어 갈 수 없음 → 0 이어야 함 (R6 오류 기준)",
          "- 연결 불가 = 그 출발 시각에 막차를 놓치거나 60분 안 열차가 없음 (R3 사유) — 오류 아님",
          "", "## 5. 원천 표기 처리 (D-094)", "",
          "- 경로 그래프 역코드 → 시각표 역코드: " + ", ".join(f"{k[0]}호선 {k[1]} → {v}" for k, v in T.CODE_ALIAS.items()),
          "- 순환선 기록 경계(같은 방향 다음 기록으로 이어 탐, 보행 0): " +
          ", ".join(f"{k}호선 {sorted(v)}" for k, v in T.LOOP_SPLIT.items()),
          "- 분기 가상 노드(예: 2_성수@seongsu_branch)는 같은 역 본선 코드로 바꿔 따라감", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--skip-engine-audit", action="store_true")
    a = ap.parse_args()
    try:
        tt = K.locate_source(a.src, K.TT_FILE)
    except FileNotFoundError as e:
        print(f"[건너뜀] {e} — 커밋된 mart 를 그대로 쓸 수 있습니다")
        return 0
    print(f"시각표: {tt}")
    t0 = time.time()
    st, clean = T.train_rows(K.read_timetable(tt))
    MART.mkdir(parents=True, exist_ok=True)
    write_parquet(st, MART / "stop_times.parquet")
    print(f"  정제 {clean['rows_kept']:,}행 · 열차 {clean['trains']:,} ({time.time() - t0:.1f}초)")
    audit = {"clean": clean, "line1_vs_3a1": compare_line1(st)}
    print(f"  1호선 3A-1 대조: {audit['line1_vs_3a1']}")
    try:
        x = K.locate_source(a.src, L9_INFO)
        audit["line9_vs_info"] = compare_line9(st, x)
        print(f"  9호선 운행정보 대조: {audit['line9_vs_info']}")
    except FileNotFoundError:
        print("  9호선 운행정보 없음 — 대조 건너뜀")
    if not a.skip_engine_audit:
        audit["engine"] = engine_audit(st)
    (MART / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
    write_report(audit)
    print(f"저장: {MART.relative_to(ROOT)}/, {REPORT.relative_to(ROOT)}")
    bad = sum(e["impossible"] for e in audit.get("engine", []))
    c1 = audit["line1_vs_3a1"]
    if bad or c1["only_new"] or c1["only_3a1"] or c1["time_diff"]:
        print(f"[확인 필요] 불가능한 연결 {bad} · 1호선 대조 {c1}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
