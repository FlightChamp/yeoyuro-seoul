"""
scripts/v2/55_compare_d111.py — D-111 결과 변화 비교와 기준 G0~G4 (사전 등록 docs/v2/preregistration_d111.md 2-3·3절)
===========================================================================================================

비교: 재계산 b (D-105e 판정 기준, 후보 탐색 v0) ↔ d111 (53 --cand v4 --tag d111). 기본 실행(base) 원자료끼리.

| 기준 | 내용 | 이 스크립트가 하는 일 |
|---|---|---|
| G0 | v0 재현 | 지정 12쌍 × 5시각 = 60질의를 지금 코드에서 cand="v0" 으로 다시 계산 → b 원자료와 확정 추천·참고 경로·상태·기대 소요 비교 |
| G1 | D-105 필수 기준 | d111 보고서(json)에서 B1·B2·B3·B7·B10·B11·B13·오류·빈칸 정차를 53 보고서와 같은 기준으로 읽음 |
| G2 | 확정 추천 나빠짐 0 | 확정 추천이 바뀐 질의마다 두 경로를 같은 평가기(v0 계산기, 그 시각 30분 칸)로 다시 평가 → 엔진 정렬 키 (round(실제,6), 체감) 비교 |
| G3 | 확정 추천 사라짐 0 | b 는 확정 추천이 있는데 d111 은 없는 질의 |
| G4 | 시간 초과·예외 0 | d111 원자료 전체(변형·B4 포함)의 오류 행 + 앱 사례 예외 + 이 스크립트 재계산 예외 |
| G5 | pytest·49·51 | 이 스크립트가 아니라 판정 패치에서 확인 (기본값을 바꾼 상태) |

자체 점검: G2 재평가에 쓰는 평가기가 맞는지 — b 확정 추천 경로를 다시 평가한 실제 시간(소수 3자리)이 b 원자료와 같아야 한다.

실행 (사용자 PC, 53 d111 실행이 끝난 뒤, 약 5~10분 (미확인), 메모리 약 0.7GB)
----
    python scripts/v2/55_compare_d111.py
    python scripts/v2/55_compare_d111.py --dry        # 연습 (질의 몇 개만, 임시 폴더에 저장, 보고 금지)

산출: reports/v2/candidates_d111_compare.{md,json}. 이미 있으면 멈춘다 (덮어쓰기 금지, 다시 하려면 --tag).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
REP = ROOT / "reports" / "v2"
MARTS = ROOT / "data" / "marts" / "v2"
NAME = "candidates_d111_compare"
BASE_TAG, NEW_TAG = "b", "d111"
KEY = ["date", "hhmm", "o", "d"]
P95_LIMIT = 2.6
EXP_TOL = 1e-6          # 저장값(소수 3자리)끼리 비교
CMP_COLS = ("best_path", "ref_path", "status", "best_expected")


def load53():
    spec = importlib.util.spec_from_file_location("m53_d111", ROOT / "scripts" / "v2" / "53_evaluate_korail_3b.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def raw_dir(tag: str) -> Path:
    return MARTS / f"korail_3b_eval_{tag}"


def read_base(tag: str) -> pd.DataFrame:
    f = raw_dir(tag) / "base.parquet"
    if not f.exists():
        raise SystemExit(f"원자료가 없습니다: {f} (53 --tag {tag} 실행 먼저)")
    return pd.read_parquet(f)


def _best(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ("best_path", "ref_path"):
        x[c] = x[c].fillna("")
    return x.set_index(KEY)


# ---------------------------------------------------------------------- 순수 비교 (테스트 대상)
def same_row(a: dict, b: dict) -> bool:
    """G0: 확정 추천·참고 경로·상태·기대 소요(저장값)가 같은가."""
    for c in ("best_path", "ref_path", "status"):
        if (a.get(c) or "") != (b.get(c) or ""):
            return False
    ea, eb = a.get("best_expected"), b.get("best_expected")
    na, nb = ea is None or pd.isna(ea), eb is None or pd.isna(eb)
    if na or nb:
        return na and nb
    return abs(float(ea) - float(eb)) <= EXP_TOL


def sort_key(actual: float, perceived: float) -> tuple:
    """엔진 정렬 키 (time_dependent.evaluate_od 와 같음)."""
    return (round(float(actual), 6), float(perceived))


def is_worse(new_key: tuple, old_key: tuple) -> bool:
    return new_key > old_key


def dist(v) -> dict:
    v = [float(x) for x in v if x is not None and not pd.isna(x)]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(v), "min": round(float(a.min()), 3), "median": round(float(np.median(a)), 3),
            "p95": round(float(np.quantile(a, 0.95)), 3), "max": round(float(a.max()), 3),
            "neg": int((a < -EXP_TOL).sum()), "pos": int((a > EXP_TOL).sum())}


def groups_of(lines: str) -> list[str]:
    g = sorted({x for x in (lines or "").split(",") if x})
    return g or ["3-B 없음"]


def gate_g1(J: dict, Jb: dict) -> dict:
    """G1: d111 보고서에서 D-105 6절 + D-105d 필수 기준 (53 보고서의 판정 표와 같은 식)."""
    B, b1, b1a, b2, b3, b7 = (J.get(k, {}) or {} for k in ("base", "B1_v23", "B1_app", "B2", "B3", "B7"))
    b2b = Jb.get("B2", {}) or {}
    p95 = B.get("sec_p95")
    items = {
        "B1": b1.get("mismatch") == 0 and b1a.get("cases") == 14 and b1a.get("same") == b1a.get("cases"),
        "B2": (not b2.get("t4_mismatch")) and b2.get("t5") == b2b.get("t5")
              and all((b2.get("same_as_3b1_report") or {"x": False}).values()),
        "B3": all(b3.get(k) == 0 for k in ("impossible", "revisit", "double_transfer")),
        "B7": not b7.get("hold") and "hold" in b7,
        "B10": B.get("best_not_connected") == 0,
        "B11": p95 is not None and p95 <= P95_LIMIT,
        "B13": B.get("best_with_ar") == 0 and (J.get("B13_app") or {}).get("violations") == 0,
        "errors": B.get("errors") == 0,
        "blank_stops": b2.get("both_missing_after_load") == 0,
    }
    return {"items": items, "pass": all(items.values()), "p95": p95,
            "values": {"B1": [b1.get("mismatch"), b1a.get("same"), b1a.get("cases")], "B3": [b3.get(k) for k in
                       ("impossible", "revisit", "double_transfer")], "B7_hold": b7.get("hold"),
                       "B10": B.get("best_not_connected"), "B13": [B.get("best_with_ar"), (J.get("B13_app") or {}).get("violations")],
                       "errors": B.get("errors"), "t5": b2.get("t5")}}


def compare_frames(b: pd.DataFrame, n: pd.DataFrame) -> dict:
    """2-3: 같은 질의끼리 결과 변화 (재평가 없이 원자료만)."""
    A, B = _best(b), _best(n)
    idx = A.index.intersection(B.index)
    A, B = A.loc[idx], B.loc[idx]
    okA, okB = A.status != "error", B.status != "error"
    hasA, hasB = A.best_path != "", B.best_path != ""
    ch = (A.best_path != B.best_path) & okA & okB
    both = ch & hasA & hasB
    dexp = (B.best_expected - A.best_expected)[both]
    by_slot = {f"{d} {h}": {"changed": int(g.sum()), "n": int(len(g))}
               for (d, h), g in ch.groupby(level=["date", "hhmm"])}
    by_group: Counter = Counter()
    n_group: Counter = Counter()
    for k in idx:
        gs = set(groups_of(A.loc[k].best_lines)) | set(groups_of(B.loc[k].best_lines))
        for g in gs:
            n_group[g] += 1
            if ch.loc[k]:
                by_group[g] += 1
    trans = Counter(f"{a}→{c}" for a, c in zip(A.status, B.status) if a != c)
    ncand = (B.n_cand - A.n_cand)
    vanished = list(map(list, A[hasA & ~hasB & okA & okB].index))
    return {"compared": int(len(idx)), "missing_in_new": int(len(_best(b).index.difference(B.index))),
            "best_changed": int(ch.sum()), "best_changed_rate": round(float(ch.mean()), 4) if len(idx) else None,
            "ref_changed": int(((A.ref_path != B.ref_path) & okA & okB).sum()),
            "by_slot": by_slot, "by_group": {g: {"changed": int(by_group[g]), "n": int(n_group[g])} for g in sorted(n_group)},
            "expected_diff": dist(dexp), "status_transitions": dict(trans.most_common()),
            "n_cand": {"b_mean": round(float(A.n_cand[okA].mean()), 3), "new_mean": round(float(B.n_cand[okB].mean()), 3),
                       "diff": dist(ncand[okA & okB])},
            "sec": {"b": dist(A.sec[okA]), "new": dist(B.sec[okB]), "increase": dist((B.sec - A.sec)[okA & okB])},
            "vanished": vanished, "changed_keys": list(map(list, A[ch].index))}


# ---------------------------------------------------------------------- 계산기가 필요한 단계
def g0_and_g2(m53, b: pd.DataFrame, n: pd.DataFrame, changed: list, dry: bool) -> dict:
    """시각마다 v0 계산기 하나: ① G0 지정 12쌍 재계산 ② 바뀐 질의의 두 확정 추천 재평가 (G2) ③ 평가기 자체 점검."""
    from yeoyuro_v2 import korail_app as KA
    from yeoyuro_v2 import time_dependent as td
    m53._init_worker("v0")
    A, B = _best(b), _best(n)
    slots = m53.SLOTS[:1] if dry else m53.SLOTS
    ods = m53.DESIGNATED[:2] if dry else m53.DESIGNATED
    g0_rows, g0_bad, g2, selfchk, errs = 0, [], [], [], []
    for date, hm in slots:
        sd = KA.service_day(date, hm)
        rows = m53.plan_task(("base", True, sd.timetable_day_type, date, hm, ods))
        for r in rows:
            k = (date, hm, r["o"], r["d"])
            g0_rows += 1
            if r["status"] == "error":
                errs.append({"stage": "G0", "key": list(k), "err": r.get("err")})
            if k not in A.index:
                g0_bad.append({"key": list(k), "why": "b 원자료에 없음"})
                continue
            old = A.loc[k].to_dict()
            if not same_row(r, old):
                g0_bad.append({"key": list(k), "now": {c: r.get(c) for c in CMP_COLS}, "b": {c: old.get(c) for c in CMP_COLS}})
        P = m53.planner("base", sd.timetable_day_type, date, True)
        m = float(sd.minute)
        ev = P.on.evaluator(td.bin_index_of(m))
        ks = [tuple(k) for k in changed if k[0] == date and k[1] == hm]
        if dry:
            ks = ks[:3]
        for k in ks:
            pa, pb = A.loc[k].best_path, B.loc[k].best_path
            if not pa or not pb:
                continue
            try:
                ra, rb = ev.evaluate(pa.split("|"), m), ev.evaluate(pb.split("|"), m)
            except Exception as e:                               # noqa: BLE001
                errs.append({"stage": "G2", "key": list(k), "err": f"{type(e).__name__}: {e}"[:200]})
                continue
            ka, kb = sort_key(ra.actual_time_min, ra.perceived_time_min), sort_key(rb.actual_time_min, rb.perceived_time_min)
            selfchk.append(abs(round(ra.actual_time_min, 3) - float(A.loc[k].best_expected)) <= 1e-3 + EXP_TOL)
            g2.append({"key": list(k), "b_key": list(ka), "new_key": list(kb), "worse": is_worse(kb, ka),
                       "d_actual": round(kb[0] - ka[0], 3), "d_perceived": round(kb[1] - ka[1], 3),
                       "b_lines": A.loc[k].best_lines, "new_lines": B.loc[k].best_lines})
        print(f"  {date} {hm}: G0 {len(rows)}질의 · G2 재평가 {len(ks)}건", flush=True)
    return {"g0": {"rows": g0_rows, "bad": g0_bad, "pass": g0_rows > 0 and not g0_bad},
            "g2": {"evaluated": len(g2), "worse": [x for x in g2 if x["worse"]], "rows": g2},
            "selfcheck": {"n": len(selfchk), "ok": int(sum(selfchk))}, "errors": errs}


def count_errors(tag: str) -> dict:
    out = {}
    for f in sorted(raw_dir(tag).glob("*.parquet")):
        if f.stem.endswith("_dry"):
            continue
        x = pd.read_parquet(f, columns=["status"])
        out[f.stem] = int((x.status == "error").sum())
    return out


# ---------------------------------------------------------------------- 보고서
def _ok(x: bool) -> str:
    return "**통과**" if x else "**미달** ⚠"


def write(S: dict, dest: Path, name: str) -> None:
    C, G = S["compare"], S["gates"]
    g1 = G["G1"]
    L = ["# D-111 결과 변화 비교 — 재계산 b(v0) ↔ d111(v4)", "",
         "- 규칙: [D-111 사전 등록](../../docs/v2/preregistration_d111.md) 2-3·3절 (실행 전 커밋 c15f3af)",
         "- 재현: `python scripts/v2/53_evaluate_korail_3b.py --workers 8 --cand v4 --tag d111` → `python scripts/v2/55_compare_d111.py`",
         f"- 비교 질의: {C['compared']:,}건 (K3B 312쌍 × 5시각) · d111 에 없는 b 질의 {C['missing_in_new']}",
         "- 실행 PC 조건: b 와 d111 이 같은 PC·같은 작업자 수인지 이 스크립트는 알 수 없음 — 시간 비교는 그 조건을 함께 적어 판정",
         "", "## 기준", "", "| 기준 | 결과 | 판정 |", "|---|---|---|",
         f"| G0 v0 재현 (지정 12쌍 × 5시각) | 같음 {S['g0']['rows'] - len(S['g0']['bad'])}/{S['g0']['rows']} | {_ok(G['G0'])} |",
         f"| G1 D-105 필수 기준 (d111) | {', '.join(f'{k} {_v(v)}' for k, v in g1['items'].items())} · p95 {g1['p95']} | {_ok(g1['pass'])} |",
         f"| G2 확정 추천 나빠짐 | {len(S['g2']['worse'])}건 / 재평가 {S['g2']['evaluated']} | {_ok(G['G2'])} |",
         f"| G3 확정 추천 사라짐 | {len(C['vanished'])}건 | {_ok(G['G3'])} |",
         f"| G4 시간 초과·예외 | 원자료 오류 {S['errors_raw']} · 앱 사례 예외 {S['app_exceptions']} · 재계산 예외 {len(S['errors_here'])} | {_ok(G['G4'])} |",
         "| G5 pytest·49·51 | 판정 패치에서 확인 | — |",
         "", f"- 평가기 자체 점검: b 확정 추천 재평가 = 저장값 {S['selfcheck']['ok']}/{S['selfcheck']['n']}"
         + ("" if S['selfcheck']['ok'] == S['selfcheck']['n'] else " ⚠ (G2 재평가 신뢰 못 함 — 판정 전에 원인 조사)"),
         f"- **G0~G4 전체: {'통과' if G['all'] else '미달'}**", "",
         "## 결과 변화 (보고만, 기준 없음)", "",
         f"- 확정 추천 바뀜: {C['best_changed']:,} / {C['compared']:,} ({100 * (C['best_changed_rate'] or 0):.1f}%) · 참고 경로 바뀜 {C['ref_changed']}",
         f"- 바뀐 건 실제 시간 차이(d111 − b, 분): {C['expected_diff']}",
         f"- 바뀐 건 체감 시간 차이(재평가): {dist([x['d_perceived'] for x in S['g2']['rows']])}",
         f"- 상태 이동: {C['status_transitions'] or '없음'}",
         f"- 후보 카드 수: b 평균 {C['n_cand']['b_mean']} → d111 평균 {C['n_cand']['new_mean']} · 차이 {C['n_cand']['diff']}",
         "", "| 시각 | 바뀜 / 질의 |", "|---|---|"]
    L += [f"| {k} | {v['changed']} / {v['n']} |" for k, v in C["by_slot"].items()]
    L += ["", "| 묶음 (두 확정 추천 중 하나라도 지나면 셈) | 바뀜 / 질의 |", "|---|---|"]
    L += [f"| {k} | {v['changed']} / {v['n']} |" for k, v in C["by_group"].items()]
    s = C["sec"]
    L += ["", "## plan 1회 시간 (초)", "", "| | p50 | p95 | 최대 |", "|---|---|---|---|",
          f"| b (v0) | {s['b'].get('median')} | {s['b'].get('p95')} | {s['b'].get('max')} |",
          f"| d111 (v4) | {s['new'].get('median')} | {s['new'].get('p95')} | {s['new'].get('max')} |",
          f"| 질의별 증가 (d111 − b) | {s['increase'].get('median')} | {s['increase'].get('p95')} | {s['increase'].get('max')} |",
          "", f"- 메모리(작업자 최대 RSS): {S['rss_mb_max'] if S['rss_mb_max'] is not None else '미측정 (psutil 없음)'}"
          + (" MB" if S['rss_mb_max'] is not None else "") + " · b 는 미측정",
          "", "## 사례 — 실제 시간이 가장 많이 줄어든 10건", "", "| 질의 | b 묶음 | d111 묶음 | 실제 차이 | 체감 차이 |", "|---|---|---|---|---|"]
    for x in sorted(S["g2"]["rows"], key=lambda r: r["d_actual"])[:10]:
        L.append(f"| {' '.join(x['key'])} | {x['b_lines'] or '-'} | {x['new_lines'] or '-'} | {x['d_actual']} | {x['d_perceived']} |")
    L += ["", "## 사례 — 지정 12쌍 × 5시각", "", "| 질의 | 바뀜 | b 실제 | d111 실제 |", "|---|---|---|---|"]
    L += [f"| {' '.join(r['key'])} | {'예' if r['changed'] else ''} | {r['b_exp']} | {r['new_exp']} |" for r in S["designated"]]
    if S["g2"]["worse"]:
        L += ["", "## G2 나빠진 질의", ""] + [f"- {x}" for x in S["g2"]["worse"]]
    if S["g0"]["bad"]:
        L += ["", "## G0 다른 질의", ""] + [f"- {x}" for x in S["g0"]["bad"]]
    if C["vanished"]:
        L += ["", "## G3 사라진 질의", ""] + [f"- {x}" for x in C["vanished"]]
    L += ["", "## 한계", "",
          "- 표본은 3-B 역이 끼는 K3B 312쌍 × 5시각 — 3-B 역이 없는 광역 질의의 변화는 이 표본으로 모름 (사전 등록 5절)",
          "- 시간은 53 실행 중 작업자 여럿이 동시에 잰 값 — 작업자 수·PC 상태가 다르면 b 와 직접 비교하기 어렵다",
          "- D-110 감사의 남은 문제(k=40 에서 더 나은 경로 11/420)는 이 비교와 관계없이 남는다", ""]
    (dest / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    (dest / f"{name}.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str) + "\n",
                                       encoding="utf-8", newline="\n")


def _v(x: bool) -> str:
    return "✓" if x else "✗"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--tag", default="", help="다시 만들 때 꼬리표 (첫 보고서는 덮어쓰지 않음)")
    a = ap.parse_args()
    name = NAME + (f"_{a.tag}" if a.tag else "")
    dest = Path(tempfile.gettempdir()) / f"{NAME}_dry" if a.dry else REP
    dest.mkdir(parents=True, exist_ok=True)
    if not a.dry and (REP / f"{name}.md").exists():
        raise SystemExit(f"{name}.md 가 이미 있습니다 — 덮어쓰지 않습니다 (다시 하려면 --tag).")
    jf, jbf = REP / f"korail_3b_eval_{NEW_TAG}.json", REP / f"korail_3b_eval_{BASE_TAG}.json"
    if not jf.exists():
        raise SystemExit(f"d111 보고서가 없습니다: {jf} (53 --cand v4 --tag d111 먼저)")
    J, Jb = json.loads(jf.read_text(encoding="utf-8")), json.loads(jbf.read_text(encoding="utf-8"))
    if J.get("cand") != "v4":
        raise SystemExit(f"d111 보고서의 후보 탐색 설정이 v4 가 아닙니다: {J.get('cand')}")
    t0 = time.time()
    b, n = read_base(BASE_TAG), read_base(NEW_TAG)
    if "cand" in n and set(n.cand.dropna()) != {"v4"}:
        raise SystemExit(f"d111 원자료의 설정이 v4 가 아닙니다: {set(n.cand.dropna())}")
    C = compare_frames(b, n)
    print(f"  비교: {C['compared']}질의 · 확정 추천 바뀜 {C['best_changed']}", flush=True)
    m53 = load53()
    X = g0_and_g2(m53, b, n, C["changed_keys"], a.dry)
    A, B = _best(b), _best(n)
    des = []
    for date, hm in m53.SLOTS:
        for o, d in m53.DESIGNATED:
            k = (date, hm, o, d)
            if k in A.index and k in B.index:
                des.append({"key": list(k), "changed": A.loc[k].best_path != B.loc[k].best_path,
                            "b_exp": A.loc[k].best_expected, "new_exp": B.loc[k].best_expected})
    err_raw = count_errors(NEW_TAG)
    app_exc = sum(len(x.get("exceptions", [])) for x in (J.get("B13_app") or {}).get("cases", []))
    rss = n.rss_mb.dropna() if "rss_mb" in n else pd.Series(dtype=float)
    g1 = gate_g1(J, Jb)
    G = {"G0": X["g0"]["pass"], "G1": g1["pass"], "G2": X["g2"]["evaluated"] > 0 and not X["g2"]["worse"]
         and X["selfcheck"]["ok"] == X["selfcheck"]["n"], "G3": not C["vanished"],
         "G4": sum(err_raw.values()) == 0 and app_exc == 0 and not X["errors"]}
    G["all"] = all(G.values())
    C.pop("changed_keys")
    S = {"compare": C, "g0": X["g0"], "g2": X["g2"], "selfcheck": X["selfcheck"], "errors_here": X["errors"],
         "errors_raw": err_raw, "app_exceptions": app_exc, "rss_mb_max": None if rss.empty else float(rss.max()),
         "designated": des, "gates": {**G, "G1": g1}, "gates_bool": G,     # gates.G1 = 항목별 표, gates_bool = 판정 값
         "elapsed_sec": round(time.time() - t0), "dry": a.dry}
    write(S, dest, name)
    print(f"G0 {G['G0']} · G1 {G['G1']} · G2 {G['G2']} · G3 {G['G3']} · G4 {G['G4']} → 전체 {'통과' if G['all'] else '미달'}")
    print(f"저장: {dest / (name + '.md')}" + ("  [연습 — 보고 금지]" if a.dry else ""))
    return 0


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
