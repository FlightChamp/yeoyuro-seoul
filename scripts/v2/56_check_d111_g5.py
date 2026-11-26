"""
scripts/v2/56_check_d111_g5.py — D-111 G5 앱 확인 (D-111c 사후 규칙: docs/v2/d111_g5_method.md)
=====================================================================================

D-111 사전 등록 G5 = "기본값을 바꾼 상태로 pytest·49·51 통과". 49·51 의 화면 검사(W1·W3·W4·H6)는 구현 전 앱과
비교하는 옛 기준이라 D-103·D-106 이 화면을 바꾼 뒤로 v0 에서도 이미 맞지 않는다. 그래서 같은 코드에서
기본 후보 탐색을 v0 과 v4 로 각각 두고 49·51 의 같은 검사를 돌려 비교한다 (pytest 는 판정 패치가 따로 돌린다).

| 묶음 | 무엇 | 기준 |
|---|---|---|
| S1 끔 화면 | W1 14사례 + H7 공휴일 3사례 — 지금 앱 화면 지문(지표·문구·안내·경고·역 목록·노선도 sha256·예외) | v0 = v4 (모두) |
| S2 켬 화면 구조 | W3 역 목록·전환, W4 노선도, W6·W7 3사례의 문구 표시·기존 카드 = 끔·예외 | v0 = v4 (시간 값 제외) |
| S3 자정 사례 | W5 | v0 = v4 |
| S4 옛 기준 비교 | 49 run_w1, 51 run_app (구현 전 앱과 비교한 결과) | v0 = v4 (차이 목록이 같음). 옛 기준 충족 여부는 보고만 |
| S5 3-B 끔 계산기 | W2 412쌍 × 41시각, H9 412쌍 × 3 (공휴일 전·후) | 원래 절대 기준: 오류 0 · 확정 추천 연결 불가 0 |
| 보고 | W6·W7 켬 화면 전체 지문이 v0 과 다른 사례 (확정 추천이 바뀌면 생길 수 있음) | 기준 없음 |

S1~S5 가 모두 맞으면 G5(49·51 부분) 통과. S1~S4 에서 차이가 나면 '보류' — 기본값을 바꾸지 않고 차이를 보고한다.

실행 (사용자 PC, 판정 패치가 부름, 약 30~40분 (미확인), 작업자 8개면 메모리 약 4~6GB (미확인))
----
    python scripts/v2/56_check_d111_g5.py --workers 8
    python scripts/v2/56_check_d111_g5.py --dry          # 연습 (사례·표본 몇 개만, 임시 폴더, 보고 금지)

산출: reports/v2/d111_g5.{md,json} (이미 있으면 멈춤). 종료 코드 0 = 통과, 2 = 보류·미달, 1 = 실행 오류.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
REP = ROOT / "reports" / "v2"
NAME = "d111_g5"
HOL_CASES = [("신림", "강남", "2026-05-01", "08:00", "calm"), ("홍대입구", "잠실", "2025-06-03", "18:00", "calm"),
             ("상봉", "건대입구", "2026-05-02", "00:30", "calm")]           # 51 run_app 과 같은 공휴일 사례
ON_CASES = [("신도림", "석계", "2026-11-25", "08:00", "calm"), ("가산디지털단지", "월계", "2026-11-25", "08:00", "calm"),
            ("금천구청", "월계", "2026-11-28", "00:30", "calm")]           # 49 W6·W7 과 같은 켬 사례


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------- 순수 함수 (테스트 대상)
def strip_sec(x):
    """시간 값(이름이 _sec 로 끝나는 키)을 뺀 사본 — 실행마다 달라지는 값."""
    if isinstance(x, dict):
        return {k: strip_sec(v) for k, v in x.items() if not str(k).endswith("_sec")}
    if isinstance(x, (list, tuple)):
        return [strip_sec(v) for v in x]
    return x


def sha(s) -> str:
    return hashlib.sha256((s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, sort_keys=True, default=str))
                          .encode("utf-8")).hexdigest()[:16]


def struct_on(c: dict) -> dict:
    """W6·W7 사례에서 비교할 구조 (문구 표시·기존 카드 = 끔·예외)."""
    return {"case": c["case"], "texts": c["texts"], "base_card_same_as_off": c["base_card_same_as_off"],
            "exceptions": c["exceptions"]}


def diff_keys(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def judge(v0: dict, v4: dict, P: dict) -> dict:
    """D-111c 판정 식."""
    a0, a4 = v0["APP"], v4["APP"]
    s1 = diff_keys(v0["off_screens"], v4["off_screens"])
    s2 = {"W3": strip_sec(a0["W3"]) != strip_sec(a4["W3"]), "W4": a0["W4"] != a4["W4"],
          "W6_W7": [struct_on(c) for c in a0["W6_W7"]] != [struct_on(c) for c in a4["W6_W7"]]}
    s3 = v0["W5"] != v4["W5"]
    s4 = {"W1": v0["old_W1"] != v4["old_W1"], "H6_H7": v0["old_H6_H7"] != v4["old_H6_H7"]}
    w2, h9 = P["W2"], P["H9"]
    s5 = {"W2_errors": w2["errors"], "W2_best_not_connected": w2["best_not_connected"],
          "H9_errors": h9["errors"], "H9_best_not_connected_after": h9["best_not_connected_after"]}
    abs4 = {"W3": a4["W3"]["dup_on"] == 0 and a4["W3"]["cleared_on_off"] and a4["W3"]["dest_kept"]
            and a4["W3"]["flash"] and not a4["W3"]["exceptions"],
            "W6_W7": all(c["base_card_same_as_off"] and not c["exceptions"] for c in a4["W6_W7"])}
    on_changed = diff_keys(v0["on_screens"], v4["on_screens"])
    G = {"S1": not s1, "S2": not any(s2.values()), "S3": not s3, "S4": not any(s4.values()),
         "S5": all(v == 0 for v in s5.values()), "abs_v4": all(abs4.values())}
    G["pass"] = all(G.values())
    return {"gates": G, "S1_diff": s1, "S2_diff": s2, "S3_diff": s3, "S4_diff": s4, "S5": s5, "abs_v4": abs4,
            "on_screens_changed": on_changed}


# ---------------------------------------------------------------------- 자식 프로세스: 한 설정의 화면 검사
def screen_fp(m49, at) -> dict:
    charts = at.get("plotly_chart")
    spec = charts[0].proto.spec if charts else ""
    return {"metrics": sha([(m.label, m.value, m.delta or "") for m in at.metric]),
            "captions": sha([c.value for c in at.caption]), "markdown": sha([m.value for m in at.markdown]),
            "info": sha([x.value for x in at.info]), "warning": sha([x.value for x in at.warning]),
            "options": sha(list(at.selectbox(key="sel_origin").options)), "map": sha(spec),
            "exceptions": [e.value[:300] for e in at.exception]}


def child(cand: str, out: Path, dry: bool) -> int:
    from yeoyuro_v2 import korail_app as KA
    KA.DEFAULT_CAND = cand                     # 앱(AppTest 는 같은 프로세스에서 실행)과 계산기 모두 이 기본값
    seen: set = set()                           # 실제로 만들어진 계산기의 (3-B 켬, 설정) — 기본값이 앱까지 갔는지 확인
    _init = KA.ExtensionPlanner.__init__

    def _rec(self, *args, **kw):
        _init(self, *args, **kw)
        seen.add((bool(self.include_3b), self.cand))
    KA.ExtensionPlanner.__init__ = _rec
    m49 = _load("m49_g5", "scripts/v2/49_check_korail_app.py")
    m51 = _load("m51_g5", "scripts/v2/51_check_holidays.py")
    w1 = m49.W1_CASES[:2] if dry else m49.W1_CASES
    hol = HOL_CASES[:1] if dry else HOL_CASES
    S: dict = {"cand": cand, "default_seen": KA.DEFAULT_CAND, "off_screens": {}, "on_screens": {}}
    t0 = time.time()
    for c in list(w1) + list(hol):
        at = m49._at(m49.APP)
        m49._search(at, *c)
        S["off_screens"][" ".join(c)] = screen_fp(m49, at)
    for c in ON_CASES[:1] if dry else ON_CASES:
        at = m49._at(m49.APP)
        at.toggle(key="ext_on").set_value(True)
        at.run()
        m49._search(at, *c)
        S["on_screens"][" ".join(c)] = screen_fp(m49, at)
    print(f"    [{cand}] 화면 지문 ({time.time() - t0:.0f}초)", flush=True)
    S["APP"] = strip_sec(m49.run_app_checks())
    S["W5"] = m49.run_w5()
    S["old_W1"] = m49.run_w1(w1)
    S["old_H6_H7"] = m51.run_app(dry)
    print(f"    [{cand}] 49·51 화면 검사 끝 ({time.time() - t0:.0f}초)", flush=True)
    S["planners_seen"] = sorted(list(x) for x in seen)
    out.write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


def planners_ok(cand: str, seen: list) -> bool:
    """3-B 켬 계산기는 그 설정, 3-B 끔 계산기는 v0 이어야 하고, 3-B 켬 계산기가 한 번 이상 만들어져야 한다."""
    s = {(bool(a), b) for a, b in seen}
    return (True, cand) in s and all(b == (cand if a else "v0") for a, b in s)


# ---------------------------------------------------------------------- 3-B 끔 계산기 (W2·H9, 한 번)
def planners(workers: int, dry: bool) -> dict:
    import pandas as pd
    m49 = _load("m49_g5p", "scripts/v2/49_check_korail_app.py")
    m51 = _load("m51_g5p", "scripts/v2/51_check_holidays.py")
    e47 = m49._e47()
    ods = [(o, d) for o, d, _ in e47.k_sample(e47.router(e47.spec(on=True)).display.station_key.unique())]
    times = m49.W2_TIMES
    if dry:
        ods, times = ods[:4], m49.W2_TIMES[::20]
    t0 = time.time()
    w2 = m49.analyze_w2(m49.run_w2(ods, times, 1 if dry else workers))
    print(f"    W2 {len(ods)}쌍 × {len(times)}시각 ({time.time() - t0:.0f}초)", flush=True)
    t0 = time.time()
    h9o = ods[:3] if dry else ods
    df_b = m51.run_pool([("before", o, d) for o, d in h9o], 1 if dry else workers)
    df_a = m51.run_pool([("after", o, d) for o, d in h9o], 1 if dry else workers)
    h9 = m51.analyze_h9(pd.concat([df_b, df_a], ignore_index=True))
    print(f"    H9 {len(h9o)}쌍 × 3 ({time.time() - t0:.0f}초)", flush=True)
    return {"W2": strip_sec({k: v for k, v in w2.items() if k != "error_examples"}) | {"error_examples": w2.get("error_examples")},
            "H9": {k: h9[k] for k in ("compared", "errors", "changed", "best_not_connected_after", "status_after")}}


# ---------------------------------------------------------------------- 보고서
def _ok(x) -> str:
    return "**같음/통과**" if x else "**다름/미달** ⚠"


def write(S: dict, dest: Path) -> None:
    J, v0 = S["judge"], S["v0"]
    G = J["gates"]
    a0 = v0["APP"]
    L = ["# D-111 G5 — 기본 후보 탐색 v0 ↔ v4 앱 확인 (49·51)", "",
         "- 규칙: [D-111c 사후 규칙](../../docs/v2/d111_g5_method.md) (G5 측정 전 커밋) · [D-111 사전 등록](../../docs/v2/preregistration_d111.md) 3절 G5",
         "- 재현: `python scripts/v2/56_check_d111_g5.py --workers 8`",
         f"- 걸린 시간: {S['elapsed_sec']}초 · 앱 안에서 만들어진 계산기 (3-B 켬, 설정): v0 {v0['planners_seen']} · v4 {S['v4']['planners_seen']}", "",
         "| 묶음 | 내용 | 결과 |", "|---|---|---|",
         f"| S1 끔 화면 지문 | {len(v0['off_screens'])}사례 | {_ok(G['S1'])} {J['S1_diff'] or ''} |",
         f"| S2 켬 화면 구조 | W3·W4·W6·W7 | {_ok(G['S2'])} {J['S2_diff']} |",
         f"| S3 자정 사례 W5 | | {_ok(G['S3'])} |",
         f"| S4 옛 기준 비교 결과 | 49 W1 · 51 H6·H7 | {_ok(G['S4'])} {J['S4_diff']} |",
         f"| S5 3-B 끔 계산기 | W2 {S['P']['W2'].get('plans')}건 · H9 {S['P']['H9'].get('compared')}건 | {_ok(G['S5'])} {J['S5']} |",
         f"| v4 절대 기준 | W3 목록 전환·W6·W7 기존 카드·예외 | {_ok(G['abs_v4'])} {J['abs_v4']} |",
         "", f"- **G5 (49·51 부분): {'통과' if G['pass'] else '보류'}** · pytest 는 판정 패치가 기본값 v4 상태로 따로 돌린다", "",
         "## 옛 기준 충족 여부 (v0, 보고만 — D-103·D-106 이 화면을 바꾼 결과)", "",
         f"- 49 W1 구현 전 앱과 같음 {v0['old_W1']['same']}/{v0['old_W1']['cases']} · 다른 항목 "
         f"{sorted({k for d in v0['old_W1']['diffs'] for k in d['keys']})}",
         f"- 49 W3 켬 역 +{a0['W3']['added_n']} (D-101 기준 +18, D-106 뒤 3-B 18역 추가) · W4 끔 점선 {a0['W4'].get('off_dotted')} (D-106)",
         f"- 51 H6 구현 전 앱과 같음 {v0['old_H6_H7']['day_same']}/{v0['old_H6_H7']['day_cases']}",
         "", "## 켬 화면 전체 지문이 달라진 사례 (보고만)", "",
         f"- {J['on_screens_changed'] or '없음'} — 확정 추천이 바뀌면 생길 수 있다 (D-111 비교 보고서의 결과 변화와 같은 원인인지는 사례별 확인)",
         ""]
    (dest / f"{NAME}.md").write_text("\n".join(str(x) for x in L) + "\n", encoding="utf-8", newline="\n")
    (dest / f"{NAME}.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=str) + "\n",
                                       encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--child", choices=["v0", "v4"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.child:
        return child(a.child, Path(a.out), a.dry)
    dest = Path(tempfile.gettempdir()) / f"{NAME}_dry" if a.dry else REP
    dest.mkdir(parents=True, exist_ok=True)
    if not a.dry and (REP / f"{NAME}.md").exists():
        raise SystemExit(f"{NAME}.md 가 이미 있습니다 — 덮어쓰지 않습니다.")
    t0 = time.time()
    res = {}
    with tempfile.TemporaryDirectory() as td:
        for cand in ("v0", "v4"):
            print(f"  [{cand}] 화면 검사 (새 프로세스)", flush=True)
            f = Path(td) / f"{cand}.json"
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", cand, "--out", str(f)]
                               + (["--dry"] if a.dry else []), cwd=str(ROOT))
            if r.returncode != 0 or not f.exists():
                print(f"[실행 오류] {cand} 화면 검사가 끝나지 않았습니다 (종료 코드 {r.returncode})")
                return 1
            res[cand] = json.loads(f.read_text(encoding="utf-8"))
            if res[cand]["default_seen"] != cand or not planners_ok(cand, res[cand]["planners_seen"]):
                print(f"[실행 오류] {cand} 프로세스 기본값 {res[cand]['default_seen']} · 만들어진 계산기 {res[cand]['planners_seen']}")
                return 1
    print("  3-B 끔 계산기 W2·H9", flush=True)
    P = planners(a.workers, a.dry)
    J = judge(res["v0"], res["v4"], P)
    S = {"v0": res["v0"], "v4": res["v4"], "P": P, "judge": J, "elapsed_sec": round(time.time() - t0), "dry": a.dry}
    write(S, dest)
    G = J["gates"]
    print(" · ".join(f"{k} {v}" for k, v in G.items()))
    print(f"저장: {dest / (NAME + '.md')}" + ("  [연습 — 보고 금지]" if a.dry else ""))
    return 0 if G["pass"] else 2


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
