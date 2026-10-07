"""
scripts/v2/28_line9_capacity_check.py
=====================================
v2.6c — 9호선 혼잡도를 1~8호선 정원 기준(1칸 160명)으로 환산해도
"9호선 급행이 가장 구조적인 회랑" 결과가 유지되는가. 기준: docs/v2/preregistration_v26c_capacity.md

C1  cap160 의 Structural 상위 10구간 중 9호선 ≥ 7 → 헤드라인 사용 가능
C2  B군 H1·H2·H3 판정이 Phase B 와 동일

Phase B 의 base evidence(data/marts/v2/l9_b1_w0off.pkl.gz)를 재사용하고 cap160 만 새로 계산한다.

실행
----
    python scripts/v2/28_line9_capacity_check.py          # 8 worker 기준 약 3분

산출물
------
    reports/v2/line9_capacity_check.md
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import line9 as L9                    # noqa: E402

_spec = importlib.util.spec_from_file_location("s27", ROOT / "scripts" / "v2" / "27_line9_evaluation.py")
S27 = importlib.util.module_from_spec(_spec)
sys.modules["s27"] = S27
_spec.loader.exec_module(S27)
_spec = importlib.util.spec_from_file_location("s23b", ROOT / "scripts" / "v2" / "23b_robustness.py")
S23B = importlib.util.module_from_spec(_spec)
sys.modules["s23b"] = S23B
_spec.loader.exec_module(S23B)

PREREG = "docs/v2/preregistration_v26c_capacity.md"
CORRIDOR7 = ["7호선 상봉→면목", "7호선 면목→사가정", "7호선 사가정→용마산", "7호선 용마산→중곡",
             "7호선 중곡→군자", "7호선 군자→어린이대공원", "7호선 어린이대공원→건대입구"]


def edge_ranking(cm: dict) -> list[str]:
    """Structural 단위의 100%+ 노출 합계로 구간 전체 순위."""
    tot = {}
    for it, c in cm.values():
        if c["type"] == "structural":
            for e, v in it["edge_exp100"].items():
                tot[e] = tot.get(e, 0.0) + v
    return [e for e, _ in sorted(tot.items(), key=lambda x: -x[1])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--allow-uncommitted", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    log = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H|%cI", "--", PREREG],
                         capture_output=True, text=True, encoding="utf-8").stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", PREREG],
                           capture_output=True, text=True, encoding="utf-8").stdout.strip()
    if (not log or dirty) and not args.allow_uncommitted:
        print(f"[중단] {PREREG} 를 먼저 커밋하세요.")
        return 2
    print(f"사전 등록: {'확인 ' + log[:8] if log and not dirty else '미확인'}")
    if args.smoke:
        S27.TAG = "_smoke"
    nA, nB = (6, 6) if args.smoke else (300, 300)
    tasks, A, B = S27.build_sample(nA, nB)
    grp = {(o, d): g for o, d, g in tasks}

    reuse = (S27.MDIR / f"l9{S27.TAG}_b1_w0off.pkl.gz").exists()
    base = S27.classify_map(S27.run(tasks, args.workers, "base", "b1_w0off"))
    print(f"[base] Phase B evidence {'재사용' if reuse else '새로 계산 (Phase B 캐시 없음)'} 완료")
    cap = S27.classify_map(S27.run(tasks, args.workers, "cap160", "cap160"))
    print("[cap160] 완료")

    rb, rc = edge_ranking(base), edge_ranking(cap)
    top_c = rc[:10]
    n9 = sum(e.startswith("9") for e in top_c)
    c1 = n9 >= 7
    Bb = {k: v for k, v in base.items() if grp[(k[0], k[1])] == "B"}
    Bc = {k: v for k, v in cap.items() if grp[(k[0], k[1])] == "B"}
    Hb, Hc = S23B.hypotheses(Bb), S23B.hypotheses(Bc)
    c2 = (Hb["h1_pass"], Hb["h2_pass"], Hb["h3_pass"]) == (Hc["h1_pass"], Hc["h2_pass"], Hc["h3_pass"])
    ks = [k for k in base if k in cap and base[k][1]["type"] != "calm"]
    keep = float(np.mean([base[k][1]["type"] == cap[k][1]["type"] for k in ks]))

    def sshare(cm, g):
        t = [c["type"] for (o, d, _), (_, c) in cm.items() if grp[(o, d)] == g and c["type"] != "calm"]
        return float(np.mean([x == "structural" for x in t])) if t else np.nan

    rank = lambda lst, e: (lst.index(e) + 1) if e in lst else "—"
    pct = S27.pct
    L = ["# v2.6c 9호선 정원 기준 보정 검사", "",
         f"- 사전 등록: `{PREREG}` " + (f"커밋 `{log[:10]}`" if log and not dirty else "**미확인**"),
         f"- 보정: 9호선 혼잡 × {L9.CAP160_FACTOR:.4f} (= 922/6 ÷ 160) · 표본 Phase B 와 동일 (A {len(A)} + B {len(B)} OD × 6시각)",
         f"- 실행 시간 {(time.time() - t0) / 60:.1f}분", "",
         "## 판정", "",
         "| 기준 | 결과 | 판정 |", "|---|---|---|",
         f"| C1 cap160 Structural 상위 10구간 중 9호선 ≥ 7 | {n9}개 | {'충족 → 9호선 급행 회랑 헤드라인 사용 가능' if c1 else '미충족 → 헤드라인은 7호선 회랑 유지, 9호선은 정원 기준에 민감'} |",
         f"| C2 B군 H1·H2·H3 판정 동일 | base ({Hb['h1_pass']}, {Hb['h2_pass']}, {Hb['h3_pass']}) / cap160 ({Hc['h1_pass']}, {Hc['h2_pass']}, {Hc['h3_pass']}) | {'동일' if c2 else '다름 — 해당 가설 정원 기준에 민감'} |",
         "", "## 함께 보고", "",
         f"- 혼잡 이동 유형 유지율 (base 대비) {pct(keep)} ({len(ks)}단위)",
         f"- 혼잡 이동 중 Structural: A군 {pct(sshare(base, 'A'))} → {pct(sshare(cap, 'A'))}, "
         f"B군 {pct(sshare(base, 'B'))} → {pct(sshare(cap, 'B'))}",
         f"- B군 H1 {pct(Hb['h1'])} → {pct(Hc['h1'])}, H2 {pct(Hb['h2'])} → {pct(Hc['h2'])}, "
         f"H3 상위 10구간 {pct(Hb['h3_obs'])} (p={Hb['h3_p']:.3f}) → {pct(Hc['h3_obs'])} (p={Hc['h3_p']:.3f})", "",
         "Structural 상위 10구간:", "",
         pd.DataFrame({"순위": range(1, 11), "base": (rb + [""] * 10)[:10], "cap160": (rc + [""] * 10)[:10]})
         .to_markdown(index=False), "",
         "v2.3 7호선 회랑의 전체 순위:", "",
         pd.DataFrame({"구간": CORRIDOR7, "base 순위": [rank(rb, e) for e in CORRIDOR7],
                       "cap160 순위": [rank(rc, e) for e in CORRIDOR7]}).to_markdown(index=False), "",
         f"- Structural 노출이 있는 구간 수: base {len(rb)}, cap160 {len(rc)}",
         "", "## 해석 원칙", "", "- 기준 미달은 그대로 보고하고 사전 등록의 조치만 따른다. 보정 계수를 바꿔 다시 돌리지 않는다."]
    rp = ROOT / "reports" / "v2" / f"line9_capacity_check{S27.TAG}.md"
    rp.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n저장: {rp}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
