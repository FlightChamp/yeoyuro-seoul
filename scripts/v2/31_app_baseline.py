"""
scripts/v2/31_app_baseline.py
=============================
IA 리팩토링 전 앱의 계산 결과·화면 표시값을 기준값 파일로 저장한다.
리팩토링 뒤 tests/test_app_ia.py 가 이 파일과 비교해 '계산 결과 불변'을 확인한다.

    python scripts/v2/31_app_baseline.py          # tests/fixtures/app_baseline.json

반드시 리팩토링 전 커밋에서 한 번만 실행한다 (파일 안에 생성 커밋을 기록).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from yeoyuro_v2.proc import utf8_stdout   # noqa: E402
import app_harness as H                    # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "app_baseline.json"


def main() -> int:
    utf8_stdout()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    out = {"commit": commit, "cases": {}}
    for c in H.CASES:
        eng = H.engine_snapshot(c)
        scr = H.screen_snapshot(H.search(c))
        if scr["exceptions"]:
            print("[중단] 화면 오류:", H.case_id(c), scr["exceptions"])
            return 1
        out["cases"][H.case_id(c)] = {"engine": eng, "screen": scr}
        r = eng["recommended"]
        print(f"{H.case_id(c):40s} 추천 {r['actual']:.1f}분 노출 {r['exp100']:.1f} · 판정 "
              f"{eng['counterfactual']['type']} · 화면 metric {len(scr['metrics'])}개")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {OUT} (생성 커밋 {commit}, 사례 {len(out['cases'])}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
