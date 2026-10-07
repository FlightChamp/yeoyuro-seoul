"""
scripts/v2/30b_check_map_workbook.py
====================================
기본 좌표 워크북(1~9호선)을 검사하고 미리보기 이미지를 만든다. 앱이 쓰는 data/master 의 노선도 csv 는 바꾸지 않는다.

1. 추가 검사: 다른 역끼리 같은 좌표(FAIL), mark_status 가 '완료' 아닌 노드 목록
2. 별도 폴더(data/interim/map_check)에서 v1 가져오기 스크립트(14) 실행 → 검증표 + 미리보기

실행
----
    python scripts/v2/30b_check_map_workbook.py
    python scripts/v2/30b_check_map_workbook.py --xlsx "<다른 워크북>"

결과
----
    reports/figures/vector_map_preview.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2 import line9 as L9
from yeoyuro_v2.proc import run_py   # Windows 하위 프로세스 한글 출력 오류 방지      # noqa: E402

DEFAULT = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(DEFAULT))
    args = ap.parse_args(argv)
    x = Path(args.xlsx)
    if not x.exists():
        print(f"[중단] 워크북이 없습니다: {x}")
        return 1
    v = pd.read_excel(x, sheet_name="Station_Visual_Nodes")
    over = []
    for (px, py), g in v.groupby(["x_px", "y_px"]):
        st = set(g.visual_node_id.str.rsplit("_L", n=1).str[0])
        if len(st) > 1:
            over.append(f"({int(px)}, {int(py)}): {', '.join(sorted(st))}")
    pending = v[v.mark_status.astype(str) != "완료"]
    print(f"[1] 노드 {len(v)}개 (노선별 {v.line_id.value_counts().sort_index().to_dict()})")
    print(f"    다른 역끼리 같은 좌표: {len(over)}곳" + ("".join(f"\n      - {o}" for o in over)))
    print(f"    '완료' 아닌 노드: {len(pending)}개" + (f" → {', '.join(pending.visual_node_id)}" if len(pending) else ""))

    root = ROOT / "data" / "interim" / "map_check"
    if root.exists():
        shutil.rmtree(root)
    L9.build_shadow_root(ROOT, root, "base")
    (root / "reports" / "data_quality").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    print("[2] 가져오기 스크립트 실행 (별도 폴더, 앱 노선도는 그대로)")
    r = run_py([str(ROOT / "scripts" / "14_import_map_workbook.py"), "--root", str(root),
                        "--xlsx", str(x), "--preview"])
    val = root / "reports" / "data_quality" / "map_workbook_validation.csv"
    if not val.exists():
        print("    [실패] 가져오기 스크립트 오류:\n", r.stdout[-2000:], r.stderr[-2000:])
        return 1
    t = pd.read_csv(val)
    print(t[["check", "status", "detail"]].to_string(index=False))
    png = root / "reports" / "figures" / "vector_map_preview.png"
    out = ROOT / "reports" / "figures" / "vector_map_preview.png"
    if png.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(png, out)
        print(f"\n미리보기: {out}")
    fails = int((t.status == "FAIL").sum()) + len(over)
    print(f"\n판정: {'통과' if fails == 0 else f'FAIL {fails}개'}"
          + (f" · 확인 필요 노드 {len(pending)}개" if len(pending) else ""))
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
