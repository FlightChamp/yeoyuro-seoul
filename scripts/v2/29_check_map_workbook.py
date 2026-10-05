"""
scripts/v2/29_check_map_workbook.py
===================================
기본 좌표 워크북(1~9호선)을 검사하고 미리보기를 만든다. 앱이 쓰는 data/master 의 노선도 csv 는 건드리지 않는다.

- 별도 폴더(data/interim/v26_mapcheck)에 1~9호선 경로 그래프를 만들고 v1 가져오기 스크립트(14)로 검증
- '확인 필요' 행(다른 역과 좌표가 같은 노드) 목록
- 미리보기: reports/figures/vector_map_preview_v26.png

실행
----
    python scripts/v2/29_check_map_workbook.py
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
from yeoyuro_v2 import line9 as L9      # noqa: E402

BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(BASE))
    args = ap.parse_args(argv)
    x = Path(args.xlsx)
    v = pd.read_excel(x, sheet_name="Station_Visual_Nodes")
    print(f"[1] 노드 {len(v)}개 · 노선 {sorted(v.line_id.unique().tolist())} · 좌표 빈칸 "
          f"{int(v.x_px.isna().sum() + v.y_px.isna().sum())}개")
    chk = v[v.mark_status.astype(str) == "확인 필요"]
    if len(chk):
        print(f"    확인 필요 {len(chk)}개:")
        for r in chk.itertuples():
            print(f"      - {r.visual_node_id} ({r.x_px}, {r.y_px}) {r.memo}")
    root = ROOT / "data" / "interim" / "v26_mapcheck"
    if root.exists():
        shutil.rmtree(root)
    L9.build_shadow_root(ROOT, root, "base")
    (root / "reports" / "data_quality").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    print("[2] 가져오기 검증 (별도 폴더)")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "14_import_map_workbook.py"), "--root", str(root),
                        "--xlsx", str(x), "--preview"], capture_output=True, text=True, encoding="utf-8")
    val = root / "reports" / "data_quality" / "map_workbook_validation.csv"
    if not val.exists():
        print("    [실패]", r.stdout[-1500:], r.stderr[-1500:])
        return 1
    t = pd.read_csv(val)
    print(t[["check", "status", "detail"]].to_string(index=False))
    png = root / "reports" / "figures" / "vector_map_preview.png"
    out = ROOT / "reports" / "figures" / "vector_map_preview_v26.png"
    if png.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(png, out)
        print(f"\n미리보기: {out}")
    fails = int((t.status == "FAIL").sum())
    print(f"\n판정: {'통과' if fails == 0 else f'FAIL {fails}개'}"
          + (f" (확인 필요 {len(chk)}개는 좌표를 고치면 사라짐)" if len(chk) else ""))
    return 0 if fails == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
