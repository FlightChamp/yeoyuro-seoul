"""
scripts/v2/29_check_map_workbook.py
===================================
기본 좌표 파일(1~9호선)을 **읽기만** 해서 검사하고 미리보기를 만든다. 사용자 파일에는 절대 쓰지 않는다.
엑셀을 열어 둔 채로 실행해도 된다 (저장한 내용 기준).

- 역명·클릭 위치 = 그 역 노선별 점 평균 + 이동량(label_dx/dy, click_dx/dy) 을 직접 계산 (수식 캐시를 믿지 않음)
- 별도 폴더(data/interim/v26_mapcheck)에서 1~9호선 경로 그래프 + v1 가져오기 스크립트(14)로 검증
- 다른 역끼리 좌표가 같은 점 목록, 미리보기 PNG

실행
----
    python scripts/v2/29_check_map_workbook.py           # 한 번 검사
    python scripts/v2/29_watch_map.py                    # 저장할 때마다 자동 검사 + 브라우저 미리보기
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2 import line9 as L9
from yeoyuro_v2.proc import run_py   # Windows 하위 프로세스 한글 출력 오류 방지                       # noqa: E402
from yeoyuro_v2.map_workbook import materialize, write_values_copy   # noqa: E402

BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
CHECK_ROOT = ROOT / "data" / "interim" / "v26_mapcheck"
PREVIEW = ROOT / "reports" / "figures" / "vector_map_preview_v26.png"


def prepare_root() -> Path:
    if not (CHECK_ROOT / "data" / "marts" / "route_edge_mart.parquet").exists():
        L9.build_shadow_root(ROOT, CHECK_ROOT, "base")
    (CHECK_ROOT / "reports" / "data_quality").mkdir(parents=True, exist_ok=True)
    (CHECK_ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    return CHECK_ROOT


def check(path: Path, quiet: bool = False, frames: dict | None = None) -> dict:
    """검사 1회. frames 를 주면 (열린 엑셀에서 읽은 표) 파일 대신 그것으로 계산한다."""
    from yeoyuro_v2.map_workbook import materialize_frames
    m = materialize_frames(frames) if frames is not None else materialize(path)
    root = prepare_root()
    png = root / "reports" / "figures" / "vector_map_preview.png"
    val = root / "reports" / "data_quality" / "map_workbook_validation.csv"
    for f in (png, val):                       # 이전 결과를 지워 둔다 → 실패하면 예전 그림이 '새 결과'로 나가지 않음
        f.unlink(missing_ok=True)
    vals = write_values_copy(m, root / "workbook_values.xlsx")
    r = run_py([str(ROOT / "scripts" / "14_import_map_workbook.py"), "--root", str(root),
                        "--xlsx", str(vals), "--preview"])
    t = pd.read_csv(val) if val.exists() else pd.DataFrame(columns=["check", "status", "detail"])
    preview_ok = png.exists()
    if png.exists():
        PREVIEW.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(png, PREVIEW)
    fails = int((t.status == "FAIL").sum()) if len(t) else -1
    from yeoyuro_v2.map_workbook import snapshot
    return {"preview_ok": preview_ok, "frames": m, "snap": snapshot(m), "ok": fails == 0 and not m["_missing"], "clash": m["_clash"], "missing": m["_missing"],
            "fails": fails, "table": t, "stderr": r.stderr[-1500:], "preview": PREVIEW,
            "nodes": len(m["Station_Visual_Nodes"])}


def report(res: dict, path: Path) -> None:
    print(f"읽은 파일: {path}  (저장 시각 {datetime.fromtimestamp(path.stat().st_mtime):%H:%M:%S})")
    print(f"[1] 노드 {res['nodes']}개 · 좌표 빈칸 {len(res['missing'])}개" +
          (f" ({', '.join(res['missing'][:10])})" if res["missing"] else ""))
    if res["clash"]:
        print(f"    다른 역끼리 같은 좌표 {len(res['clash'])}곳 (엑셀에서도 주황색으로 보임):")
        for x, y, ks in res["clash"]:
            print(f"      - ({x}, {y}): {', '.join(ks)}")
    if res["fails"] < 0:
        print("[2] 가져오기 검증 실패:", res["stderr"])
        return
    print("[2] 가져오기 검증")
    print(res["table"][["check", "status", "detail"]].to_string(index=False))
    print(f"\n미리보기: {res['preview']}")
    print(f"판정: {'통과' if res['ok'] else 'FAIL ' + str(res['fails']) + '개'}"
          + (f" · 겹침 {len(res['clash'])}곳" if res["clash"] else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(BASE))
    args = ap.parse_args(argv)
    x = Path(args.xlsx)
    res = check(x)
    report(res, x)
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
