"""
scripts/v2/29_diagnose_map_edit.py
==================================
"좌표를 고쳤는데 미리보기가 그대로" 일 때 원인을 찾는다. 아무 파일도 고치지 않는다 (읽기 전용).

확인하는 것
-----------
1. 검사 스크립트가 읽는 기본 파일의 경로·수정 시각, 그 안의 확인 대상 노드 좌표
2. 프로젝트 폴더·다운로드 폴더에 있는 다른 좌표 워크북 사본들 (다른 파일을 고쳤을 가능성)
3. 엑셀이 아직 파일을 열고 있는지 (~$ 잠금 파일)
4. 검사 스크립트가 동기화 기능이 있는 최신 버전인지
5. 미리보기 그림이 기본 파일보다 나중에 만들어졌는지

실행
----
    python scripts/v2/29_diagnose_map_edit.py
    python scripts/v2/29_diagnose_map_edit.py --nodes 흑석_L9 봉은사_L9 개화_L9
"""

from __future__ import annotations

import sys

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
PREVIEW = ROOT / "reports" / "figures" / "vector_map_preview_v26.png"
CHECK = ROOT / "scripts" / "v2" / "29_check_map_workbook.py"


def ts(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def coords(p: Path, nodes: list[str]) -> str:
    try:
        v = pd.read_excel(p, sheet_name="Station_Visual_Nodes")
    except Exception as e:                          # noqa: BLE001
        return f"(읽기 실패: {type(e).__name__})"
    v = v.set_index("visual_node_id")
    out = []
    for n in nodes:
        if n in v.index:
            out.append(f"{n}=({v.at[n, 'x_px']}, {v.at[n, 'y_px']})")
        else:
            out.append(f"{n}=없음")
    return ", ".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", nargs="+", default=["흑석_L9", "봉은사_L9", "동작_L9", "종합운동장_L9"])
    args = ap.parse_args(argv)
    print("[1] 검사 스크립트가 읽는 기본 파일")
    if not BASE.exists():
        print(f"    없음: {BASE}")
        return 1
    print(f"    {BASE}")
    print(f"    수정 시각 {ts(BASE)} · 좌표 {coords(BASE, args.nodes)}")

    print("\n[2] 다른 좌표 워크북 사본 (여기를 고쳤다면 기본 파일에는 반영되지 않음)")
    places = [ROOT, Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads",
              Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop",
              Path(os.environ.get("USERPROFILE", Path.home())) / "Documents"]
    found = []
    for base in places:
        if not base.exists():
            continue
        try:
            for p in base.rglob("*vector_map_coordinate_workbook*.xlsx"):
                if p.resolve() != BASE.resolve() and not p.name.startswith("~$") and "archive" not in p.parts \
                        and "interim" not in p.parts:
                    found.append(p)
        except PermissionError:
            pass
    if not found:
        print("    없음")
    for p in sorted(found, key=lambda q: q.stat().st_mtime, reverse=True)[:8]:
        print(f"    - {p}\n      수정 {ts(p)} · {coords(p, args.nodes)}")
    newer = [p for p in found if p.stat().st_mtime > BASE.stat().st_mtime]
    if newer:
        print(f"    [의심] 기본 파일보다 나중에 수정된 사본이 {len(newer)}개 있습니다. 그 파일을 고쳤을 가능성이 큽니다.")

    print("\n[3] 엑셀이 기본 파일을 아직 열고 있는가")
    lock = BASE.parent / ("~$" + BASE.name)
    print(f"    {'예 — 엑셀을 완전히 닫은 뒤 다시 실행하세요' if lock.exists() else '아니오'}")

    print("\n[4] 검사 스크립트 버전")
    txt = CHECK.read_text(encoding="utf-8") if CHECK.exists() else ""
    print(f"    {'최신 (동기화 기능 있음)' if 'def sync' in txt else '예전 버전 — 동기화 zip(yeoyuro_v2_6_map_sync) 적용 필요'}")

    print("\n[5] 미리보기 그림")
    if PREVIEW.exists():
        late = PREVIEW.stat().st_mtime >= BASE.stat().st_mtime
        print(f"    수정 시각 {ts(PREVIEW)} · {'기본 파일보다 나중 (정상)' if late else '기본 파일보다 이전 → 검사가 끝까지 돌지 않았거나 실패'}")
    else:
        print("    없음 — 검사 스크립트가 아직 미리보기를 만들지 않았음")

    print("\n[판단 도움]")
    print("    - [1]의 좌표가 고친 값이 아니면: 기본 파일이 아닌 다른 파일을 고쳤거나, 고친 뒤 zip 압축 해제로 덮어써짐")
    print("    - [1]은 고친 값인데 그림이 그대로면: 이미지 창을 닫고 다시 열기 (사진 앱이 예전 그림을 보여 줄 수 있음)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
