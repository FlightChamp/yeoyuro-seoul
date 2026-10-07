"""
scripts/v2/29_edit_map.py
=========================
노선도 좌표 편집을 시작하는 단일 명령.

1. 고쳐야 할 **정확한 엑셀 파일** 경로를 출력하고, 그 파일을 엑셀로 연다
2. 파일 탐색기에서 그 파일을 선택한 상태로 보여 준다 (위치 확인용)
3. 다운로드 폴더 등에 같은 이름의 다른 사본이 있으면 경고한다 (그걸 고치면 반영되지 않음)
4. 실시간 미리보기(29_watch_map) 를 켠다 — 엑셀에서 Ctrl+S 할 때마다 브라우저 그림이 바뀐다

실행
----
    python scripts/v2/29_edit_map.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"


def main() -> int:
    if not BASE.exists():
        print(f"[중단] 기본 파일이 없습니다: {BASE}")
        return 1
    print("=" * 70)
    print("고칠 파일 (이 파일만 고치세요):")
    print(f"  {BASE}")
    print("=" * 70)
    home = Path(os.environ.get("USERPROFILE", Path.home()))
    others = []
    for d in (home / "Downloads", home / "Desktop", home / "Documents"):
        if d.exists():
            others += [p for p in d.glob("*vector_map_coordinate_workbook*.xlsx") if not p.name.startswith("~$")]
    if others:
        print("[주의] 아래 파일들은 예전 사본입니다. 이 파일들을 고치면 미리보기에 반영되지 않습니다:")
        for p in others:
            print(f"  - {p}")
        print("  (정리하려면: Remove-Item 로 지우거나 다른 이름으로 바꾸세요)")
    if sys.platform.startswith("win"):
        os.startfile(str(BASE))                                            # 엑셀로 열기
        subprocess.Popen(["explorer", "/select,", str(BASE)])             # 탐색기에서 위치 보여 주기
        print("\n엑셀과 파일 탐색기를 열었습니다. 엑셀 창 제목이 'yeoyuro_seoul_vector_map_coordinate_workbook' 인지 확인하세요.")
    print("\n고치는 칸 (노란 칸):")
    print("  - 역 위치  : Station_Visual_Nodes 시트의 x_px, y_px  (예: 흑석_L9 행)")
    print("  - 역명만   : Station_Labels 시트의 label_dx, label_dy  (이동량, 위로 40px 이면 dy = -40)")
    print("  고친 뒤 Ctrl+S → 2~4초 뒤 브라우저 미리보기가 바뀝니다.\n")
    spec = importlib.util.spec_from_file_location("watch29", ROOT / "scripts" / "v2" / "29_watch_map.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w.main()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
