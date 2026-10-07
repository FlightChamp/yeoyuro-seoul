"""
scripts/v2/30_build_app_data.py
===============================
앱(Streamlit)이 쓰는 1~9호선 데이터 폴더 data/app_v26 를 만든다.

- 경로 그래프: 1~8호선 mart + 9호선 mart (환승 ×1.81, 혼잡 3개년 중앙값 = v2.6 기본값)
- 노선도: 기본 좌표 엑셀(1~9호선)을 v1 가져오기 스크립트(14)로 변환
- 분석 스크립트(v2.0~v2.5)가 읽는 data/marts·data/master 는 건드리지 않는다 (결과 재현성 유지)

앱은 data/app_v26 를 먼저 읽고, 없는 파일만 기존 폴더에서 읽는다.
엑셀 좌표를 고치면 앱이 저장 시각을 보고 노선도만 자동으로 다시 가져온다 (이 스크립트를 다시 돌릴 필요 없음).

실행
----
    python scripts/v2/30_build_app_data.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2 import line9 as L9
from yeoyuro_v2.proc import run_py   # Windows 하위 프로세스 한글 출력 오류 방지                                    # noqa: E402
from yeoyuro_v2.map_workbook import materialize, write_values_copy    # noqa: E402

APP = ROOT / "data" / "app_v26"
WORKBOOK = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"


def import_map(app: Path = APP, workbook: Path = WORKBOOK) -> tuple[bool, str]:
    """엑셀 → 노선도 csv (app/data/master/map_*.csv). 반환 (성공 여부, 메시지)."""
    (app / "reports" / "data_quality").mkdir(parents=True, exist_ok=True)
    (app / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    vals = write_values_copy(materialize(workbook), app / "reports" / "workbook_values.xlsx")
    r = run_py([str(ROOT / "scripts" / "14_import_map_workbook.py"), "--root", str(app),
                        "--xlsx", str(vals)])
    val = app / "reports" / "data_quality" / "map_workbook_validation.csv"
    if r.returncode != 0 or not val.exists():
        return False, (r.stdout + r.stderr)[-1500:]
    import pandas as pd
    t = pd.read_csv(val)
    fails = t[t.status == "FAIL"]
    return fails.empty, "검증 통과" if fails.empty else "; ".join(f"{x.check}: {x.detail}" for x in fails.itertuples())


def main() -> int:
    if APP.exists():
        shutil.rmtree(APP)
    L9.build_shadow_root(ROOT, APP, "base")
    ok, msg = import_map()
    n = sum(1 for _ in (APP / "data").rglob("*") if _.is_file())
    print(f"앱 데이터 폴더: {APP} (파일 {n}개)")
    print(f"노선도 가져오기: {'성공' if ok else '실패'} — {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
