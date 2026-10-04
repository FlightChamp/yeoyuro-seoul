"""여유로 서울 v2 패키지. v1 코드(scripts/, app/streamlit/)는 수정하지 않고 읽기만 한다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_v1(root: Path | None = None):
    """v1 모듈을 불러온다. 반환: (scorer_module, station_routing_module, display_df)."""
    root = Path(root or ROOT)
    for p in (root / "app" / "streamlit", root / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import station_routing as sr  # noqa: E402
    mod = sr.load_scorer_class(root)
    display = sr.load_display_master(root)
    return mod, sr, display
