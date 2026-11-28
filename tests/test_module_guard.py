"""
tests/test_module_guard.py — D-113: 바뀐 저장소 모듈 다시 불러오기 (app/streamlit/module_guard.py).

2026-11-28 Streamlit Cloud: push 뒤 새 앱 파일 + 옛 yeoyuro_v2 모듈 → AttributeError (Reboot 로 해결).
임시 폴더의 가짜 패키지로, 파일이 바뀌면 다음 실행에서 새 판을 읽는지 본다.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app" / "streamlit")]
import module_guard as MG                       # noqa: E402


def _write(p: Path, text: str, bump: float = 0.0):
    p.write_text(text, encoding="utf-8")
    if bump:
        t = os.path.getmtime(p) + bump
        os.utime(p, (t, t))


def test_reload_after_file_change(tmp_path, monkeypatch):
    pkg = tmp_path / "fakepkg_d113"
    pkg.mkdir()
    _write(pkg / "__init__.py", "")
    _write(pkg / "mod.py", "VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, MG._REG_NAME, None)          # 시험마다 새 기록
    sys.modules.pop(MG._REG_NAME)
    try:
        m = importlib.import_module("fakepkg_d113.mod")
        assert m.VALUE == 1
        MG.record(tmp_path)
        cleared = []
        assert MG.refresh(tmp_path, lambda: cleared.append(1)) == [] and not cleared   # 안 바뀌면 아무것도 안 함
        assert sys.modules["fakepkg_d113.mod"] is m
        _write(pkg / "mod.py", "VALUE = 2\nNEW = True\n", bump=10)
        changed = MG.refresh(tmp_path, lambda: cleared.append(1))
        assert changed == ["fakepkg_d113.mod"] and cleared == [1]
        assert "fakepkg_d113.mod" not in sys.modules and "fakepkg_d113" not in sys.modules
        m2 = importlib.import_module("fakepkg_d113.mod")
        assert m2.VALUE == 2 and m2.NEW is True                  # 새 판을 읽음
    finally:
        for n in [n for n in sys.modules if n.startswith("fakepkg_d113")]:
            sys.modules.pop(n, None)
        sys.modules.pop(MG._REG_NAME, None)


def test_only_repo_modules(tmp_path):
    names = MG._repo_modules(ROOT)
    assert all(str(ROOT.resolve()) in f for f in names.values())
    assert "pandas" not in names and "module_guard" not in names and "__main__" not in names


def test_app_uses_guard():
    src = (ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py").read_text(encoding="utf-8")
    i_guard = src.index("_MG.refresh(ROOT")
    assert i_guard < src.index("from yeoyuro_v2 import korail_app")       # 다른 저장소 모듈보다 먼저
    assert "_MG.record(ROOT)" in src
