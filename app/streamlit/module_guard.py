"""
app/streamlit/module_guard.py — 바뀐 저장소 모듈을 다시 불러오기 (D-113)
=====================================================================

왜: Streamlit Cloud 에서 push 뒤 앱 파일(app/streamlit/…)은 새로 읽혔지만, 이미 메모리에 있던 yeoyuro_v2 모듈은
옛 판이 남아 `AttributeError: module 'yeoyuro_v2.korail_app' has no attribute 'norm_groups'` 가 났다 (2026-11-28,
D-112a push 뒤, Reboot 로 해결). 앱 폴더 밖 모듈은 Streamlit 이 바뀐 것을 알아채지 못하는 것으로 보인다 (미확인).

하는 일: 앱이 다시 실행될 때마다 저장소 안 파일에서 불러온 모듈의 파일 수정 시각을 처음 본 값과 비교한다.
하나라도 바뀌었으면 저장소 모듈을 모두 메모리에서 내리고 Streamlit 캐시를 비운다 → 이어지는 import 가 새 판을 읽는다.
바뀐 것이 없으면 아무것도 하지 않는다 (평소 비용 = 파일 시각 확인 수십 번).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Callable

_REG_NAME = "_yeoyuro_module_registry"
_SELF = __name__


def _registry() -> dict:
    reg = sys.modules.get(_REG_NAME)
    if reg is None:
        reg = types.ModuleType(_REG_NAME)
        reg.seen = {}
        sys.modules[_REG_NAME] = reg
    return reg.seen


def _repo_modules(root: Path) -> dict:
    """{모듈 이름: 파일} — 저장소(root) 안 .py 파일에서 불러온 모듈만 (이 모듈·__main__ 은 뺌)."""
    base = str(Path(root).resolve())
    out = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or name in (_SELF, "__main__", _REG_NAME) or not f.endswith(".py"):
            continue
        try:
            rf = str(Path(f).resolve())
        except OSError:
            continue
        if rf.startswith(base + os.sep) and "site-packages" not in rf:
            out[name] = rf
    return out


def _mtime(f: str) -> float | None:
    try:
        return os.path.getmtime(f)
    except OSError:
        return None


def record(root: Path) -> None:
    """지금 메모리에 있는 저장소 모듈의 파일 시각을 처음 본 값으로 적어 둔다 (이미 적힌 것은 그대로)."""
    seen = _registry()
    for name, f in _repo_modules(root).items():
        if name not in seen:
            seen[name] = (f, _mtime(f))


def refresh(root: Path, clear: Callable[[], None] | None = None) -> list[str]:
    """바뀐 모듈이 있으면 저장소 모듈을 모두 내리고 clear() (캐시 비우기)를 부른다. 반환: 바뀐 모듈 이름."""
    seen = _registry()
    changed = sorted(n for n, (f, t) in seen.items() if n in sys.modules and _mtime(f) != t)
    if not changed:
        return []
    for name in _repo_modules(root):
        sys.modules.pop(name, None)
    seen.clear()
    if clear is not None:
        try:
            clear()
        except Exception:                                # noqa: BLE001 — 캐시 비우기 실패는 다시 불러오기를 막지 않음
            pass
    return changed


def streamlit_clear() -> None:
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
