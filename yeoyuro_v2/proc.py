"""
yeoyuro_v2/proc.py
==================
하위 프로세스로 파이썬 스크립트를 실행할 때 쓰는 공통 함수.

Windows 에서 다른 프로그램(Streamlit 앱, 검사 스크립트)이 파이썬 스크립트를 하위 프로세스로 실행하면
출력 인코딩이 cp1252 같은 영어 전용 코드페이지로 잡혀, 스크립트가 한글을 print 하는 순간
UnicodeEncodeError 로 멈춘다 (v2.6 에서 실제 발생: 노선도 가져오기가 '완료' 출력에서 실패).
→ 하위 프로세스는 항상 UTF-8 로 입출력하게 강제한다.
"""
from __future__ import annotations

import os
import subprocess
import sys


def utf8_env() -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_py(args: list[str]) -> subprocess.CompletedProcess:
    """python <args...> 를 UTF-8 로 실행하고 출력을 문자열로 받는다."""
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=utf8_env())


def utf8_stdout() -> None:
    """이 스크립트 자신의 print 도 어떤 환경에서든 한글에서 멈추지 않게."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
