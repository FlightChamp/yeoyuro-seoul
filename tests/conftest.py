"""
tests/conftest.py — 모든 테스트 공통 설정.

점검용 강제 대체 환경변수(YEOYURO_FORCE_DEGRADE)가 셸에 남아 있으면 앱 테스트 전체가 v1 간이 계산으로 돌아
기준값 테스트가 대량으로 실패한다 (2026-11-18 실제 발생). 테스트마다 시작 전에 비운다.
강제 대체를 검사하는 테스트는 monkeypatch.setenv 로 그 테스트 안에서만 다시 켠다 (D-077).
"""
import pytest


@pytest.fixture(autouse=True)
def _clear_force_degrade(monkeypatch):
    monkeypatch.delenv("YEOYURO_FORCE_DEGRADE", raising=False)
