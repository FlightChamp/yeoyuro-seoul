"""
yeoyuro_v2/provenance.py
========================
Phase 1-3 (D-075) — 경로 결과의 **계산 출처(provenance)** 를 기록한다.

문제
----
앱은 v2 시간 진행형 계산이 실패하면 v1(출발 시각 고정) 값이나 미리 계산한 대표 결과로 조용히 바꿔
보여 줬다. 앱이 멈추지 않는 장점은 있지만, 사용자도 개발자도 지금 화면이 어떤 계산의 결과인지 알 수 없었다.
또 v2 재계산이 후보 일부를 고친 뒤 실패하면 v1·v2 값이 섞인 결과가 그대로 화면에 나갈 수 있었다.

이 모듈이 하는 일
-----------------
1. 결과 dict 에 `provenance` = {source, reason, error_type} 를 붙인다.
     source: v2_time_dependent | v1_static | precomputed
2. v2 재계산은 **복사본**에서 하고, 실패하면 원본(v1 값)을 그대로 돌려준다 → 값이 섞이지 않는다.
3. v2 가 아닌 결과는 logger `yeoyuro.route` 에 WARNING 으로 남긴다 (Streamlit Cloud 로그에서 확인).
4. 화면 안내 문구(notice_ko)를 한곳에서 만든다.

fallback 자체는 유지한다 (앱이 멈추는 것보다 낫다). 바뀌는 것은 '조용히'가 아니라 '알리고' 바꾼다는 점이다.
Streamlit 에 의존하지 않으므로 pytest 로 검증한다 (tests/test_provenance.py).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

LOGGER = logging.getLogger("yeoyuro.route")


class Source(str, Enum):
    """결과를 만든 계산 방식."""
    V2 = "v2_time_dependent"     # 정상: 구간 진입 시각 기준 시간 진행형 재계산
    V1 = "v1_static"             # 대체: 출발 시각 하나로 모든 구간을 판정한 v1 값
    PRECOMPUTED = "precomputed"  # 대체: 미리 계산해 둔 대표 시나리오(08:30 등)


# 대체 사유 코드 → 화면용 설명
REASON_KO = {
    "import_error": "v2 계산 모듈을 불러오지 못함",
    "evaluator_unavailable": "v2 평가기를 만들지 못함",
    "evaluator_error": "v2 평가기 생성 중 오류",
    "retime_error": "v2 재계산 중 오류",
    "search_error": "경로 탐색 중 오류",
    "search_not_ok": "경로 탐색 결과 없음",
    "forced": "점검용 강제 대체(YEOYURO_FORCE_DEGRADE)",
}


@dataclass(frozen=True)
class Provenance:
    source: Source
    reason: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def degraded(self) -> bool:
        return self.source is not Source.V2

    def as_dict(self) -> dict:
        return {"source": self.source.value, "reason": self.reason, "error_type": self.error_type}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Provenance"]:
        if not d:
            return None
        return cls(Source(d["source"]), d.get("reason"), d.get("error_type"))


def attach(result: Optional[dict], prov: Provenance, *, context: str = "",
           exc: Optional[BaseException] = None) -> Optional[dict]:
    """결과에 provenance 를 붙이고, 대체 결과면 경고 로그를 남긴다."""
    if prov.degraded:
        LOGGER.warning("route result degraded: source=%s reason=%s error=%s %s",
                       prov.source.value, prov.reason, prov.error_type, context,
                       exc_info=(type(exc), exc, exc.__traceback__) if exc else None)
    if result is not None:
        result["provenance"] = prov.as_dict()
    return result


def retime_with_provenance(result: dict,
                           evaluator_factory: Callable[[], Any],
                           retime_fn: Callable[..., dict],
                           *, depart_min: float, mode: str, context: str = "",
                           force_degrade: bool = False) -> dict:
    """v1 탐색 결과를 v2 로 재계산하고 출처를 기록한다.

    evaluator_factory : 인자 없이 v2 평가기를 돌려주는 함수 (None 이면 평가기 없음)
    retime_fn         : app_bridge.retime_result 와 같은 시그니처 (result, factory, depart_min, mode)
    force_degrade     : 점검용. True 면 v2 를 건너뛰고 v1 대체 경로를 탄다 (화면·로그 확인용)

    재계산은 깊은 복사본에서 한다. 중간에 실패해도 원본 result 는 v1 값 그대로라 값이 섞이지 않는다.
    """
    if force_degrade:
        return attach(result, Provenance(Source.V1, "forced"), context=context)
    try:
        evaluator = evaluator_factory()
    except Exception as e:                                   # noqa: BLE001 — 대체 사유로 기록
        return attach(result, Provenance(Source.V1, "evaluator_error", type(e).__name__),
                      context=context, exc=e)
    if evaluator is None:
        return attach(result, Provenance(Source.V1, "evaluator_unavailable"), context=context)
    try:
        out = retime_fn(copy.deepcopy(result), lambda _bin: evaluator, depart_min, mode)
    except Exception as e:                                   # noqa: BLE001
        return attach(result, Provenance(Source.V1, "retime_error", type(e).__name__),
                      context=context, exc=e)
    return attach(out, Provenance(Source.V2), context=context)


def notice_ko(prov_dict: Optional[dict]) -> Optional[str]:
    """대체 결과일 때 화면에 띄울 문구. 정상(v2)이거나 정보가 없으면 None."""
    prov = Provenance.from_dict(prov_dict)
    if prov is None or not prov.degraded:
        return None
    why = REASON_KO.get(prov.reason or "", prov.reason or "알 수 없음")
    if prov.error_type:
        why += " · %s" % prov.error_type
    if prov.source is Source.V1:
        return ("이 결과는 **출발 시각 고정 방식(v1)** 으로 계산했습니다 — 구간마다 실제 진입 시각의 혼잡도를 "
                "반영하는 v2 계산이 이번 요청에서 실패했습니다 (%s). 소요시간·혼잡도가 v2 결과와 다를 수 있습니다." % why)
    return ("지금 조건으로 계산하지 못해 **미리 계산한 대표 시나리오**를 보여 드립니다 (%s). "
            "입력한 날짜·시각과 다른 조건의 결과일 수 있습니다." % why)
