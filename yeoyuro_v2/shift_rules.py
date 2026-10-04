"""
yeoyuro_v2/shift_rules.py
=========================
Route Shift 판정 규칙. 가중치 점수 없이 사전 고정 임계값의 AND/OR 만 쓴다.

기본값은 v1 채택 조건(09 스크립트 MAX_TIME_LOSS_MIN 등)을 계승한다.
값을 바꾸려면 ThresholdSet 을 새로 만들어 id 를 붙이고 docs/v2/decision_log.md 에 기록한다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ThresholdSet:
    set_id: str = "T0_v1_inherit"
    min_exposure_drop_min: float = 5.0     # exposure_100 감소 (분)  ┐ 둘 중 하나
    min_max_cong_drop_pp: float = 15.0     # 최대 기대혼잡 감소 (%p) ┘
    max_time_loss_min: float = 15.0        # 실제 소요 증가
    max_perceived_excess_min: float = 5.0  # 체감시간 초과
    max_extra_transfer: int = 1
    max_edge_jaccard: float = 0.65         # directed-edge 유사도 상한
    calm_gate_exposure_min: float = 3.0    # base exposure_100 < 이 값 AND max < 100 이면 Calm

    def as_dict(self) -> dict:
        return asdict(self)


T0 = ThresholdSet()


def edge_jaccard(a: list[str], b: list[str]) -> float:
    ea, eb = set(zip(a, a[1:])), set(zip(b, b[1:]))
    u = ea | eb
    return len(ea & eb) / len(u) if u else 0.0


def is_calm(base, th: ThresholdSet = T0) -> bool:
    return base.exposure[100] < th.calm_gate_exposure_min and base.max_congestion < 100


def route_shift(base, cands, th: ThresholdSet = T0):
    """Route Shift 판정.

    Returns
    -------
    (success: bool, best_candidate | None, binding_constraint: str)
    binding_constraint: 성공이면 'ok'. 실패면 노출 감소가 가장 큰 후보가 처음 걸린 조건.
    """
    checks = []
    for c in cands:
        if c.path == base.path:
            continue
        exp_drop = base.exposure[100] - c.exposure[100]
        cong_drop = base.max_congestion - c.max_congestion
        failed = []
        if not (exp_drop >= th.min_exposure_drop_min or cong_drop >= th.min_max_cong_drop_pp):
            failed.append("insufficient_reduction")
        if c.actual_time_min - base.actual_time_min > th.max_time_loss_min:
            failed.append("time_loss")
        if c.perceived_time_min - base.perceived_time_min > th.max_perceived_excess_min:
            failed.append("perceived_excess")
        if c.transfer_count - base.transfer_count > th.max_extra_transfer:
            failed.append("extra_transfer")
        if edge_jaccard(base.path, c.path) > th.max_edge_jaccard:
            failed.append("near_duplicate")
        checks.append((c, exp_drop, cong_drop, failed))
    if not checks:
        return False, None, "no_candidate"
    ok = [x for x in checks if not x[3]]
    if ok:
        best = max(ok, key=lambda x: (x[1], x[2]))
        return True, best[0], "ok"
    top = max(checks, key=lambda x: (x[1], x[2]))
    return False, top[0], top[3][0]
