"""
yeoyuro_v2/temporal_shift.py
============================
v2.2 — Time Shift 진단: "조금 일찍 / 늦게 출발하면 같은 이동의 혼잡 노출이 줄어드는가?"

판정 (ThresholdSet, D-015·D-016)
--------------------------------
후보 시각 = 기준 출발 ± 30 / 60 / 90 분 (운행 창 밖은 제외)
성공 조건 (모두 만족):
  - 감소: exposure_100 감소 ≥ 5분  또는  최대 기대혼잡 감소 ≥ 15%p   (Route Shift 와 동일한 OR 규칙)
  - 소요시간 증가 ≤ 5분
  - 같은 경로, 또는 그 시각의 최속 경로가 directed-edge Jaccard ≥ 0.8 인 유사 경로
기준이 Calm 이면 판정하지 않는다 (type = "calm").

Top 3 정렬 (가중치 없이 사전식, D-015)
  1) 이동폭 |shift| 작은 순  — 생활 패턴을 덜 바꾸는 쪽이 먼저
  2) exposure_100 감소 큰 순
  3) 늦게 출발하는 쪽 우선 (같은 폭이면 '조금 늦게'가 대부분 실행하기 쉽다)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .shift_rules import T0, ThresholdSet, edge_jaccard, is_calm, reduction_ok
from .time_dependent import N_BINS, bin_index_of, fmt_min


@dataclass
class ShiftRow:
    shift_min: int
    depart_min: float
    depart: str
    same_route: bool
    jaccard: float
    path: list
    actual_time_min: float
    perceived_time_min: float
    max_congestion: float
    exposure_100_min: float
    exposure_130_min: float
    actual_delta_min: float
    perceived_delta_min: float
    max_cong_drop_pp: float
    exposure_100_drop_min: float
    exposure_130_drop_min: float
    success: bool
    fail_reason: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("path")
        return d


def _row(base, r, shift, t, same, jac, th: ThresholdSet) -> ShiftRow:
    exp_drop = base.exposure[100] - r.exposure[100]
    cong_drop = base.max_congestion - r.max_congestion
    dt = r.actual_time_min - base.actual_time_min
    reason = "ok"
    if not reduction_ok(base.exposure[100], r.exposure[100], base.max_congestion, r.max_congestion, th):
        reason = "insufficient_reduction"
    elif dt > th.time_shift_max_time_increase_min:
        reason = "time_increase"
    elif not same and jac < th.time_shift_similar_jaccard:
        reason = "route_changed"
    return ShiftRow(
        shift_min=shift, depart_min=t, depart=fmt_min(t), same_route=same, jaccard=round(jac, 3),
        path=list(r.path), actual_time_min=round(r.actual_time_min, 2),
        perceived_time_min=round(r.perceived_time_min, 2),
        max_congestion=round(r.max_congestion, 1), exposure_100_min=round(r.exposure[100], 2),
        exposure_130_min=round(r.exposure[130], 2), actual_delta_min=round(dt, 2),
        perceived_delta_min=round(r.perceived_time_min - base.perceived_time_min, 2),
        max_cong_drop_pp=round(cong_drop, 1), exposure_100_drop_min=round(exp_drop, 2),
        exposure_130_drop_min=round(base.exposure[130] - r.exposure[130], 2),
        success=reason == "ok", fail_reason=reason)


def evaluate_time_shift(evaluator, path: list[str], depart_min: float,
                        th: ThresholdSet = T0, reopt=None) -> dict:
    """같은 경로를 ± 시각으로 옮겨 평가하고 Time Shift 판정을 돌려준다.

    reopt : 선택. reopt(t) -> 그 시각의 최속 경로(TDResult). 주면 경로가 바뀌는 경우도 후보로 본다.
    """
    base = evaluator.evaluate(path, depart_min)
    shifts = sorted({s for w in th.time_shift_windows if w <= th.time_shift_primary_window
                     for s in (-w, w)})
    rows: list[ShiftRow] = []
    skipped = []
    for s in shifts:
        t = depart_min + s
        b = bin_index_of(t)
        if b < 0 or b >= N_BINS:
            skipped.append(s)
            continue
        r = evaluator.evaluate(path, t)
        rows.append(_row(base, r, s, t, True, 1.0, th))
        if reopt is not None:
            f = reopt(t)
            if f is not None and f.path != list(path):
                rows.append(_row(base, f, s, t, False, edge_jaccard(path, f.path), th))

    calm = is_calm(base, th)
    ok = [x for x in rows if x.success]
    best_per_shift = {}
    for x in ok:
        k = x.shift_min
        if k not in best_per_shift or x.exposure_100_drop_min > best_per_shift[k].exposure_100_drop_min:
            best_per_shift[k] = x
    top = sorted(best_per_shift.values(),
                 key=lambda x: (abs(x.shift_min), -x.exposure_100_drop_min, -x.shift_min))[:3]

    if calm:
        ttype, binding, top = "calm", "calm", []
    elif top:
        ttype, binding = "time_shiftable", "ok"
    else:
        ttype = "not_time_shiftable"
        if rows:
            worst = max(rows, key=lambda x: (x.exposure_100_drop_min, x.max_cong_drop_pp))
            binding = worst.fail_reason
        else:
            binding = "out_of_window"
    return {
        "base": base, "rows": rows, "top": top, "type": ttype, "binding_constraint": binding,
        "min_shift_min": top[0].shift_min if top else None, "skipped_shifts": skipped,
        "threshold_set_id": th.set_id,
    }


def shift_curve(evaluator, path: list[str], depart_min: float, span: int = 90, step: int = 15):
    """시각별 곡선 (UI 용). 판정 없이 값만 돌려준다."""
    out = []
    for s in range(-span, span + 1, step):
        t = depart_min + s
        b = bin_index_of(t)
        if b < 0 or b >= N_BINS:
            continue
        r = evaluator.evaluate(path, t)
        out.append({"shift_min": s, "depart": fmt_min(t), "actual_time_min": round(r.actual_time_min, 1),
                    "max_congestion": round(r.max_congestion, 1),
                    "exposure_100_min": round(r.exposure[100], 1),
                    "exposure_100_min_p90": round(r.exposure_p90[100], 1)})
    return out
