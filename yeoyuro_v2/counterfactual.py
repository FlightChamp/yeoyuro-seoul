"""
yeoyuro_v2/counterfactual.py
============================
Counterfactual card — 한 OD × 출발 시각에 대해 "지금 / 경로 변경 / 시간 조정 / 둘 다" 를 계산한다.

원칙
----
- 기준(지금) = 시간 진행형 엔진의 최속 경로. v2.3 OD 분류와 같은 기준이다.
- 유형 판정 = v2.3 사전 등록 기준 T1 + od_shiftability.classify() 그대로 (카드와 분석 리포트가 어긋나지 않게).
- 카드에 보여 줄 '구체적 대안'은 classify() 와 같은 조건·같은 우선순위로 고른다
    · 경로 변경: 조건 통과 후보 중 100%+ 노출을 가장 많이 줄이는 경로
    · 시간 조정: 조건 통과 시각 중 이동 폭이 가장 작은 시각 (같으면 더 많이 줄이는 쪽)
  tests/test_counterfactual.py 가 'card 의 성공 여부 = classify 결과' 를 확인한다.
- 둘 다(경로+시각 동시 변경) = 강건성 검증 R2(joint) 와 같은 조건. 주 분류에는 넣지 않는 참고 선택지 (D-035).
- 이 모듈은 Streamlit 을 모른다. 앱은 결과 dict 만 받아 그린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .od_shiftability import build_evidence, classify, is_oow, _m
from .shift_rules import T1, ThresholdSet, reduction_ok
from .time_dependent import N_BINS, bin_index_of, fmt_min

SHIFTS = (-90, -60, -30, 30, 60, 90)          # v2.3 과 같은 후보 시각 (판정은 T1 주 창 ±60분 안만)
JOINT_SHIFTS = (-60, -30, 30, 60)              # R2 joint 와 같은 범위

ROUTE_FAIL_KO = {
    "insufficient_reduction": "혼잡 노출이 충분히 줄지 않음",
    "time_loss": "소요시간이 15분 넘게 늘어남",
    "perceived_excess": "체감시간이 5분 넘게 늘어남",
    "extra_transfer": "환승이 2회 이상 늘어남",
    "near_duplicate": "기존 경로와 거의 같은 경로",
    "no_candidate": "다른 후보 경로가 없음",
}
TIME_FAIL_KO = {
    "insufficient_reduction": "혼잡 노출이 충분히 줄지 않음",
    "time_increase": "소요시간이 5분 넘게 늘어남",
    "route_changed": "그 시각에는 다른 경로가 최속",
    "out_of_window": "±60분 안에 비교할 시각이 없음",
}
TYPE_KO = {
    "calm": "여유로운 이동",
    "route_shiftable": "경로로 피할 수 있는 혼잡",
    "time_shiftable": "시간으로 피할 수 있는 혼잡",
    "dual": "경로·시간 모두 효과 있음",
    "structural": "경로·시간만으로는 피하기 어려운 혼잡",
    "invalid": "판정 불가 (운행 자료 시간대 밖)",
}


@dataclass
class Option:
    """카드 한 칸. ok=False 면 metrics 는 '가장 가까웠던 후보' (없으면 None)."""
    kind: str                      # now / route / time / joint
    ok: bool
    path: list = field(default_factory=list)
    depart: str = ""
    shift_min: int = 0
    actual: float | None = None
    perceived: float | None = None
    exp100: float | None = None
    max_cong: float | None = None
    transfers: int | None = None
    reason: str = ""               # 실패 이유 (한국어)

    def delta(self, base: "Option") -> dict:
        if self.actual is None:
            return {}
        return {"actual": self.actual - base.actual, "exp100": self.exp100 - base.exp100,
                "max": self.max_cong - base.max_cong, "transfers": self.transfers - base.transfers}


def _opt(kind, ok, r, depart_min, shift=0, reason=""):
    m = _m(r) if not isinstance(r, dict) else r
    return Option(kind, ok, list(getattr(r, "path", [])), fmt_min(depart_min + shift), shift,
                  m["actual"], m["perceived"], m["exp100"], m["max"], m["transfers"], reason)


def _route_fails(b: dict, c: dict, th: ThresholdSet) -> list[str]:
    """classify() 의 Route 조건과 같은 순서."""
    f = []
    if not reduction_ok(b["exp100"], c["exp100"], b["max"], c["max"], th):
        f.append("insufficient_reduction")
    if c["actual"] - b["actual"] > th.max_time_loss_min:
        f.append("time_loss")
    if c["perceived"] - b["perceived"] > th.max_perceived_excess_min:
        f.append("perceived_excess")
    if c["transfers"] - b["transfers"] > th.max_extra_transfer:
        f.append("extra_transfer")
    if c["jac"] > th.max_edge_jaccard:
        f.append("near_duplicate")
    return f


def _time_fails(b: dict, x: dict, th: ThresholdSet) -> list[str]:
    """classify() 의 Time 조건과 같은 순서."""
    f = []
    if not reduction_ok(b["exp100"], x["exp100"], b["max"], x["max"], th):
        f.append("insufficient_reduction")
    if x["actual"] - b["actual"] > th.time_shift_max_time_increase_min:
        f.append("time_increase")
    if not x["same"] and x["jac"] < th.time_shift_similar_jaccard:
        f.append("route_changed")
    return f


def counterfactual(router, origin: str, dest: str, depart_min: float, th: ThresholdSet = T1,
                   with_joint: bool = True) -> dict | None:
    """반환 dict: type, type_ko, now, route, time, joint(Option), classify(raw), threshold_set_id.
    경로를 찾지 못하면 None."""
    res = router.evaluate_od(origin, dest, depart_min)
    if not res:
        return None
    b0 = bin_index_of(depart_min)
    ev = router.evaluator(b0)
    base = ev.evaluate(res[0].path, depart_min)
    rows, tres = [], {}
    for s in SHIFTS:
        t = depart_min + s
        if not (0 <= bin_index_of(t) < N_BINS):
            continue
        same = ev.evaluate(base.path, t)
        rows.append((s, True, same))
        tres[(s, True)] = same
        f = router.evaluate_od(origin, dest, t)
        if f and f[0].path != base.path:
            rows.append((s, False, f[0]))
            tres[(s, False)] = f[0]
    evd = build_evidence(base, res[1:], rows)
    cl = classify(evd, th)
    b = evd["base"]
    now = _opt("now", True, base, depart_min)

    # 경로 변경
    route_objs = [c for c in res[1:] if c.path != base.path]
    best_ok, best_fail = None, None
    for r, c in zip(route_objs, evd["route"]):
        if is_oow(c):
            continue
        fails = _route_fails(b, c, th)
        drop = b["exp100"] - c["exp100"]
        if not fails:
            if best_ok is None or drop > best_ok[0]:
                best_ok = (drop, r)
        elif best_fail is None or drop > best_fail[0]:
            best_fail = (drop, r, fails[0])
    if best_ok:
        route = _opt("route", True, best_ok[1], depart_min)
    elif best_fail:
        route = _opt("route", False, best_fail[1], depart_min, reason=ROUTE_FAIL_KO[best_fail[2]])
    else:
        route = Option("route", False, reason=ROUTE_FAIL_KO["no_candidate"])

    # 시간 조정 (T1 주 창 안)
    t_ok, t_fail = None, None
    for (s, same, r), x in zip(rows, evd["time"]):
        if abs(s) > th.time_shift_primary_window or is_oow(x):
            continue
        fails = _time_fails(b, x, th)
        drop = b["exp100"] - x["exp100"]
        if not fails:
            k = (abs(s), -drop)
            if t_ok is None or k < t_ok[0]:
                t_ok = (k, s, r)
        elif t_fail is None or drop > t_fail[0]:
            t_fail = (drop, s, r, fails[0])
    if t_ok:
        time_opt = _opt("time", True, t_ok[2], depart_min, t_ok[1])
    elif t_fail:
        time_opt = _opt("time", False, t_fail[2], depart_min, t_fail[1], TIME_FAIL_KO[t_fail[3]])
    else:
        time_opt = Option("time", False, reason=TIME_FAIL_KO["out_of_window"])

    # 둘 다 (경로 + 시각 동시, R2 joint 조건)
    joint = Option("joint", False, reason="계산하지 않음")
    if with_joint and cl["type"] not in ("calm", "invalid"):
        best = None
        for s in JOINT_SHIFTS:
            t = depart_min + s
            if not (0 <= bin_index_of(t) < N_BINS):
                continue
            for r in router.evaluate_od(origin, dest, t):
                if r.out_of_window or r.path == base.path:
                    continue
                ok = (reduction_ok(b["exp100"], r.exposure[100], b["max"], r.max_congestion, th)
                      and r.actual_time_min - b["actual"] <= th.max_time_loss_min
                      and r.perceived_time_min - b["perceived"] <= th.max_perceived_excess_min
                      and r.transfer_count - b["transfers"] <= th.max_extra_transfer)
                if ok:
                    drop = b["exp100"] - r.exposure[100]
                    if best is None or (abs(s), -drop) < best[0]:
                        best = ((abs(s), -drop), s, r)
        joint = (_opt("joint", True, best[2], depart_min, best[1]) if best
                 else Option("joint", False, reason="±60분 안에서 경로와 시각을 함께 바꿔도 기준을 넘는 대안 없음"))

    # 판정과 카드 선택이 같은 규칙인지 (테스트로도 확인)
    if cl["type"] not in ("calm", "invalid"):
        assert route.ok == bool(cl["route_ok"]) and time_opt.ok == bool(cl["time_ok"]), "classify 와 카드 불일치"
    return {"type": cl["type"], "type_ko": TYPE_KO[cl["type"]], "now": now, "route": route,
            "time": time_opt, "joint": joint, "classify": cl, "threshold_set_id": th.set_id,
            "origin": origin, "destination": dest, "depart": fmt_min(depart_min)}
