"""
yeoyuro_v2/od_shiftability.py
=============================
v2.3 — OD 단위 유형 분류. 계산(evidence 수집)과 판정(classify)을 분리한다.

evidence  : 엔진으로 한 번만 계산하는 사실값 (기준 경로, 대안 경로들, ± 시각 평가값)
classify  : evidence + ThresholdSet -> 유형. 순수 함수라 sensitivity 를 재계산 없이 돌릴 수 있다.

유형 (preregistration_v23.md §3)
    calm / route_shiftable / time_shiftable / dual / structural
"""

from __future__ import annotations

from .shift_rules import ThresholdSet, edge_jaccard, reduction_ok

TYPES = ("calm", "route_shiftable", "time_shiftable", "dual", "structural")
INVALID = "invalid"          # 기준 경로가 혼잡 자료 범위(05:30~01:00) 밖을 지남 → 분석 제외 (D-036)
_WIN_START, _WIN_END = 330.0, 1500.0


def _m(r) -> dict:
    """TDResult -> evidence 용 최소 dict."""
    return {"actual": r.actual_time_min, "perceived": r.perceived_time_min,
            "transfers": r.transfer_count, "exp100": r.exposure[100],
            "exp100_p90": r.exposure_p90[100], "exp130": r.exposure[130],
            "max": r.max_congestion, "arrive": r.arrive_min, "depart": r.depart_min,
            "oow": bool(r.out_of_window)}


def is_oow(m: dict) -> bool:
    """자료 범위 밖 여부. 'oow' 가 없는 이전 evidence 는 도착·출발 시각으로 판정한다."""
    if m.get("oow"):
        return True
    return m["arrive"] > _WIN_END + 1e-9 or m.get("depart", _WIN_START) < _WIN_START - 1e-9


def build_evidence(base, route_cands, time_rows) -> dict:
    """base: TDResult, route_cands: [TDResult], time_rows: [(shift, same_route, TDResult)]"""
    return {
        "base": _m(base),
        "base_path": list(base.path),
        "route": [{**_m(c), "jac": edge_jaccard(base.path, c.path)} for c in route_cands
                  if c.path != base.path],
        "time": [{**_m(r), "shift": s, "same": same,
                  "jac": 1.0 if same else edge_jaccard(base.path, r.path)} for s, same, r in time_rows],
    }


def classify(ev: dict, th: ThresholdSet) -> dict:
    b = ev["base"]
    if is_oow(b):
        return {"type": INVALID, "route_ok": None, "time_ok": None, "route_bind": INVALID,
                "time_bind": INVALID, "route_drop": None, "time_drop": None, "time_shift": None}
    key = "exp100_p90" if th.use_p90 else "exp100"
    be, bm = b[key], b["max"]
    if be < th.calm_gate_exposure_min and bm < 100:
        return {"type": "calm", "route_ok": None, "time_ok": None, "route_bind": "calm",
                "time_bind": "calm", "route_drop": None, "time_drop": None, "time_shift": None}

    # Route
    r_ok, r_drop, r_bind, best_fail = False, None, "no_candidate", None
    for c in ev["route"]:
        if is_oow(c):
            continue                                   # 자료 범위 밖 후보는 판정에 쓰지 않음
        red = reduction_ok(be, c[key], bm, c["max"], th)
        fails = []
        if not red:
            fails.append("insufficient_reduction")
        if c["actual"] - b["actual"] > th.max_time_loss_min:
            fails.append("time_loss")
        if c["perceived"] - b["perceived"] > th.max_perceived_excess_min:
            fails.append("perceived_excess")
        if c["transfers"] - b["transfers"] > th.max_extra_transfer:
            fails.append("extra_transfer")
        if c["jac"] > th.max_edge_jaccard:
            fails.append("near_duplicate")
        drop = be - c[key]
        if not fails:
            if not r_ok or drop > r_drop:
                r_ok, r_drop = True, drop
            r_bind = "ok"
        elif not r_ok and (best_fail is None or drop > best_fail[0]):
            best_fail = (drop, fails[0])
    if not r_ok and best_fail:
        r_bind = best_fail[1]

    # Time
    t_ok, t_drop, t_shift, t_bind, best_tfail = False, None, None, "out_of_window", None
    for x in ev["time"]:
        if abs(x["shift"]) > th.time_shift_primary_window or is_oow(x):
            continue
        fails = []
        if not reduction_ok(be, x[key], bm, x["max"], th):
            fails.append("insufficient_reduction")
        if x["actual"] - b["actual"] > th.time_shift_max_time_increase_min:
            fails.append("time_increase")
        if not x["same"] and x["jac"] < th.time_shift_similar_jaccard:
            fails.append("route_changed")
        drop = be - x[key]
        if not fails:
            k = (abs(x["shift"]), -drop)
            if not t_ok or k < (abs(t_shift), -t_drop):
                t_ok, t_drop, t_shift = True, drop, x["shift"]
            t_bind = "ok"
        elif not t_ok and (best_tfail is None or drop > best_tfail[0]):
            best_tfail = (drop, fails[0])
    if not t_ok and best_tfail:
        t_bind = best_tfail[1]

    if r_ok and t_ok:
        typ = "dual"
    elif r_ok:
        typ = "route_shiftable"
    elif t_ok:
        typ = "time_shiftable"
    else:
        typ = "structural"
    return {"type": typ, "route_ok": r_ok, "time_ok": t_ok, "route_bind": r_bind, "time_bind": t_bind,
            "route_drop": r_drop, "time_drop": t_drop, "time_shift": t_shift}
