"""
yeoyuro_v2/app_bridge.py
========================
v2.1 — 앱(쾌적 경로 페이지)의 결과를 시간 진행형으로 다시 계산한다.

v1 앱 흐름은 그대로 둔다:
    find_route_by_station()  ->  후보 경로 집합 (v1 Yen + 다양성 필터)
이 모듈이 그 뒤에 끼어든다:
    retime_result()          ->  같은 후보를 step 정책으로 재평가하고,
                                 최속·추천·대안·시간대안 판정을 v2 값으로 다시 한다.

경로 대안 판정은 v1 규칙(시간손실 ≤15분, 최대혼잡 감소 ≥15%p, 체감 ≤최속+5분)에 값만 시간 진행형.
시간 대안은 temporal_shift.evaluate_time_shift 를 v2.3 사전 등록 기준 T1(±30/60, 노출 감소 기준, Top 3)로 쓴다.
time_alternative_td 는 v2.1 방식 비교용으로 남겨 둔다.

Streamlit 에 의존하지 않으므로 pytest 로 검증할 수 있다.
"""

from __future__ import annotations

from .shift_rules import T1
from .temporal_shift import evaluate_time_shift
from .time_dependent import (TimeDependentEvaluator, BIN_START_MIN, BIN_WIDTH_MIN,
                             N_BINS, bin_index_of, bin_label, fmt_min)

V1_MAX_TIME_LOSS = 15.0
V1_MIN_CONG_DROP = 15.0
V1_MAX_PERCEIVED_EXCESS = 5.0
V1_TIME_ALT_WINDOW = 120
V1_TIME_ALT_MIN_GAIN = 10.0

MODE_WEIGHTS = {
    "fast":         {"actual": 1.0, "perceived": 0.0, "max_cong": 0.00, "tpen": 0.3, "seat": 0.0},
    "calm":         {"actual": 0.0, "perceived": 1.0, "max_cong": 0.05, "tpen": 0.5, "seat": 0.0},
    "min_transfer": {"actual": 0.5, "perceived": 0.5, "max_cong": 0.00, "tpen": 3.0, "seat": 0.0},
    "balanced":     {"actual": 0.3, "perceived": 0.7, "max_cong": 0.03, "tpen": 0.8, "seat": 0.5},
}


def attach_td(ev: dict, evaluator: TimeDependentEvaluator, depart_min: float) -> dict:
    """후보 dict 에 v2 값을 덮어쓴다. v1 원래 값은 ev['v1'] 에 보존한다."""
    path = ev.get("path") or []
    if len(path) < 2:
        return ev
    r = evaluator.evaluate(path, depart_min, keep_trace=True)
    if "v1" not in ev:
        ev["v1"] = {k: ev.get(k) for k in ("actual_time_min", "perceived_time_min",
                                          "max_congestion", "avg_congestion",
                                          "transfer_wait_min")}
    ev["actual_time_min"] = round(r.actual_time_min, 1)
    ev["perceived_time_min"] = round(r.perceived_time_min, 1)
    ev["max_congestion"] = round(r.max_congestion, 1)
    ev["avg_congestion"] = round(r.avg_congestion_edge, 1)
    ev["transfer_wait_min"] = round(r.transfer_wait_min, 1)
    ev["exposure_80_min"] = round(r.exposure[80], 1)
    ev["exposure_100_min"] = round(r.exposure[100], 1)
    ev["exposure_130_min"] = round(r.exposure[130], 1)
    ev["exposure_100_min_p90"] = round(r.exposure_p90[100], 1)
    ev["depart_min"] = depart_min
    ev["arrive_min"] = r.arrive_min
    ev["edge_congestion"] = {(t.u, t.v): t.congestion for t in r.trace if t.kind == "ride"}
    ev["td_policy"] = r.policy
    # 승차 구간 대기 표기(segments)도 도착 bin 기준 값으로 맞춘다
    waits = [t for t in r.trace if t.kind == "transfer_wait"]
    wi = 0
    for sg in ev.get("segments") or []:
        if sg.get("kind") == "transfer":
            sg["wait_min"] = round(waits[wi].t_end - waits[wi].t_start, 1) if wi < len(waits) else 0.0
            wi += 1
    return ev


def _score(c: dict, mode: str) -> float:
    w = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["calm"])
    return (w["actual"] * c["actual_time_min"] + w["perceived"] * c["perceived_time_min"]
            + w["max_cong"] * (c["max_congestion"] or 0)
            + w["tpen"] * c.get("transfer_penalty_min", 0.0)
            - w["seat"] * c.get("seat_chance_score", 0.0))


def time_alternative_td(path: list[str], evaluator_for_bin, depart_min: float,
                        window_min: int = V1_TIME_ALT_WINDOW,
                        min_gain: float = V1_TIME_ALT_MIN_GAIN):
    """같은 경로를 출발 시각을 30분 단위로 옮겨 시간 진행형으로 평가한다.

    v1 time_alternative 와 같은 기준(최대혼잡 최소, 개선 ≥10%p)이지만,
    각 후보 시각에서도 구간 진입 시각을 누적한다.
    evaluator_for_bin(b) -> 그 bin 을 출발 bin 으로 하는 TimeDependentEvaluator
    """
    b0 = bin_index_of(depart_min)
    rows = []
    for k in range(-(window_min // BIN_WIDTH_MIN), window_min // BIN_WIDTH_MIN + 1):
        b = b0 + k
        if b < 0 or b >= N_BINS:
            continue
        t = depart_min + k * BIN_WIDTH_MIN
        r = evaluator_for_bin(b).evaluate(path, t)
        rows.append({"shift_min": k * BIN_WIDTH_MIN, "depart_min": t, "depart": fmt_min(t),
                     "time_bin": bin_label(b), "max_congestion": round(r.max_congestion, 1),
                     "exposure_100_min": round(r.exposure[100], 1),
                     "actual_time_min": round(r.actual_time_min, 1)})
    cur = next((x for x in rows if x["shift_min"] == 0), None)
    if not cur:
        return None
    best = min(rows, key=lambda x: (x["max_congestion"], abs(x["shift_min"])))
    gain = round(cur["max_congestion"] - best["max_congestion"], 1)
    if best["shift_min"] == 0 or gain < min_gain:
        return None
    return {"best_time_bin": best["time_bin"], "best_depart": best["depart"],
            "best_depart_min": best["depart_min"],
            "best_max_congestion": best["max_congestion"],
            "current_max_congestion": cur["max_congestion"], "gain_pp": gain,
            "best_exposure_100_min": best["exposure_100_min"],
            "current_exposure_100_min": cur["exposure_100_min"],
            "profile": rows, "td": True}


def retime_result(result: dict, evaluator_for_bin, depart_min: float, mode: str) -> dict:
    """find_route_by_station 결과 전체를 v2 값으로 다시 판정한다."""
    if not result or not result.get("ok"):
        return result
    ev0 = evaluator_for_bin(bin_index_of(depart_min))
    cands = [attach_td(dict(c), ev0, depart_min) for c in result["candidates"]]
    fastest_src = result.get("fastest")
    fastest = attach_td(dict(fastest_src), ev0, depart_min) if fastest_src else None
    pool = cands + ([fastest] if fastest and all(fastest["path"] != c["path"] for c in cands) else [])
    fastest = min(pool, key=lambda c: c["actual_time_min"])
    cands.sort(key=lambda c: _score(c, mode))
    rec = cands[0]

    alt = None
    for c in cands:
        if c["path"] in (fastest["path"], rec["path"]):
            continue
        tl = c["actual_time_min"] - fastest["actual_time_min"]
        cd = (fastest["max_congestion"] or 0) - (c["max_congestion"] or 0)
        pe = c["perceived_time_min"] - fastest["perceived_time_min"]
        if tl <= V1_MAX_TIME_LOSS and cd >= V1_MIN_CONG_DROP and pe <= V1_MAX_PERCEIVED_EXCESS:
            alt = dict(c)
            alt["time_loss_vs_fastest"] = round(tl, 1)
            alt["comfort_gain_vs_fastest"] = round(cd, 1)
            break

    # v2.2: Time Shift 판정은 대안 유무와 상관없이 항상 계산한다 (화면 노출은 대안이 없을 때).
    ts = evaluate_time_shift(ev0, rec["path"], depart_min, T1)   # v2.3 사전 등록 기준 (D-022)
    time_alt = None
    if alt is None and ts["top"]:
        b = ts["top"][0]
        time_alt = {"best_time_bin": bin_label(bin_index_of(b.depart_min)), "best_depart": b.depart,
                    "best_depart_min": b.depart_min, "shift_min": b.shift_min,
                    "best_max_congestion": b.max_congestion,
                    "current_max_congestion": round(ts["base"].max_congestion, 1),
                    "gain_pp": b.max_cong_drop_pp,
                    "best_exposure_100_min": b.exposure_100_min,
                    "current_exposure_100_min": round(ts["base"].exposure[100], 1),
                    "actual_delta_min": b.actual_delta_min, "td": True, "v22": True}

    out = dict(result)
    out.update({"recommended": rec, "fastest": fastest, "alternative": alt,
                "time_alternative": time_alt, "candidates": cands,
                "time_shift": {"type": ts["type"], "binding_constraint": ts["binding_constraint"],
                               "top": [x.as_dict() for x in ts["top"]],
                               "base_exposure_100_min": round(ts["base"].exposure[100], 1),
                               "base_max_congestion": round(ts["base"].max_congestion, 1)},
                "engine": "v2_time_dependent", "depart_min": depart_min})
    return out
