"""
station_routing.py
==================
**역 단위(station-level)** 경로 탐색. 사용자 입력과 내부 그래프를 분리한다.

원칙
----
여유로 서울 은 사용자에게 역 단위 입력을 제공하지만, 내부 그래프와 혼잡도 계산은
호선별 station_uid 단위로 유지한다. 환승역에서 어떤 노선을 처음 탈지는 사용자가
고정하지 않는 한 알고리즘이 후보로 비교하며, **최초 승차 노선 선택은 환승으로
계산하지 않는다.**

가상 노드
---------
사용자가 "서울역"을 고르면 1호선/4호선 노드를 모두 출발 후보로 연다.

    V_ORIGIN_서울역 --(origin_access, 0분)--> 1_서울역
    V_ORIGIN_서울역 --(origin_access, 0분)--> 4_서울역
    2_강남 --(destination_exit, 0분)--> V_DEST_강남

origin_access / destination_exit 은 **환승 횟수에 포함하지 않는다.** 기본 penalty 0.
추후 역 출입구·승강장 정보가 생기면 access penalty 를 추가할 수 있다.

이렇게 하면 "서울역 1호선 → 4호선 환승" 으로 시작하는 경로가 나오지 않는다.
가상 노드에서 4호선으로 바로 진입하는 비용이 0 이므로 환승 경로가 지배당한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ORIGIN_PREFIX = "V_ORIGIN_"

# 환승 후 다음 노선 대기시간의 상한(분).
# 심야처럼 배차가 매우 긴 시간대에 대기시간이 경로 점수를 지배하지 않게 막는다.
MAX_TRANSFER_WAIT_MIN = 12.0
DEST_PREFIX = "V_DEST_"


def load_scorer_class(root: Path):
    for cand in (root / "scripts" / "09_route_scoring_prototype.py",
                 root / "09_route_scoring_prototype.py",
                 Path(__file__).with_name("09_route_scoring_prototype.py")):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("rs09", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("09_route_scoring_prototype.py 를 찾을 수 없습니다.")


_HEADWAY_CACHE: dict = {}


def load_headway(root: Path):
    """30분 bin 평균 배차간격 -> 환승 대기시간 조회표.

    station 단위를 우선 쓰고, 없거나 미운행이면 line 단위로 내려간다.
    최초 승차 전 대기시간에는 쓰지 않는다(이번 MVP 범위 밖).
    """
    key = str(Path(root).resolve())
    if key in _HEADWAY_CACHE:
        return _HEADWAY_CACHE[key]

    def _load(name):
        base = Path(root) / "data" / "marts"
        for ext in (".parquet", ".csv.gz", ".csv"):
            q = base / (name + ext)
            if q.exists():
                return pd.read_parquet(q) if ext == ".parquet" else pd.read_csv(q)
        return None

    st = _load("headway_station_30min")
    ln = _load("headway_line_30min")
    station_map, line_map = {}, {}
    if st is not None:
        ok = st[st["is_operating"] == 1] if "is_operating" in st.columns else st
        for r in ok.itertuples():
            if pd.isna(r.expected_wait_min):
                continue
            station_map[(str(r.line_id), r.station_name, r.direction,
                         r.day_type, r.time_bin)] = float(r.expected_wait_min)
    if ln is not None:
        col = ("avg_expected_wait_min" if "avg_expected_wait_min" in ln.columns
               else "expected_wait_min")
        for r in ln.itertuples():
            v = getattr(r, col, None)
            if v is None or pd.isna(v):
                continue
            line_map[(str(r.line_id), r.direction, r.day_type, r.time_bin)] = float(v)
    out = {"station": station_map, "line": line_map,
           "available": bool(station_map or line_map)}
    _HEADWAY_CACHE[key] = out
    return out


def expected_wait(headway: dict, line, station, direction, day_type, time_bin):
    """(노선, 역, 방향, 요일유형, 시간대) 예상 대기시간(분). 없으면 None."""
    if not headway or not headway.get("available") or not direction:
        return None
    v = headway["station"].get((str(line), station, direction, day_type, time_bin))
    if v is None:
        v = headway["line"].get((str(line), direction, day_type, time_bin))
    if v is None:
        return None
    return round(min(float(v), MAX_TRANSFER_WAIT_MIN), 1)


def apply_transfer_wait(ev: dict, headway: dict, day_type: str, time_bin: str):
    """환승 후 다음 노선 승차 전 예상 대기시간을 경로에 더한다.

    **최초 승차 전 대기시간은 더하지 않는다.** 사용자가 승강장에 도착하는 시점을
    알 수 없어 출발 시각 입력의 의미가 흐려지기 때문이다.
    환승이 발생한 경우에만, 갈아탄 노선의 배차간격으로 기대 대기시간을 계산한다.
    """
    segs = ev.get("segments") or []
    total = 0.0
    for i, sg in enumerate(segs):
        if sg.get("kind") != "transfer":
            continue
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if not nxt or nxt.get("kind") != "ride":
            continue
        w = expected_wait(headway, nxt.get("line"), nxt.get("from"),
                          nxt.get("direction"), day_type, time_bin)
        sg["wait_min"] = w
        if w:
            total += w
    ev["transfer_wait_min"] = round(total, 1)
    # 대기를 더하기 '전' 값이 승차+중간정차+환승도보 시간이다.
    # 여기서 또 빼면 이중 차감이 된다. 세부 항목(running/dwell/walk)은
    # evaluate() 가 이미 분리해 넣어두므로 그대로 둔다.
    ev["ride_time_min"] = round(ev["actual_time_min"], 1)
    ev.setdefault("dwell_time_min", 0.0)
    ev.setdefault("dwell_stop_count", 0)
    ev.setdefault("running_time_min", ev["ride_time_min"])
    ev.setdefault("transfer_walk_min", 0.0)
    if total:
        ev["actual_time_min"] = round(ev["actual_time_min"] + total, 2)
        ev["perceived_time_min"] = round(ev["perceived_time_min"] + total, 2)
    return ev


def load_display_master(root: Path) -> pd.DataFrame:
    p = root / "data" / "master" / "station_display_master.csv"
    if not p.exists():
        raise FileNotFoundError(
            "station_display_master.csv 가 없습니다. 12_build_display_masters.py 실행 필요.")
    return pd.read_csv(p)


def candidates_of(display: pd.DataFrame, station_key: str) -> list[str]:
    row = display[display["station_key"] == station_key]
    if row.empty:
        return []
    return [x for x in str(row.iloc[0]["candidate_station_uids"]).split(";") if x]


def station_of(node: str) -> str:
    return node.split("_", 1)[1].split("@")[0]


def line_of(node: str) -> str:
    return node.split("_", 1)[0]


def has_station_revisit(path: list[str]) -> bool:
    """같은 역을 떨어진 위치에서 다시 지나는 경로인지 본다.

    6호선에서 '구산 -> 응암(순환끝) -> 새절 -> 응암(본선) -> 역촌' 같은 U턴이 생긴다.
    새절에서 반대 방향 열차로 갈아타는 셈인데 그래프에는 비용이 없어서, 배차가 긴
    시간대에는 정직한 재승차 환승보다 싸게 나온다.
    분기역의 계통 환승(강동->강동, 성수->성수)은 **연속**이므로 허용한다.
    """
    names = [station_of(n) for n in path if not is_virtual(n)]
    seen, prev = set(), None
    for nm in names:
        if nm != prev:
            if nm in seen:
                return True
            seen.add(nm)
        prev = nm
    return False


def is_virtual(node: str) -> bool:
    return node.startswith(ORIGIN_PREFIX) or node.startswith(DEST_PREFIX)


# --------------------------------------------------------------------------
# 노선별 방면 라벨. 프로젝트 구간의 종점을 기준으로 한다.
# 임의 추측이 아니라 route_edge_mart 의 direction / branch_code 에서 계산한다.
DIRECTION_LABEL = {
    ("1", "main", "up"): "청량리 방면", ("1", "main", "down"): "서울역 방면",
    ("2", "main", "inner"): "내선순환", ("2", "main", "outer"): "외선순환",
    ("2", "seongsu_branch", "inner"): "성수 방면",
    ("2", "seongsu_branch", "outer"): "신설동 방면",
    ("2", "seongsu_branch_east", "inner"): "성수 방면",
    ("2", "seongsu_branch_east", "outer"): "신설동 방면",
    ("2", "sinjeong_branch", "inner"): "까치산 방면",
    ("2", "sinjeong_branch", "outer"): "신도림 방면",
    ("3", "main", "up"): "지축 방면", ("3", "main", "down"): "오금 방면",
    ("4", "main", "up"): "불암산 방면", ("4", "main", "down"): "남태령 방면",
    ("5", "main", "up"): "방화 방면", ("5", "main", "down"): "하남검단산 방면",
    ("5", "macheon_branch", "up"): "방화 방면",
    ("5", "macheon_branch", "down"): "마천 방면",
    ("6", "main", "up"): "응암 방면", ("6", "main", "down"): "신내 방면",
    ("6", "eungam_loop", "down"): "응암순환 방면",
    ("7", "main", "up"): "장암 방면", ("7", "main", "down"): "온수 방면",
    ("8", "main", "up"): "암사역사공원 방면", ("8", "main", "down"): "모란 방면",
}


def _edge_index(scorer):
    """(from_node, to_node) -> 승차 엣지 메타. 방면/혼잡도 계산용."""
    idx = getattr(scorer, "_edge_meta_index", None)
    if idx is None:
        idx = {}
        for r in scorer.ride.itertuples():
            idx[(r.from_node, r.to_node)] = {
                "line": str(r.line_id),
                "direction": r.direction if pd.notna(r.direction) else None,
                "branch": getattr(r, "branch_code", "main"),
                "congestion": float(r.congestion),
                "time": float(r.travel_time_min),
            }
        scorer._edge_meta_index = idx
    return idx


def bound_label(line: str, branch: str, direction) -> str:
    """종점 기준 방면 문구(예: '방화 방면'). 참고용으로만 보관한다.

    화면에는 쓰지 않는다. 지선·순환 때문에 예외가 계속 늘고, 잘못 표기할 위험이 있다.
    """
    if not direction:
        return ""
    for b in (branch, "main"):
        lab = DIRECTION_LABEL.get((str(line), b, direction))
        if lab:
            return lab
    return ""


def describe_path(scorer, path: list[str]) -> list[dict]:
    """내부 노드를 사용자용 **구조화 segment** 로 바꾼다. station_uid 를 노출하지 않는다.

    ride    : line, direction_label, from/to, station_count, duration_min,
              max_congestion, stations(전체 정차역)
    transfer: at, from_line, to_line, duration_min
    """
    real = [n for n in path if not is_virtual(n)]
    if not real:
        return []
    emeta = _edge_index(scorer)
    segs = []
    cur = {"kind": "ride", "line": line_of(real[0]), "from": station_of(real[0]),
           "stations": [station_of(real[0])], "n_stops": 0, "minutes": 0.0,
           "max_congestion": 0.0, "dirs": [], "branches": []}

    for i, (u, v) in enumerate(zip(real, real[1:])):
        e = next((x for x in scorer.adj.get(u, []) if x["to"] == v), None)
        if e is None:
            continue
        if e["kind"] == "transfer" and _is_through_pass(scorer, real, i, u, v):
            # 직결 분기 통과. 같은 열차이므로 segment 를 끊지 않는다.
            # 분기역은 중간 정차역으로 남는다.
            continue
        if e["kind"] == "transfer":
            cur["to"] = station_of(u)
            segs.append(cur)
            segs.append({"kind": "transfer", "at": station_of(u),
                         "from_line": cur["line"], "to_line": line_of(v),
                         "minutes": round(e["time"], 1)})
            cur = {"kind": "ride", "line": line_of(v), "from": station_of(v),
                   "stations": [station_of(v)], "n_stops": 0, "minutes": 0.0,
                   "max_congestion": 0.0, "dirs": [], "branches": []}
        else:
            m = emeta.get((u, v), {})
            cur["n_stops"] += 1
            cur["minutes"] += float(e.get("time", 0.0))
            c = m.get("congestion")
            if c is not None and not pd.isna(c):
                cur["max_congestion"] = max(cur["max_congestion"], float(c))
            if m.get("direction"):
                cur["dirs"].append(m["direction"])
            cur["branches"].append(m.get("branch", "main"))
            cur["stations"].append(station_of(v))
    cur["to"] = station_of(real[-1])
    segs.append(cur)

    out = []
    for sgm in segs:
        if sgm["kind"] != "ride":
            out.append(sgm)
            continue
        if sgm["n_stops"] == 0 and len(segs) > 1:
            continue                      # 0정거장 구간은 표시하지 않는다
        dirs = sgm.pop("dirs")
        branches = sgm.pop("branches")
        d = max(set(dirs), key=dirs.count) if dirs else None
        b = max(set(branches), key=branches.count) if branches else "main"
        sgm["direction"] = d
        sgm["branch"] = b
        # 방면 표기는 **경로상 바로 다음 역** 을 쓴다.
        # 종점 기준(방화 방면 등)은 지선·순환에서 예외가 많고 오표기 위험이 있는데,
        # 다음 역은 경로에서 직접 나오므로 추론이 개입하지 않는다.
        stns = sgm.get("stations") or []
        nxt = stns[1] if len(stns) >= 2 else None
        sgm["next_station"] = nxt
        # 1정거장 구간은 다음 역 = 도착역이라 '동묘앞 방면 신설동 → 동묘앞' 처럼
        # 같은 역명이 두 번 나온다. 그 경우 방면 표기를 생략한다.
        sgm["direction_label"] = ("%s 방면" % nxt) if (nxt and nxt != sgm["to"]) else ""
        sgm["bound_label"] = bound_label(sgm["line"], b, d)   # 참고용(비표시)
        # 중간역 정차시간. evaluate() 와 같은 규칙(엣지 k개 -> 중간역 k-1개)을 쓴다.
        # 여기서 더하지 않으면 카드 총계와 segment 합계가 어긋난다.
        dwell_min = _dwell_min(scorer)
        sgm["dwell_stop_count"] = max(sgm["n_stops"] - 1, 0)
        sgm["dwell_min"] = round(sgm["dwell_stop_count"] * dwell_min, 1)
        sgm["running_min"] = round(sgm["minutes"], 1)
        sgm["minutes"] = round(sgm["minutes"] + sgm["dwell_min"], 1)
        sgm["max_congestion"] = round(sgm["max_congestion"], 1)
        out.append(sgm)
    return out


def segments_to_text(segs: list[dict]) -> list[str]:
    """폴백/테스트용 텍스트. 화면은 타임라인 카드로 렌더링한다."""
    out = []
    for i, s in enumerate(segs):
        if s["kind"] == "ride":
            verb = "승차" if i == 0 else "탑승"
            d = (" %s" % s["direction_label"]) if s.get("direction_label") else ""
            out.append("%s에서 %s호선%s %s → %s (%d개 역)"
                       % (s["from"], s["line"], d, verb, s["to"], s["n_stops"]))
        else:
            out.append("%s에서 %s호선으로 환승 (약 %.0f분)"
                       % (s["at"], s["to_line"], s["minutes"]))
    return out


def find_route_by_station(scorer, display: pd.DataFrame,
                          origin_station_name: str,
                          destination_station_name: str,
                          preference_mode: str = "calm",
                          k: int = 5) -> dict:
    """역 단위 입력으로 경로를 찾는다.

    반환 dict:
        ok, fastest, alternative, time_alternative, candidates, n_origin_nodes, ...
    각 후보에는 segments(사용자 문구용)와 transfer_count 가 들어간다.
    origin_access / destination_exit 은 환승으로 세지 않는다.
    """
    o_nodes = candidates_of(display, origin_station_name)
    d_nodes = candidates_of(display, destination_station_name)
    if not o_nodes or not d_nodes:
        return {"ok": False, "reason": "역 정보를 찾을 수 없습니다."}
    if origin_station_name == destination_station_name:
        return {"ok": False, "reason": "출발역과 도착역이 같습니다."}

    v_o = ORIGIN_PREFIX + origin_station_name
    v_d = DEST_PREFIX + destination_station_name

    orig_adj = scorer.adj
    orig_nodes = scorer.nodes
    aug = dict(orig_adj)
    aug[v_o] = [{"to": n, "cost": 0.0, "time": 0.0, "cong": float("nan"),
                 "event": 0.0, "kind": "origin_access"} for n in o_nodes if n in orig_nodes]
    for n in d_nodes:
        if n not in orig_nodes:
            continue
        aug[n] = list(aug.get(n, [])) + [
            {"to": v_d, "cost": 0.0, "time": 0.0, "cong": float("nan"),
             "event": 0.0, "kind": "destination_exit"}]
    if not aug[v_o]:
        return {"ok": False, "reason": "출발역이 그래프에 없습니다."}

    try:
        scorer.adj = aug
        scorer.nodes = set(aug) | {e["to"] for lst in aug.values() for e in lst}
        paths_t = scorer._yen(v_o, v_d, K=k, weight="time")
        paths_c = scorer._yen(v_o, v_d, K=k, weight="cost")
    finally:
        scorer.adj = orig_adj
        scorer.nodes = orig_nodes

    seen, raw, n_revisit = set(), [], 0
    for p in paths_t + paths_c:
        real = tuple(n for n in p if not is_virtual(n))
        if len(real) < 2 or real in seen:
            continue
        seen.add(real)
        if has_station_revisit(list(real)):
            n_revisit += 1
            continue
        raw.append(list(real))
    if not raw:
        return {"ok": False, "reason": "프로젝트 범위 내에서 연결되는 경로가 없습니다."}

    headway = load_headway(getattr(scorer, "root", Path(".")))
    cands = []
    for p in raw:
        ev = scorer.evaluate(p)
        ev["segments"] = describe_path(scorer, p)
        ev["boarding_line"] = line_of(p[0])
        ev["alighting_line"] = line_of(p[-1])
        # 환승 대기시간을 더한 뒤 점수를 매긴다(대기가 순위에 반영되어야 한다).
        apply_transfer_wait(ev, headway, scorer.day_type, scorer.time_bin)
        cands.append(ev)

    from importlib import import_module  # noqa
    w = _MODE_WEIGHTS.get(preference_mode, _MODE_WEIGHTS["calm"])
    for c in cands:
        c["route_score"] = (w["actual"] * c["actual_time_min"]
                            + w["perceived"] * c["perceived_time_min"]
                            + w["max_cong"] * (c["max_congestion"] or 0)
                            + w["tpen"] * c["transfer_penalty_min"]
                            - w["seat"] * c["seat_chance_score"])
    cands.sort(key=lambda c: c["route_score"])

    # 다양성 필터 (directed edge 기준)
    def edge_set(ev):
        p = ev["path"]
        return set(zip(p, p[1:]))
    kept = []
    for c in cands:
        e = edge_set(c)
        if all(_jaccard(e, edge_set(k2)) <= 0.65 for k2 in kept):
            kept.append(c)
        if len(kept) >= 4:
            break
    if not kept:
        kept = cands[:1]

    fastest = min(cands, key=lambda c: c["actual_time_min"])
    alt = None
    for c in kept:
        if c["path"] == fastest["path"]:
            continue
        tl = c["actual_time_min"] - fastest["actual_time_min"]
        cd = (fastest["max_congestion"] or 0) - (c["max_congestion"] or 0)
        pe = c["perceived_time_min"] - fastest["perceived_time_min"]
        if tl <= 15.0 and cd >= 15.0 and pe <= 5.0:
            alt = dict(c)
            alt["time_loss_vs_fastest"] = round(tl, 1)
            alt["comfort_gain_vs_fastest"] = round(cd, 1)
            break

    time_alt = None
    if alt is None:
        try:
            time_alt = scorer.time_alternative(kept[0]["path"])
        except Exception:
            time_alt = None

    return {"ok": True,
            "n_revisit_filtered": n_revisit,
            "origin_station": origin_station_name,
            "destination_station": destination_station_name,
            "preference_mode": preference_mode,
            "n_origin_nodes": len(aug[v_o]),
            "n_destination_nodes": len([n for n in d_nodes if n in orig_nodes]),
            "recommended": kept[0],
            "fastest": fastest,
            "alternative": alt,
            "time_alternative": time_alt,
            "candidates": kept}


_MODE_WEIGHTS = {
    "fast":         {"actual": 1.0, "perceived": 0.0, "max_cong": 0.00, "tpen": 0.3, "seat": 0.0},
    "calm":         {"actual": 0.0, "perceived": 1.0, "max_cong": 0.05, "tpen": 0.5, "seat": 0.0},
    "min_transfer": {"actual": 0.5, "perceived": 0.5, "max_cong": 0.00, "tpen": 3.0, "seat": 0.0},
    "balanced":     {"actual": 0.3, "perceived": 0.7, "max_cong": 0.03, "tpen": 0.8, "seat": 0.5},
}


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0

# ---------------------------------------------------------------- 환승 방면 표기
def _norm_station_token(s) -> str:
    """역명·방면 표기를 비교용 토큰으로 정규화한다.

    '강동 방면' -> '강동',  '서울역' -> '서울',  '동대문역사문화공원 방면' -> 그대로.
    끝의 '역' 만 떼므로 역명 중간의 '역' 은 보존된다.
    """
    s = "" if s is None else str(s)
    s = s.replace("방면", "")
    s = "".join(s.split())
    if len(s) > 1 and s.endswith("역"):
        s = s[:-1]
    return s


def is_valid_toward(station, toward) -> bool:
    """방면 표기가 환승역 자기 자신을 가리키면 무효로 본다.

    원본 `수도권 도시철도 환승 데이터` 의 강동 5→5 두 행은
    `하차 열차 방면` 이 '강동 방면'(자기 역명)으로 기재돼 있다.
    substring 이 아니라 정규화 후 완전 일치로 판단하므로
    '동대문' 과 '동대문역사문화공원' 은 서로 다른 토큰이 된다.
    """
    t = _norm_station_token(toward)
    return bool(t) and t != _norm_station_token(station)


def transfer_basis_text(station, arrive_toward, depart_toward) -> str:
    """환승 방면 기준 문구. 무효한 방면만 생략하고 나머지는 그대로 쓴다.

    원본 값을 고쳐 쓰지 않는다. 경로 기준 방면 유도는 v1.1 과제다.
    """
    parts = []
    if is_valid_toward(station, arrive_toward):
        parts.append("%s 하차" % str(arrive_toward).strip())
    if is_valid_toward(station, depart_toward):
        parts.append("%s 승차" % str(depart_toward).strip())
    if not parts:
        return "환승 위치 기준"
    return " · ".join(parts) + " 기준"


def _dwell_min(scorer) -> float:
    """RouteScorer 가 정의된 모듈의 DEFAULT_DWELL_TIME_MIN.

    load_scorer_class() 는 module_from_spec + exec_module 로 모듈을 만들고
    sys.modules 에 등록하지 않는다. 따라서 sys.modules 조회로는 찾을 수 없고,
    클래스 객체의 __globals__ 를 통해 모듈 전역을 직접 읽어야 한다.
    """
    g = getattr(type(scorer).evaluate, "__globals__", {})
    try:
        return float(g.get("DEFAULT_DWELL_TIME_MIN", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_through_pass(scorer, path, i, u, v) -> bool:
    """분기 직결 통과 판정. scorer 모듈의 규칙을 그대로 쓴다.

    load_scorer_class() 가 만든 모듈은 sys.modules 에 등록되지 않으므로
    클래스의 __globals__ 를 통해 읽는다(_dwell_min 과 같은 방식).
    """
    g = getattr(type(scorer).evaluate, "__globals__", {})
    fn = g.get("is_through_pass")
    return bool(fn(path, i, u, v)) if fn else False
