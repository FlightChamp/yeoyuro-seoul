"""
scripts/v2/24_build_mobility_atlas.py
=====================================
v2.4 — Mobility Atlas: v2.3 OD 판정 결과를 '구간(역→역, 방향 있음)' 단위로 모은다.

원칙 (기획서 §4, docs/v2/mobility_atlas.md)
- 단일 Stress 점수를 만들지 않는다. 해석 가능한 지표(dimension)를 따로 둔다.
- 비율 지표는 지지 표본이 작으면(혼잡 통과 < MIN_SUPPORT) 값을 비운다. 작은 표본의 100% 를 지도에 칠하지 않는다.
- 모든 값은 무작위 OD 표본 위의 집계다. '이 구간을 지나는 모든 승객'의 값이 아니다.

입력
----
    data/marts/v2/od_shiftability_evidence.pkl.gz   (v2.3 실행 시 생성, 커밋 안 됨)
    data/marts/v2/od_shiftability_mart.parquet      (v2.3, 커밋됨)
    data/marts/route_edge_mart.parquet               (v1)

출력
----
    data/marts/v2/mobility_atlas_mart.parquet   (long: section × time_group × dimension, 커밋 대상)
    reports/v2/mobility_atlas_report.md

dimension
---------
    congestion_exposure        통과 1회당 평균 100%+ 노출(분)               (OD-count)
    demand_weighted_exposure   같은 값을 수요 proxy 로 가중                   (proxy)
    exposed_traversal_share    통과 중 이 구간에서 100%+ 를 겪은 비율
    structural_share           혼잡 통과 중 Structurally constrained 이동 비율
    alternative_opportunity    혼잡 통과 중 Route 성공(Route-shiftable+Dual) 비율
    temporal_shift_benefit     혼잡 통과 중 Time 성공(Time-shiftable+Dual) 비율
  '혼잡 통과' = 그 구간에서 실제로 100%+ 노출이 있었던 통과 (이동 전체가 혼잡인 것이 아니라)
  (transfer_burden: 환승역 단위 대기·도보 evidence 를 저장하지 않아 v2.4 에서 제외, D-024
   event_sensitivity: v2.5)

실행
----
    python scripts/v2/24_build_mobility_atlas.py
"""

from __future__ import annotations

import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MIN_SUPPORT = 5
GROUPS = {"am": ("07:30", "08:00", "08:30"), "pm": ("17:30", "18:00", "18:30")}
DIMS = ["congestion_exposure", "demand_weighted_exposure", "exposed_traversal_share",
        "structural_share", "alternative_opportunity", "temporal_shift_benefit"]
DIM_KO = {"congestion_exposure": "통과 1회당 100%+ 노출(분)",
          "demand_weighted_exposure": "수요 proxy 가중 노출(분)",
          "exposed_traversal_share": "100%+ 를 겪은 통과 비율",
          "structural_share": "구조적 혼잡 이동 비율",
          "alternative_opportunity": "경로 변경으로 피할 수 있는 비율",
          "temporal_shift_benefit": "시간 조정으로 피할 수 있는 비율"}


def station_of(node: str) -> str:
    return node.split("_", 1)[1].split("@")[0]


def main() -> int:
    evp = ROOT / "data" / "marts" / "v2" / "od_shiftability_evidence.pkl.gz"
    if not evp.exists():
        print(f"[중단] {evp} 가 없습니다. 먼저 python scripts/v2/23_build_od_shiftability.py 를 실행하세요.")
        return 1
    with gzip.open(evp, "rb") as f:
        items = pickle.load(f)
    mart = pd.read_parquet(ROOT / "data" / "marts" / "v2" / "od_shiftability_mart.parquet")
    if len(mart) != len(items):
        print("[중단] evidence 와 od_shiftability_mart 의 행 수가 다릅니다. 23 스크립트를 다시 실행하세요.")
        return 1
    key = ["origin", "destination", "base_time"]
    ev_keys = [(it["origin"], it["destination"], it["base_time"]) for it in items]
    if list(map(tuple, mart[key].values)) != ev_keys:
        mart = mart.set_index(key).loc[ev_keys].reset_index()

    redge = pd.read_parquet(ROOT / "data" / "marts" / "route_edge_mart.parquet")
    ride = {(r.from_node, r.to_node): str(r.line_id) for r in redge.itertuples()}

    rows = []
    for it, m in zip(items, mart.itertuples(index=False)):
        path = it["base_path"]
        typ = m.final_shiftability_type
        hot = typ != "calm"
        for u, v in zip(path, path[1:]):
            line = ride.get((u, v))
            if line is None:
                continue                                  # 환승·직결 통과 edge 제외
            label = f"{line}호선 {station_of(u)}→{station_of(v)}"
            rows.append({
                "from_node": u, "to_node": v, "line_id": line, "section": label,
                "from_station": station_of(u), "to_station": station_of(v),
                "base_time": it["base_time"], "w": m.demand_proxy_w,
                "exp100": it["edge_exp100"].get(label, 0.0), "hot": hot,
                "structural": typ == "structural",
                "route_ok": bool(m.route_shift_success) if hot else False,
                "time_ok": bool(m.time_shift_success) if hot else False,
            })
    tr = pd.DataFrame(rows)
    tr["time_group"] = tr.base_time.map({t: g for g, ts in GROUPS.items() for t in ts})

    def agg(g):
        # '혼잡 통과' = 이 구간에서 실제로 100%+ 를 겪은 통과. 이동 전체의 판정을, 혼잡을 겪지 않은
        # 구간에까지 칠하면 지도가 왜곡된다 (예: 구조적 이동의 출발 구간이 구조적으로 보임). D-025
        hot = g[g.exp100 > 0]
        n_hot = len(hot)
        ok = n_hot >= MIN_SUPPORT
        wsum = g.w.sum()
        return pd.Series({
            "n_traversals": len(g), "n_hot_traversals": n_hot,
            "low_support": not ok,
            "congestion_exposure": g.exp100.mean(),
            "demand_weighted_exposure": (g.w * g.exp100).sum() / wsum if wsum > 0 else np.nan,
            "exposed_traversal_share": (g.exp100 > 0).mean(),
            "structural_share": hot.structural.mean() if ok else np.nan,
            "alternative_opportunity": hot.route_ok.mean() if ok else np.nan,
            "temporal_shift_benefit": hot.time_ok.mean() if ok else np.nan,
            "exposure_sum_min": g.exp100.sum(),
        })

    sec_cols = ["from_node", "to_node", "line_id", "section", "from_station", "to_station"]
    parts = []
    for grp, sub in [("all", tr), *[(g, tr[tr.time_group == g]) for g in GROUPS]]:
        a = sub.groupby(sec_cols).apply(agg, include_groups=False).reset_index()
        a["time_group"] = grp
        parts.append(a)
    wide = pd.concat(parts, ignore_index=True)
    long = wide.melt(id_vars=sec_cols + ["time_group", "n_traversals", "n_hot_traversals", "low_support"],
                     value_vars=DIMS, var_name="atlas_dimension", value_name="value")
    long["day_type"] = "weekday"
    long["threshold_set_id"] = mart.threshold_set_id.iloc[0]
    out = ROOT / "data" / "marts" / "v2" / "mobility_atlas_mart.parquet"
    long.to_parquet(out, index=False)

    # ---------------------------------------------------------------- 리포트
    W = wide[wide.time_group == "all"]
    sup = W[~W.low_support]
    top_exp = W.sort_values("exposure_sum_min", ascending=False).head(10)
    top_struct = sup[(sup.structural_share > 0) & (sup.exposure_sum_min >= 30)].sort_values(["structural_share", "n_hot_traversals"],
                                                          ascending=False).head(10)
    low_alt = sup.sort_values(["alternative_opportunity", "n_hot_traversals"], ascending=[True, False]).head(10)
    corr = sup[["congestion_exposure", "structural_share", "alternative_opportunity",
                "temporal_shift_benefit"]].corr(method="spearman").round(2)
    by_line = (W.groupby("line_id").agg(구간수=("section", "size"), 노출합계=("exposure_sum_min", "sum"))
                .assign(노출비중=lambda d: (d.노출합계 / d.노출합계.sum()).round(3)).sort_values("노출합계",
                                                                                     ascending=False))
    show = ["section", "n_traversals", "n_hot_traversals", "congestion_exposure", "exposure_sum_min",
            "structural_share", "alternative_opportunity", "temporal_shift_benefit"]
    L = [
        "# v2.4 Mobility Atlas 리포트",
        "",
        f"- 입력: v2.3 od_shiftability (ThresholdSet `{mart.threshold_set_id.iloc[0]}`), 단위 {len(items):,}건",
        f"- 구간(방향 있음) {W.shape[0]:,}개, 통과 기록 {len(tr):,}건, "
        f"혼잡 통과(그 구간에서 100%+ 노출) ≥ {MIN_SUPPORT} 인 구간 {len(sup):,}개 (비율 지표는 이 구간만 계산)",
        "- 단일 종합 점수는 만들지 않는다. 지표별로 따로 본다.",
        "- 모든 값은 무작위 OD 표본의 기준 경로가 그 구간을 지날 때의 집계다. 그 구간 전체 승객의 값이 아니다.",
        "",
        "## 1. 노출이 큰 구간 (100%+ 노출 합계 상위 10)",
        "",
        top_exp[show].round(3).to_markdown(index=False),
        "",
        "## 2. 구조적 혼잡 비율이 높은 구간 (혼잡 통과 ≥ 5, 노출 합계 30분 이상)",
        "",
        top_struct[show].round(3).to_markdown(index=False) if len(top_struct) else "(없음)",
        "",
        "## 3. 경로 대안이 가장 적은 구간 (혼잡 통과 ≥ 5)",
        "",
        low_alt[show].round(3).to_markdown(index=False),
        "",
        "## 4. 노선별 노출 비중",
        "",
        by_line.to_markdown(),
        "",
        "## 5. 지표 간 순위 상관 (Spearman, 혼잡 통과 ≥ 5 구간)",
        "",
        corr.to_markdown(),
        "",
        "상관이 낮을수록 지표가 서로 다른 정보를 담는다는 뜻이며, 하나의 점수로 합치지 않는 근거가 된다.",
        "단, structural_share 와 temporal_shift_benefit 은 정의상 겹친다 (Structural = Route·Time 모두 실패) — "
        "이 둘의 강한 음의 상관은 발견이 아니라 정의의 결과다.",
    ]
    rdir = ROOT / "reports" / "v2"
    (rdir / "mobility_atlas_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n저장: {out} ({len(long):,}행)\n저장: {rdir / 'mobility_atlas_report.md'}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()                       # Windows 에서 한글 출력 때문에 멈추지 않게
    raise SystemExit(main())
