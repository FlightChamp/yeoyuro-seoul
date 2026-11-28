"""
tests/test_map_click_consistency.py — D-113: 노선도 클릭 영역·역 이름이 자기 역 점에 붙어 있는가 (앱이 그리는 csv 기준).

2026-11-28 사용자 발견: 9호선 공항시장~선유도·석촌~중앙보훈병원 클릭 위치가 숫자로 고정돼 역 점을 옮긴 뒤에도 옛 자리에 남아
이웃 역이 선택됐다 (예: 마곡나루 점 → 신방화, 중앙보훈병원 점 → 둔촌오륜). 799ff0c·e9db118 에서 수식으로 되돌림.
이 테스트가 있으면 map_save_commit.py 의 전체 테스트 단계에서 같은 문제가 걸린다.

기준: 클릭 영역·역 이름 위치에서 가장 가까운 역 점이 자기 역이고, 클릭은 클릭 반경 안, 역 이름은 20 단위 안.
예외(일부러 옮김, 역 이름과 클릭을 같은 자리로): 한양대·뚝섬 — 자기 역이 가장 가깝고 35 단위 안이면 통과.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "app_v26" / "data" / "master"
EXCEPT = {"한양대": 35, "뚝섬": 35}
need = pytest.mark.skipif(not (MASTER / "map_click_areas.csv").exists(), reason="앱 노선도 csv 없음")


def _pts():
    vn = pd.read_csv(MASTER / "map_visual_nodes.csv")
    return {k: list(zip(g.x_px.astype(float), g.y_px.astype(float))) for k, g in vn.groupby("station_key")}


def _check(df, xc, yc, limit_of):
    pts = _pts()

    def dist(k, c):
        return min(math.dist(p, c) for p in pts[k])
    bad = []
    for r in df.itertuples():
        k = r.station_key
        if k not in pts:
            bad.append((k, "역 점 없음"))
            continue
        c = (float(getattr(r, xc)), float(getattr(r, yc)))
        near = min(pts, key=lambda s: dist(s, c))
        lim = EXCEPT.get(k, limit_of(r))
        if near != k or dist(k, c) > lim:
            bad.append((k, round(dist(k, c)), near, lim))
    return bad


@need
def test_click_areas_on_own_station():
    ca = pd.read_csv(MASTER / "map_click_areas.csv")
    assert len(ca) >= 270
    assert _check(ca, "click_x_px", "click_y_px", lambda r: float(r.click_radius_px)) == []


@need
def test_labels_on_own_station():
    lb = pd.read_csv(MASTER / "map_labels.csv")
    assert _check(lb, "label_x_px", "label_y_px", lambda r: 20.0) == []
