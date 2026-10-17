"""노선도 역명 겹침 회귀 테스트 — 기본 좌표 엑셀(라벨 이동량 포함) 기준으로 역명끼리 겹치는 쌍이 거의 없어야 한다.

scripts/v2/39_fix_label_overlaps.py 와 같은 크기 계산(1920×1080 발표 화면 기준)을 쓴다. 노선도 좌표를 고친 뒤
이 테스트가 실패하면 39 스크립트를 --apply 로 다시 실행하면 된다.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"


@pytest.mark.skipif(not BASE.exists(), reason="노선도 좌표 엑셀 없음")
def test_station_labels_do_not_overlap():
    spec = importlib.util.spec_from_file_location("fixlab", ROOT / "scripts" / "v2" / "39_fix_label_overlaps.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    lab, vn, segs = m.load(BASE)
    B = m.base_boxes(lab)
    boxes = {k: m.box(b, b["dx"], b["dyv"]) for k, b in B.items()}
    ks = list(boxes)
    def real(a, b, px=3.0):                         # 화면 3px 이상 실제로 겹칠 때만 (닿는 정도는 제외)
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * m.SCALE >= px and h * m.SCALE >= px
    pairs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:] if real(boxes[a], boxes[b])]
    assert len(pairs) <= 1, pairs
    # 시연 역 주변은 겹침 0
    demo = {"동대문역사문화공원", "동대문", "을지로3가", "을지로4가", "충무로", "신당"}
    assert not [p for p in pairs if set(p) & demo], pairs
