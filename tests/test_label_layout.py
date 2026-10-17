"""노선도 역명 레이어 회귀 테스트 (D-067).

역명은 캔버스 좌표 단위 SVG 이미지라 확대 배율과 상관없이 역명끼리의 겹침이 같다.
기본 좌표(map_labels.csv)에서 역명끼리 겹치는 쌍이 하나도 없어야 한다.
좌표를 고친 뒤 이 테스트가 실패하면 yeoyuro_v2/map_labels.py 의 LABEL_FONT_UNITS 를 줄이거나 해당 역 좌표를 옮긴다.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import pytest

from yeoyuro_v2 import map_labels as ml

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "app_v26" / "data" / "master" / "map_labels.csv"
pytestmark = pytest.mark.skipif(not LABELS.exists(), reason="map_labels.csv 없음")


def _lb():
    return pd.read_csv(LABELS)


def test_no_station_labels_overlap():
    assert ml.overlapping_pairs(_lb()) == []


def test_all_labels_rendered_inside_canvas():
    lb = _lb()
    boxes = ml.label_boxes(lb)
    assert len(boxes) == lb["station_key"].nunique()
    for k, (x0, y0, x1, y1) in boxes.items():
        assert 0 <= x0 and x1 <= 5120 and 0 <= y0 and y1 <= 2880, k


def test_transfer_labels_larger_and_estimate_conservative():
    assert ml.font_size(12.0) > ml.font_size(10.5) == ml.LABEL_FONT_UNITS
    assert ml.text_width("동대문역사문화공원", 10) == pytest.approx(85.5)


def test_all_labels_below_marker_by_default():
    lb = _lb()
    boxes = ml.label_boxes(lb)
    assert all(boxes[r.station_key][1] > r.label_y_px for r in lb.itertuples())   # 모든 역명이 역 점 아래 (사용자 요청)


def test_above_anchor_supported():
    lb = pd.DataFrame({"station_key": ["가"], "label_x_px": [100], "label_y_px": [100],
                       "label_size_px": [10.5], "label_anchor": ["above"]})
    assert ml.label_boxes(lb)["가"][3] < 100


def test_svg_layout_image_scales_with_axes():
    lb = _lb()
    img = ml.layout_image(lb, 5120, 2880)
    assert img["xref"] == "x" and img["yref"] == "y" and img["sizex"] == 5120 and img["layer"] == "above"
    svg = base64.b64decode(img["source"].split(",", 1)[1]).decode("utf-8")
    assert svg.count("<text ") == len(ml.label_rows(lb))
    assert "동대문역사문화공원" in svg and 'paint-order="stroke"' in svg


def test_draw_map_uses_label_image_not_text_traces():
    src = (ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py").read_text(encoding="utf-8")
    body = src[src.index("def draw_map("):src.index("from html import escape as _esc")]
    assert 'mode="text"' not in body and "_map_labels.layout_image" in body
