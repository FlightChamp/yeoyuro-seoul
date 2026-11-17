"""
yeoyuro_v2/map_labels.py
========================
노선도 역명 레이어 (D-067).

Plotly 의 text trace 는 글자 크기가 화면 픽셀로 고정이라 확대해도 글자가 커지지 않는다.
그래서 전체 화면에서 겹치지 않게 줄이면 확대해도 계속 작다.
지도 앱처럼 "전체 보기에서는 아주 작게, 확대하면 함께 커지게" 하려고, 역명을 캔버스 좌표 단위의
SVG 한 장으로 그려 Plotly layout image(xref/yref = 축) 로 얹는다. 이미지가 데이터 좌표에 묶여 있으므로
확대하면 글자도 같은 비율로 커지고, 벡터라 확대해도 흐려지지 않는다.

글자 크기가 캔버스 단위이므로 역명끼리의 겹침은 확대 배율과 무관하다 → 아래 label_boxes 로 한 번만 검사하면 된다.
Streamlit 에 의존하지 않는다 (pytest 로 검증).
"""

from __future__ import annotations

import base64
from html import escape
from itertools import combinations

# 일반역 역명 글자 크기 (캔버스 좌표 단위, 캔버스 5120×2880).
# 1920×1080 화면 전체 보기는 0.279 화면px / 캔버스 단위 → 17.9 단위 ≈ 5px (사용자 지정). 2배 확대 ≈ 10px, 3배 ≈ 15px.
# 모든 역명은 역 점 아래. 이 크기에서 닿던 4곳은 워크북 label_dx 로 가로만 2~13 단위 옮겼다 (D-068).
LABEL_FONT_UNITS = 17.9
BASE_SIZE_PX = 10.5            # 워크북 label_size_px 기준값 (환승역 12·13 은 이 비율로 키운다)
LABEL_DY_NORMAL = 20           # 역 점 중심 → 역명 윗변 (캔버스 단위). 26 → 20: 역 점에 더 가깝게
LABEL_DY_MAJOR = 30            # 환승역(마커가 커서 조금 더 띄움). 36 → 30
ABOVE_ANCHORS = {"above", "top", "위"}   # 워크북 Station_Labels.label_anchor 가 이 값이면 역명을 역 점 위에 둔다
LINE_HEIGHT = 1.15             # 글자 상자 높이 = 글자 크기 × 이 값
HALO = 0.24                    # 흰 테두리 두께 = 글자 크기 × 이 값 (노선 위에서도 읽히게)
TEXT_COLOR = "#111827"
FONT_FAMILY = ("'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR','Noto Sans CJK KR',"
               "'NanumGothic',sans-serif")


def char_em(c: str) -> float:
    """글자 폭(em) 추정. 한글 0.95, 숫자·영문 0.62.
    Noto Sans CJK KR Bold 실측 한글 0.92 · 숫자 0.59 에 약 3% 여유. 맑은 고딕(Windows) 실측은 미확인."""
    return 0.62 if c.isascii() else 0.95


def text_width(text: str, size: float) -> float:
    return sum(char_em(c) for c in text) * size


def font_size(label_size_px: float) -> float:
    try:
        s = float(label_size_px)
    except (TypeError, ValueError):
        s = BASE_SIZE_PX
    if s != s:                                  # NaN
        s = BASE_SIZE_PX
    return LABEL_FONT_UNITS * s / BASE_SIZE_PX


def label_rows(lb) -> list[tuple[str, float, float, float, float]]:
    """(역명, 가운데 x, 윗변 y, 글자 크기, 폭). lb = map_labels (station_key, label_x_px, label_y_px, label_size_px)."""
    out = []
    for r in lb.itertuples():
        try:
            x, y = float(r.label_x_px), float(r.label_y_px)
        except (TypeError, ValueError):
            continue
        if x != x or y != y:
            continue
        size = font_size(getattr(r, "label_size_px", BASE_SIZE_PX))
        dy = LABEL_DY_MAJOR if size > LABEL_FONT_UNITS + 1e-9 else LABEL_DY_NORMAL
        name = str(r.station_key)
        if str(getattr(r, "label_anchor", "") or "").strip().lower() in ABOVE_ANCHORS:
            # 위: 글자 상자 아랫변이 역 점 중심에서 dy 만큼 위 (상자 아래쪽 여백 약 0.12em 은 빼서 간격을 맞춘다)
            top = y - dy - size * LINE_HEIGHT + size * 0.12
        else:
            top = y + dy
        out.append((name, x, top, size, text_width(name, size)))
    return out


def label_boxes(lb) -> dict[str, tuple[float, float, float, float]]:
    """역명 → (x0, y0, x1, y1) 캔버스 좌표 상자."""
    return {n: (x - w / 2, top, x + w / 2, top + s * LINE_HEIGHT) for n, x, top, s, w in label_rows(lb)}


def overlapping_pairs(lb, margin_ratio: float = 0.05) -> list[tuple[str, str]]:
    """서로 겹치는 역명 쌍. margin = 글자 크기 × margin_ratio 만큼 붙어 있어도 겹침으로 본다."""
    rows = label_rows(lb)
    boxes = [(n, x - w / 2, top, x + w / 2, top + s * LINE_HEIGHT, s * margin_ratio)
             for n, x, top, s, w in rows]
    pairs = []
    for a, b in combinations(boxes, 2):
        m = max(a[5], b[5])
        if a[1] - m < b[3] and b[1] - m < a[3] and a[2] - m < b[4] and b[2] - m < a[4]:
            pairs.append((a[0], b[0]))
    return pairs


def label_svg(lb, width: int, height: int) -> str:
    """캔버스 전체(0..width, 0..height)를 덮는 투명 SVG. 역명마다 흰 테두리 + 진한 글자."""
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" preserveAspectRatio="none">',
             f'<g font-family="{escape(FONT_FAMILY, quote=True)}" font-weight="700" text-anchor="middle" '
             f'fill="{TEXT_COLOR}" stroke="#FFFFFF" stroke-linejoin="round" paint-order="stroke">']
    for name, x, top, size, _ in label_rows(lb):
        base = top + size * 0.92               # 알파벳 기준선 (글자 윗변 + 약 0.92em)
        parts.append(f'<text x="{x:.1f}" y="{base:.1f}" font-size="{size:.2f}" '
                     f'stroke-width="{size * HALO:.2f}">{escape(name)}</text>')
    parts.append("</g></svg>")
    return "".join(parts)


def label_image_uri(lb, width: int, height: int) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(label_svg(lb, width, height).encode("utf-8")).decode("ascii")


def layout_image(lb, width: int, height: int) -> dict:
    """fig.add_layout_image(**layout_image(...)) 용 인자. y 축이 아래로 증가(range 뒤집힘)하는 노선도 기준."""
    return dict(source=label_image_uri(lb, width, height), xref="x", yref="y",
                x=0, y=0, sizex=width, sizey=height, xanchor="left", yanchor="top",
                sizing="stretch", layer="above")
