"""Spatial Facility PoC(37) — 합성 도면·합성 좌표만 쓴다 (실제 시설 위치 아님)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
shapefile = pytest.importorskip("shapefile")
pytestmark = pytest.mark.filterwarnings("ignore:Glyph")
from yeoyuro_v2.spatial_poc import (floor_name, lonlat_to_3857, place_facilities, read_floor,   # noqa: E402
                                    read_floors, render_2d)

PRJ = 'PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984"]]'
LON, LAT = 127.0, 37.57                      # 합성 테스트 좌표 (실제 시설 아님)


def _square(cx, cy, h):
    return [[(cx - h, cy - h), (cx - h, cy + h), (cx + h, cy + h), (cx + h, cy - h), (cx - h, cy - h)]]


def _write_floor(folder: Path, floor: str, z: float):
    cx, cy = lonlat_to_3857(LON, LAT)
    w = shapefile.Writer(str(folder / f"01_0159_test_{floor}"), shapeType=shapefile.POLYGON, encoding="cp949")
    for f in ("fid", "floor", "etype", "atype"):
        w.field(f, "C", 40)
    for f in ("area", "z_min", "z_max", "z_index"):
        w.field(f, "N", 18, 6)
    w.poly(_square(cx, cy, 50)); w.record("a", floor, "unclassified", "unclassified", 10000, z, z, z)
    w.poly(_square(cx + 10, cy, 3)); w.record("b", floor, "stair", "stair", 36, z, z + 5, z + 5)
    w.close()
    (folder / f"01_0159_test_{floor}.prj").write_text(PRJ)


def test_projection_known_values():
    x, y = lonlat_to_3857(180, 0)
    assert math.isclose(x, 20037508.34, rel_tol=1e-9) and abs(y) < 1e-6
    assert floor_name(-1) == "B1F" and floor_name(-3) == "B3F" and floor_name(1) is None and floor_name("x") is None


def test_read_floor(tmp_path):
    _write_floor(tmp_path, "B2F", 16.7)
    f = read_floor(tmp_path / "01_0159_test_B2F.shp")
    assert f["floor"] == "B2F" and f["crs"] == "EPSG:3857" and len(f["polygons"]) == 2
    assert {p["etype"] for p in f["polygons"]} == {"unclassified", "stair"}
    assert f["polygons"][0]["z_min"] == pytest.approx(16.7)


def test_place_facilities_floor_and_bbox(tmp_path):
    for fl, z in (("B1F", 34.3), ("B2F", 16.7)):
        _write_floor(tmp_path, fl, z)
    floors = read_floors(tmp_path)
    fac = pd.DataFrame([
        {"시설명": "합성A", "위도": LAT, "경도": LON, "층번호": -1},          # B1 도면 안
        {"시설명": "합성B", "위도": LAT, "경도": LON + 0.01, "층번호": -2},   # 약 880 m 밖 → 그리지 않음
        {"시설명": "합성C", "위도": LAT, "경도": LON, "층번호": -3},          # 도면 없는 층 → 제외
    ])
    p = place_facilities(fac, floors)
    assert list(p.name) == ["합성A", "합성B"]
    assert list(p.in_bbox) == [True, False] and list(p.floor) == ["B1F", "B2F"]
    assert place_facilities(None, floors).empty


def test_render_2d_without_facilities(tmp_path):
    _write_floor(tmp_path, "B1F", 34.3)
    out = render_2d(read_floors(tmp_path), tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 1000


def test_glb_summary_and_render(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from yeoyuro_v2.spatial_poc import glb_summary, render_3d
    box = trimesh.creation.box(extents=(10, 2, 4))
    p = tmp_path / "Test_B1.glb"
    box.export(p)
    s = glb_summary(p)
    assert s["faces"] == 12 and s["up_axis_range_y"] == [-1.0, 1.0]
    assert render_3d([p], tmp_path / "f3.png").exists()
