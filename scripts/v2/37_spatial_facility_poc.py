"""
scripts/v2/37_spatial_facility_poc.py
=====================================
공모전 Spatial Facility PoC — 동묘앞역 2D 도면(SHP)·3D 모델(GLB) 실데이터 읽기·렌더링,
안전시설 위치정보가 개방되면 같은 도면에 겹치는 구조.

    pip install pyshp trimesh      (처음 한 번)
    python scripts/v2/37_spatial_facility_poc.py [--facilities <안전시설 CSV>]
    입력: data/raw/spatial_dongmyo/ (없으면 다운로드 폴더의 공모전 '2D3D 샘플' zip 을 찾아 푼다)
    출력: reports/v2/fig_spatial_2d.png · fig_spatial_3d.png · spatial_poc.md · spatial_poc.json

--facilities: 공모전 안전시설 명세 열(시설명 또는 시설유형, 위도, 경도, 층번호)을 가진 CSV.
              주지 않으면 시설 레이어는 비워 둔다 (시설 위치를 임의로 만들지 않는다).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from yeoyuro_v2.spatial_poc import (FLOORS, glb_summary, locate, place_facilities, read_floors,   # noqa: E402
                                    render_2d, render_3d)

SRC = ROOT / "data" / "raw" / "spatial_dongmyo"
OUT = ROOT / "reports" / "v2"


def korean_font() -> str | None:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR") if n in have), None)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--facilities", default=None, help="안전시설 CSV (위도, 경도, 층번호, 시설명)")
    a = ap.parse_args()
    try:
        import shapefile, trimesh  # noqa: F401
    except ImportError:
        print("[중단] pyshp·trimesh 가 필요합니다 → pip install pyshp trimesh")
        return 1
    src = locate(Path(a.src))
    font = korean_font()

    floors = read_floors(src)
    fac = pd.read_csv(a.facilities, encoding="utf-8-sig") if a.facilities else None
    placed = place_facilities(fac, floors)
    f2 = render_2d(floors, OUT / "fig_spatial_2d.png", placed, font=font)

    glbs = sorted(src.rglob("*.glb"))
    g = [glb_summary(p) for p in glbs]
    f3 = render_3d(glbs, OUT / "fig_spatial_3d.png", font=font) if glbs else None

    # 수직 정합: 2D 층 바닥 높이(z_min 최솟값) ≈ 3D 층 모델의 가장 낮은 Y (1 m 이내면 일치)
    vert = []
    for fl in FLOORS:
        if fl not in floors:
            continue
        z = float(np.nanmin([p["z_min"] for p in floors[fl]["polygons"]]))      # 2D 층의 가장 낮은 면 = 바닥
        m = next((x for x in g if fl[:2] in x["file"]), None)
        if m:
            lo, hi = m["up_axis_range_y"]
            vert.append({"floor": fl, "shp_floor_z": round(z, 2), "glb_floor_y": lo, "diff_m": round(abs(z - lo), 2),
                         "match": bool(abs(z - lo) <= 1.0)})
    res = {
        "source": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
        "floors": {k: {"crs": v["crs"], "polygons": len(v["polygons"]),
                       "etype": pd.Series([p["etype"] for p in v["polygons"]]).value_counts().to_dict(),
                       "bbox_m": [round(v["bbox"][2] - v["bbox"][0], 1), round(v["bbox"][3] - v["bbox"][1], 1)]}
                   for k, v in floors.items()},
        "glb": g, "vertical_check": vert,
        "facilities": {"input": a.facilities, "rows": 0 if fac is None else int(len(fac)),
                       "placed": int(placed.in_bbox.sum()) if len(placed) else 0},
        "figures": [str(f2.relative_to(ROOT))] + ([str(f3.relative_to(ROOT))] if f3 else []),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spatial_poc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(res)

    print("=" * 72)
    for k, v in res["floors"].items():
        print(f"{k}: {v['crs']} · 폴리곤 {v['polygons']}개 · 범위 {v['bbox_m'][0]}×{v['bbox_m'][1]} m · {v['etype']}")
    for x in g:
        print(f"{x['file']}: 메시 {x['meshes']} · 면 {x['faces']:,} · 높이(Y) {x['up_axis_range_y']}")
    for v in vert:
        print(f"수직 정합 {v['floor']}: 2D 바닥 {v['shp_floor_z']} m vs 3D 바닥 {v['glb_floor_y']} m (차이 {v['diff_m']} m) → {'일치' if v['match'] else '불일치'}")
    print(f"안전시설 레이어: 입력 {res['facilities']['rows']}행 · 도면 위 {res['facilities']['placed']}개"
          + ("" if a.facilities else " (입력 없음 → 비워 둠)"))
    print("그림:", ", ".join(res["figures"]))
    print("=" * 72)
    return 0


def write_md(res: dict) -> None:
    L = ["# Spatial Facility PoC — 동묘앞역 2D·3D (공모전 제공 샘플)", "",
         "- 범위: 실데이터 읽기·렌더링과, 안전시설 위치(위도·경도·층번호)가 개방되면 같은 층 도면에 겹치는 구조까지.",
         "- 하지 않은 것: 시설 위치 임의 생성, 설명자료 예시값 사용, 3D 길찾기·동선 최적화.", "",
         "| 층 | 좌표계 | 폴리곤 | 범위 (m) | 분류 |", "|---|---|---|---|---|"]
    for k, v in res["floors"].items():
        L.append(f"| {k} | {v['crs']} | {v['polygons']} | {v['bbox_m'][0]} × {v['bbox_m'][1]} | {v['etype']} |")
    L += ["", "| 3D 파일 | 메시 | 면 | 높이(Y) 범위 |", "|---|---|---|---|"]
    L += [f"| {x['file']} | {x['meshes']} | {x['faces']:,} | {x['up_axis_range_y']} |" for x in res["glb"]]
    L += ["", "수직 정합 (2D 바닥 높이 vs 3D 바닥 Y, 1 m 이내 = 일치): " + ", ".join(
              f"{v['floor']} {v['shp_floor_z']} / {v['glb_floor_y']} m → {'일치' if v['match'] else '불일치'}" for v in res["vertical_check"]),
          "", "확인 필요 (미확인): 3D 모델의 수평 좌표계와 2D(EPSG:3857)의 정합 규칙, 안전시설 X/Y/Z 좌표의 기준. "
          "위도·경도·층번호 경로는 좌표 변환만으로 2D 도면에 겹칠 수 있다.",
          f"", f"안전시설 레이어: 입력 {res['facilities']['rows']}행, 도면 위 {res['facilities']['placed']}개."]
    (OUT / "spatial_poc.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
