"""
yeoyuro_v2/spatial_poc.py
=========================
공모전 Spatial Facility PoC — 공모전 제공 역사 도면(동묘앞역 2D SHP · 3D GLB)을 실제로 읽어 렌더링하고,
안전시설 위치정보(위도·경도·층번호)가 개방되면 같은 도면 위에 겹쳐 그리는 구조를 보인다.

원칙
- 안전시설 원본을 받지 않았으므로 시설 위치를 임의로 만들지 않는다. 설명자료 예시값도 실제 관측값처럼 쓰지 않는다.
  시설 레이어는 '입력이 있을 때만' 그린다. (테스트는 합성 좌표로만 한다)
- 3D 는 읽기·렌더 가능 여부 확인과 발표용 그림 1장까지만. 3D 길찾기·동선 최적화는 하지 않는다.

좌표
- SHP: EPSG:3857 (WGS 84 Web Mercator, .prj 확인). 속성 z_min/z_max = 층 높이(m).
- 안전시설 명세: 위도·경도(WGS 84) + 층번호(-1=B1 …) + X/Y/Z(모델 좌표로 보이나 기준 미확인).
  → 위도·경도를 EPSG:3857 로 바꿔 같은 층 도면에 겹친다. X/Y/Z 는 기준이 확인되면 3D 에 쓴다.
- GLB: glTF 관례(Y 위쪽). Y 범위가 SHP 층 높이와 맞는지 확인한다. 수평 좌표는 지역 좌표계이며 SHP 와의 정합 규칙은 미확인.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FLOORS = ("B1F", "B2F", "B3F")
FLOOR_NO = {"B1F": -1, "B2F": -2, "B3F": -3}
R_EARTH = 6378137.0


# ---------------------------------------------------------------- 입력 찾기
def locate(folder: Path) -> Path:
    """폴더에 SHP/GLB 가 있으면 그대로, 없으면 다운로드 폴더의 공모전 2D·3D 샘플 zip 을 찾아 풀어 둔다."""
    folder = Path(folder)
    if folder.is_dir() and any(folder.rglob("*.shp")):
        return folder
    dl = Path.home() / "Downloads"
    for zp in sorted(dl.glob("*.zip"), key=lambda q: q.stat().st_mtime, reverse=True) if dl.is_dir() else []:
        try:
            z = zipfile.ZipFile(zp)
            names = z.namelist()
        except (zipfile.BadZipFile, OSError):
            continue
        if not any(n.endswith("Dongmyo_B1.glb") for n in names):
            continue
        for i in z.infolist():
            if i.is_dir():
                continue
            name = i.filename if i.flag_bits & 0x800 else i.filename.encode("cp437").decode("cp949", "replace")
            out = folder / Path(name).name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(i))
        print(f"[샘플] {zp.name} 을(를) {folder} 에 풀었습니다")
        return folder
    raise FileNotFoundError(f"동묘앞 2D·3D 샘플을 찾지 못했습니다. 공모전 2D3D 샘플 zip 을 다운로드 폴더나 {folder} 에 넣어 주세요")


# ---------------------------------------------------------------- 2D
def read_floor(shp_path: Path) -> dict:
    """SHP 한 층 → {'floor','crs','polygons':[{'rings':[np.ndarray], 'etype', 'z_min', 'z_max'}], 'bbox'}"""
    import shapefile                                   # pyshp (순수 파이썬, GDAL 불필요)
    shp_path = Path(shp_path)
    r = shapefile.Reader(str(shp_path.with_suffix("")), encoding="cp949")
    names = [f[0] for f in r.fields[1:]]
    polys = []
    for sr in r.iterShapeRecords():
        pts = np.asarray(sr.shape.points, dtype=float)
        if len(pts) < 3:
            continue
        parts = list(sr.shape.parts) + [len(pts)]
        rings = [pts[a:b] for a, b in zip(parts[:-1], parts[1:]) if b - a >= 3]
        rec = dict(zip(names, sr.record))
        polys.append({"rings": rings, "etype": str(rec.get("etype", "")), "z_min": float(rec.get("z_min") or np.nan),
                      "z_max": float(rec.get("z_max") or np.nan), "area": float(rec.get("area") or np.nan)})
    prj = shp_path.with_suffix(".prj")
    crs = "EPSG:3857" if prj.exists() and "Web_Mercator" in prj.read_text(errors="replace") else "unknown"
    m = re.search(r"(B\dF)", shp_path.stem)
    return {"floor": m.group(1) if m else shp_path.stem, "crs": crs, "polygons": polys, "bbox": list(r.bbox)}


def read_floors(folder: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(folder).rglob("*.shp")):
        f = read_floor(p)
        out[f["floor"]] = f
    return out


def lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    x = R_EARTH * math.radians(lon)
    y = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def floor_name(no) -> str | None:
    try:
        n = int(no)
    except (TypeError, ValueError):
        return None
    return f"B{-n}F" if n < 0 else None


def place_facilities(fac: pd.DataFrame | None, floors: dict[str, dict], margin: float = 50.0) -> pd.DataFrame:
    """안전시설 표(명세 열: 위도, 경도, 층번호, 시설명/시설유형) → 도면 좌표·층 배정.
    도면 범위(+margin m) 밖이면 in_bbox=False 로 남기고 그리지 않는다."""
    cols = ["name", "floor", "x", "y", "in_bbox"]
    if fac is None or len(fac) == 0:
        return pd.DataFrame(columns=cols)
    rows = []
    for r in fac.itertuples(index=False):
        d = r._asdict()
        lat, lon = d.get("위도"), d.get("경도")
        fl = floor_name(d.get("층번호"))
        if lat is None or lon is None or fl is None or fl not in floors:
            continue
        x, y = lonlat_to_3857(float(lon), float(lat))
        b = floors[fl]["bbox"]
        inside = (b[0] - margin <= x <= b[2] + margin) and (b[1] - margin <= y <= b[3] + margin)
        rows.append({"name": d.get("시설명") or d.get("시설유형") or "", "floor": fl, "x": x, "y": y, "in_bbox": inside})
    return pd.DataFrame(rows, columns=cols)


ETYPE_STYLE = {"stair": ("#C2410C", "계단"), "column": ("#6B7280", "기둥"), "inner_wall": ("#374151", "내벽"),
               "unclassified": ("#9CA3AF", "공간 (미분류)")}


def render_2d(floors: dict[str, dict], out: Path, facilities: pd.DataFrame | None = None, font: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, PathPatch
    from matplotlib.path import Path as MPath
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    xs = [b for f in floors.values() for b in (f["bbox"][0], f["bbox"][2])]
    ys = [b for f in floors.values() for b in (f["bbox"][1], f["bbox"][3])]
    x0, y0 = min(xs), min(ys)
    order = [f for f in FLOORS if f in floors]
    fig, axes = plt.subplots(1, len(order), figsize=(5.2 * len(order), 4.6))
    axes = np.atleast_1d(axes)
    for ax, fl in zip(axes, order):
        f = floors[fl]
        for p in sorted(f["polygons"], key=lambda q: -(q["area"] if np.isfinite(q["area"]) else 0)):
            verts, codes = [], []
            for ring in p["rings"]:
                ring = ring - [x0, y0]
                verts += list(ring) + [ring[0]]
                codes += [MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 1) + [MPath.CLOSEPOLY]
            color = ETYPE_STYLE.get(p["etype"], ETYPE_STYLE["unclassified"])[0]
            hi = p["etype"] in ("stair", "column", "inner_wall")
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=0.85 if hi else 0.12,
                                   edgecolor=color if hi else "#6B7280", linewidth=0.6 if hi else 0.4))
        if facilities is not None and len(facilities):
            q = facilities[(facilities.floor == fl) & facilities.in_bbox]
            ax.scatter(q.x - x0, q.y - y0, s=46, marker="^", color="#1D4ED8", edgecolor="white", linewidth=0.8, zorder=5)
        z = np.nanmedian([p["z_min"] for p in f["polygons"]])
        ax.set_title(f"{fl}  (도면 높이 약 {z:.1f} m · 폴리곤 {len(f['polygons'])}개)", fontsize=11)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xlabel("동쪽 (m)", fontsize=8)
        ax.set_ylabel("북쪽 (m)", fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [Patch(facecolor=c, alpha=0.85 if k != "unclassified" else 0.2, edgecolor=c, label=l)
               for k, (c, l) in ETYPE_STYLE.items()]
    has_fac = facilities is not None and len(facilities) and bool(facilities.in_bbox.any())
    handles.append(Patch(facecolor="white", edgecolor="#1D4ED8", linestyle="--",
                         label="안전시설 (위도·경도·층번호 개방 시 표시)" if not has_fac else "안전시설"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.5, frameon=False)
    fig.suptitle("동묘앞역 층별 역사 도면 — 공모전 제공 2D SHP 실데이터 렌더링 (안전시설 레이어는 데이터 개방 후 겹침)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 3D
def glb_summary(path: Path) -> dict:
    import trimesh
    s = trimesh.load(str(path), force="scene")
    geos = list(s.geometry.values())
    b = s.bounds
    return {"file": Path(path).name, "meshes": len(geos), "vertices": int(sum(len(g.vertices) for g in geos)),
            "faces": int(sum(len(g.faces) for g in geos if hasattr(g, "faces"))),
            "bounds_min": [round(float(v), 2) for v in b[0]], "bounds_max": [round(float(v), 2) for v in b[1]],
            "up_axis_range_y": [round(float(b[0][1]), 2), round(float(b[1][1]), 2)]}


def render_3d(paths: list[Path], out: Path, max_faces: int = 60000, font: str | None = None) -> Path:
    """GLB 층들을 한 장의 비스듬한 그림으로 (Y 위쪽 → 화면 위쪽). 면 수가 많으면 균등 표본으로 줄인다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if font:
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
    colors = {"B1": "#60A5FA", "B2": "#F59E0B", "B3": "#10B981"}
    fig = plt.figure(figsize=(10, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    rng = np.random.default_rng(0)
    for p in paths:
        m = trimesh.load(str(p), force="mesh")
        tri = m.vertices[m.faces]
        if len(tri) > max_faces:
            tri = tri[rng.choice(len(tri), max_faces, replace=False)]
        tri = tri[:, :, [0, 2, 1]] * [1, -1, 1]       # glTF (x, y-up, z) → (x, -z, y) : 위쪽 = 높이
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        shade = 0.55 + 0.45 * np.abs(n @ np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]))
        key = re.search(r"B\d", Path(p).stem)
        base = np.array(matplotlib.colors.to_rgb(colors.get(key.group(0) if key else "B1", "#9CA3AF")))
        fc = np.clip(base[None, :] * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0, alpha=0.9))
        allv.append(tri.reshape(-1, 3))
    v = np.vstack(allv)
    lo, hi = v.min(0), v.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], (hi[2] - lo[2]) * 3))
    ax.view_init(elev=28, azim=-58)
    ax.set_axis_off()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="upper left", fontsize=9, frameon=False)
    ax.set_title("동묘앞역 3D 역사 모델 (공모전 제공 GLB, 높이 3배 과장) — 읽기·렌더 확인용", fontsize=11)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _trim_white(out)
    return out


def _trim_white(path: Path, pad: int = 24) -> None:
    """3D 그림의 빈 여백을 잘라낸다 (발표 자료용)."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        l, t, r, b = bbox
        im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height))).save(path)
