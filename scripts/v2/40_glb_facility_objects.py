"""
scripts/v2/40_glb_facility_objects.py
=====================================
공모전 제공 동묘앞역 3D 모델(GLB) 안의 시설 객체를 세고(개체 단위), 안전문 이름 규칙을 점검하고,
빠른하차정보와 공모전 설명자료 예시의 시설 번호 일치를 확인한다 (D-069).
3D 모델을 2D 도면(SHP)에 맞추는 변환을 찾고 검증한다 (D-070): 층별 면적 겹침(IoU), 경계 거리, 안전문 간격(m),
시설 객체가 같은 층 도면 공간 안에 들어가는 비율. 변환 값과 검증 수치만 저장한다(시설 좌표는 저장하지 않는다).

    python scripts/v2/40_glb_facility_objects.py
    입력: data/raw/spatial_dongmyo/*.glb (trimesh 필요), data/raw/fast_exit/fast_exit_all.csv
    출력: reports/v2/glb_objects.json  (개수·점검 결과만. 좌표는 저장하지 않는다)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import math                                                                                      # noqa: E402

import numpy as np                                                                               # noqa: E402

from yeoyuro_v2.glb_objects import (CLUSTER_GAP, _glb_xy, _shp_edge_points, apply_registration,   # noqa: E402
                                    fit_glb_to_shp, psd_layout_check, read_folder, read_instances)
from yeoyuro_v2.spatial_poc import R_EARTH, read_floors                                          # noqa: E402

SRC = ROOT / "data" / "raw" / "spatial_dongmyo"
FAST_EXIT = ROOT / "data" / "raw" / "fast_exit" / "fast_exit_all.csv"
OUT = ROOT / "reports" / "v2" / "glb_objects.json"

# 공모전 설명자료(「지하철 혼잡도 및 안전시설 데이터 활용 서비스 발굴 공모전」 설명자료) 2.2.1 엘리베이터 예시 행
SPEC_ELEVATOR_EXAMPLE = {"시설아이디": "1041999", "승강기일련번호": "0000-946", "역명": "동묘앞", "상세위치": "신설동 방면6-2"}


def _raster(polys_xy, box, res=0.5):
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = box
    im = Image.new("L", (int((x1 - x0) / res), int((y1 - y0) / res)), 0)
    dr = ImageDraw.Draw(im)
    for poly in polys_xy:
        dr.polygon([((x - x0) / res, (y1 - y) / res) for x, y in poly], fill=255)
    return np.array(im) > 0


def registration_check(inst: list[dict], psd: dict) -> dict:
    import trimesh
    from matplotlib.path import Path as MPath
    from scipy.spatial import cKDTree
    floors = read_floors(SRC)
    reg = fit_glb_to_shp(SRC, floors)
    s, R, t = reg["scale"], np.array(reg["R"]), np.array(reg["t"])
    yc = np.mean([f["bbox"][1] for f in floors.values()])
    lat = math.degrees(2 * math.atan(math.exp(yc / R_EARTH)) - math.pi / 2)
    k = 1 / math.cos(math.radians(lat))                    # 웹 메르카토르 1단위 = 실제 1/k m
    per_floor = {}
    for p in sorted(SRC.glob("*.glb")):
        fl = p.stem[-2:] + "F"
        if fl not in floors:
            continue
        bb = floors[fl]["bbox"]
        box = (bb[0] - 30, bb[1] - 30, bb[2] + 30, bb[3] + 30)
        A = _raster([r for poly in floors[fl]["polygons"] for r in poly["rings"]], box)
        sc = trimesh.load(str(p))
        tris = []
        for n in sc.graph.nodes_geometry:
            if not n.startswith(("b0", "flr", "str", "stl")):
                continue
            T, g = sc.graph[n]
            m = sc.geometry[g].copy()
            m.apply_transform(T)
            v = (s * (R @ (m.vertices[:, [0, 2]] * [1, -1]).T)).T + t
            tris += [v[f] for f in m.faces]
        B = _raster(tris, box)
        d = cKDTree(_shp_edge_points(floors[fl])).query((s * (R @ _glb_xy(p).T)).T + t)[0] / k
        per_floor[fl[:2]] = {"iou": round(float((A & B).sum() / (A | B).sum()), 3),
                             "boundary_dist_median_m": round(float(np.median(d)), 2),
                             "boundary_dist_p90_m": round(float(np.percentile(d, 90)), 2)}
    inside = {}
    for o in inst:
        x, y = apply_registration(reg, o["x"], o["z"])
        ok = any(MPath(r).contains_point((x, y)) for poly in floors[o["floor"] + "F"]["polygons"] for r in poly["rings"])
        a = inside.setdefault(o["kind"], [0, 0])
        a[0] += int(ok)
        a[1] += 1
    steps = [v["step"] for v in psd.values() if v.get("step")]
    # 안전문 면 ↔ 방면 점검 (D-071): 각 면의 10번 칸 → 1번 칸 방향(= 진행 방향, 1번 칸이 앞)과 진행 방향 기준 좌우
    psd_dir = {}
    doors = {o["psd"]: apply_registration(reg, o["x"], o["z"]) for o in inst if o.get("psd")}
    if doors:
        mid = {sd: np.mean([v for k2, v in doors.items() if k2[0] == sd], axis=0) for sd in {k2[0] for k2 in doors}}
        for sd in sorted(mid):
            c1 = np.mean([v for k2, v in doors.items() if k2[0] == sd and k2[1] == 1], axis=0)
            c10 = np.mean([v for k2, v in doors.items() if k2[0] == sd and k2[1] == 10], axis=0)
            dv = (c1 - c10) / np.linalg.norm(c1 - c10)
            other = [m for k2, m in mid.items() if k2 != sd]
            left = float((mid[sd] - other[0]) @ np.array([-dv[1], dv[0]])) if other else None
            psd_dir[str(sd)] = {"travel_bearing_deg": round(float((90 - np.degrees(np.arctan2(dv[1], dv[0]))) % 360), 1),
                                "on_left_of_travel": None if left is None else bool(left > 0)}
    return {**reg, "psd_direction": psd_dir, "lat_deg": round(lat, 5), "mercator_k": round(k, 4),
            "model_unit_m": round(s / k, 4), "per_floor": per_floor,
            "psd_pitch_m": round(float(np.mean(steps)) * s / k, 2) if steps else None,
            "objects_inside_floor_polygon": {kk: {"inside": a[0], "n": a[1]} for kk, a in inside.items()}}


def main() -> int:
    if not SRC.exists() or not list(SRC.glob("*.glb")):
        print("GLB 없음:", SRC)
        return 1
    nodes = read_folder(SRC)
    inst = read_instances(SRC)
    by = Counter((i["floor"], i["kind"]) for i in inst)
    psd = psd_layout_check(nodes)
    res = {
        "source": sorted(p.name for p in SRC.glob("*.glb")),
        "cluster_gap": CLUSTER_GAP,
        "instances": [{"floor": f, "kind": k, "n": n} for (f, k), n in sorted(by.items())],
        "instances_total_by_kind": dict(Counter(i["kind"] for i in inst)),
        "psd_layout": {str(k): v for k, v in psd.items()},
        # 두 면의 1번 칸이 서로 반대쪽 끝 → 진행 방향 기준 칸 번호(두 선로가 반대 방향)와 모순 없음
        "psd_car1_opposite_ends": (len(psd) == 2 and len({v["car1_end"] for v in psd.values()}) == 2
                                   and all(v["monotonic"] for v in psd.values())),
    }
    if FAST_EXIT.exists():
        fe = pd.read_csv(FAST_EXIT, dtype=str, encoding="utf-8-sig")
        hit = fe[(fe.facNo == SPEC_ELEVATOR_EXAMPLE["시설아이디"]) & (fe.elvtrNo == SPEC_ELEVATOR_EXAMPLE["승강기일련번호"])]
        res["spec_example_match"] = {
            "spec": SPEC_ELEVATOR_EXAMPLE,
            "fast_exit_rows": hit[["stnNm", "lineNm", "qckgffVhclDoorNo", "facPstnNm", "facNo", "elvtrNo"]].to_dict("records"),
            "match": bool(len(hit)) and bool((hit.stnNm == SPEC_ELEVATOR_EXAMPLE["역명"]).all())
                     and bool(hit.facPstnNm.str.replace(" ", "").eq(SPEC_ELEVATOR_EXAMPLE["상세위치"].replace(" ", "")).any()),
        }
    shp = sorted(SRC.glob("*.shp"))
    if shp:
        res["registration"] = registration_check(inst, psd)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("instances_total_by_kind", "psd_car1_opposite_ends")}, ensure_ascii=False))
    if "registration" in res:
        r = res["registration"]
        print("3D→2D 정합: 배율 %.4f · 회전 %.1f° · 모델 1단위 = %.3f m · 안전문 간격 %.2f m"
              % (r["scale"], r["angle_deg"], r["model_unit_m"], r["psd_pitch_m"]))
        print("  층별:", r["per_floor"])
        print("  도면 공간 안에 들어간 시설:", r["objects_inside_floor_polygon"])
    if "spec_example_match" in res:
        print("설명자료 예시 = 빠른하차정보:", res["spec_example_match"]["match"])
    print("저장:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
