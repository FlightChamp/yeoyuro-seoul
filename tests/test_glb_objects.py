"""3D 모델(GLB) 속 시설 객체 읽기 (D-069). 원본 GLB 가 없는 환경(공개 저장소)에서는 건너뛴다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yeoyuro_v2 import glb_objects as g

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "spatial_dongmyo"
need_glb = pytest.mark.skipif(not list(SRC.glob("*.glb")) if SRC.exists() else True, reason="GLB 원본 없음")


def test_quaternion_and_cluster_helpers():
    m = g._node_matrix({"translation": [1, 2, 3], "rotation": [0, 0, 0, 1], "scale": [2, 2, 2]})
    assert np.allclose(m @ [1, 1, 1, 1], [3, 4, 5, 1])
    r = g._quat_to_mat([0, 0.7071068, 0, 0.7071068])          # Y 축 90도
    assert np.allclose(r @ [1, 0, 0], [0, 0, -1], atol=1e-6)
    lab = g._cluster(np.array([[0, 0], [1, 0], [10, 0], [10.5, 0]]), 2.0)
    assert len(set(lab)) == 2 and lab[0] == lab[1] and lab[2] == lab[3]


@need_glb
def test_node_level_objects_and_psd_naming():
    objs = g.read_folder(SRC)
    s = g.summary(objs)
    assert s["승강장 안전문"] == 80
    assert s["구호용품보관함"] == 1 and s["SOS 비상통화장치"] == 1        # 노드 단위(개체는 합쳐져 있음)
    gh = next(o for o in objs if o.kind == "구호용품보관함")
    assert any("구호용품" in m for m in gh.materials)                     # 재질 이름 근거
    lay = g.psd_layout_check(objs)
    assert set(lay) == {0, 1}
    assert all(v["n"] == 40 and v["cars"] == 10 and v["doors_per_car"] == 4 and v["monotonic"] for v in lay.values())
    assert {v["car1_end"] for v in lay.values()} == {"high", "low"}      # 두 면의 1번 칸이 반대쪽 끝


@need_glb
def test_instances_match_trimesh_world_frame():
    trimesh = pytest.importorskip("trimesh")
    inst = g.read_instances(SRC)
    by = {}
    for i in inst:
        by[i["kind"]] = by.get(i["kind"], 0) + 1
    assert by["구호용품보관함"] == 5 and by["SOS 비상통화장치"] == 6 and by["승강장 안전문"] == 80
    # 모든 개체가 해당 층 메시 경계 안에 있다 (임의 좌표가 아님)
    for p in sorted(SRC.glob("*.glb")):
        fl = p.stem[-2:]
        b = trimesh.load(str(p), force="mesh").bounds
        for i in inst:
            if i["floor"] == fl:
                assert np.all(np.array([i["x"], i["y"], i["z"]]) >= b[0] - 1e-6)
                assert np.all(np.array([i["x"], i["y"], i["z"]]) <= b[1] + 1e-6)
    # JSON 계산(노드 경계)과 trimesh 계산이 같은 좌표계 — 안전문 하나로 확인
    node = next(o for o in g.read_folder(SRC) if o.node == "psd_00_0101")
    one = next(i for i in inst if i["node"] == "psd_00_0101")
    assert np.allclose(node.center, [one["x"], one["y"], one["z"]], atol=0.5)


def test_report_json_if_present():
    rep = ROOT / "reports" / "v2" / "glb_objects.json"
    if not rep.exists():
        pytest.skip("glb_objects.json 없음")
    d = json.loads(rep.read_text(encoding="utf-8"))
    assert d["instances_total_by_kind"]["승강장 안전문"] == 80
    assert d["psd_car1_opposite_ends"] is True
    assert d["spec_example_match"]["match"] is True
    assert "x" not in json.dumps(d["instances"])                       # 좌표는 저장하지 않는다


def test_registration_helpers():
    reg = {"scale": 2.0, "R": [[0, -1], [1, 0]], "t": [10, 20]}          # 90도 회전, 배율 2
    assert g.apply_registration(reg, 1.0, 0.0) == (10.0, 22.0)           # (x, -z) = (1, 0) → R·(1,0) = (0,1)
    s, R, t = g._umeyama(np.array([[0, 0], [1, 0], [0, 1]], float), np.array([[5, 5], [5, 7], [3, 5]], float))
    assert abs(s - 2) < 1e-9 and np.allclose(R, [[0, -1], [1, 0]]) and np.allclose(t, [5, 5])


def test_inside_floor_point_in_polygon():
    floor = {"polygons": [{"etype": "x", "rings": [[(0, 0), (10, 0), (10, 10), (0, 10)]]},
                          {"etype": "y", "rings": [[(20, 0), (30, 0), (25, 8)]]}]}
    assert g.inside_floor(floor, 5, 5) and g.inside_floor(floor, 25, 2)
    assert not g.inside_floor(floor, 15, 5) and not g.inside_floor(floor, 5, 11)


def test_registration_report_quality():
    rep = ROOT / "reports" / "v2" / "glb_objects.json"
    if not rep.exists():
        pytest.skip("glb_objects.json 없음")
    r = json.loads(rep.read_text(encoding="utf-8")).get("registration")
    if not r:
        pytest.skip("정합 결과 없음")
    assert r["per_floor"]["B2"]["iou"] >= 0.8 and min(v["iou"] for v in r["per_floor"].values()) >= 0.65
    assert 4.5 <= r["psd_pitch_m"] <= 5.5                                 # 열차 1칸 약 20 m · 문 4개
    ins = r["objects_inside_floor_polygon"]
    assert ins["승강장 안전문"]["inside"] == ins["승강장 안전문"]["n"]


def test_elevator_doors_mapping_unit():
    inst = [{"psd": (0, 6, 2), "node": "psd_00_0602", "floor": "B2", "x": 1.0, "y": 2.0, "z": 3.0},
            {"psd": (1, 1, 2), "node": "psd_01_0102", "floor": "B2", "x": 4.0, "y": 5.0, "z": 6.0}]
    rows = [{"line": 1, "toward": "신설동", "car": 6, "door": 2, "elevator_no": "0000-946"},
            {"line": 1, "toward": "동대문", "car": 1, "door": 2, "elevator_no": "0000-944"},
            {"line": 6, "toward": "신당", "car": 3, "door": 3, "elevator_no": "0000-962"}]      # 6호선: 안전문 없음
    out = g.elevator_doors(inst, rows, "동묘앞")
    assert [(o["toward"], o["psd_node"]) for o in out] == [("신설동", "psd_00_0602"), ("동대문", "psd_01_0102")]


def test_psd_direction_report():
    rep = ROOT / "reports" / "v2" / "glb_objects.json"
    if not rep.exists():
        pytest.skip("glb_objects.json 없음")
    r = (json.loads(rep.read_text(encoding="utf-8")).get("registration") or {}).get("psd_direction")
    if not r:
        pytest.skip("방향 점검 없음")
    # 0면 = 신설동(동북동) 방면, 1면 = 동대문(서남서) 방면, 두 면 모두 진행 방향 왼쪽(1호선 좌측통행)
    assert 30 <= r["0"]["travel_bearing_deg"] <= 110 and 210 <= r["1"]["travel_bearing_deg"] <= 290
    assert r["0"]["on_left_of_travel"] and r["1"]["on_left_of_travel"]
    assert g.PSD_SIDE_BY_TOWARD[("1", "동묘앞", "신설동")] == 0 and g.PSD_SIDE_BY_TOWARD[("1", "동묘앞", "동대문")] == 1
