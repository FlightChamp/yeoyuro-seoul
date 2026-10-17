"""
yeoyuro_v2/glb_objects.py
=========================
공모전 제공 역사 3D 모델(GLB) 안에 이름이 붙은 '시설 객체'를 읽는다 (D-069).

동묘앞 GLB 는 층 구조 외에, 일부 시설을 별도 노드로 모델링해 두었다. 한 노드에 같은 종류 여러 개가
합쳐져 있으므로(예: b02_gh 하나에 보관함 5개), 메시를 연결 요소로 나눈 뒤 가까운 요소(4 단위 이내)를 한 개체로 묶는다.
  - b02_gh  : 재질 이름 '구호용품'          → 구호용품보관함 (재질 이름 근거)
  - b02_sh  : 재질 이름 'FireT'             → 소화 설비 (추정)
  - b03_sh  : b02_sh 와 같은 이름 규칙·재질  → 소화 설비 (추정)
  - b02_sos : 노드 이름 'sos'               → SOS 비상통화장치 (이름 근거)
  - psd_SS_CCDD (80개)                      → 승강장 안전문(PSD). 이름 규칙상 면 SS · 칸 CC · 문 DD 로 보이나
                                              공식 정의는 미확인. 두 면의 1번 칸이 서로 반대쪽 끝에 있어
                                              진행 방향 기준 칸 번호와 모순되지 않는다.

위치는 모델 원본의 노드 변환(이동·회전·크기) × 메시 경계상자에서 계산한다. 임의 좌표를 만들지 않는다.
좌표계: glTF 원본(Y-up). 앱의 3D 보기는 (x, -z, y) 로 그리므로 to_plot() 으로 같은 변환을 한다.
read_objects(): GLB JSON 만 읽는 노드 단위 결과 (trimesh 불필요).
read_instances(): trimesh 로 노드 메시를 개체 단위로 나눈 결과 (3D 보기와 같은 의존성).
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# (판정 함수, 시설 종류, 근거 수준, 근거 설명)
_RULES = [
    (lambda n, mats: n.startswith("psd_"), "승강장 안전문", "이름",
     "노드 이름 psd(Platform Screen Door) · 이름 규칙 psd_면_칸문"),
    (lambda n, mats: any("구호용품" in m for m in mats), "구호용품보관함", "재질 이름",
     "재질 이름 '구호용품'"),
    (lambda n, mats: n.endswith("_sos"), "SOS 비상통화장치", "이름",
     "노드 이름 'sos'"),
    (lambda n, mats: re.fullmatch(r"b0\d_sh", n) is not None, "소화 설비(추정)", "추정",
     "노드 이름 'sh' · B2 객체 재질 이름 'FireT'"),
]
PSD_RE = re.compile(r"psd_(\d\d)_(\d\d)(\d\d)$")


@dataclass
class GlbObject:
    floor: str                    # 'B1' / 'B2' / 'B3'
    node: str                     # GLB 노드 이름
    kind: str                     # 시설 종류(표시용)
    evidence_level: str           # '재질 이름' / '이름' / '추정'
    evidence: str                 # 근거 설명
    center: tuple[float, float, float]           # glTF 원본 좌표 (Y-up)
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]
    materials: list[str] = field(default_factory=list)
    psd: tuple[int, int, int] | None = None      # (면, 칸, 문) — 이름 규칙 해석, 공식 정의 미확인

    def to_plot(self) -> tuple[float, float, float]:
        """앱 3D 보기 좌표 (x, -z, y)."""
        x, y, z = self.center
        return x, -z, y


def _read_glb_json(path: Path) -> dict:
    b = path.read_bytes()
    if b[:4] != b"glTF":
        raise ValueError("GLB 아님: %s" % path.name)
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n])


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(node: dict) -> np.ndarray:
    m = np.eye(4)
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T     # glTF 는 열 우선
    t = np.array(node.get("translation", [0, 0, 0]), dtype=float)
    r = _quat_to_mat(node.get("rotation", [0, 0, 0, 1]))
    s = np.diag(node.get("scale", [1, 1, 1]))
    m[:3, :3] = r @ s
    m[:3, 3] = t
    return m


def _mesh_bounds(j: dict, mesh_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    lo, hi = None, None
    for p in j["meshes"][mesh_idx]["primitives"]:
        acc = j["accessors"][p["attributes"]["POSITION"]]
        if "min" not in acc or "max" not in acc:
            continue
        a, b = np.array(acc["min"], float), np.array(acc["max"], float)
        lo = a if lo is None else np.minimum(lo, a)
        hi = b if hi is None else np.maximum(hi, b)
    return None if lo is None else (lo, hi)


def _world_nodes(j: dict):
    """(노드, 월드 변환) — 장면 그래프를 따라 부모 변환을 곱한다."""
    nodes = j.get("nodes", [])
    roots = j["scenes"][j.get("scene", 0)]["nodes"] if j.get("scenes") else range(len(nodes))
    stack = [(i, np.eye(4)) for i in roots]
    while stack:
        i, parent = stack.pop()
        w = parent @ _node_matrix(nodes[i])
        yield nodes[i], w
        stack.extend((c, w) for c in nodes[i].get("children", []))


def read_objects(path: Path) -> list[GlbObject]:
    """GLB 한 개에서 시설 객체를 찾는다. 규칙에 맞지 않는 노드(바닥·벽·천장 등)는 건너뛴다."""
    path = Path(path)
    j = _read_glb_json(path)
    floor = (re.search(r"B\d", path.stem) or re.search(r".*", path.stem)).group(0)
    out = []
    for node, w in _world_nodes(j):
        if "mesh" not in node:
            continue
        name = str(node.get("name", ""))
        mats = sorted({j["materials"][p["material"]].get("name", "")
                       for p in j["meshes"][node["mesh"]]["primitives"] if "material" in p})
        rule = next((r for r in _RULES if r[0](name, mats)), None)
        if rule is None:
            continue
        bounds = _mesh_bounds(j, node["mesh"])
        if bounds is None:
            continue
        lo, hi = bounds
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        wc = (w @ corners.T).T[:, :3]
        blo, bhi = wc.min(axis=0), wc.max(axis=0)
        m = PSD_RE.match(name)
        out.append(GlbObject(floor=floor, node=name, kind=rule[1], evidence_level=rule[2], evidence=rule[3],
                             center=tuple(float(v) for v in (blo + bhi) / 2),
                             bbox=(tuple(float(v) for v in blo), tuple(float(v) for v in bhi)),
                             materials=mats, psd=tuple(int(g) for g in m.groups()) if m else None))
    return out


def read_folder(folder: Path) -> list[GlbObject]:
    objs = []
    for p in sorted(Path(folder).glob("*.glb")):
        objs += read_objects(p)
    return objs


def summary(objs: list[GlbObject]) -> dict[str, int]:
    out: dict[str, int] = {}
    for o in objs:
        out[o.kind] = out.get(o.kind, 0) + 1
    return out


def psd_layout_check(objs: list[GlbObject]) -> dict:
    """안전문 이름 규칙(면·칸·문) 점검: 면별 칸·문 수, 1번 칸이 어느 끝에 있는지, 문 간격."""
    res = {}
    for side in sorted({o.psd[0] for o in objs if o.psd}):
        ds = sorted((o for o in objs if o.psd and o.psd[0] == side), key=lambda o: (o.psd[1], o.psd[2]))
        xs = np.array([o.center[0] for o in ds])
        car1 = float(np.mean([o.center[0] for o in ds if o.psd[1] == 1]))
        res[side] = {"n": len(ds), "cars": len({o.psd[1] for o in ds}), "doors_per_car": len({o.psd[2] for o in ds}),
                     "car1_x": car1, "car1_end": "high" if car1 > float(np.mean(xs)) else "low",
                     "monotonic": bool(np.all(np.diff(xs) > 0) or np.all(np.diff(xs) < 0)),
                     "step": float(np.median(np.abs(np.diff(xs)))) if len(xs) > 1 else None}
    return res


CLUSTER_GAP = 4.0     # 같은 개체로 묶는 연결 요소 간 최대 거리 (모델 수평 단위)


def _cluster(points: np.ndarray, gap: float) -> np.ndarray:
    """단일 연결 군집 (scipy 없이). points: (n, 2) → 군집 번호 (n,)"""
    n = len(points)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        d = np.hypot(*(points[i + 1:] - points[i]).T)
        for k in np.nonzero(d <= gap)[0]:
            a, b = find(i), find(i + 1 + k)
            if a != b:
                parent[b] = a
    return np.array([find(i) for i in range(n)])


def read_instances(folder: Path, gap: float = CLUSTER_GAP) -> list[dict]:
    """시설 객체를 개체 단위로. [{floor, kind, evidence_level, evidence, node, x, y, z(glTF), n_parts, psd}]"""
    import trimesh
    nodes = {(o.floor, o.node): o for o in read_folder(folder)}
    out = []
    for p in sorted(Path(folder).glob("*.glb")):
        floor = (re.search(r"B\d", p.stem) or re.search(r".*", p.stem)).group(0)
        sc = trimesh.load(str(p))
        if not isinstance(sc, trimesh.Scene):
            continue
        groups: dict[str, list] = {}
        for gnode in sc.graph.nodes_geometry:
            T, gname = sc.graph[gnode]
            # trimesh 는 여러 primitive 를 '노드이름_해시' 로 나눈다 → 원래 노드 이름으로 되돌린다
            base = gnode if (floor, gnode) in nodes else gnode.rsplit("_", 1)[0]
            if (floor, base) not in nodes:
                continue
            m = sc.geometry[gname].copy()
            m.apply_transform(T)
            groups.setdefault(base, []).append(m)
        for base, parts in groups.items():
            o = nodes[(floor, base)]
            if o.psd:                                  # 안전문은 노드 하나 = 문 하나
                lo = np.min([m.bounds[0] for m in parts], axis=0)
                hi = np.max([m.bounds[1] for m in parts], axis=0)
                c = (lo + hi) / 2
                out.append(dict(floor=floor, kind=o.kind, evidence_level=o.evidence_level, evidence=o.evidence,
                                node=base, x=float(c[0]), y=float(c[1]), z=float(c[2]), n_parts=1, psd=o.psd))
                continue
            comps = [c for m in parts for c in m.split(only_watertight=False)]
            if not comps:
                continue
            cen = np.array([c.bounds.mean(axis=0) for c in comps])
            lab = _cluster(cen[:, [0, 2]], gap)
            for k in np.unique(lab):
                sel = [comps[i] for i in np.nonzero(lab == k)[0]]
                lo = np.min([c.bounds[0] for c in sel], axis=0)
                hi = np.max([c.bounds[1] for c in sel], axis=0)
                c = (lo + hi) / 2
                out.append(dict(floor=floor, kind=o.kind, evidence_level=o.evidence_level, evidence=o.evidence,
                                node=base, x=float(c[0]), y=float(c[1]), z=float(c[2]), n_parts=len(sel), psd=None))
    return out


# ---------------------------------------------------------------- 3D 모델 → 2D 도면 정합 (D-070)
# 공식 정합 규칙은 없다. 두 원본의 모양만으로 닮음변환(회전·균일 배율·이동, 뒤집기 없음)을 찾는다:
#   3D 수평 좌표 (x, -z) → SHP 좌표(EPSG:3857). 세 층을 함께 맞춘 변환 하나를 쓴다.
# 검증은 scripts/v2/40 에서: 층별 면적 겹침(IoU), 경계 거리, 안전문 간격(m) — 결과는 reports/v2/glb_objects.json.

def _glb_xy(path: Path, prefixes=("b0", "flr", "str", "stl")) -> np.ndarray:
    import trimesh
    sc = trimesh.load(str(path))
    pts = []
    for n in sc.graph.nodes_geometry:
        if not n.startswith(prefixes):
            continue
        T, g = sc.graph[n]
        m = sc.geometry[g].copy()
        m.apply_transform(T)
        pts.append(m.vertices[:, [0, 2]] * [1, -1])
    return np.vstack(pts) if pts else np.empty((0, 2))


def _shp_edge_points(floor: dict, step: float = 0.5) -> np.ndarray:
    out = []
    for p in floor["polygons"]:
        for r in p["rings"]:
            for a, b in zip(r[:-1], r[1:]):
                n = max(1, int(np.hypot(*(b - a)) / step))
                t = np.linspace(0, 1, n, endpoint=False)[:, None]
                out.append(a + t * (b - a))
    return np.vstack(out)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """닮음변환 (뒤집기 없음): dst ≈ s·R·src + t"""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, 1.0 if np.linalg.det(U @ Vt) > 0 else -1.0])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((a ** 2).sum() / len(src))
    return float(s), R, md - s * R @ ms


def fit_glb_to_shp(folder: Path, floors: dict, n_glb: int = 8000, n_shp: int = 20000, iters: int = 80) -> dict:
    """세 층을 함께 맞춘 닮음변환. floors = spatial_poc.read_floors(...) 결과 {'B1F': {...}, ...}."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    G, S = {}, {}
    for p in sorted(Path(folder).glob("*.glb")):
        fl = re.search(r"B\d", p.stem).group(0) + "F"
        if fl not in floors:
            continue
        g = _glb_xy(p)
        s = _shp_edge_points(floors[fl])
        G[fl] = g[rng.choice(len(g), min(len(g), n_glb), replace=False)]
        S[fl] = s[rng.choice(len(s), min(len(s), n_shp), replace=False)]
    trees = {k: cKDTree(v) for k, v in S.items()}

    def icp(params, keys):
        s, R, t = params
        for _ in range(iters):
            src, dst = [], []
            for k in keys:
                y = (s * (R @ G[k].T)).T + t
                d, idx = trees[k].query(y)
                keep = d < np.percentile(d, 80)
                src.append(G[k][keep])
                dst.append(S[k][idx[keep]])
            s, R, t = _umeyama(np.vstack(src), np.vstack(dst))
        d = np.concatenate([trees[k].query((s * (R @ G[k].T)).T + t)[0] for k in keys])
        return (s, R, t), float(np.median(d))

    # 시작값: 가장 큰 층(승강장 층)의 주성분 방향을 맞춘 네 가지 후보 중 가장 잘 맞는 것
    base = max(G, key=lambda k: len(S[k]))
    cg, cs = G[base].mean(0), S[base].mean(0)
    wg, vg = np.linalg.eigh(np.cov((G[base] - cg).T))
    ws, vs = np.linalg.eigh(np.cov((S[base] - cs).T))
    best = None
    for f1 in (1, -1):
        for f2 in (1, -1):
            A = vs[:, ::-1] @ np.diag([f1, f2]) @ vg[:, ::-1].T
            if np.linalg.det(A) < 0:
                continue
            s0 = float(np.sqrt(ws.max() / wg.max()))
            cand, med = icp((s0, A, cs - s0 * A @ cg), [base])
            if best is None or med < best[1]:
                best = (cand, med)
    (s, R, t), _ = icp(best[0], list(G))
    return {"scale": s, "R": R.tolist(), "t": [float(v) for v in t],
            "angle_deg": float(np.degrees(np.arctan2(R[1][0], R[0][0])))}


def apply_registration(reg: dict, x: float, z: float) -> tuple[float, float]:
    """3D 원본 수평 좌표 (x, z) → SHP 좌표 (EPSG:3857)."""
    R = np.array(reg["R"])
    p = reg["scale"] * R @ np.array([x, -z]) + np.array(reg["t"])
    return float(p[0]), float(p[1])


# ---------------------------------------------------------------- 엘리베이터 승하차 문 (D-071)
# 동묘앞 1호선 승강장 안전문(psd_면_칸문)의 면 ↔ 열차 방면.
#   근거 ① 두 면의 1번 칸이 서로 반대쪽 끝 → 칸 번호가 진행 방향 기준(1번 칸 = 앞)
#        ② 3D→2D 정합 뒤, 0면의 1번 칸은 북동쪽 끝 → 0면 열차는 북동쪽(신설동)으로 진행
#        ③ 0면은 그 진행 방향의 왼쪽 → 1호선 서울역~청량리 좌측통행과 일치 (서울시 내 손안에 서울 기사)
#   ②·③ 은 scripts/v2/40 이 매번 다시 계산해 glb_objects.json 의 psd_direction 에 남긴다.
PSD_SIDE_BY_TOWARD = {("1", "동묘앞", "신설동"): 0, ("1", "동묘앞", "동대문"): 1}


def elevator_doors(instances: list[dict], mart_rows: list[dict], station: str) -> list[dict]:
    """빠른하차정보 엘리베이터 칸-문 → 같은 번호의 승강장 안전문 위치. 안전문이 없는 노선(예: 6호선)은 건너뛴다."""
    psd = {o["psd"]: o for o in instances if o.get("psd")}
    out = []
    for r in mart_rows:
        side = PSD_SIDE_BY_TOWARD.get((str(r["line"]), station, r["toward"]))
        if side is None or r.get("door") is None:
            continue
        o = psd.get((side, int(r["car"]), int(r["door"])))
        if o is None:
            continue
        out.append(dict(floor=o["floor"], x=o["x"], y=o["y"], z=o["z"], line=str(r["line"]), toward=r["toward"],
                        car=int(r["car"]), door=int(r["door"]), elevator_no=r.get("elevator_no"), psd_node=o["node"]))
    return out
