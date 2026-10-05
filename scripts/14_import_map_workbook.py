"""
14_import_map_workbook.py
=========================
`yeoyuro_seoul_vector_map_coordinate_workbook.xlsx` 를 프로젝트 마스터로 가져오고 검증한다.

이 워크북이 노선도 좌표의 **source of truth** 다. 좌표가 어색해 보여도 자동 레이아웃으로
대체하지 않는다. 문제가 있으면 validation warning 으로 남기고 최소 보정만 한다.

좌표계
------
    기준 캔버스 5120 x 2880 (이미지 좌표계: y 는 아래로 증가)
    SVG   viewBox="0 0 5120 2880"
    Plotly x 0~5120, y 0~2880 이며 y축을 reversed 로 두어 이미지 좌표계와 맞춘다.

pct 컬럼은 엑셀 수식 문자열일 수 있으므로 **신뢰하지 않고 x_px/y_px 에서 재계산**한다.

출력
----
    data/master/map_visual_nodes.csv     지도 렌더링용 노선별 좌표
    data/master/map_click_areas.csv      투명 클릭 hitbox (station_key 기준)
    data/master/map_labels.csv           역명 라벨 (station_key 기준)
    data/master/map_line_segments.csv    노선 폴리라인 세그먼트 (좌표 포함)
    data/master/map_transfer_links.csv   환승 connector
    data/master/station_display_master.csv  candidate_station_uids 를 채워 갱신
    reports/data_quality/map_workbook_validation.csv
    reports/figures/vector_map_preview.png

사용법
------
    python scripts/14_import_map_workbook.py --root . --xlsx <경로> --preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENC = "utf-8-sig"
CANVAS_W, CANVAS_H = 5120, 2880

LINE_COLORS = {
    "1": "#0052A4", "2": "#009D3E", "3": "#EF7C1C", "4": "#00A5DE",
    "5": "#996CAC", "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C",
    "9": "#BDB092",   # v2.6 9호선 (노선 색 추가만, 1~8호선 결과 영향 없음)
}

SHEETS = ["Station_Display_Master", "Station_Visual_Nodes", "Station_Click_Areas",
          "Station_Labels", "Line_Sequences", "Transfer_Links"]


def load_mart(base: Path, name: str):
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = base / (name + ext)
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    return None


class MapWorkbookImporter:

    def __init__(self, root: Path, xlsx: Path):
        self.root = root
        self.xlsx = xlsx
        self.master = root / "data" / "master"
        self.marts = root / "data" / "marts"
        self.report = root / "reports" / "data_quality"
        self.figs = root / "reports" / "figures"
        for d in (self.master, self.report):
            d.mkdir(parents=True, exist_ok=True)
        self.checks: list[dict] = []
        self.notes: list[str] = []

    def add(self, name, ok, detail=""):
        self.checks.append({"check": name, "status": "PASS" if ok else "FAIL",
                            "detail": str(detail)})

    def warn(self, name, detail):
        self.checks.append({"check": name, "status": "WARN", "detail": str(detail)})

    # ---------- 읽기 ----------
    def read(self):
        xl = pd.ExcelFile(self.xlsx)
        missing = [s for s in SHEETS if s not in xl.sheet_names]
        self.add("필수 시트 존재", not missing, missing or "6개 모두 존재")
        d = {s: xl.parse(s) for s in SHEETS if s in xl.sheet_names}

        # pct 는 엑셀 수식일 수 있으므로 px 에서 재계산한다
        for key, xc, yc in (("Station_Visual_Nodes", "x_px", "y_px"),
                            ("Station_Click_Areas", "click_x_px", "click_y_px"),
                            ("Station_Labels", "label_x_px", "label_y_px")):
            if key in d:
                t = d[key]
                t[xc] = pd.to_numeric(t[xc], errors="coerce")
                t[yc] = pd.to_numeric(t[yc], errors="coerce")
                t[xc.replace("_px", "_pct")] = t[xc] / CANVAS_W * 100
                t[yc.replace("_px", "_pct")] = t[yc] / CANVAS_H * 100
        self.notes.append("pct 컬럼은 엑셀 수식 대신 px 에서 재계산했다")
        return d

    # ---------- 검증 ----------
    def validate(self, d):
        vn, ls, tl = d["Station_Visual_Nodes"], d["Line_Sequences"], d["Transfer_Links"]
        ca, lb, dm = d["Station_Click_Areas"], d["Station_Labels"], d["Station_Display_Master"]
        ids = set(vn["visual_node_id"])

        self.add("visual node 좌표 결측 없음",
                 int(vn["x_px"].isna().sum()) == 0 and int(vn["y_px"].isna().sum()) == 0,
                 "노드 %d개" % len(vn))

        bad = ls[~ls["visual_node_id"].isin(ids)]
        self.add("Line_Sequences 참조 유효", len(bad) == 0,
                 "무효 %d건 %s" % (len(bad), sorted(set(bad["visual_node_id"]))[:5]))

        refs = set(tl["from_visual_node_id"]) | set(tl["to_visual_node_id"])
        bad2 = sorted(refs - ids)
        self.add("Transfer_Links 참조 유효", not bad2, "무효 %s" % bad2[:5])

        out = vn[(vn["x_px"] < 0) | (vn["x_px"] > CANVAS_W)
                 | (vn["y_px"] < 0) | (vn["y_px"] > CANVAS_H)]
        self.add("좌표가 캔버스 범위 안(0~5120, 0~2880)", len(out) == 0,
                 "x %d~%d / y %d~%d" % (vn["x_px"].min(), vn["x_px"].max(),
                                        vn["y_px"].min(), vn["y_px"].max()))

        loop = ls[(ls["line_id"] == 2) & (ls["branch_code"] == "main")].sort_values("path_order")
        first, last = loop["visual_node_id"].iloc[0], loop["visual_node_id"].iloc[-1]
        self.add("2호선 main loop 닫힘 가능", first == "시청_L2" and last == "충정로_L2",
                 "%s ... %s → 렌더링 시 마지막-처음 연결" % (first, last))

        # 클릭/라벨은 station_key 기준 1행
        for nm, t in (("Station_Click_Areas", ca), ("Station_Labels", lb)):
            dup = t["station_key"].duplicated().sum()
            self.add("%s station_key 유일" % nm, dup == 0, "중복 %d" % dup)

        # 그래프와 역 키 대조
        ride = load_mart(self.marts, "route_edge_mart")
        if ride is not None:
            nodes = sorted(set(ride["from_node"]) | set(ride["to_node"]))
            keys = {n.split("_", 1)[1].split("@")[0] for n in nodes}
            wb = set(dm["station_key"])
            self.add("워크북 역 = 라우팅 그래프 역", wb == keys,
                     "워크북 %d / 그래프 %d / 차집합 %s"
                     % (len(wb), len(keys), sorted(wb ^ keys)[:5]))
        else:
            self.warn("라우팅 그래프 대조", "route_edge_mart 없음 → 건너뜀")

        # 좌표 중복 경고 (같은 station_key 의 다른 호선 노드가 완전히 겹침)
        dupxy = (vn.groupby(["x_px", "y_px"])["visual_node_id"]
                 .agg(list).reset_index())
        dupxy = dupxy[dupxy["visual_node_id"].map(len) > 1]
        if len(dupxy):
            same_station = sum(1 for v in dupxy["visual_node_id"]
                               if len({x.rsplit("_L", 1)[0] for x in v}) == 1)
            self.warn("좌표가 완전히 겹치는 visual node",
                      "%d쌍 (그중 동일 역 %d쌍은 정상)" % (len(dupxy), same_station))

    # ---------- 산출 ----------
    def build(self, d):
        vn, ls, tl = d["Station_Visual_Nodes"], d["Line_Sequences"], d["Transfer_Links"]
        ca, lb, dm = d["Station_Click_Areas"], d["Station_Labels"], d["Station_Display_Master"]
        pos = {r.visual_node_id: (float(r.x_px), float(r.y_px)) for r in vn.itertuples()}

        # --- 노선 폴리라인 세그먼트 ---
        segs = []
        for (line, branch), g in ls.groupby(["line_id", "branch_code"]):
            g = g.sort_values("path_order")
            ids = g["visual_node_id"].tolist()

            # 6호선 응암순환: 시퀀스가 구산 -> 새절 로 바로 넘어가 응암 복귀가 보이지 않는다.
            # 지도에서 '응암-역촌-불광-독바위-연신내-구산-응암-새절' 흐름이 보이도록
            # 구산 다음에 응암을 한 번 되돌려 넣는다. (최소 보정)
            if str(line) == "6" and "구산_L6" in ids and "새절_L6" in ids:
                i = ids.index("구산_L6")
                if ids[i + 1] == "새절_L6":
                    ids = ids[:i + 1] + ["응암_L6"] + ids[i + 1:]
                    self.warn("6호선 응암순환 렌더링 보정",
                              "구산_L6 다음에 응암_L6 을 삽입해 loop 복귀를 표시")

            pairs = list(zip(ids, ids[1:]))
            # 2호선 main 은 순환선이므로 마지막-처음을 연결해 loop 를 닫는다
            if str(line) == "2" and branch == "main":
                pairs.append((ids[-1], ids[0]))

            for a, b in pairs:
                if a not in pos or b not in pos:
                    continue
                segs.append({"line_id": str(line), "branch_code": branch,
                             "from_visual_node_id": a, "to_visual_node_id": b,
                             "x0": pos[a][0], "y0": pos[a][1],
                             "x1": pos[b][0], "y1": pos[b][1],
                             "color": LINE_COLORS.get(str(line), "#888")})
        seg = pd.DataFrame(segs)

        # --- 환승 connector 좌표 ---
        tl = tl.copy()
        tl["x0"] = tl["from_visual_node_id"].map(lambda v: pos.get(v, (np.nan,) * 2)[0])
        tl["y0"] = tl["from_visual_node_id"].map(lambda v: pos.get(v, (np.nan,) * 2)[1])
        tl["x1"] = tl["to_visual_node_id"].map(lambda v: pos.get(v, (np.nan,) * 2)[0])
        tl["y1"] = tl["to_visual_node_id"].map(lambda v: pos.get(v, (np.nan,) * 2)[1])

        # --- candidate_station_uids 채우기 ---
        ride = load_mart(self.marts, "route_edge_mart")
        if ride is not None:
            nodes = sorted(set(ride["from_node"]) | set(ride["to_node"]))
            bykey = {}
            for n in nodes:
                bykey.setdefault(n.split("_", 1)[1].split("@")[0], []).append(n)
            dm = dm.copy()
            dm["candidate_station_uids"] = dm["station_key"].map(
                lambda k: ";".join(sorted(set(bykey.get(k, [])), key=lambda x: ("@" in x, x))))
            n_empty = int((dm["candidate_station_uids"] == "").sum())
            self.add("candidate_station_uids 채움", n_empty == 0,
                     "미채움 %d / %d" % (n_empty, len(dm)))
            dm["n_lines"] = dm["available_lines"].astype(str).str.count(",") + 1
            dm["is_branch_node"] = dm["station_key"].isin(["성수", "신도림", "강동", "응암"])
        return seg, tl, dm

    # ---------- 미리보기 ----------
    def preview(self, vn, seg, tl, lb):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
        except ImportError:
            self.notes.append("matplotlib 미설치 → 미리보기 생략")
            return None
        for c in ("NanumGothic", "Malgun Gothic", "AppleGothic",
                  "Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"):
            if any(c in f.name for f in font_manager.fontManager.ttflist):
                matplotlib.rcParams["font.family"] = c
                break
        matplotlib.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(17.0, 9.6))
        for r in seg.itertuples():
            ax.plot([r.x0, r.x1], [r.y0, r.y1], color=r.color, lw=5.5,
                    solid_capstyle="round", zorder=1)
        for r in tl.itertuples():
            if bool(r.connector_visible) and not pd.isna(r.x0):
                ax.plot([r.x0, r.x1], [r.y0, r.y1], color="#9CA3AF", lw=2.2, zorder=2)
        tr = vn[vn["is_transfer_station"] == True]
        nm = vn[vn["is_transfer_station"] != True]
        ax.scatter(nm["x_px"], nm["y_px"], s=34, c="white",
                   edgecolors="#4B5563", linewidths=1.4, zorder=3)
        ax.scatter(tr["x_px"], tr["y_px"], s=110, c="white",
                   edgecolors="#111827", linewidths=2.4, zorder=4)
        for r in lb.itertuples():
            ax.annotate(r.station_key, (r.label_x_px, r.label_y_px),
                        xytext=(0, -16), textcoords="offset points",
                        ha="center", fontsize=7.6,
                        fontweight="bold" if str(r.label_weight) == "bold" else "normal",
                        color="#111827", zorder=5)
        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(CANVAS_H, 0)          # 이미지 좌표계 (y 아래로 증가)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Yeoyuro Seoul vector subway map — workbook 5120x2880", fontsize=12)
        fig.tight_layout()
        self.figs.mkdir(parents=True, exist_ok=True)
        p = self.figs / "vector_map_preview.png"
        fig.savefig(p, dpi=105, facecolor="white")
        plt.close(fig)
        return p

    # ---------- 실행 ----------
    def run(self, do_preview: bool):
        d = self.read()
        self.validate(d)
        seg, tl, dm = self.build(d)

        outs = {}
        for name, df in (("map_visual_nodes", d["Station_Visual_Nodes"]),
                         ("map_click_areas", d["Station_Click_Areas"]),
                         ("map_labels", d["Station_Labels"]),
                         ("map_line_segments", seg),
                         ("map_transfer_links", tl),
                         ("station_display_master", dm)):
            p = self.master / (name + ".csv")
            df.to_csv(p, index=False, encoding=ENC)
            outs[name] = p

        chk = pd.DataFrame(self.checks)
        pc = self.report / "map_workbook_validation.csv"
        chk.to_csv(pc, index=False, encoding=ENC)
        outs["validation"] = pc
        png = self.preview(d["Station_Visual_Nodes"], seg, tl, d["Station_Labels"]) \
            if do_preview else None
        if png:
            outs["preview"] = png

        n_fail = int((chk.status == "FAIL").sum())
        n_warn = int((chk.status == "WARN").sum())
        print("=" * 88)
        print(" 14_import_map_workbook - 완료")
        print("=" * 88)
        print(chk.to_string(index=False))
        print("\n  PASS %d / FAIL %d / WARN %d"
              % (int((chk.status == "PASS").sum()), n_fail, n_warn))
        for n in self.notes:
            print("  - " + n)
        print("\n[렌더링 세그먼트]")
        print(seg.groupby(["line_id", "branch_code"]).size().to_string())
        print("\n[생성 파일]")
        for k, v in outs.items():
            print("  %-24s %s" % (k, v))
        print("=" * 88)
        return 1 if n_fail else 0


# 노선도 좌표 워크북 파일명. 프로젝트 리브랜딩(MetroCalm -> 여유로 서울) 이전
# 이름으로 저장된 워크북도 계속 읽을 수 있도록 legacy 이름을 함께 둔다.
WORKBOOK_NAME = "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
WORKBOOK_LEGACY = "metrocalm_vector_map_coordinate_workbook.xlsx"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--xlsx", default=WORKBOOK_NAME)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args(argv)
    x = Path(args.xlsx)
    if not x.exists():
        # 신규 파일명을 먼저 찾고, 없으면 리브랜딩 이전 파일명으로 되짚는다.
        cand = list(Path(args.root).rglob(WORKBOOK_NAME))
        if not cand:
            cand = list(Path(args.root).rglob(WORKBOOK_LEGACY))
            if cand:
                print("[warning] 이전 파일명의 워크북을 사용합니다: %s" % cand[0].name)
        if not cand:
            print("워크북 xlsx 를 찾을 수 없습니다: %s" % args.xlsx)
            return 2
        x = cand[0]
    return MapWorkbookImporter(Path(args.root), x).run(args.preview)


if __name__ == "__main__":
    sys.exit(main())
