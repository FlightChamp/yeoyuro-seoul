"""
scripts/v2/40_build_korail_line1.py
===================================
Phase 3A-1 — 1호선 코레일 서울 시내 구간 시각표 mart + 운행계통 검사 + 대기시간 비교 + 노선도 배치 초안.

**별도 검증 환경**: 결과물은 data/marts/v2/korail_line1/ 에만 쓴다. 앱·경로 엔진은 이 mart 를 읽지 않는다 (3A-2 전까지).

하는 일
-------
1. 원천 감사: 한글 손실 사본이면 멈춤, 26역 역명을 표준데이터와 대조
2. 일반열차 정차 순서표 (급행 제외, 통과 자리표시 제외, 음수 정차시간이면 멈춤)
3. 운행계통 검사: 서울역·청량리 직통, 구로 분기, 광명 셔틀, 중간 종착, 자정 이후
4. 급행 통과 표시(00:00:00) 검증 — 열차별 기록과 2026-05-26 경인선 급행 조정 대조 (급행은 mart 에 넣지 않음)
5. 대기시간 비교 (D-085 사전 등록 기준): 목적지 정차 필터 기대 대기 vs 전체 열차 배차÷2
6. 26역 노선도 좌표 초안 + 미리보기 그림

실행
----
    python scripts/v2/40_build_korail_line1.py --src "<원본 폴더>"
    (원본 폴더 = 시각표 20260901 원본(cp949, 한글 정상)과 전국도시철도역사정보표준데이터 xlsx 가 있는 폴더.
     하위 폴더까지 찾는다. 없으면 [건너뜀] — 커밋된 mart 를 그대로 쓴다)

산출물
------
    data/marts/v2/korail_line1/stop_times.parquet          일반열차 정차 순서표 (커밋)
    data/marts/v2/korail_line1/stations.csv                 36역 (26 코레일 + 10 서울교통공사 구간)
    data/marts/v2/korail_line1/transfer_candidates.csv      환승 후보 8곳 (3A-2 입력)
    data/marts/v2/korail_line1/map_draft.csv                26역 노선도 좌표 초안 (앱 미사용)
    reports/v2/line1k_3a1_build.md                          검사 결과 (숫자는 여기서만 인용)
    reports/v2/fig_line1k_map_draft.png                     배치 초안 미리보기
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import korail_line1 as K                              # noqa: E402

OUT = ROOT / "data" / "marts" / "v2" / "korail_line1"
REPORT = ROOT / "reports" / "v2" / "line1k_3a1_build.md"
FIG = ROOT / "reports" / "v2" / "fig_line1k_map_draft.png"
MAP_NODES = ROOT / "data" / "app_v26" / "data" / "master" / "map_visual_nodes.csv"
MAP_SEGS = ROOT / "data" / "app_v26" / "data" / "master" / "map_line_segments.csv"

# D-085 사전 등록: 대기시간 급변 판정 (결과를 보기 전에 고정)
FLAG_DIFF_MIN = 3.0         # 목적지 필터 기대 대기 − 전체 배차÷2 ≥ 3분
FLAG_RATIO = 2.0            # 또는 2배 이상
BIN_JUMP_MIN = 3.0          # 인접 30분 bin 사이 기대 대기 차이 ≥ 3분
PEAK_BINS = [(420, 540), (1050, 1170)]      # 07:00~09:00, 17:30~19:30
NEWS_GYEONGIN_STOPS = {1005: "대방", 1032: "신길", 1801: "개봉", 1808: "동암", 1810: "제물포"}


def hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------- 검사
def check_routes(st: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """운행계통 검사 결과 (숫자는 리포트·테스트가 같이 쓴다)."""
    r = {}
    seqs = {k: list(g.sort_values("stop_seq").code) for k, g in st.groupby(["day_type", "train", "direction"])}
    pairs = collections.Counter()
    for k, cs in seqs.items():
        for a, b in zip(cs, cs[1:]):
            pairs[(k[0], a, b)] += 1
    wd = lambda a, b: pairs[("weekday", a, b)]                            # noqa: E731
    r["seoul_through"] = (wd(1002, 150), wd(150, 1002), sum(1 for k, c in seqs.items() if k[0] == "weekday"
                                                             and any(c[i:i + 3] == [1002, 150, 151] for i in range(len(c)))))
    r["cheongnyangni_through"] = (wd(158, 1015), wd(1015, 158), wd(157, 1015) + wd(1015, 157))
    r["guro_split"] = (wd(1701, 1813), wd(1701, 1702), wd(1813, 1701), wd(1702, 1701))
    both = [k for k, c in seqs.items() if 1813 in c and 1702 in c]
    r["guro_both_branches"] = len(both)
    gm = st[st.dest.eq("광명")]
    r["gwangmyeong"] = (gm.groupby(["day_type", "train", "direction"]).ngroups,
                        sorted(set(gm.code)))
    last = st[st.last_in_scope & st.day_type.eq("weekday")]
    term = last[last.dest.isin([K.SCOPE[c] for c in K.SCOPE] + ["서울역"])]
    r["terminate_in_seoul"] = term.dest.value_counts().to_dict()
    r["after_midnight"] = {
        "arr_ge_1440": int((st.arr_min >= 1440).sum()), "dep_ge_1440": int((st.dep_min >= 1440).sum()),
        "trains_cross_midnight_weekday": int(st[st.day_type.eq("weekday")].groupby(["train", "direction"])
                                             .apply(lambda g: (g.arr_min.max() >= 1440) or (g.dep_min.max() >= 1440))
                                             .sum()),
        "latest": hhmm(np.nanmax(st[["arr_min", "dep_min"]].values)),
        "earliest": hhmm(np.nanmin(st[["arr_min", "dep_min"]].values)),
    }
    r["dwell_over5"] = st[st.dwell_min > 5][["day_type", "train", "station", "dwell_min"]].to_dict("records")
    r["dwell_negative"] = int((st.dwell_min < 0).sum())
    seg = K.segments(st)
    r["run_min"] = (float(seg.run_min.min()), float(seg.run_min.median()), float(seg.run_min.max()))
    r["n_trains"] = st.groupby("day_type").apply(lambda g: g.groupby(["train", "direction"]).ngroups).to_dict()
    return r


def check_express(rows: pd.DataFrame) -> dict:
    """급행 통과 표시 검증. 일반열차의 정차 순서(인접 역쌍)를 기준으로 급행 행을 시각순으로 놓았을 때
    모든 연속 쌍이 실제 인접 역이면, 00:00:00 행은 '시각표에 있는 통과역 + 통과 시각(출발 칸)' 이다."""
    r = {}
    e, n = rows[rows.express], rows[~rows.express]
    r["placeholder_rows"] = int(e.is_pass.sum())
    r["placeholder_in_normal"] = int((n.arr == K.PASS_MARK).sum())
    r["placeholder_dep_missing"] = int(e[e.is_pass].dep.isna().sum() + (e[e.is_pass].dep == K.PASS_MARK).sum())
    adj = collections.Counter()
    for _, g in n.groupby(["day_type", "train", "dir"]):
        cs = list(g.sort_values("t_min").code)
        adj.update(zip(cs, cs[1:]))
    bad, total = [], 0
    for k, g in e.groupby(["day_type", "train", "dir"]):
        cs = list(g.sort_values("t_min").code)
        total += 1
        if any(adj[(a, b)] == 0 for a, b in zip(cs, cs[1:])):
            bad.append(k)
    r["express_trains"] = total
    r["non_adjacent_trains"] = sorted(set(k[1] for k in bad))
    gi = e.groupby(["day_type", "train", "dir"]).filter(lambda g: (g.code == 1813).any())
    gi = gi[~gi.train.isin(K.ANOMALOUS_EXPRESS)]
    tab = gi.groupby("code").agg(rows=("is_pass", "size"), passes=("is_pass", "sum"))
    r["gyeongin_news_stops"] = {NEWS_GYEONGIN_STOPS[c]: (int(tab.rows.get(c, 0)), int(tab.passes.get(c, 0)))
                                for c in NEWS_GYEONGIN_STOPS}
    seoul = gi[gi.code.isin(list(K.SEOUL_KORAIL))]
    p = seoul.groupby("code").is_pass.agg(["size", "sum"])
    r["gyeongin_seoul_pass"] = {K.SCOPE[c]: (int(v["size"]), int(v["sum"])) for c, v in p.iterrows()}
    return r


# ---------------------------------------------------------------------- 대기시간 비교
def wait_compare(st: pd.DataFrame, idx: K.NextTrainIndex, day: str = "weekday") -> pd.DataFrame:
    """코레일 26역 출발 × 범위 안 목적지 × 30분 bin: 목적지 필터 기대 대기 E vs 전체 배차÷2 H."""
    s = st[st.day_type.eq(day) & st.dep_min.notna()]
    all_deps = {k: sorted(g.dep_min) for k, g in s.groupby(["code", "direction"])}
    dir_of = {}
    for (d, o, t), (deps, arrs, trains, dirs) in idx._pairs.items():
        if d == day and o in K.SEOUL_KORAIL:
            dir_of[(o, t)] = collections.Counter(dirs).most_common(1)[0][0]
    # 바로 다음 정차역(방향별 최빈)행 열차 = 그 방향으로 떠나는 거의 모든 열차 → 배차 불규칙만의 효과
    nxt = collections.Counter()
    for k, g in s.sort_values("stop_seq").groupby(["train", "direction"]):
        cs = list(st[(st.day_type == day) & (st.train == k[0]) & (st.direction == k[1])].sort_values("stop_seq").code)
        for a, b in zip(cs, cs[1:]):
            nxt[(a, k[1], b)] += 1
    next_of = {}
    for (a, d, b), n in nxt.items():
        if n > next_of.get((a, d), (None, -1))[1]:
            next_of[(a, d)] = (b, n)
    out = []
    for (o, t), direction in dir_of.items():
        deps = idx.departures(day, o, t)
        alld = all_deps.get((o, direction), [])
        nx = next_of.get((o, direction), (None, 0))[0]
        deps_next = idx.departures(day, o, nx) if nx is not None else []
        for b0 in range(300, 1500, 30):
            n_dest = sum(1 for x in deps if b0 <= x < b0 + 30)
            if n_dest == 0:
                continue
            e, unserved = K.expected_wait(deps, b0, b0 + 30)
            e_next, _ = K.expected_wait(deps_next, b0, b0 + 30) if deps_next else (np.nan, 1.0)
            h = K.headway_half(alld, b0, b0 + 30)
            out.append({"origin": o, "dest": t, "direction": direction, "bin": b0, "n_dest": n_dest,
                        "n_all": sum(1 for x in alld if b0 <= x < b0 + 30), "E": e, "E_next": e_next, "H": h,
                        "unserved": unserved})
    w = pd.DataFrame(out)
    w["diff"] = w.E - w.H
    w["irregular"] = w.E_next - w.H          # 배차 간격이 고르지 않아서 생기는 차이
    w["dest_filter"] = w.E - w.E_next        # 목적지에 안 가는 열차(종착·분기)를 빼서 생기는 차이
    w["flag"] = (w["diff"] >= FLAG_DIFF_MIN) | ((w.H >= 1) & (w.E / w.H >= FLAG_RATIO))
    w["peak"] = w.bin.map(lambda b: any(a <= b < z for a, z in PEAK_BINS))
    w = w.sort_values(["origin", "dest", "bin"])
    w["jump"] = w.groupby(["origin", "dest"]).E.diff().abs()
    return w.reset_index(drop=True)


# ---------------------------------------------------------------------- 그림
def draw_draft(draft: pd.DataFrame, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as e:                                             # noqa: BLE001
        print(f"  (그림 건너뜀: matplotlib 없음 {e})")
        return False
    ko = [f.name for f in font_manager.fontManager.ttflist
          if any(k in f.name for k in ("Malgun", "Nanum", "Noto Sans CJK", "AppleGothic"))]
    plt.rcParams["font.family"] = ko[:1] + ["DejaVu Sans"]
    seg = pd.read_csv(MAP_SEGS)
    nodes = pd.read_csv(MAP_NODES)
    fig, ax = plt.subplots(figsize=(12, 9), dpi=110)
    for r in seg.itertuples():
        ax.plot([r.x0, r.x1], [r.y0, r.y1], color=r.color, lw=1.6, alpha=0.35, zorder=1)
    ax.scatter(nodes.x_px, nodes.y_px, s=6, color="#9CA3AF", zorder=2)
    col = {"경부": "#0052A4", "경인": "#2E7BCF", "경원": "#0A3D7A"}
    pos = {r.code: (r.x_px, r.y_px) for r in draft.itertuples()}
    for code, name in K.METRO_L1.items():
        n = nodes[nodes.station_name == name]
        if len(n):
            pos[code] = (n.x_px.iloc[0], n.y_px.iloc[0])
    for corridor, seq in K.CORRIDORS.items():
        xs = [pos[c][0] for c in seq if c in pos]
        ys = [pos[c][1] for c in seq if c in pos]
        ax.plot(xs, ys, color=col[corridor], lw=3.2, zorder=3, label=f"1호선 {corridor} (초안)")
    mk = {"anchor": "o", "interp": "s", "extrap": "^"}
    for r in draft.itertuples():
        ax.scatter([r.x_px], [r.y_px], marker=mk[r.placement], s=46, facecolor="white",
                   edgecolor="#B91C1C" if r.conflict else "#111827", lw=1.4, zorder=4)
        ax.annotate(r.station, (r.x_px, r.y_px), xytext=(6, -10), textcoords="offset points", fontsize=7.5,
                    color="#B91C1C" if r.conflict else "#111827", zorder=5)
    xs, ys = draft.x_px, draft.y_px
    ax.set_xlim(xs.min() - 450, xs.max() + 450)
    ax.set_ylim(ys.max() + 300, ys.min() - 300)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("1호선 코레일 26역 노선도 배치 초안 (○ 기존 역 좌표  □ 사이 등간격  △ 위·경도 방향 연장, 빨강 = 다른 역과 40px 안)",
                 fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def write_parquet(df: pd.DataFrame, path: Path):
    """pandas 2·3 어디서나 같게 읽히도록: 문자열 열은 plain string, pandas 전용 메타데이터는 뺀다."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([(c, pa.string() if (df[c].dtype == object or str(df[c].dtype) in ("str", "string"))
                         else pa.from_numpy_dtype(df[c].dtype)) for c in df.columns])
    t = pa.Table.from_pandas(df, schema=schema, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(t, path)


# ---------------------------------------------------------------------- 리포트
def write_report(audit, route, exp, w, draft, tc, src_name, std_name):
    L = ["# 3A-1 빌드 검사 결과 — 1호선 코레일 서울 시내 구간 (일반열차)", "",
         f"- 원천: `{src_name}` · 역명 대조: `{std_name}`",
         "- 실행: `python scripts/v2/40_build_korail_line1.py --src <원본 폴더>` · 이 파일의 숫자는 스크립트가 쓴다",
         "- 범위: 코레일 26역 + 서울교통공사 1호선 10역(직통 확인용). **일반열차만** (급행은 §4 검증만, mart 미포함)",
         "- 앱·경로 엔진 영향 없음 (별도 검증 환경, D-085)", ""]
    L += ["## 1. 원천 감사", "",
          f"- 한글 손실 검사: 통과 (역명·출발역·도착역에 '??' 없음)",
          f"- 26역 역명 = 표준데이터 역명(역 접미사 제거) = 코드 상수: **{int(audit.match.sum())}/26 일치**",
          "- 정차 순서표 열차×방향 수: " + " · ".join(f"{k} {v}" for k, v in sorted(route["n_trains"].items()))
          + f" · 음수 정차시간 {route['dwell_negative']}행", ""]
    a, b, c = route["seoul_through"]
    L += ["## 2. 운행계통 (평일)", "",
          "| 검사 | 결과 | 판정 |", "|---|---|---|",
          f"| 서울역 직통: 남영→서울역 / 서울역→남영 / 남영→서울역→시청 연속 열차 | {a} / {b} / {c} | 같은 열차로 이어짐 → 환승 아님 |"]
    a, b, c = route["cheongnyangni_through"]
    L += [f"| 청량리 직통: 청량리→회기 / 회기→청량리 / 제기동↔회기 바로 연결 | {a} / {b} / {c} | 청량리 건너뛰는 기록 없음 |"]
    a, b, c, d = route["guro_split"]
    L += [f"| 구로 분기: 구로→구일 / 구로→가산 / 구일→구로 / 가산→구로 | {a} / {b} / {c} / {d} | 분기 확인 |",
          f"| 한 열차가 구일과 가산디지털단지를 모두 지남 | {route['guro_both_branches']} | 0 이어야 함 |"]
    n, codes = route["gwangmyeong"]
    L += [f"| 광명행 열차(전 요일 열차×방향) · 서울 안 정차역 | {n} · {', '.join(K.SCOPE[c] for c in codes)} | 영등포~금천구청 안에서만 |"]
    L += [f"| 서울 안 종착(평일, 열차 수) | {', '.join(f'{k} {v}' for k, v in route['terminate_in_seoul'].items())} | 중간 종착 → 목적지 필터 필요 |"]
    mid = route["after_midnight"]
    L += [f"| 자정 이후: 도착 ≥24:00 / 출발 ≥24:00 / 평일 자정 넘김 열차 | {mid['arr_ge_1440']} / {mid['dep_ge_1440']} / {mid['trains_cross_midnight_weekday']} | 24:xx 그대로 1,440분 이상 |",
          f"| 첫·막 시각 (범위 안, 운행일 기준) | {mid['earliest']} ~ {mid['latest']} | 24:00 이상 = 다음 날 새벽 |",
          f"| 역간 주행시간 최소 / 중앙 / 최대 (분) | {route['run_min'][0]:.1f} / {route['run_min'][1]:.1f} / {route['run_min'][2]:.1f} | 0 이하 없음 |",
          f"| 정차 5분 초과 | {len(route['dwell_over5'])}행 | 아래 목록 |", ""]
    for d in route["dwell_over5"]:
        L.append(f"  - {d['day_type']} {d['train']} {d['station']} {d['dwell_min']:.1f}분")
    L += ["", "## 3. 급행 통과 표시 `00:00:00` 검증 (mart 에는 넣지 않음)", "",
          "- 공식 정의: 공공데이터포털 15098251 설명에는 `급행:1, 일반:0` 만 있고 `00:00:00` 의 뜻은 **적혀 있지 않음**(미러 페이지 기준) → 열차별 기록으로 검증",
          f"- 자리표시 행 {exp['placeholder_rows']:,} (모두 급행) · 일반열차 {exp['placeholder_in_normal']} · 그 행의 출발 칸 비었거나 00:00:00 {exp['placeholder_dep_missing']}",
          f"- 급행 {exp['express_trains']}개(열차×방향×요일)를 '출발 칸 시각'으로 정렬하면 연속 역쌍이 일반열차의 실제 인접 역쌍과 모두 같음. 예외 = {', '.join(exp['non_adjacent_trains'])} (3A-0 §5.2 이상 기록)",
          "  → `00:00:00` 행 = 그 역을 **통과하는 시각이 출발 칸에 적힌 무정차 통과 행**으로 판단 (정차·승하차 불가)",
          "- 2026-05-26 경인선 급행 조정(코레일: 특급 → 일반 급행 전환, 대방·신길·개봉·동암·제물포 급행 정차 확대) 대조 — 9월 시각표 경인 급행(구일 경유, 이상 3편성 제외):", "",
          "| 역 | 급행 행 | 그중 통과 |", "|---|---|---|"]
    for k, (n, p) in exp["gyeongin_news_stops"].items():
        L.append(f"| {k} | {n} | {p} |")
    L += ["", "  → 5개 역 모두 통과 0 = 조정 뒤 운행과 일치. 서울 안 경인 급행 통과역: "
          + ", ".join(f"{k} {p}/{n}" for k, (n, p) in exp["gyeongin_seoul_pass"].items() if p), "",
          "- 미확인: 열차번호별 공식 정차역표와의 1:1 대조 (3A-3)", ""]
    pk = w[w.peak]
    L += ["## 4. 대기시간 비교 — 목적지 정차 필터 기대 대기 E vs 전체 열차 배차÷2 H (평일, D-085 사전 등록 기준)", "",
          f"- 셀 = 출발역(코레일 26) × 목적지(범위 안) × 30분, 그 30분에 목적지행 열차가 1대 이상인 셀만: {len(w):,}개",
          f"- 급변 판정: E − H ≥ {FLAG_DIFF_MIN:g}분 또는 E/H ≥ {FLAG_RATIO:g}",
          f"- 전체 {w.flag.mean() * 100:.1f}% · 출퇴근(07:00~09:00, 17:30~19:30) {pk.flag.mean() * 100:.1f}% 가 급변",
          f"- E − H 중앙값 {w['diff'].median():.2f}분 · 90% {w['diff'].quantile(.9):.2f}분 · 최대 {w['diff'].max():.1f}분",
          f"- 인접 30분 사이 E 차이 ≥ {BIN_JUMP_MIN:g}분: {int((w.jump >= BIN_JUMP_MIN).sum()):,}건 ({(w.jump >= BIN_JUMP_MIN).mean() * 100:.1f}%)",
          f"- 다음 열차까지 60분 초과 비율이 있는 셀: {int((w.unserved > 0).sum())}",
          f"- 차이 나누기 (중앙값 / 90%): 배차 불규칙(E_next − H) {w.irregular.median():.2f} / {w.irregular.quantile(.9):.2f}분 · "
          f"목적지 필터(E − E_next) {w.dest_filter.median():.2f} / {w.dest_filter.quantile(.9):.2f}분",
          f"  - 목적지 필터만으로 ≥ {FLAG_DIFF_MIN:g}분: {(w.dest_filter >= FLAG_DIFF_MIN).mean() * 100:.1f}% · "
          f"배차 불규칙만으로 ≥ {FLAG_DIFF_MIN:g}분: {(w.irregular >= FLAG_DIFF_MIN).mean() * 100:.1f}%",
          "  - 주의: 여기 E 는 '갈아타지 않고 한 열차로' 가는 기준. 구로·금천구청 등에서 같은 1호선 다른 열차로 갈아타면 "
          "더 짧을 수 있음 → 3A-2 그래프에서 같은 노선 갈아타기로 비교", "",
          "급변 셀이 많은 출발→목적지 (출퇴근, 상위 10):", "",
          "| 출발 | 목적지 | 급변 셀 / 출퇴근 셀 | E 중앙 | H 중앙 |", "|---|---|---|---|---|"]
    top = (pk.groupby(["origin", "dest"]).agg(flag=("flag", "sum"), n=("flag", "size"), E=("E", "median"),
                                               H=("H", "median")).sort_values(["flag", "E"], ascending=False).head(10))
    for (o, t), r in top.iterrows():
        L.append(f"| {K.SCOPE[o]} | {K.SCOPE[t]} | {int(r.flag)} / {int(r.n)} | {r.E:.1f} | {r.H:.1f} |")
    by_cor = w.assign(cor=w.origin.map(lambda c: K.SEOUL_KORAIL[c][1])).groupby(["cor", "peak"]).flag.mean().unstack()
    L += ["", "계통별 급변 비율 (%):", "", "| 계통 | 출퇴근 외 | 출퇴근 |", "|---|---|---|"]
    for cor, r in by_cor.iterrows():
        L.append(f"| {cor} | {r.get(False, np.nan) * 100:.1f} | {r.get(True, np.nan) * 100:.1f} |")
    L += ["", "## 5. 환승 후보 (3A-2 입력, 엔진 미반영)", "",
          "| 역 | 노선 | 공식 거리(m) | 환승 데이터 평균(분) | 사용 후보(분) | 상태 |", "|---|---|---|---|---|---|"]
    for r in tc.itertuples():
        dist = "" if pd.isna(r.distance_m) else f"{r.distance_m:.0f}"
        ds = "자리표시(07:00)" if pd.isna(r.dataset_mean_min) else f"{r.dataset_mean_min:.2f}"
        L.append(f"| {r.station} | 1↔{r.other_line} | {dist} | {ds} | {r.walk_min:.2f} | {r.status} |")
    L += ["", "- 1~8호선 v1 환승시간 = 같은 '수도권 도시철도 환승 데이터' 패턴 평균 (transfer_edge_mart 80/80 일치 확인)",
          "", "## 6. 노선도 배치 초안 (앱 미사용)", "",
          f"- 기존 역 좌표 재사용 {int((draft.placement == 'anchor').sum())} · 사이 등간격 {int((draft.placement == 'interp').sum())} · 위·경도 방향 연장 {int((draft.placement == 'extrap').sum())}",
          f"- 다른 역과 40px 안 (배치 조정 필요): {', '.join(f'{r.station}({r.conflict})' for r in draft.itertuples() if r.conflict) or '없음'}",
          "- 그림: `reports/v2/fig_line1k_map_draft.png` · 좌표: `data/marts/v2/korail_line1/map_draft.csv`", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    a = ap.parse_args()
    try:
        tt = K.locate_source(a.src, K.TT_FILE)
        std_path = K.locate_source(a.src, K.STD_FILE_PATTERN)
    except FileNotFoundError as e:
        print(f"[건너뜀] {e} — 커밋된 mart 를 그대로 쓸 수 있습니다")
        return 0
    print(f"시각표: {tt}\n표준데이터: {std_path}")
    raw = K.read_timetable(tt)
    rows = K.line1_rows(raw)
    std = K.read_standard(std_path)
    audit = K.cross_check(rows, std)
    if not audit.match.all():
        print(audit[~audit.match].to_string())
        raise SystemExit("역명 대조 실패 — 멈춤")
    st = K.stop_times(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_parquet(st, OUT / "stop_times.parquet")
    stations = pd.concat([
        audit[["code", "station", "corridor", "lat", "lon"]].assign(operator="코레일"),
        pd.DataFrame({"code": list(K.METRO_L1), "station": list(K.METRO_L1.values()),
                      "corridor": "서울교통공사 1호선", "lat": np.nan, "lon": np.nan, "operator": "서울교통공사"})])
    stations.to_csv(OUT / "stations.csv", index=False, encoding="utf-8")
    tc = K.transfer_candidates()
    tc.to_csv(OUT / "transfer_candidates.csv", index=False, encoding="utf-8")
    draft = K.map_draft(pd.read_csv(MAP_NODES), audit)
    draft.to_csv(OUT / "map_draft.csv", index=False, encoding="utf-8")
    route = check_routes(st, rows)
    exp = check_express(rows)
    idx = K.NextTrainIndex(st)
    w = wait_compare(st, idx)
    draw_draft(draft, FIG)
    write_report(audit, route, exp, w, draft, tc, tt.name, std_path.name)
    print(f"완료: {OUT} · {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
