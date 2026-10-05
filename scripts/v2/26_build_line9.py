"""
scripts/v2/26_build_line9.py
============================
v2.6 Phase A — 9호선 데이터 감사 + v1 호환 mart + 시나리오별 shadow root 생성.

하는 일
-------
1. 출처 무결성: 시각표 한글 손실 확인, 역코드 38개 복구, 운영사 운행정보와 전 구간 소요 교차검증
2. 역간 운행시간의 시간대 변동성 측정 → 기준(line9.RUN_VAR_*) 으로 bin 사용 여부 결정
3. 정차(대피), 배차, 일반↔급행 연계 대기, 환승(보정 proxy), 혼잡(3개년 중앙값 / max_3y) mart
4. 수요 proxy 를 OA-12252 최근 12개월로 1~9호선 통일 + v2.3 가중 결과 변화량
5. shadow root (1~8호선 + 9호선 합본) 5개: base, tf100, tf160, tf278, max3y
6. 스모크 검사: 9호선 경유 OD 몇 개를 실제로 경로 탐색

실행
----
    python scripts/v2/26_build_line9.py --src "<원본 폴더>"
    (원본 폴더 = 시각표·9호선 혼잡도·역사정보·운행정보·환승역거리·OA-12252 CSV 가 있는 폴더.
     하위 폴더까지 찾고, 파일 이름의 공백·밑줄·괄호 차이는 무시한다.
     원본이 없으면 [건너뜀] 후 종료 — 커밋된 mart 는 그대로 쓸 수 있다)

산출물
------
    data/marts/v2/line9/*.parquet                 9호선 단독 mart (커밋)
    data/marts/v2/demand_station_hour_oa.parquet  1~9호선 통일 수요 proxy (커밋)
    data/interim/v26/<scenario>/                  합본 shadow root (재생성, 커밋 안 함)
    reports/v2/line9_data_audit.md
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yeoyuro_v2 import line9 as L9                                  # noqa: E402
from yeoyuro_v2.od_shiftability import TYPES, classify              # noqa: E402
from yeoyuro_v2.shift_rules import T1                                # noqa: E402

OA_FILE = "서울시_지하철_호선별_역별_시간대별_승하차_인원_정보.csv"
OA_LINES = ["1호선", "2호선", "3호선", "4호선", "5호선", "6호선", "7호선", "8호선",
            "9호선", "9호선2~3단계", "9호선2단계"]
FORCED = {"총신대입구(이수)": "이수", "청량리(서울시립대입구)": "청량리"}
RENAMES = {"당고개": "불암산", "뚝섬유원지": "자양"}
SCENARIOS = L9.SCENARIOS


def canon_oa(n: str) -> str:
    n = str(n).strip()
    if n in FORCED:
        return FORCED[n]
    b = re.sub(r"\(.*\)$", "", n).strip() or n
    return RENAMES.get(b, b)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="원본 파일 폴더 (여러 개 가능, 하위 폴더까지 탐색)")
    ap.add_argument("--oa-from", type=int, default=202507)
    ap.add_argument("--oa-to", type=int, default=202606)
    args = ap.parse_args(argv)
    src = [Path(x) for x in args.src]
    t0 = time.time()
    miss = L9.missing_sources(src)
    if miss:
        print(f"[건너뜀] 원본 폴더 {', '.join(map(str, src))} 에서 다음 파일을 찾지 못했습니다:")
        for n in miss:
            print("   -", n)
        print("커밋된 9호선 mart(data/marts/v2/line9)와 감사 리포트는 그대로 사용할 수 있습니다.")
        print("원본으로 다시 만들려면 위 파일이 있는 폴더를 --src 로 지정하세요.")
        return 3
    out9 = ROOT / "data" / "marts" / "v2" / "line9"
    out9.mkdir(parents=True, exist_ok=True)
    L = ["# v2.6 9호선 데이터 감사 리포트", "",
         f"- 원본 폴더: `{', '.join(map(str, src))}` · 실행 `python scripts/v2/26_build_line9.py --src <원본>`", ""]

    # ------------------------------------------------------------ 1. 출처 무결성
    codes = L9.station_codes(src)
    sm = L9.station_master(src)
    tt = L9.timetable(src, codes)
    express = set(tt[tt.pattern == "9X"].station.unique())
    assert len(codes) == 38 and set(codes) == set(range(4101, 4139)), "역코드 38개 복구 실패"
    assert list(sm.code) == list(range(4101, 4139)), "역사정보 역번호 순서 불일치"
    assert (sm.station.values == [codes[c] for c in sm.code]).all(), "역사정보와 운행정보 역명 불일치"
    seg = L9.segments(tt)

    k = pd.read_excel(L9.locate(src, L9.KRIC_FILE), header=3)

    def kric_total(kind, day, f, to):
        out = []
        for _, r in k[(k.운행유형 == kind) & (k.요일구분 == day)].iterrows():
            dep = {int(m.group(1)): int(m.group(2)) * 60 + int(m.group(3))
                   for m in re.finditer(r"(\d{4})-(\d+):(\d+)", str(r.정거장출발시각))}
            arr = {int(m.group(1)): int(m.group(2)) * 60 + int(m.group(3))
                   for m in re.finditer(r"(\d{4})-(\d+):(\d+)", str(r.정거장도착시각))}
            if f in dep and to in arr:
                out.append(arr[to] - dep[f])
        return np.median(out) if out else np.nan, len(out)

    def tt_total(pat, day, f, to, dr):
        g = tt[(tt.pattern == pat) & (tt.day_type == day) & (tt.direction == dr)]
        out = []
        for _, x in g.groupby("train"):
            c = set(x.code)
            if f in c and to in c:
                out.append(x[x.code == to].a.iloc[0] - x[x.code == f].p.iloc[0])
        return np.median(out) if out else np.nan, len(out)

    cv = []
    for pat, kind, f in (("9L", "일반", 4101), ("9X", "급행", 4102)):
        for day, kday in (("weekday", "평일"), ("sunday", "휴일")):
            a, na = tt_total(pat, day, f, 4138, "up")
            b, nb = kric_total(kind, kday, f, 4138)
            cv.append({"패턴": pat, "요일": day, "시각표(2026-09) 분": round(a, 1), "열차": na,
                       "운영사(2026-02) 분": round(b, 1), "열차 ": nb})
    cv = pd.DataFrame(cv)
    L += ["## 1. 출처 무결성", "",
          f"- 시각표 9호선 {len(tt):,}행, 역명 한글 손실 행 {int(tt.name_corrupted.sum()):,}개 → 역코드 4101~4138 로 38개 역 복구",
          f"- 운영사 운행정보 역코드 38개 = 역사정보 역번호 901~938 (순서·역명 일치)",
          f"- 급행 정차역 {len(express)}개: {', '.join(sorted(express))}", "",
          "전 구간(개화/김포공항 → 중앙보훈병원) 소요 중앙값 교차검증:", "", cv.to_markdown(index=False), ""]

    # ------------------------------------------------------------ 2. 운행시간 변동성
    rv = L9.run_variability(seg)
    L += ["## 2. 역간 운행시간의 시간대 변동성 (기준은 계산 전 고정, D-040)", "",
          f"- 구간별 30분 bin 중앙값이 전체 중앙값에서 ±{L9.RUN_VAR_SECTION_TOL_MIN}분 이내인 구간: "
          f"{pct(rv['section_share_within_tol'])} (기준 ≥ {pct(L9.RUN_VAR_SECTION_SHARE)})",
          f"- 구간별 최대 편차 {rv['max_section_dev']:.2f}분, 전 구간 합의 bin 별 최대 상대 편차 "
          f"{pct(rv['total_max_rel_dev'])} (기준 ≤ {pct(L9.RUN_VAR_TOTAL_TOL)})",
          f"- 요일유형 간 구간 운행시간 최대 차이 {rv['day_type_max_dev']:.2f}분",
          f"- **판정: {'변동 작음 → 패턴별 운행시간 중앙값 + 시간대별 정차 사용' if rv['small'] else '변동 큼 → 요일·30분 bin 별 운행시간 사용 (v2_line9_run, 열차 3대 미만 bin 은 평일 중앙값)'}**", "",
          "전 구간(개화→중앙보훈병원, 일반·평일) 운행시간 합의 시간대별 변화 (정차 제외):", "",
          (lambda t: t.to_markdown(index=False))(
              seg[(seg.pattern == "9L") & (seg.day_type == "weekday") & (seg.direction == "up")]
              .groupby(["time_bin", "from_code"]).run.median().groupby(level=0).agg(["sum", "size"])
              .query("size >= 36").reset_index().rename(columns={"time_bin": "출발 bin", "sum": "운행시간 합(분)",
                                                               "size": "구간 수"})
              .assign(**{"운행시간 합(분)": lambda x: x["운행시간 합(분)"].round(1)})
              .iloc[::3]), ""]

    # ------------------------------------------------------------ 3. mart
    redges = L9.route_edges(seg, sm)
    dwell = L9.dwell_table(seg)
    runtab = L9.run_table(seg)
    v1_hw = pd.read_parquet(ROOT / "data" / "marts" / "headway_station_30min.parquet")
    bins = sorted(v1_hw.time_bin.unique(), key=lambda b: ((int(b[:2]) - 5) % 24) * 60 + int(b[3:5]))
    hw_st, hw_ln = L9.headway_tables(tt, bins)
    sw = L9.switch_wait_table(tt)
    mod_c = {}
    spec = importlib.util.spec_from_file_location("v1s", ROOT / "scripts" / "09_route_scoring_prototype.py")
    v1s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1s)
    lk_med, lk_max, cong_long = L9.congestion_lookup(src, v1s.C0, v1s.KAPPA, express)
    v1_tr = pd.read_parquet(ROOT / "data" / "marts" / "transfer_edge_mart.parquet")
    vols = pd.read_csv(L9.locate(src, "서울교통공사_환승역환승인원정보_20251130.csv"), encoding="cp949")
    vols = pd.DataFrame({"station_name": vols.iloc[:, 1].map(L9.canon),
                         "weekday_volume": pd.to_numeric(vols.iloc[:, 2], errors="coerce")})
    tr = {}
    for key, f in L9.TRANSFER_FACTORS.items():
        tr[key], raw_tr = L9.transfer_edges(src, f, v1_tr, vols, express)
    for name, df in [("route_edges", redges), ("dwell", dwell), ("run", runtab), ("headway_station", hw_st),
                     ("headway_line", hw_ln), ("switch_wait", sw), ("congestion_median", lk_med),
                     ("congestion_max3y", lk_max)] + [(f"transfer_{k_}", v) for k_, v in tr.items()]:
        df.to_parquet(out9 / f"{name}.parquet", index=False)

    ov = dwell[(dwell.pattern == "9L") & (dwell.day_type == "weekday")]
    ovs = ov.groupby("station").apply(lambda g: pd.Series({
        "정차≥2분 bin 비율": (g.dwell_min >= 2).mean(), "최대 정차(분)": g.dwell_min.max()}),
        include_groups=False)
    ovs = ovs[ovs["정차≥2분 bin 비율"] > 0].sort_values("정차≥2분 bin 비율", ascending=False)
    hw8 = hw_st[(hw_st.day_type == "weekday") & (hw_st.time_bin == "08:00~08:30") & (hw_st.direction == "up")
                & hw_st.station_name.isin(["당산", "여의도", "고속터미널", "종합운동장", "가양"])]
    swc = sw[(sw.day_type == "weekday") & (sw.direction == "up") & (sw.from_line == "9L")
             & sw.time_bin.isin(["07:30~08:00", "08:00~08:30", "08:30~09:00"])]
    swc = swc.groupby("station").wait_min.mean().rename("일반→급행 연계 대기(분)").to_frame()
    xh = hw_st[(hw_st.line_id == "9X") & (hw_st.day_type == "weekday") & (hw_st.direction == "up")
               & hw_st.time_bin.isin(["07:30~08:00", "08:00~08:30", "08:30~09:00"])] \
        .groupby("station_name").expected_wait_min.mean().rename("급행 배차/2(분)")
    swc = swc.join(xh).round(2)
    L += ["## 3. 정차·배차·연계 대기", "",
          "일반열차 정차 2분 이상(급행 대피) 역 (평일, 30분 bin 기준):", "", ovs.round(2).to_markdown(), "",
          "평일 08:00~08:30 중앙보훈병원행 출발 열차 수:", "",
          hw8.pivot_table(index="station_name", columns="line_id", values="n_departures").to_markdown(), "",
          "일반 → 급행 갈아타기: 시각표 연계 대기 vs 급행 배차/2 (평일 07:30~09:00, 중앙보훈병원행):", "",
          swc.to_markdown(), ""]

    # 환승 보정 계수 재현
    d = pd.read_csv(L9.locate(src, L9.TRANSFER_FILE), encoding="cp949")
    d["sec"] = d["환승소요시간"].map(lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))
    d["from_line"], d["to_line"] = d["호선"].astype(str), d["환승노선"].str.extract(r"(\d)")[0]
    d["stn"] = d["환승역명"].map(L9.canon)
    m = v1_tr[v1_tr.edge_type == "transfer"].astype({"from_line": str, "to_line": str}).merge(
        d, left_on=["station_name", "from_line", "to_line"], right_on=["stn", "from_line", "to_line"])
    ratio = m.transfer_time_min / (m.sec / 60)
    t9 = raw_tr.assign(거리시간_분=(raw_tr.sec / 60).round(2))
    for key, f in L9.TRANSFER_FACTORS.items():
        t9[f"×{f:.2f}"] = (t9.sec / 60 * f).round(2)
    L += ["## 4. 환승 보행시간 (보정 proxy — 실측 아님, D-042)", "",
          f"- 원천: 환승역거리 ÷ 1.2m/s. 보행속도 실측 분포 평균 {(d['환승거리'] / d.sec).mean():.3f}m/s",
          f"- 보정 계수: 1~8호선 환승 {len(m)}쌍에서 v1 실측 ÷ 거리시간 중앙값 **{ratio.median():.2f}** "
          f"(IQR {ratio.quantile(.25):.2f}~{ratio.quantile(.75):.2f}) → 기본 1.81, sensitivity 1.00/1.60/2.78",
          "- v1 환승 데이터(수도권 도시철도 환승)의 9호선 값은 '10:00' 등 자리표시 값이라 쓰지 않음", "",
          t9[["station", "from_line", "환승거리", "거리시간_분"] + [c for c in t9.columns if c.startswith("×")]]
          .to_markdown(index=False), "",
          "- 일반↔급행 같은 역 갈아타기: 도보 0분 근사 (방향별 승강장 1개, D-046). 대기는 위 연계 대기 표", ""]

    # 혼잡
    nyear = cong_long.dropna(subset=["v"]).groupby("year").size()
    gap = (lk_max.congestion_median - lk_med.congestion_median)
    pk = lk_med[(lk_med.day_type == "weekday") & lk_med.time_bin_index.between(4, 6)]
    term = cong_long[(cong_long.station == "개화") & (cong_long.direction == "down") & (cong_long.pattern == "9L")]
    L += ["## 5. 혼잡도 (3개년 중앙값 = baseline, 3개년 최댓값 = max_3y 시나리오, D-045)", "",
          f"- 연도별 유효 셀: {nyear.to_dict()} · 0 값은 종착/미운행으로 결측 처리",
          f"- 방향 확인: 하선(=down, 개화행) 개화 역 값이 사실상 결측 → 종착역 = 개화 (상선 = 중앙보훈병원행 = 시각표 UP)"
          f" [{'일치' if (term.v.isna().mean() >= 0.95 and term.v.fillna(0).max() <= 5) else '불일치'}: "
          f"결측 {pct(term.v.isna().mean())}, 남은 값 최대 {term.v.max():.1f}%]",
          f"- max_3y − 중앙값: 중앙값 {gap.median():.1f}, p90 {gap.quantile(.9):.1f}%p",
          f"- 평일 07:30~09:00 최대 기대 혼잡 (중앙값 기준): 일반 {pk[pk.line_id == '9L'].congestion_median.max():.1f}, "
          f"급행 {pk[pk.line_id == '9X'].congestion_median.max():.1f}",
          "- 정원 기준: 9호선 100% = 6칸 922명(1칸 약 154명), 1~8호선 100% = 1칸 160명 → 같은 % 에서 9호선 인원이 약 4% 적음",
          "- 9호선은 p90 을 정의하지 않음 (congestion_p90 결측). p90 기반 분석은 9호선 구간을 제외하고 해석", ""]

    # ------------------------------------------------------------ 4. 수요 proxy 통일
    oa = pd.read_csv(L9.locate(src, OA_FILE), encoding="cp949")
    oa = oa[(oa["사용월"] >= args.oa_from) & (oa["사용월"] <= args.oa_to) & oa["호선명"].isin(OA_LINES)]
    oa["station_key"] = oa["지하철역"].map(canon_oa)
    hc = [c for c in oa.columns if "시-" in c]
    lg = oa.melt(id_vars=["사용월", "station_key"], value_vars=hc, var_name="c", value_name="n")
    lg["hour"] = lg.c.str[:2].astype(int).replace({0: 24, 1: 25})
    lg["kind"] = np.where(lg.c.str.contains("승차"), "boardings", "alightings")
    days = sum(pd.Period(str(ym), "M").days_in_month for ym in sorted(oa["사용월"].unique()))
    dem = lg.groupby(["station_key", "hour", "kind"]).n.sum().unstack("kind").reset_index()
    dem[["boardings", "alightings"]] = (dem[["boardings", "alightings"]] / days).round(1)
    disp = pd.read_csv(ROOT / "data" / "master" / "station_display_master.csv")
    disp9 = L9.display_rows(disp, sm, express)
    dem = dem[dem.station_key.isin(set(disp9.station_key))]
    dem["day_type"] = "all_days"
    dem["months"] = f"{args.oa_from}~{args.oa_to}"
    dem["n_days"] = days
    dem.to_parquet(ROOT / "data" / "marts" / "v2" / "demand_station_hour_oa.parquet", index=False)
    missing = sorted(set(disp9.station_key) - set(dem.station_key))

    # v2.3 가중 결과 변화
    spec = importlib.util.spec_from_file_location("s23", ROOT / "scripts" / "v2" / "23_build_od_shiftability.py")
    s23 = importlib.util.module_from_spec(spec)
    sys.modules["s23"] = s23
    spec.loader.exec_module(s23)
    evp = ROOT / "data" / "marts" / "v2" / "od_shiftability_evidence.pkl.gz"
    comp = None
    if evp.exists():
        with gzip.open(evp, "rb") as f:
            items = pickle.load(f)
        cl = [classify(it, T1) for it in items]
        keep = [c["type"] != "invalid" for c in cl]
        items = [it for it, kp in zip(items, keep) if kp]
        types = np.array([c["type"] for c, kp in zip(cl, keep) if kp])
        w_old = s23.demand_weights(items)
        w_new = s23.demand_weights(items, path=ROOT / "data" / "marts" / "v2" / "demand_station_hour_oa.parquet",
                                   day_type="all_days")
        comp = pd.DataFrame([{"유형 (혼잡 이동 중)": t,
                              "기존 proxy (서울교통공사 평일)": pct(s23.shares(types, w_old, True)[t]),
                              "OA-12252 (1~9호선 통일)": pct(s23.shares(types, w_new, True)[t]),
                              "변화": f"{100 * (s23.shares(types, w_new, True)[t] - s23.shares(types, w_old, True)[t]):+.1f}%p"}
                             for t in TYPES if t != "calm"])
        corr = np.corrcoef(w_old, w_new)[0, 1]
    L += ["## 6. 수요 proxy 통일 (OA-12252, D-041)", "",
          f"- 기간 {args.oa_from}~{args.oa_to} ({days}일), 대상 역 {dem.station_key.nunique()}개 "
          f"(9호선 포함), 매칭 안 된 역 {len(missing)}개{': ' + ', '.join(missing) if missing else ''}",
          "- OA-12252 는 월 합계라 평일/주말 구분 없음 → 일평균(전체 요일). 같은 시간대 안의 상대 가중치로만 사용", ""]
    if comp is not None:
        L += [f"- v2.3 단위별 가중치 상관 (기존 vs OA): {corr:.3f}", "", comp.to_markdown(index=False), ""]
    else:
        L += ["- v2.3 evidence 가 없어 비교 생략 (23 스크립트를 먼저 실행하면 비교됨)", ""]

    # ------------------------------------------------------------ 5. shadow root
    disp9.to_csv(out9 / "station_display_master_with9.csv", index=False, encoding="utf-8-sig")
    sm[["station", "code", "lat", "lon", "platform_type"]].to_csv(out9 / "station_latlon.csv", index=False,
                                                                  encoding="utf-8-sig")
    (out9 / "RUN_BY_BIN").unlink(missing_ok=True)
    if not rv["small"]:
        (out9 / "RUN_BY_BIN").write_text("v2_line9_run 사용 (D-040 변동성 기준 미달)", encoding="utf-8")
    for sc in SCENARIOS:
        L9.build_shadow_root(ROOT, ROOT / "data" / "interim" / "v26" / sc, sc)
    L += ["## 7. shadow root", "", "| 시나리오 | 환승 계수 | 9호선 혼잡 |", "|---|---|---|"] + \
         [f"| {sc} | {L9.TRANSFER_FACTORS[tf]:.2f} | {cong} |" for sc, (tf, cong) in SCENARIOS.items()] + \
         ["", "위치: `data/interim/v26/<시나리오>/` (1~8호선 mart + 9호선 mart 합본, 재생성 대상)", ""]

    # ------------------------------------------------------------ 6. 스모크
    from yeoyuro_v2 import load_v1
    from yeoyuro_v2.time_dependent import TDRouter, parse_hhmm
    mod, sr, _ = load_v1(ROOT)
    broot = ROOT / "data" / "interim" / "v26" / "base"
    router = TDRouter(broot, mod, sr, disp9, "weekday", policy="step", k=5, multi_bin=True)
    sm_rows = []
    for o, dd, tm in [("김포공항", "종합운동장", "08:00"), ("가양", "고속터미널", "08:00"),
                      ("여의도", "고속터미널", "08:10"), ("개화", "신논현", "07:40"), ("노원", "수유", "08:00")]:
        res = router.evaluate_od(o, dd, parse_hhmm(tm))
        for i, r in enumerate(res[:3]):
            lines = []
            for n_ in r.path:
                ln = n_.split("_", 1)[0]
                if not lines or lines[-1] != ln:
                    lines.append(ln)
            sm_rows.append({"OD": f"{o}→{dd} {tm}", "순위": i + 1, "노선": ">".join(lines),
                            "소요(분)": round(r.actual_time_min, 1), "정차(분)": round(r.dwell_time_min, 1),
                            "환승대기(분)": round(r.transfer_wait_min, 1), "100%+노출(분)": round(r.exposure[100], 1),
                            "최대혼잡": round(r.max_congestion, 1), "범위밖": r.out_of_window})
    L += ["## 8. 스모크 검사 (base, 평일, 시간 진행형)", "", pd.DataFrame(sm_rows).to_markdown(index=False), "",
          "## 9. 게이트", "",
          f"- A 역간 소요: {'통과' if (cv['시각표(2026-09) 분'] - cv['운영사(2026-02) 분']).abs().max() <= 2 else '확인 필요'} "
          "(두 출처 전 구간 차이 ≤ 2분)",
          "- B 혼잡 정합성: 조건부 통과 (정의 동일, 정원 4% 차이·집계 주기 차이 문서화, p90 미정의)",
          f"- C 배차: 통과 (9L/9X × 역 × 방향 × 요일 × 30분, 운행 bin {int(hw_st.is_operating.sum()):,}개)",
          "", f"실행 시간 {(time.time() - t0) / 60:.1f}분"]
    rp = ROOT / "reports" / "v2" / "line9_data_audit.md"
    rp.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n저장: {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
