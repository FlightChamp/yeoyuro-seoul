"""
05b_did_event_effect.py
=======================
이벤트 효과를 **대조군을 둔 이중차분(DiD-lite)** 으로 재추정한다.

왜 필요한가
-----------
05 의 spike_ratio 는 '같은 역의 평소 대비 배수'다. 대조군이 없는 단순 전후 비교라
그날 서울 전체 통행량이 많았거나 날씨가 좋았다면 배율이 부풀려진다.

    spike_ratio = 여의나루 당일 / 여의나루 평소                 (전후 비교)
    DiD         = (처치군 변화) - (대조군 변화)                  (순효과)

대조군 선정
-----------
1. 어떤 이벤트의 anchor/impact 역도 아닌 역
2. **spillover 제외**: 처치역과 같은 노선이거나 3정거장 이내면 뺀다.
   이벤트 인파가 인접역으로 번지면 대조군이 오염되어 DiD 가 과소추정된다.
3. 평소 이용 규모가 처치역과 가까운 역 N개 (log 총량 기준 최근접)
4. **평행추세 검증**: 이벤트 4주 전 구간에서 처치군과 대조군의 추세가 평행한지 본다.
   평행하지 않으면 DiD 가정이 깨지므로 `parallel_trend_ok = False` 로 표시한다.

absolute_lift 를 함께 낸다
--------------------------
비율만 보면 왜곡된다.
    평소 10명 -> 100명   : 10배지만 실제 증가 90명
    평소 3,000명 -> 6,000명 : 2배지만 실제 증가 3,000명
지하철 혼잡 관점에서는 후자가 더 중요할 수 있다.

출력
----
    data/marts/event_did_mart.parquet
    reports/model/event_did_report.md
    reports/model/event_did_control_groups.csv

사용법
------
    python scripts/05b_did_event_effect.py --root .
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ENC = "utf-8-sig"
N_CONTROL = 5              # 처치역당 대조군 수
SPILLOVER_CODE_GAP = 3     # 같은 노선에서 이 정거장 이내는 spillover 로 보고 제외
PRE_WEEKS = 4              # 평행추세 검증 구간
BASELINE_WEEKS = 8
PARALLEL_TOL = 0.15        # 사전 추세 차이가 이 비율 이내면 평행으로 본다


def to_date(v):
    if pd.isna(v) or str(v).strip() in ("", "없음", "nan"):
        return None
    return datetime.strptime(str(v).strip()[:10], "%Y-%m-%d").date()


def daterange(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


class DiDEstimator:

    def __init__(self, root: Path):
        self.root = root
        self.mart = root / "data" / "marts"
        self.master = root / "data" / "master"
        self.report = root / "reports" / "model"
        self.report.mkdir(parents=True, exist_ok=True)
        self.notes = []

    # ---------- 로드 ----------
    def load_events(self):
        ec = pd.read_csv(self.master / "event_calendar_master.csv")
        rows = []
        for r in ec.itertuples():
            s = to_date(r.impact_start_date) or to_date(r.official_start_date)
            e = to_date(r.impact_end_date) or to_date(r.official_end_date)
            if not s or not e:
                continue
            anchors = [x for x in str(r.anchor_stations).split(";") if x and x != "nan"]
            impacts = [x for x in str(r.impact_stations).split(";") if x and x != "nan"]
            rows.append({"event_id": r.event_id, "event_type": r.event_type,
                         "event_name": r.event_name, "start": s, "end": e,
                         "anchors": anchors, "impacts": impacts})
        ue = pd.read_csv(self.master / "university_event_master.csv")
        for r in ue.itertuples():
            s, e = to_date(r.exam_start_date), to_date(r.exam_end_date)
            if not s or not e:
                continue
            rows.append({"event_id": r.event_id, "event_type": r.event_type,
                         "event_name": "%s %s %s" % (r.university, r.academic_year, r.event_type),
                         "start": s, "end": e,
                         "anchors": [x for x in str(r.impact_stations).split(";") if x],
                         "impacts": []})
        ev = pd.DataFrame(rows)
        self.notes.append("이벤트 %d건 로드" % len(ev))
        return ev

    def load_daily(self):
        """일별 역 총량. 05 와 달리 방향 구분 없이 역 단위로 본다(대조군 매칭 목적)."""
        pq = self.mart / "ridership_hourly_mart.parquet"
        gz = self.mart / "ridership_hourly_mart.csv.gz"
        cols = ["date", "line_id", "station_uid", "station_name", "direction",
                "passenger_count"]
        parts = []
        if pq.exists():
            import pyarrow.parquet as pqm
            f = pqm.ParquetFile(pq)
            for b in f.iter_batches(batch_size=2_000_000, columns=cols):
                parts.append(self._agg(b.to_pandas()))
        elif gz.exists():
            for c in pd.read_csv(gz, usecols=cols, chunksize=2_000_000):
                parts.append(self._agg(c))
        else:
            raise FileNotFoundError("ridership_hourly_mart 없음. 01 을 먼저 실행하세요.")
        d = (pd.concat(parts, ignore_index=True)
             .groupby(["date", "line_id", "station_uid", "station_name"], as_index=False)
             ["passenger_count"].sum())
        d["date"] = pd.to_datetime(d["date"]).dt.date
        d["dow"] = pd.to_datetime(d["date"]).dt.day_of_week
        self.notes.append("일별 역총량 %d행 / 역 %d개" % (len(d), d["station_name"].nunique()))
        return d

    @staticmethod
    def _agg(df):
        return (df.groupby(["date", "line_id", "station_uid", "station_name"], as_index=False)
                ["passenger_count"].sum())

    # ---------- 대조군 ----------
    def build_control_pool(self, daily, ev):
        treated = set()
        for r in ev.itertuples():
            treated |= set(r.anchors) | set(r.impacts)
        scale = (daily.groupby(["station_name", "line_id"])["passenger_count"]
                 .mean().reset_index(name="mean_daily"))
        scale["station_code"] = pd.to_numeric(
            daily.groupby(["station_name", "line_id"])["station_uid"].first()
            .reset_index(drop=True).astype(str).str.split("_").str[-1], errors="coerce")
        scale = scale.merge(
            daily.groupby(["station_name", "line_id"])["station_uid"].first().reset_index(),
            on=["station_name", "line_id"], how="left")
        scale["station_code"] = pd.to_numeric(
            scale["station_uid"].astype(str).str.split("_").str[-1], errors="coerce")
        scale["is_treated"] = scale["station_name"].isin(treated)
        scale["log_scale"] = np.log1p(scale["mean_daily"])
        self.notes.append("처치역 %d개 / 후보 대조역 %d개"
                          % (int(scale["is_treated"].sum()), int((~scale["is_treated"]).sum())))
        return scale

    def pick_controls(self, scale, station_name):
        """규모가 가까우면서 spillover 위험이 없는 역 N개."""
        t = scale[scale["station_name"] == station_name]
        if t.empty:
            return [], "처치역이 승하차에 없음"
        t = t.iloc[0]
        pool = scale[~scale["is_treated"]].copy()
        # spillover 제외: 같은 노선이면서 역번호가 가까운 역
        same_line = (pool["line_id"].astype(str) == str(t["line_id"]))
        near = (pool["station_code"] - t["station_code"]).abs() <= SPILLOVER_CODE_GAP
        n_excl = int((same_line & near).sum())
        pool = pool[~(same_line & near)]
        pool["dist"] = (pool["log_scale"] - t["log_scale"]).abs()
        sel = pool.nsmallest(N_CONTROL, "dist")
        return sel["station_name"].tolist(), "spillover 제외 %d개" % n_excl

    # ---------- DiD ----------
    def estimate(self, daily, ev, scale):
        series = {}
        for st, g in daily.groupby("station_name"):
            series[st] = dict(zip(g["date"], g["passenger_count"]))

        def baseline(st, target, exclude_dates):
            s = series.get(st)
            if not s:
                return None, 0
            vals = []
            for k in range(1, BASELINE_WEEKS + 1):
                for d in (target - timedelta(weeks=k), target + timedelta(weeks=k)):
                    if d in exclude_dates:
                        continue
                    v = s.get(d)
                    if v is not None and not pd.isna(v):
                        vals.append(float(v))
            if len(vals) < 4:
                return None, len(vals)
            return float(np.median(vals)), len(vals)

        def pre_trend(st, target, exclude_dates):
            """이벤트 직전 PRE_WEEKS 주의 같은 요일 값 대비 baseline 비율 평균."""
            s = series.get(st)
            if not s:
                return None
            ratios = []
            for k in range(1, PRE_WEEKS + 1):
                d = target - timedelta(weeks=k)
                if d in exclude_dates:
                    continue
                v = s.get(d)
                b, _ = baseline(st, d, exclude_dates)
                if v is not None and b and b > 0:
                    ratios.append(float(v) / b)
            return float(np.mean(ratios)) if ratios else None

        # 이벤트일 집합 (baseline 오염 방지)
        ev_days = {}
        for r in ev.itertuples():
            for st in set(r.anchors) | set(r.impacts):
                ev_days.setdefault(st, set()).update(daterange(r.start, r.end))
        all_ev_days = set()
        for s in ev_days.values():
            all_ev_days |= s

        rows, ctrl_rows = [], []
        for r in ev.itertuples():
            for st in r.anchors:
                controls, note = self.pick_controls(scale, st)
                if not controls:
                    continue
                ctrl_rows.append({"event_id": r.event_id, "treated_station": st,
                                  "controls": ";".join(controls), "note": note})
                for d in daterange(r.start, r.end):
                    t_act = series.get(st, {}).get(d)
                    t_base, n_b = baseline(st, d, ev_days.get(st, set()))
                    if t_act is None or not t_base or t_base <= 0:
                        continue
                    t_ratio = float(t_act) / t_base

                    c_ratios, c_lifts = [], []
                    for c in controls:
                        c_act = series.get(c, {}).get(d)
                        c_base, _ = baseline(c, d, all_ev_days)
                        if c_act is None or not c_base or c_base <= 0:
                            continue
                        c_ratios.append(float(c_act) / c_base)
                        c_lifts.append(float(c_act) - c_base)
                    if not c_ratios:
                        continue
                    c_ratio = float(np.median(c_ratios))

                    # 평행추세 검증
                    t_pre = pre_trend(st, d, ev_days.get(st, set()))
                    c_pre = [pre_trend(c, d, all_ev_days) for c in controls]
                    c_pre = [x for x in c_pre if x is not None]
                    if t_pre is not None and c_pre:
                        gap = abs(t_pre - float(np.median(c_pre)))
                        parallel_ok = bool(gap <= PARALLEL_TOL)
                    else:
                        gap, parallel_ok = np.nan, None

                    rows.append({
                        "event_id": r.event_id, "event_type": r.event_type,
                        "event_name": r.event_name, "date": d, "station_name": st,
                        "n_controls": len(c_ratios),
                        "treated_actual": float(t_act),
                        "treated_baseline": round(t_base, 1),
                        "spike_ratio": round(t_ratio, 3),
                        "absolute_lift": round(float(t_act) - t_base, 1),
                        "control_ratio": round(c_ratio, 3),
                        "did_ratio": round(t_ratio - c_ratio, 3),
                        "did_ratio_normalized": round(t_ratio / c_ratio, 3) if c_ratio else np.nan,
                        "did_absolute_lift": round(
                            float(t_act) - t_base * c_ratio, 1),
                        "pre_trend_treated": round(t_pre, 3) if t_pre else np.nan,
                        "pre_trend_control": round(float(np.median(c_pre)), 3) if c_pre else np.nan,
                        "pre_trend_gap": round(gap, 3) if not pd.isna(gap) else np.nan,
                        "parallel_trend_ok": parallel_ok,
                        "n_baseline_days": n_b,
                    })
        return pd.DataFrame(rows), pd.DataFrame(ctrl_rows)

    # ---------- 리포트 ----------
    def write_report(self, did, ctrl, paths):
        L = []
        A = L.append
        A("# 이벤트 효과 이중차분(DiD-lite) 리포트\n")
        A("05 의 spike_ratio 는 대조군 없는 전후 비교다. 그날 서울 전체 통행량이 많았다면"
          " 배율이 부풀려진다. 여기서는 대조군을 두고 순효과를 분리한다.\n")
        A("```\nDiD = (처치군 변화) - (대조군 변화)\n```\n")

        A("## 1. 대조군 선정 규칙\n")
        A("1. 어떤 이벤트의 anchor/impact 역도 아닌 역")
        A("2. **spillover 제외**: 같은 노선에서 %d정거장 이내는 뺀다."
          " 인파가 인접역으로 번지면 대조군이 오염되어 효과가 과소추정된다." % SPILLOVER_CODE_GAP)
        A("3. 평소 이용 규모(log 일평균)가 가장 가까운 %d개" % N_CONTROL)
        A("4. **평행추세 검증**: 이벤트 직전 %d주 추세 차이가 %.2f 이내면 가정 충족\n"
          % (PRE_WEEKS, PARALLEL_TOL))

        A("## 2. 전후 비교 vs 이중차분\n")
        peak = (did.sort_values("spike_ratio", ascending=False)
                .groupby(["event_id", "event_name", "station_name"], as_index=False).first())
        top = peak.nlargest(15, "spike_ratio")[
            ["event_name", "station_name", "date", "spike_ratio", "control_ratio",
             "did_ratio_normalized", "absolute_lift", "did_absolute_lift",
             "parallel_trend_ok"]]
        A(top.to_markdown(index=False))
        A("")
        A("- `spike_ratio` 전후 비교 배율")
        A("- `control_ratio` 같은 날 대조군의 배율 (도시 전체 트렌드)")
        A("- `did_ratio_normalized` = spike / control — **순효과 배율**")
        A("- `absolute_lift` 실제 증가 인원, `did_absolute_lift` 트렌드 보정 후 증가 인원\n")

        A("## 3. 비율 vs 절대량\n")
        A("비율만 보면 왜곡된다. 평소 10명이 100명이 되면 10배지만 증가는 90명이고,"
          " 3,000명이 6,000명이 되면 2배지만 증가는 3,000명이다."
          " 지하철 혼잡 관점에서는 후자가 더 중요할 수 있다.\n")
        by_abs = peak.nlargest(10, "did_absolute_lift")[
            ["event_name", "station_name", "spike_ratio", "did_ratio_normalized",
             "did_absolute_lift"]]
        A("**절대 증가 인원 기준 상위 10**\n")
        A(by_abs.to_markdown(index=False))
        A("")

        A("## 4. 평행추세 가정\n")
        n_ok = int((did["parallel_trend_ok"] == True).sum())
        n_bad = int((did["parallel_trend_ok"] == False).sum())
        n_na = int(did["parallel_trend_ok"].isna().sum())
        A("- 충족 %d행 / 위반 %d행 / 판정불가 %d행" % (n_ok, n_bad, n_na))
        A("- 위반한 행은 DiD 가정이 깨진 것이므로 효과 추정을 신뢰하지 않는다.\n")
        if n_bad:
            bad = (did[did["parallel_trend_ok"] == False]
                   .nlargest(8, "spike_ratio")[
                       ["event_name", "station_name", "date", "spike_ratio",
                        "pre_trend_treated", "pre_trend_control", "pre_trend_gap"]])
            A("**평행추세 위반 사례**\n")
            A(bad.to_markdown(index=False))
            A("")

        A("## 5. 이벤트 유형별 순효과\n")
        g = (peak.groupby("event_type")
             .agg(n=("event_id", "size"),
                  median_spike=("spike_ratio", "median"),
                  median_did=("did_ratio_normalized", "median"),
                  median_abs_lift=("did_absolute_lift", "median"))
             .round(3))
        A(g.to_markdown())
        A("")
        A("> `median_spike` 보다 `median_did` 가 작으면, 전후 비교가 도시 전체 트렌드를"
          " 흡수해 과대추정하고 있었다는 뜻이다.\n")

        A("## 6. 대조군 구성\n")
        A(ctrl.head(20).to_markdown(index=False))
        A("")

        A("## 7. 한계\n")
        A("- 회귀 기반 DiD 가 아니라 비율 차분 방식(DiD-lite)이다. 표준오차를 내지 않는다.")
        A("- 대조군은 규모 유사성만으로 매칭했다. 상권 성격·요일 패턴은 통제하지 않았다.")
        A("- 평행추세 판정 임계 %.2f 는 근거 있는 상수가 아니라 초기값이다." % PARALLEL_TOL)
        A("- spillover 는 같은 노선 %d정거장 기준으로만 제외했다."
          " 환승으로 이어진 다른 노선 역은 통제하지 못한다.\n" % SPILLOVER_CODE_GAP)

        if self.notes:
            A("## 8. 처리 노트\n")
            for n in self.notes:
                A("- " + n)
            A("")
        A("## 9. 산출물\n")
        for k, v in paths.items():
            A("- `%s` : %s" % (k, v))

        p = self.report / "event_did_report.md"
        p.write_text("\n".join(L), encoding=ENC)
        return p

    # ---------- 실행 ----------
    def run(self):
        ev = self.load_events()
        daily = self.load_daily()
        scale = self.build_control_pool(daily, ev)
        did, ctrl = self.estimate(daily, ev, scale)
        if did.empty:
            print("DiD 결과가 비어 있습니다. 이벤트 역이 승하차에 있는지 확인하세요.")
            return

        paths = {}
        try:
            p = self.mart / "event_did_mart.parquet"
            did.to_parquet(p, index=False)
        except Exception:
            p = self.mart / "event_did_mart.csv.gz"
            did.to_csv(p, index=False, encoding=ENC, compression="gzip")
        paths["event_did_mart"] = p
        p2 = self.report / "event_did_control_groups.csv"
        ctrl.to_csv(p2, index=False, encoding=ENC)
        paths["control_groups"] = p2
        paths["report"] = self.write_report(did, ctrl, paths)

        peak = (did.sort_values("spike_ratio", ascending=False)
                .groupby(["event_id", "event_name", "station_name"], as_index=False).first())
        print("=" * 100)
        print(" 05b_did_event_effect - 완료")
        print("=" * 100)
        for n in self.notes:
            print("  " + n)
        print("\n  DiD 행 %d / 이벤트-역 조합 %d" % (len(did), len(peak)))
        print("  평행추세 충족 %d / 위반 %d / 판정불가 %d"
              % (int((did.parallel_trend_ok == True).sum()),
                 int((did.parallel_trend_ok == False).sum()),
                 int(did.parallel_trend_ok.isna().sum())))
        print("\n[전후 비교 vs DiD - 상위 12]")
        print(peak.nlargest(12, "spike_ratio")[
            ["event_name", "station_name", "spike_ratio", "control_ratio",
             "did_ratio_normalized", "absolute_lift", "did_absolute_lift",
             "parallel_trend_ok"]].to_string(index=False))
        print("\n[이벤트 유형별]")
        print((peak.groupby("event_type")
               .agg(n=("event_id", "size"), median_spike=("spike_ratio", "median"),
                    median_did=("did_ratio_normalized", "median"),
                    median_abs_lift=("did_absolute_lift", "median")).round(3)).to_string())
        print("\n[생성 파일]")
        for k, v in paths.items():
            print("  %-20s %s" % (k, v))
        print("=" * 100)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    DiDEstimator(Path(args.root)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
