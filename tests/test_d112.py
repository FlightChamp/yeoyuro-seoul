"""
tests/test_d112.py — D-112a: 3-B 정차표 실행 모드·공개 정책·비공개 내려받기·노선 부분 집합 (사전 등록 preregistration_d112.md)

가짜 토큰·합성 정차표·모의 내려받기 응답만 쓴다 (D-112 6절). 실제 저장소·토큰·네트워크는 쓰지 않는다.

| 묶음 | 검사 |
|---|---|
| 모드 | local 은 환경 변수·표시 파일로만, 그 밖은 public (fail-closed) |
| 정책 | 배포 정책 파일은 모든 노선 꺼짐 · C1~C3·GC→GJ 의존 · 운영사·노선 짝이 틀리면 꺼짐 |
| 내려받기 | 성공·해시·인증·크기·예외·토큰 없음 — 토큰은 헤더에만, 반환값·로그에 없음 · 꺼진 운영사는 요청 0 |
| 검사 | 실제 정차표 운영사별 통과 · 열·코드·시각 역행·운영사 섞임 실패 |
| 계산기 | 공개 모드에서 받은 자료가 없으면 로컬 파일이 있어도 3-B 끔 · 부분 집합 경로에 꺼진 노선 노드 없음 · 전부 = 로컬 |
| 앱 | 공개 모드·Secrets 없음 → 3-A 만 (로컬 파일이 있어도) · SB 만 켬 → SB 역만, 안내 문구 |
| git | 표시 파일·꾸러미 폴더 무시 |
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app" / "streamlit")]
from yeoyuro_v2 import data_release as DR       # noqa: E402
from yeoyuro_v2 import korail_app as KA          # noqa: E402

DATA9 = ROOT / "data" / "app_v26"
APP = ROOT / "app" / "streamlit" / "yeoyuro_seoul_app.py"
HAS_APP = (DATA9 / "data" / "marts" / "route_edge_mart.parquet").exists()
HAS_3B = KA.k3b_ready(ROOT)
need_app = pytest.mark.skipif(not HAS_APP, reason="app_v26 데이터 없음")
need_3b = pytest.mark.skipif(not (HAS_APP and HAS_3B), reason="3-B 상세 정차표 없음 (P4)")
FAKE = "ghp_FAKE_TOKEN_d112_DO_NOT_USE"


# ---------------------------------------------------------------------- 합성 자료
def _synthetic(op: str) -> pd.DataFrame:
    from yeoyuro_v2 import korail_3b as K3
    rows = []
    groups = sorted(DR.OPERATOR_GROUPS[op])
    for dtp in sorted(DR.DAY_TYPES):
        for g in groups:
            codes = [c for _, c in list(K3.AR_CODE.values())[:3]] if g == "AR" else list(K3.CODE.values())[:3]
            for i, c in enumerate(codes):
                rows.append(dict(day_type=dtp, group=g, line=g, train=f"T{g}1", direction="UP", stop_seq=i + 1,
                                 code=int(c), arr_min=600.0 + 3 * i, dep_min=600.5 + 3 * i))
    df = pd.DataFrame(rows)
    for c in ("day_type", "group", "line", "train", "direction"):
        df[c] = df[c].astype("string")
    return df[list(DR.COLUMNS)]


def _bytes(df: pd.DataFrame) -> bytes:
    b = io.BytesIO()
    df.to_parquet(b, index=False)
    return b.getvalue()


def _policy(enabled=("GJ", "GC", "SB", "AR"), korail: bytes | None = None, arex: bytes | None = None, **over) -> DR.Policy:
    ops = {"korail": {"groups": ["GJ", "GC", "SB"], "repo": "o/k", "path": "k.parquet", "ref": "main",
                      "sha256": hashlib.sha256(korail or b"").hexdigest(), "rows": 0},
           "arex": {"groups": ["AR"], "repo": "o/a", "path": "a.parquet", "ref": "main",
                    "sha256": hashlib.sha256(arex or b"").hexdigest(), "rows": 0}}
    grs = {g: {"operator": "arex" if g == "AR" else "korail", "enabled": g in enabled, "status": "허용",
               "decision": "D-TEST", "evidence": "합성 시험", "trains": 0} for g in ("GJ", "GC", "SB", "AR")}
    grs["GC"]["depends_on"] = ["GJ"]
    for k, v in over.items():
        grs[k].update(v)
    return DR.Policy(ops, grs, [])


class _Http:
    """모의 HTTP — 요청을 기록하고 정해 둔 응답을 돌려준다."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, url, headers, timeout, max_bytes):
        self.calls.append((url, dict(headers)))
        r = self.table[url.split("/contents/")[1].split("?")[0]]
        if isinstance(r, Exception):
            raise r
        return r


SECRETS = {"k3b_remote": {"korail": {"token": FAKE}, "arex": {"token": FAKE}}}


# ---------------------------------------------------------------------- 모드
def test_mode_is_public_unless_explicit(tmp_path):
    assert DR.resolve_mode(tmp_path, environ={}) == "public"                       # 아무것도 없으면 공개
    assert DR.resolve_mode(tmp_path, environ={DR.ENV_MODE: "LOCAL"}) == "local"
    assert DR.resolve_mode(tmp_path, environ={DR.ENV_MODE: "dev"}) == "public"     # 모르는 값 = 공개
    (tmp_path / DR.LOCAL_FLAG).write_text("x")
    assert DR.resolve_mode(tmp_path, environ={}) == "local"
    assert DR.resolve_mode(tmp_path, environ={DR.ENV_MODE: "public"}) == "public"  # 환경 변수가 표시 파일보다 앞섬


def test_public_mode_ignores_local_file(tmp_path):
    (tmp_path / "data" / "marts" / "v2" / "korail_3b").mkdir(parents=True)
    (tmp_path / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet").write_bytes(b"x")
    src = DR.resolve_k3b(tmp_path, environ={}, policy_path=ROOT / "config" / "data_release.toml",
                         http_get=lambda *a: pytest.fail("요청하면 안 됨"))
    assert src.mode == "public" and src.groups == frozenset() and src.stop_times is None and src.requested == ()
    loc = DR.resolve_k3b(tmp_path, environ={DR.ENV_MODE: "local"})
    assert loc.mode == "local" and loc.groups == DR.ALL_GROUPS


# ---------------------------------------------------------------------- 정책
def test_shipped_policy_all_disabled():
    pol = DR.load_policy()
    assert not pol.problems
    g, why = DR.policy_groups(pol)
    assert g == frozenset() and set(why) == DR.ALL_GROUPS and set(why.values()) == {"policy"}
    for s in pol.groups.values():
        assert s["enabled"] is False and s["status"] == "보류" and s["decision"] == "D-107"
    for s in pol.operators.values():
        assert s["repo"] == "" and s["sha256"] == ""
    text = (ROOT / "config" / "data_release.toml").read_text(encoding="utf-8")
    assert "token =" not in text and "ghp_" not in text                         # 정책 파일에 토큰 자리 없음
    assert FAKE not in (ROOT / "config" / "data_release.toml").read_text(encoding="utf-8")


def test_policy_rules():
    g, _ = DR.policy_groups(_policy())
    assert g == DR.ALL_GROUPS
    g, why = DR.policy_groups(_policy(enabled=("GC", "SB")))                     # GC 는 GJ 없이 못 켬
    assert g == {"SB"} and why["GC"] == "policy"
    g, why = DR.policy_groups(_policy(GJ={"status": "보류"}))                   # C1
    assert "GJ" not in g and "GC" not in g
    g, _ = DR.policy_groups(_policy(SB={"evidence": ""}))                        # C2
    assert "SB" not in g
    g, _ = DR.policy_groups(_policy(AR={"decision": " "}))
    assert "AR" not in g
    bad = DR.Policy(*(lambda p: (p.operators, p.groups))(_policy()), ["operator:korail"])
    g, why = DR.policy_groups(bad)
    assert g == {"AR"} and why["GJ"] == "config"


def test_policy_file_problems(tmp_path):
    f = tmp_path / "p.toml"
    f.write_text('[operators.korail]\ngroups = ["GJ", "AR"]\n[groups.AR]\noperator = "korail"\n', encoding="utf-8")
    pol = DR.load_policy(f)
    assert "operator:korail" in pol.problems and "group:AR" in pol.problems
    assert DR.load_policy(tmp_path / "없음.toml").problems == ["config"]


# ---------------------------------------------------------------------- 내려받기
def test_fetch_ok_and_token_only_in_header(caplog):
    body = _bytes(_synthetic("korail"))
    http = _Http({"k.parquet": (200, body)})
    spec = _policy(korail=body).operators["korail"]
    out, code = DR.fetch_operator("korail", spec, FAKE, http)
    assert code == "ok" and out == body
    url, headers = http.calls[0]
    assert url.startswith("https://api.github.com/repos/o/k/contents/") and FAKE not in url
    assert headers["Authorization"] == f"Bearer {FAKE}"


@pytest.mark.parametrize("resp,code", [((401, b""), "auth"), ((404, b""), "auth"), ((500, b""), "network"),
                                       ((200, b"other"), "hash"), ((200, b"x" * (DR.MAX_BYTES + 1)), "size"),
                                       (RuntimeError(f"boom {FAKE}"), "network"), (TimeoutError(FAKE), "timeout")])
def test_fetch_failures_hide_token(resp, code, caplog):
    caplog.set_level(logging.DEBUG)
    spec = _policy(korail=b"good").operators["korail"]
    out, got = DR.fetch_operator("korail", spec, FAKE, _Http({"k.parquet": resp}))
    assert out is None and got == code and got in DR.CODES
    assert FAKE not in caplog.text


def test_fetch_config_and_secret():
    spec = _policy(korail=b"x").operators["korail"]
    assert DR.fetch_operator("korail", {**spec, "repo": ""}, FAKE, _Http({}))[1] == "config"
    assert DR.fetch_operator("korail", {**spec, "sha256": "abc"}, FAKE, _Http({}))[1] == "config"
    assert DR.fetch_operator("korail", spec, None, _Http({}))[1] == "secret"


def test_load_public_requests_only_enabled_operators(caplog):
    kb, ab = _bytes(_synthetic("korail")), _bytes(_synthetic("arex"))
    http = _Http({"k.parquet": (200, kb), "a.parquet": (200, ab)})
    src = DR.load_public(_policy(enabled=("SB",), korail=kb, arex=ab), SECRETS, http)
    assert src.groups == {"SB"} and src.requested == ("korail",) and len(http.calls) == 1
    assert set(src.stop_times.group.unique()) == {"SB"}                        # 켜진 노선 행만
    src = DR.load_public(_policy(enabled=(), korail=kb, arex=ab), SECRETS, http)
    assert src.groups == frozenset() and src.requested == () and len(http.calls) == 1
    src = DR.load_public(_policy(korail=kb, arex=ab), {}, http)                 # Secrets 없음
    assert src.groups == frozenset() and set(src.reasons.values()) == {"secret"}
    assert FAKE not in caplog.text


def test_load_public_partial_failure_and_mixing(caplog):
    kb, ab = _bytes(_synthetic("korail")), _bytes(_synthetic("arex"))
    mixed = _bytes(pd.concat([_synthetic("korail"), _synthetic("arex")], ignore_index=True))
    http = _Http({"k.parquet": (200, mixed), "a.parquet": (200, ab)})
    src = DR.load_public(_policy(korail=mixed, arex=ab), SECRETS, http)        # 코레일 꾸러미에 AR 섞임
    assert src.groups == {"AR"} and src.reasons["GJ"] == "schema" and src.reasons["SB"] == "schema"
    http = _Http({"k.parquet": (200, kb), "a.parquet": (401, b"")})
    src = DR.load_public(_policy(korail=kb, arex=ab), SECRETS, http)
    assert src.groups == {"GJ", "GC", "SB"} and src.reasons == {"AR": "auth"}
    assert FAKE not in caplog.text and "T" + "GJ1" not in caplog.text          # 정차표 행 내용도 로그에 없음


# ---------------------------------------------------------------------- 데이터 검사
def test_validate_synthetic():
    k = _synthetic("korail")
    assert DR.validate_stop_times(k, "korail") == []
    assert DR.validate_stop_times(k, "arex") == ["group"]
    assert DR.validate_stop_times(k.drop(columns="code"), "korail") == ["columns"]
    x = k.copy()
    x.loc[0, "code"] = 999999
    assert "code" in DR.validate_stop_times(x, "korail")
    x = k.copy()
    x.loc[x.index[1], ["arr_min", "dep_min"]] = 100.0                           # 시각 역행
    assert "time_order" in DR.validate_stop_times(x, "korail")
    assert "rows" in DR.validate_stop_times(k, "korail", expect_rows=len(k) + 1)
    assert "trains" in DR.validate_stop_times(k, "korail", expect_trains={"GJ": 2})
    assert "day_type" in DR.validate_stop_times(k[k.day_type != "sunday"], "korail")


@pytest.mark.skipif(not HAS_3B, reason="3-B 상세 정차표 없음 (P4)")
def test_validate_real_split():
    m = __import__("importlib").util
    spec = m.spec_from_file_location("m57", ROOT / "scripts" / "v2" / "57_split_k3b_packages.py")
    m57 = m.module_from_spec(spec)
    spec.loader.exec_module(m57)
    st = pd.read_parquet(KA_K3B())
    parts = m57.split(st)
    assert sum(len(x) for x in parts.values()) == len(st)
    for op, df in parts.items():
        assert DR.validate_stop_times(df, op) == [], op
        assert set(df.group.unique()) == set(DR.OPERATOR_GROUPS[op])


def KA_K3B() -> Path:
    return ROOT / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet"


# ---------------------------------------------------------------------- 계산기
@pytest.fixture(scope="module")
def parts():
    import station_routing as sr
    from yeoyuro_v2.line9 import register_v1_directions
    mod = sr.load_scorer_class(ROOT)
    register_v1_directions(mod)
    disp = pd.read_csv(DATA9 / "data" / "master" / "station_display_master.csv")
    return mod, sr, disp


def _planner(parts, **kw):
    mod, sr, disp = parts
    return KA.ExtensionPlanner(DATA9, mod, sr, disp, "weekday", query_date="2026-10-14", repo_root=ROOT, **kw)


@need_app
def test_planner_public_without_data_never_reads_local(parts, monkeypatch):
    real = pd.read_parquet

    def guard(p, *a, **k):
        assert "korail_3b/stop_times" not in str(p).replace("\\", "/"), "공개 모드가 로컬 3-B 파일을 읽음"
        return real(p, *a, **k)
    monkeypatch.setattr(pd, "read_parquet", guard)
    p = _planner(parts, include_3b=True, k3b_stop_times=None, k3b_local=False)
    assert p.groups_3b == frozenset() and p.include_3b is False and p.on.status_3b == "off"


ODS = [("회기", "수서"), ("공덕", "청량리"), ("선릉", "모란"), ("김포공항", "서울역"), ("상봉", "용산")]


def _nodes_ok(path, G) -> bool:
    for n in path:
        pre = n.split("_", 1)[0]
        if pre in ("GJ", "GC", "SB", "AR") and pre not in G:
            return False
    return True


@need_3b
@pytest.mark.parametrize("G", [frozenset({"GJ"}), frozenset({"SB"}), frozenset({"AR"}),
                               frozenset({"GJ", "GC", "SB"}), frozenset()])
def test_subset_paths_only_enabled_lines(parts, G):
    from yeoyuro_v2 import korail_app as K
    st = pd.read_parquet(KA_K3B())
    p = _planner(parts, include_3b=G, k3b_stop_times=st, k3b_local=False)
    assert p.groups_3b == G and p.include_3b == bool(G)
    nodes = set(p.on.pieces.korail_nodes)
    assert all(n.split("_", 1)[0] in G | {"1"} or n.startswith("1_") for n in nodes), sorted(nodes)[:5]
    for o, d in ODS:
        r = p.plan(o, d, K.service_day("2026-10-14", "08:00"))
        for c in r.candidates:
            assert _nodes_ok(c.path, G), (o, d, c.path)
    disp = parts[2]
    x = K.extended_display(disp, G)
    lines = {ln for v in x.available_lines.astype(str) for ln in v.split(",")}
    assert lines & {"GJ", "SB", "AR"} <= G
    uids = {u for v in x.candidate_station_uids.astype(str) for u in v.split(";")}
    assert not any(u.startswith("GC_") for u in uids) or "GC" in G


@need_3b
def test_subset_gc_only_is_reported_not_crashing(parts):
    """GC 만 켜기는 정책이 거부한다 (D-112 5절). 엔진에 넣었을 때 멈추지 않는지만 본다 — 경춘 열차가 GJ_ 노드를 만든다."""
    st = pd.read_parquet(KA_K3B())
    p = _planner(parts, include_3b={"GC"}, k3b_stop_times=st, k3b_local=False)
    nodes = set(p.on.pieces.korail_nodes)
    assert {"GC_상봉", "GC_망우"} <= nodes and "GJ_회기" in nodes and not any(n.startswith("SB_") for n in nodes)


@need_3b
def test_all_given_equals_local(parts):
    from yeoyuro_v2 import korail_app as K
    st = pd.read_parquet(KA_K3B())
    a = _planner(parts, include_3b=True)
    b = _planner(parts, include_3b=KA.ALL_3B, k3b_stop_times=st, k3b_local=False)
    for o, d in ODS:
        sd = K.service_day("2026-10-14", "08:00")
        ra, rb = a.plan(o, d, sd), b.plan(o, d, sd)
        key = lambda r: (r.status, None if r.best is None else tuple(r.best.path),           # noqa: E731
                         None if r.best is None else round(r.best.expected_min, 6),
                         None if getattr(r, "reference", None) is None else tuple(r.reference.path))
        assert key(ra) == key(rb), (o, d)


# ---------------------------------------------------------------------- 앱
def _route(at):
    at.session_state["nav_page"] = "경로 추천"
    at.session_state["view_route"] = "경로"


def _fresh_cache():
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()


@need_app
def test_app_public_without_secrets_is_3a_only(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv(DR.ENV_MODE, "public")
    _fresh_cache()
    try:
        at = AppTest.from_file(str(APP), default_timeout=600)
        _route(at)
        at.run()
        at.toggle(key="ext_on").set_value(True)
        at.run()
        assert not at.exception
        assert KA.K3B_MISSING in " ".join(c.value for c in at.caption)
        opts = " ".join(at.selectbox(key="sel_origin").options)
        assert "월계" in opts and "야탑" not in opts and "서빙고" not in opts         # 3-A 는 있고 3-B 는 없음
    finally:
        _fresh_cache()


@need_3b
def test_app_public_sb_only(monkeypatch):
    from streamlit.testing.v1 import AppTest
    st = pd.read_parquet(KA_K3B())
    fake = DR.K3BSource("public", frozenset({"SB"}), st[st.group == "SB"].reset_index(drop=True), {}, ("korail",))
    monkeypatch.setenv(DR.ENV_MODE, "public")
    monkeypatch.setattr(DR, "resolve_k3b", lambda *a, **k: fake)
    _fresh_cache()
    try:
        at = AppTest.from_file(str(APP), default_timeout=900)
        _route(at)
        at.run()
        at.toggle(key="ext_on").set_value(True)
        at.run()
        assert not at.exception
        caps = " ".join(c.value for c in at.caption)
        assert KA.K3B_PARTIAL in caps and "수인분당선" in caps and KA.K3B_MISSING not in caps
        opts = " ".join(at.selectbox(key="sel_origin").options)
        assert "야탑" in opts and "서빙고" not in opts and "양원" not in opts          # SB 역만
        spec = at.get("plotly_chart")[0].proto.spec.replace(" ", "")
        assert '"dash":"dot"' in spec                                             # 꺼진 GJ·AR 은 점선으로 남음
        at.selectbox(key="sel_origin").set_value("왕십리")
        at.selectbox(key="sel_dest").set_value("야탑")
        at.date_input[0].set_value(dt.date(2026, 10, 14))
        at.time_input[0].set_value(dt.time(8, 0))
        at.run()
        [b for b in at.button if b.label == "경로 검색"][0].click()
        at.run()
        assert not at.exception
        mds = " ".join(m.value for m in at.markdown)
        assert "경의중앙" not in mds.split("광역철도 포함 경로")[-1]               # 확장 카드에 꺼진 노선 없음
    finally:
        _fresh_cache()


# ---------------------------------------------------------------------- git
def test_gitignore_private_paths():
    for p in (DR.LOCAL_FLAG, "data/private/k3b_packages/korail_3b_korail.parquet",
              "data/marts/v2/korail_3b/stop_times.parquet"):
        r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", p])
        assert r.returncode == 0, p
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "data/marts/v2/korail_3b", "data/private"],
                             capture_output=True, text=True).stdout.split()
    assert tracked == []
