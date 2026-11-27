"""
yeoyuro_v2/data_release.py — 3-B 정차표 실행 모드·공개 정책·비공개 내려받기 (D-112, 구현 D-112a)
=================================================================================================

사전 등록: docs/v2/preregistration_d112.md. 이 모듈은 이용허락을 판단하지 않는다 — 정책 파일에 적힌 결정만 따른다.

| 모드 | 정하는 방법 | 3-B 자료 |
|---|---|---|
| local (로컬 개발) | 환경 변수 YEOYURO_DATA_MODE=local, 또는 저장소 맨 위 표시 파일 `.yeoyuro_local_dev` (git 무시) | 로컬 정차표 파일 그대로 (지금 동작) |
| public (기본) | 그 밖의 모든 경우 | config/data_release.toml 의 C1~C6 을 모두 통과한 노선만, 비공개 저장소에서 받음 |

fail-closed: public 모드는 로컬 3-B 파일을 열지 않는다. Secrets 가 없거나, 받기·해시·검사가 실패하면 그 노선만 끈다.
노출 방지: 실패는 일반 코드(auth·network·timeout·size·hash·schema·policy·config)로만 남긴다. 토큰·응답 본문·정차표 행·
예외 원문은 로그·반환값·화면 어디에도 넣지 않는다. 토큰은 HTTP 헤더에만 넣는다 (URL 에 넣지 않음).
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

LOG = logging.getLogger("yeoyuro.data_release")
ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "config" / "data_release.toml"
LOCAL_FLAG = ".yeoyuro_local_dev"
ENV_MODE = "YEOYURO_DATA_MODE"
ALL_GROUPS = frozenset({"GJ", "GC", "SB", "AR"})
OPERATOR_GROUPS = {"korail": frozenset({"GJ", "GC", "SB"}), "arex": frozenset({"AR"})}
API_HOST = "api.github.com"
MAX_BYTES = 5 * 2**20            # 정차표 전체가 약 120KB (2026-11-28) — 넉넉한 상한
TIMEOUT_SEC = 15
COLUMNS = {"day_type": "str", "group": "str", "line": "str", "train": "str", "direction": "str",
           "stop_seq": "int", "code": "int", "arr_min": "float", "dep_min": "float"}
DAY_TYPES = {"weekday", "saturday", "sunday"}
CODES = ("auth", "network", "timeout", "size", "hash", "schema", "policy", "config", "secret")


# ---------------------------------------------------------------------- 모드
def resolve_mode(root: Path | None = None, environ: Mapping[str, str] | None = None) -> str:
    """'local' 은 명시적으로 켤 때만. 값이 이상하거나 없으면 'public' (fail-closed)."""
    env = os.environ if environ is None else environ
    v = str(env.get(ENV_MODE, "")).strip().lower()
    if v == "local":
        return "local"
    if v == "public":
        return "public"
    return "local" if (Path(root or ROOT) / LOCAL_FLAG).is_file() else "public"


# ---------------------------------------------------------------------- 정책
@dataclass
class Policy:
    operators: dict
    groups: dict
    problems: list = field(default_factory=list)      # 정책 파일 자체의 문제 (있으면 그 항목은 꺼짐)


def load_policy(path: Path | None = None) -> Policy:
    """정책 파일을 읽는다. 읽지 못하면 빈 정책 (모든 노선 꺼짐)."""
    try:
        raw = tomllib.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001 — 파일 없음·문법 오류: 전부 끔
        LOG.warning("data_release policy=config")
        return Policy({}, {}, ["config"])
    ops, grs = dict(raw.get("operators", {})), dict(raw.get("groups", {}))
    problems = []
    for op, spec in ops.items():
        if op not in OPERATOR_GROUPS or set(spec.get("groups", [])) != set(OPERATOR_GROUPS[op]):
            problems.append(f"operator:{op}")
    for g, spec in grs.items():
        if g not in ALL_GROUPS or g not in OPERATOR_GROUPS.get(spec.get("operator", ""), ()):
            problems.append(f"group:{g}")
    return Policy(ops, grs, problems)


def policy_groups(pol: Policy) -> tuple[frozenset, dict]:
    """C1·C2·C3 와 의존성으로 켤 수 있는 노선 → (노선 집합, {노선: 끈 이유 코드})."""
    why: dict = {}
    ok = set()
    for g in sorted(ALL_GROUPS):
        s = pol.groups.get(g)
        if s is None or f"group:{g}" in pol.problems or f"operator:{s.get('operator')}" in pol.problems:
            why[g] = "config"
        elif not (s.get("enabled") is True and s.get("status") == "허용" and str(s.get("decision", "")).strip()
                  and str(s.get("evidence", "")).strip()):
            why[g] = "policy"
        else:
            ok.add(g)
    for g in sorted(ok):                                  # depends_on (GC → GJ)
        dep = set(pol.groups[g].get("depends_on", []))
        if not dep <= ok:
            ok.discard(g)
            why[g] = "policy"
    return frozenset(ok), why


# ---------------------------------------------------------------------- 데이터 검사 (C6)
def _station_codes(groups) -> set:
    from yeoyuro_v2 import korail_3b as K3
    out = set()
    if set(groups) & {"GJ", "GC", "SB"}:
        out |= {int(c) for c in K3.CODE.values()}
    if "AR" in groups:
        out |= {int(c) for _, c in K3.AR_CODE.values()}
    return out


def validate_stop_times(df: pd.DataFrame, operator: str, expect_rows: int | None = None,
                        expect_trains: Mapping[str, int] | None = None) -> list[str]:
    """정차표 검사 (D-112 4절). 문제 목록을 돌려준다 — 내용(행)은 넣지 않고 항목 이름만."""
    bad = []
    if list(df.columns) != list(COLUMNS):
        return ["columns"]
    kinds = {"str": pd.api.types.is_string_dtype, "int": pd.api.types.is_integer_dtype,
             "float": pd.api.types.is_float_dtype}
    for c, k in COLUMNS.items():
        if not (kinds[k](df[c]) or (k == "str" and df[c].dtype == object)):
            bad.append(f"dtype:{c}")
    if bad:
        return bad
    allowed = OPERATOR_GROUPS.get(operator)
    if allowed is None or not set(df.group.unique()) <= set(allowed):
        return ["group"]                                   # 다른 운영사 노선이 섞이면 전체를 쓰지 않음
    if set(df.day_type.unique()) != DAY_TYPES:        # 휴일 시각표는 토·일 두 벌로 들어 있음 (3B-1)
        bad.append("day_type")
    if not set(df.code.astype(int).unique()) <= _station_codes(allowed):
        bad.append("code")
    t = df[["arr_min", "dep_min"]]
    if ((t < 0) | (t > 30 * 60)).any().any():
        bad.append("time_range")
    x = df.sort_values(["day_type", "group", "train", "direction", "stop_seq"])
    tm = x.dep_min.fillna(x.arr_min)
    prev = tm.groupby([x.day_type, x.group, x.train, x.direction]).shift()
    if ((tm < prev) & prev.notna() & tm.notna()).any():
        bad.append("time_order")
    if expect_rows is not None and len(df) != int(expect_rows):
        bad.append("rows")
    if expect_trains:
        got = df.groupby("group").train.nunique().to_dict()
        if any(int(got.get(g, 0)) != int(n) for g, n in expect_trains.items()):
            bad.append("trains")
    return bad


# ---------------------------------------------------------------------- 내려받기 (C4·C5)
def _urllib_get(url: str, headers: dict, timeout: float, max_bytes: int) -> tuple[int, bytes]:
    """기본 HTTP GET — (상태 코드, 본문 앞부분 max_bytes+1 바이트). 예외는 호출한 쪽이 코드로 바꾼다."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 — https·정해진 호스트만 (fetch_operator)
            return int(r.status), r.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        return int(e.code), b""


def fetch_operator(op: str, spec: Mapping, token: str | None,
                   http_get: Callable[[str, dict, float, int], tuple[int, bytes]] | None = None) -> tuple[bytes | None, str]:
    """운영사 꾸러미 하나 → (바이트 또는 None, 코드). 코드는 'ok' 또는 CODES 중 하나. 토큰·본문은 밖으로 내지 않는다."""
    repo, path, ref, sha = (str(spec.get(k, "")).strip() for k in ("repo", "path", "ref", "sha256"))
    if not repo or "/" not in repo or not path or len(sha) != 64:
        return None, "config"
    if not token:
        return None, "secret"
    url = f"https://{API_HOST}/repos/{repo}/contents/{path}" + (f"?ref={ref}" if ref else "")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.raw",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "yeoyuro-seoul"}
    try:
        status, body = (http_get or _urllib_get)(url, headers, TIMEOUT_SEC, MAX_BYTES)
    except TimeoutError:
        return None, "timeout"
    except Exception as e:                               # noqa: BLE001 — 원문은 버리고 코드만
        return None, "timeout" if "timed out" in type(e).__name__.lower() + str(type(e)) else "network"
    if status in (401, 403, 404):
        return None, "auth"
    if status != 200:
        return None, "network"
    if len(body) > MAX_BYTES:
        return None, "size"
    if hashlib.sha256(body).hexdigest() != sha.lower():
        return None, "hash"
    return body, "ok"


# ---------------------------------------------------------------------- 공개 모드 3-B 자료
@dataclass
class K3BSource:
    mode: str                                  # 'local' | 'public'
    groups: frozenset                          # 실제로 켜진 3-B 노선
    stop_times: pd.DataFrame | None            # public 이고 켜진 노선이 있을 때만 (켜진 노선 행만)
    reasons: dict = field(default_factory=dict)  # {노선: 끈 이유 코드} — 화면·로그에는 이 코드만
    requested: tuple = ()                      # 요청을 보낸 운영사 (검사용)


def _secret_token(secrets: Mapping | None, op: str) -> str | None:
    try:
        sec = (secrets or {}).get("k3b_remote", {})
        tok = (sec or {}).get(op, {}).get("token")
        return str(tok) if tok else None
    except Exception:                                    # noqa: BLE001 — Secrets 모양이 이상하면 없는 것과 같다
        return None


def load_public(policy: Policy, secrets: Mapping | None,
                http_get: Callable | None = None) -> K3BSource:
    """public 모드: C1~C6 을 통과한 노선만. 정책에서 켜진 노선이 없는 운영사는 요청하지 않는다."""
    want, why = policy_groups(policy)
    frames, got, requested = [], set(), []
    for op in sorted(OPERATOR_GROUPS):
        need = want & OPERATOR_GROUPS[op]
        if not need:
            continue
        spec = policy.operators.get(op, {})
        requested.append(op)
        body, code = fetch_operator(op, spec, _secret_token(secrets, op), http_get)
        if body is None:
            for g in need:
                why[g] = code
            LOG.warning("data_release operator=%s code=%s", op, code)
            continue
        try:
            df = pd.read_parquet(io.BytesIO(body))
        except Exception:                                # noqa: BLE001
            df = None
        exp_tr = {g: policy.groups[g].get("trains") for g in OPERATOR_GROUPS[op]
                  if g in policy.groups and policy.groups[g].get("trains")}
        bad = ["parse"] if df is None else validate_stop_times(df, op, spec.get("rows") or None, exp_tr or None)
        if bad:
            for g in need:
                why[g] = "schema"
            LOG.warning("data_release operator=%s code=schema items=%s", op, ",".join(bad))
            continue
        frames.append(df[df.group.isin(need)])
        got |= need
    for g in got:
        why.pop(g, None)
    st = pd.concat(frames, ignore_index=True) if frames else None
    return K3BSource("public", frozenset(got), st, why, tuple(requested))


def resolve_k3b(root: Path | None = None, secrets: Mapping | None = None, environ: Mapping | None = None,
                policy_path: Path | None = None, http_get: Callable | None = None) -> K3BSource:
    """앱이 부르는 한 곳. local 모드는 지금 동작(로컬 파일 있으면 전부), public 모드는 정책·내려받기."""
    base = Path(root or ROOT)
    if resolve_mode(base, environ) == "local":
        has = (base / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet").exists()
        return K3BSource("local", ALL_GROUPS if has else frozenset(), None,
                         {} if has else {g: "local_missing" for g in ALL_GROUPS})
    return load_public(load_policy(policy_path or base / "config" / "data_release.toml"), secrets, http_get)
