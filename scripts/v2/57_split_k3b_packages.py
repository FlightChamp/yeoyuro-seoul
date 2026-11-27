"""
scripts/v2/57_split_k3b_packages.py — 3-B 정차표를 운영사별 꾸러미로 나눈다 (D-112 2절, 구현 D-112a)
=====================================================================================================

로컬 정차표(data/marts/v2/korail_3b/stop_times.parquet, git 무시) → 운영사별 parquet + manifest.
결과는 data/private/k3b_packages/ (git 무시)에만 쓴다. 이 스크립트는 업로드하지 않고 네트워크도 쓰지 않는다.
이용허락이 확보되기 전에는 결과를 어디에도 올리지 않는다 (D-112 7절).

    python scripts/v2/57_split_k3b_packages.py           # 꾸러미 만들기 + 검사
    python scripts/v2/57_split_k3b_packages.py --check   # 만들지 않고 검사 결과만

출력: 운영사별 행 수·노선별 열차 수·SHA-256. 활성화할 때(새 D 번호) 이 값을 config/data_release.toml 에 적는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from yeoyuro_v2 import data_release as DR          # noqa: E402
from yeoyuro_v2 import korail_3b as K3             # noqa: E402

SRC = ROOT / "data" / "marts" / "v2" / "korail_3b" / "stop_times.parquet"
OUT = ROOT / "data" / "private" / "k3b_packages"


def split(st: pd.DataFrame) -> dict:
    """{운영사: 정차표} — 열 순서·자료형은 원본 그대로, 행 순서는 (day_type, group, train, stop_seq)."""
    out = {}
    for op, groups in sorted(DR.OPERATOR_GROUPS.items()):
        x = st[st.group.isin(groups)].sort_values(["day_type", "group", "train", "stop_seq"]).reset_index(drop=True)
        out[op] = x[list(DR.COLUMNS)]
    return out


def manifest(op: str, df: pd.DataFrame, body: bytes, src_sha: str) -> dict:
    return {"schema": 1, "operator": op, "groups": sorted(DR.OPERATOR_GROUPS[op]),
            "rows": int(len(df)), "trains": {g: int(n) for g, n in df.groupby("group").train.nunique().items()},
            "valid_from": {g: K3.VALID_FROM[g] for g in sorted(DR.OPERATOR_GROUPS[op])},
            "sha256": hashlib.sha256(body).hexdigest(), "source_sha256": src_sha}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not SRC.exists():
        print(f"로컬 정차표가 없습니다: {SRC.relative_to(ROOT)} (로컬 개발 환경에서만 실행)")
        return 1
    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    st = pd.read_parquet(SRC)
    parts = split(st)
    ok = sum(len(x) for x in parts.values()) == len(st)
    print(f"원본 {len(st):,}행 → " + " · ".join(f"{op} {len(x):,}행" for op, x in parts.items())
          + ("  (합계 일치)" if ok else "  ⚠ 합계 불일치"))
    if not a.check:
        OUT.mkdir(parents=True, exist_ok=True)
    for op, df in parts.items():
        bad = DR.validate_stop_times(df, op)
        f = OUT / DR.load_policy().operators.get(op, {}).get("path", f"korail_3b_{op}.parquet")
        if a.check:
            print(f"  {op}: 검사 {'통과' if not bad else bad}")
            continue
        df.to_parquet(f, index=False)
        body = f.read_bytes()
        m = manifest(op, df, body, src_sha)
        (OUT / f"{f.stem}.manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n",
                                                     encoding="utf-8", newline="\n")
        print(f"  {op}: {f.relative_to(ROOT)} · 검사 {'통과' if not bad else bad} · 열차 {m['trains']} · sha256 {m['sha256']}")
        ok &= not bad
    print("※ 이 파일들은 git 이 무시합니다. 이용허락 확보 전에는 업로드하지 마세요 (D-112 7절).")
    return 0 if ok else 2


if __name__ == "__main__":
    from yeoyuro_v2.proc import utf8_stdout
    utf8_stdout()
    raise SystemExit(main())
