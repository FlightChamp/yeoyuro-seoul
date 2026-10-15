"""
scripts/v2/34_fetch_fast_exit.py
================================
공모전 Facility PoC 준비 — 공공데이터포털 「서울교통공사_빠른하차정보」 OpenAPI 를 전부 받아 파일로 저장한다.

    python scripts/v2/34_fetch_fast_exit.py --key "<일반 인증키(Decoding)>"
    → data/raw/fast_exit/fast_exit_all.csv · fast_exit_all.json · page1_raw.txt

- API: https://apis.data.go.kr/B553766/inout/getFstExit (REST, JSON+XML)
- 요청 변수 이름이 공개 명세에서 확인되지 않아, 흔히 쓰는 조합(dataType=JSON / type=json)을 차례로 시도한다.
- 응답 구조를 모르는 상태에서도 '항목 목록'을 찾아 표로 저장한다. 실패하면 원문(page1_raw.txt)을 남긴다.
- 인증키는 파일에 저장하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://apis.data.go.kr/B553766/inout/getFstExit"
OUT = ROOT / "data" / "raw" / "fast_exit"
ROWS = 1000
FORMATS = ({"dataType": "JSON"}, {"type": "json"}, {})


def utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def call(key: str, page: int, fmt: dict, timeout: int = 30) -> str:
    q = {"serviceKey": key, "pageNo": page, "numOfRows": ROWS, **fmt}
    url = URL + "?" + urllib.parse.urlencode(q, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": "yeoyuro-seoul/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------- 응답 해석
def _xml_to_obj(el: ET.Element):
    kids = list(el)
    if not kids:
        return (el.text or "").strip()
    out: dict = {}
    for k in kids:
        v = _xml_to_obj(k)
        if k.tag in out:
            if not isinstance(out[k.tag], list):
                out[k.tag] = [out[k.tag]]
            out[k.tag].append(v)
        else:
            out[k.tag] = v
    return out


def parse(text: str):
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        return json.loads(t)
    return {ET.fromstring(t).tag: _xml_to_obj(ET.fromstring(t))}


def find_items(obj) -> list[dict]:
    """응답 안에서 가장 큰 '딕셔너리 목록'을 찾는다 (item 이 1개면 딕셔너리 하나로 오는 경우도 처리)."""
    best: list[dict] = []

    def walk(o, parent_key=""):
        nonlocal best
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            if len(o) > len(best):
                best = o
        if isinstance(o, dict):
            if parent_key in ("item", "row", "items") and o and all(not isinstance(v, (dict, list)) for v in o.values()):
                if not best:
                    best = [o]
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, parent_key)

    walk(obj)
    return best


def find_value(obj, names: tuple[str, ...]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and not isinstance(v, (dict, list)):
                return v
            r = find_value(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_value(v, names)
            if r is not None:
                return r
    return None


# ---------------------------------------------------------------- 실행
def fetch_all(key: str) -> tuple[list[dict], dict, str]:
    last = ""
    for fmt in FORMATS:
        try:
            text = call(key, 1, fmt)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:500].decode('utf-8', 'replace')}"
            continue
        except Exception as e:                       # 네트워크 오류
            last = f"{type(e).__name__}: {e}"
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "page1_raw.txt").write_text(text, encoding="utf-8")
        last = text[:500]
        try:
            obj = parse(text)
        except Exception:
            continue
        items = find_items(obj)
        if not items:
            continue
        total = find_value(obj, ("totalCount", "total_count", "list_total_count"))
        total = int(total) if str(total or "").isdigit() else None
        rows, page = list(items), 1
        while total and len(rows) < total and page < 50:
            page += 1
            time.sleep(0.3)
            more = find_items(parse(call(key, page, fmt)))
            if not more:
                break
            rows += more
        return rows, fmt, ""
    return [], {}, last


def col_like(cols: list[str], *words: str) -> str | None:
    for c in cols:
        if any(w.lower() in c.lower() for w in words):
            return c
    return None


def main() -> int:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=os.environ.get("DATA_GO_KR_KEY", ""), help="공공데이터포털 일반 인증키 (Decoding)")
    a = ap.parse_args()
    if not a.key:
        print("[중단] --key 에 공공데이터포털 '일반 인증키(Decoding)' 를 넣어 주세요.")
        return 1
    rows, fmt, err = fetch_all(a.key.strip())
    print("=" * 72)
    if not rows:
        print("[실패] 항목을 받지 못했습니다. 서버 응답 앞부분:")
        print(err or "(빈 응답)")
        print(f"원문: {OUT / 'page1_raw.txt'}  ← 이 파일을 첨부해 주세요")
        print("- SERVICE_KEY_IS_NOT_REGISTERED_ERROR: 승인 직후에는 키가 활성화되기까지 1~2시간 걸릴 수 있습니다.")
        print("- Decoding 키로 안 되면 Encoding 키로 한 번 더 시도해 주세요.")
        print("=" * 72)
        return 1
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "fast_exit_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (OUT / "fast_exit_all.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"받은 행: {len(rows)}  (요청 형식 {fmt or '기본'})")
    print(f"저장: {OUT / 'fast_exit_all.csv'}")
    print(f"열 ({len(cols)}개): {', '.join(cols)}")
    line_c = col_like(cols, "line", "호선")
    fac_c = col_like(cols, "fclt", "facl", "설비", "시설", "type")
    stn_c = col_like(cols, "stnnm", "stn_nm", "역명", "stationnm", "statnnm")
    if line_c:
        cnt: dict = {}
        for r in rows:
            cnt[r.get(line_c)] = cnt.get(r.get(line_c), 0) + 1
        print(f"호선별 행 수 ({line_c}): {dict(sorted(cnt.items(), key=lambda x: str(x[0])))}")
    if fac_c:
        cnt = {}
        for r in rows:
            cnt[r.get(fac_c)] = cnt.get(r.get(fac_c), 0) + 1
        print(f"설비 종류 ({fac_c}): {cnt}")
    print("-" * 72)
    print("앞 3행:")
    for r in rows[:3]:
        print("  ", r)
    print("-" * 72)
    print("4호선 서울역 · 동대문역사문화공원 · 수유 행 (엘리베이터 포함 여부 확인용):")
    hit = 0
    for r in rows:
        s = " ".join(str(v) for v in r.values())
        if ("4" in str(r.get(line_c, "")) or "4호선" in s) and any(n in s for n in ("서울역", "동대문역사문화공원", "수유")):
            print("  ", r)
            hit += 1
            if hit >= 15:
                break
    if not hit:
        print("   (해당 행을 찾지 못함 — 열 이름을 보고 제가 맞추겠습니다)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
