"""
tests/test_docs_links.py — docs/v2 문서의 상대 링크가 실제 파일을 가리키는지 검사한다 (Phase 2-3, D-083).

문서를 옮기거나 이름을 바꾸면 링크가 조용히 깨진다. 외부 URL(http)과 문서 안 앵커(#)는 검사하지 않는다.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs" / "v2").glob("*.md"))
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    broken = []
    for target in LINK.findall(doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#", 1)[0]
        if path and not (doc.parent / path).resolve().exists():
            broken.append(target)
    assert not broken, "%s 의 깨진 링크: %s" % (doc.name, broken)
