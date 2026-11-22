"""
yeoyuro_v2/map_planned.py
=========================
노선도 기본 파일에 '예정' 노선(Phase 3)을 함께 두기 위한 정의. (D-090)

- 기본 파일의 Station_Visual_Nodes · Line_Sequences · Transfer_Links 에 `phase` 열이 있다.
  비어 있으면 운영 중(1~9호선), '3-A' · '3-B' 이면 예정 노선.
- **앱·경로 그래프 검증은 예정 행을 빼고** 읽는다 (map_workbook.materialize 기본값) → 앱 노선도 불변.
- 좌표 편집 미리보기(29_watch_map · 29_check_map_workbook)는 예정 행까지 그린다.

범위 (D-085, D-089)
-------------------
- 3-A 1호선(코레일): 경부 서울역–금천구청, 경인 구로–온수, 경원 청량리–도봉산
- 3-B 경의중앙선 용산–양원·용산–수색, 수인분당선 왕십리–복정, 공항철도 서울역–김포공항
  (경의중앙선 서울역–신촌–가좌 지선, 수인분당선 왕십리–청량리는 제외)
역 순서는 노선 구조 기준. 3-B 는 시각표 확보 뒤 다시 확인한다.
"""
from __future__ import annotations

PHASE_COL = "phase"

# line_id → (이름, 색)
LINE_INFO = {
    "1": ("1호선", "#0052A4"), "2": ("2호선", "#009D3E"), "3": ("3호선", "#EF7C1C"), "4": ("4호선", "#00A5DE"),
    "5": ("5호선", "#996CAC"), "6": ("6호선", "#CD7C2F"), "7": ("7호선", "#747F00"), "8": ("8호선", "#E6186C"),
    "9": ("9호선", "#BDB092"),
    "GJ": ("경의중앙선", "#77C4A3"), "SB": ("수인분당선", "#F5A200"), "AR": ("공항철도", "#0090D2"),
}
LINE_COLORS = {k: v[1] for k, v in LINE_INFO.items()}

# (line_id, branch_code, phase, 표시 이름, [역 순서])
PLANNED = [
    ("1", "gyeongbu", "3-A", "1호선 경부선(코레일)",
     ["서울역", "남영", "용산", "노량진", "대방", "신길", "영등포", "신도림", "구로", "가산디지털단지", "독산", "금천구청"]),
    ("1", "gyeongin", "3-A", "1호선 경인선(코레일)", ["구로", "구일", "개봉", "오류동", "온수"]),
    ("1", "gyeongwon", "3-A", "1호선 경원선(코레일)",
     ["청량리", "회기", "외대앞", "신이문", "석계", "광운대", "월계", "녹천", "창동", "방학", "도봉", "도봉산"]),
    ("GJ", "jungang", "3-B", "경의중앙선 중앙선 쪽",
     ["용산", "이촌", "서빙고", "한남", "옥수", "응봉", "왕십리", "청량리", "회기", "중랑", "상봉", "망우", "양원"]),
    ("GJ", "gyeongui", "3-B", "경의중앙선 경의선 쪽",
     ["용산", "효창공원앞", "공덕", "서강대", "홍대입구", "가좌", "디지털미디어시티", "수색"]),
    ("SB", "main", "3-B", "수인분당선",
     ["왕십리", "서울숲", "압구정로데오", "강남구청", "선정릉", "선릉", "한티", "도곡", "구룡", "개포동",
      "대모산입구", "수서", "복정"]),
    ("AR", "main", "3-B", "공항철도", ["서울역", "공덕", "홍대입구", "디지털미디어시티", "마곡나루", "김포공항"]),
]
LINES_ROWS = [  # Lines 시트에 덧붙이는 행 (설명용)
    ("1", "1호선(코레일)", "#0052A4", "경부 서울역~금천구청 · 경인 구로~온수 · 경원 청량리~도봉산",
     "gyeongbu; gyeongin; gyeongwon", "phase 3-A 예정 — 앱 노선도·경로 계산에는 아직 없음"),
    ("GJ", "경의중앙선", "#77C4A3", "용산~양원 · 용산~수색 (서울역 지선 제외, D-089)", "jungang; gyeongui",
     "phase 3-B 예정"),
    ("SB", "수인분당선", "#F5A200", "왕십리~복정 (청량리 연장 제외, D-089)", "main", "phase 3-B 예정"),
    ("AR", "공항철도", "#0090D2", "서울역~김포공항", "main", "phase 3-B 예정"),
]


def node_id(station: str, line: str) -> str:
    return f"{station}_L{line}"


def planned_nodes() -> list[tuple[str, str, str, str, str]]:
    """(visual_node_id, station, line_id, branch_codes, phase) — 예정 노선의 노선별 점 (중복 없음, 순서 유지)."""
    out: dict[str, list] = {}
    for line, br, phase, _, stations in PLANNED:
        for s in stations:
            nid = node_id(s, line)
            if nid not in out:
                out[nid] = [nid, s, line, br, phase]
            elif br not in out[nid][3].split(";"):
                out[nid][3] += ";" + br
    return [tuple(v) for v in out.values()]


def is_planned(v) -> bool:
    return v is not None and str(v).strip() not in ("", "nan", "None")
