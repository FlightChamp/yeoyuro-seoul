"""
yeoyuro_seoul_app.py — 여유로 서울 쾌적 경로 추천
==========================================

실행
----
    streamlit run app/streamlit/yeoyuro_seoul_app.py -- --root .

설계 원칙
---------
1. 사용자 입력과 지도 클릭은 **역 단위(station-level)** 다. "1호선 서울역"이 아니라 "서울역".
2. 내부 그래프와 혼잡도 계산은 호선별 station_uid 단위를 유지한다.
   환승역에서 어떤 노선을 처음 탈지는 알고리즘이 후보로 비교하며,
   **최초 승차 노선 선택은 환승으로 계산하지 않는다.**
3. 마트 로딩은 @st.cache_data, 그래프/스코어러 객체는 @st.cache_resource.
4. 계산 실패 시 precomputed demo 로 자연스럽게 fallback. 에러 창을 띄우지 않는다.
5. 혼잡도는 실시간이 아니라 **과거 패턴 기반 기대 혼잡 위험도**임을 상시 명시한다.
6. 환승 호차/문은 원본에 있는 조합만 표시한다(추정 금지).
7. 대안이 없으면 억지 추천하지 않고 no_meaningful_alternative 정책을 표시한다.
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------
# 경로
# --------------------------------------------------------------------------
def resolve_root() -> Path:
    for i, a in enumerate(sys.argv):
        if a == "--root" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "marts").exists():
            return p
    return Path(".").resolve()


ROOT = resolve_root()
MARTS = ROOT / "data" / "marts"
MASTER = ROOT / "data" / "master"
REPORTS = ROOT / "reports"
# v2.6: 앱 전용 1~9호선 데이터 폴더 (scripts/v2/30_build_app_data.py 가 생성).
# 있으면 앱은 여기를 먼저 읽고, 없는 파일만 기존 data/ 에서 읽는다. 분석 스크립트용 data/ 는 그대로 둔다.
APP_DATA = ROOT / "data" / "app_v26"
DATA_ROOT = APP_DATA if (APP_DATA / "data" / "marts" / "route_edge_mart.parquet").exists() else ROOT
MAP_WORKBOOK = ROOT / "data" / "master" / "yeoyuro_seoul_vector_map_coordinate_workbook.xlsx"
for extra in (ROOT, ROOT / "scripts", ROOT / "app" / "streamlit"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

st.set_page_config(page_title="여유로 서울", page_icon="🚇", layout="wide",
                   initial_sidebar_state="expanded")

# 브랜드 색
PRIMARY = "#002D56"     # PANTONE 7463C
SECONDARY = "#8D7150"   # PANTONE 873C
BASE_BG = "#FFFFFF"
LIGHT = "#F3F4F6"
DANGER = "#C62828"
WARN = "#EF6C00"

LINE_COLORS = {
    "1": "#0052A4", "2": "#009D3E", "3": "#EF7C1C", "4": "#00A5DE",
    "5": "#996CAC", "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C",
    "9": "#BDB092", "9L": "#BDB092", "9X": "#A08A4E",   # v2.6 9호선 (급행은 조금 진하게)
}


def line_label(x) -> str:
    """노선 표시 이름. 내부 id 9L/9X 를 사용자에게 그대로 보이지 않는다."""
    return {"9L": "9호선 일반", "9X": "9호선 급행"}.get(str(x), "%s호선" % x)


def line_sort_key(x):
    """노선 정렬 (1~9, 9 일반 → 9 급행). int() 로 정렬하면 9L/9X 에서 오류가 난다."""
    x = str(x)
    return (int(x[0]) if x[:1].isdigit() else 99, x)

DISCLAIMER = ("표시된 혼잡도는 실시간 측정값이 아닌 **과거 패턴 기반 기대 혼잡도**입니다. "
              "혼잡도는 정원 대비 승차 인원 비율이며, 100%는 열차 1칸 160명(좌석 54 + 입석 106) 기준입니다.")

MODE_LABEL = {"calm": "혼잡 회피", "fast": "빠른 도착",
              "min_transfer": "환승 최소", "balanced": "균형"}

DIRECTION_KO = {"up": "상행", "down": "하행", "inner": "내선", "outer": "외선"}
DAY_TYPE_KO = {"weekday": "평일", "saturday": "토요일", "sunday": "일요일"}

# ---------------------------------------------------------------------------
# 노선별 혼잡 단면용 서비스 계통 정의.
#   화면 표시명과 내부 조회 키(line_id / branch / direction)를 분리한다.
#   역 순서는 station_code 로 정렬하면 안 된다.
#   성수지선이 성수(211) -> 용답(244) -> 신답(245) -> 용두(250) -> 신설동(246) 처럼
#   번호 순서와 운행 순서가 다르기 때문이다. 그래프를 걸어서 순서를 만든다.
#   directions: (표시명, 내부 direction 코드, 시작역, 끝역)
# ---------------------------------------------------------------------------
SERVICE_PATTERNS = [
    {"label": "1호선", "line": "1", "scope": "main",
     "directions": [("서울역 방면", "down", "청량리", "서울역"),
                    ("청량리 방면", "up", "서울역", "청량리")]},
    {"label": "2호선 본선", "line": "2", "scope": "line2_main",
     "directions": [("내선순환", "inner", "시청", "충정로"),
                    ("외선순환", "outer", "시청", "을지로입구")]},
    {"label": "2호선 성수지선", "line": "2", "scope": "seongsu",
     "directions": [("신설동 방면", "outer", "성수", "신설동"),
                    ("성수 방면", "inner", "신설동", "성수")]},
    {"label": "2호선 신정지선", "line": "2", "scope": "sinjeong",
     "directions": [("까치산 방면", "inner", "신도림", "까치산"),
                    ("신도림 방면", "outer", "까치산", "신도림")]},
    {"label": "3호선", "line": "3", "scope": "main",
     "directions": [("오금 방면", "down", "지축", "오금"),
                    ("지축 방면", "up", "오금", "지축")]},
    {"label": "4호선", "line": "4", "scope": "main",
     "directions": [("남태령 방면", "down", "불암산", "남태령"),
                    ("불암산 방면", "up", "남태령", "불암산")]},
    {"label": "5호선 방화–하남검단산", "line": "5", "scope": "hanam",
     "directions": [("하남검단산 방면", "down", "방화", "하남검단산"),
                    ("방화 방면", "up", "하남검단산", "방화")]},
    {"label": "5호선 방화–마천", "line": "5", "scope": "macheon",
     "directions": [("마천 방면", "down", "방화", "마천"),
                    ("방화 방면", "up", "마천", "방화")]},
    {"label": "6호선", "line": "6", "scope": "main",
     "directions": [("응암순환 → 신내 방면", "down", "응암", "신내"),
                    ("신내 → 응암 방면", "up", "신내", "응암")]},
    {"label": "7호선", "line": "7", "scope": "main",
     "directions": [("온수 방면", "down", "장암", "온수"),
                    ("장암 방면", "up", "온수", "장암")]},
    {"label": "8호선", "line": "8", "scope": "main",
     "directions": [("모란 방면", "down", "암사역사공원", "모란"),
                    ("암사역사공원 방면", "up", "모란", "암사역사공원")]},
]

# 2호선 지선 전용 역 (본선 순서에 섞이면 안 된다)
SEONGSU_ONLY = {"용답", "신답", "용두", "신설동"}
SINJEONG_ONLY = {"도림천", "양천구청", "신정네거리", "까치산"}
# 5호선 강동 이후 분기
HANAM_ONLY = {"길동", "굽은다리", "명일", "고덕", "상일동", "강일", "미사",
              "하남풍산", "하남시청", "하남검단산"}
MACHEON_ONLY = {"둔촌동", "올림픽공원", "방이", "오금", "개롱", "거여", "마천"}

# 내부 노드명을 사용자 표시명으로 (내부 표기를 그대로 노출하지 않는다)
DISPLAY_STATION_NAME = {"응암S": "응암(순환 시작)"}

# ---------------------------------------------------------------------------
# 전역 스타일. 화이트 베이스 + 네이비 Primary + 골드 Secondary.
# 노선색은 노선 구분에만 쓰고, 혼잡 위험은 주황/빨강으로 제한한다.
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
  .stApp {{ background: {BASE_BG}; }}
  h1, h2, h3 {{ color: {PRIMARY}; }}
  .badge {{ display:inline-block; padding:1px 8px; margin-right:4px;
           border-radius:10px; color:#fff; font-size:0.78rem; font-weight:600; }}

  /* ---------- 사이드바: 네이비 배경 + 골드 포인트 ---------- */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {PRIMARY} 0%, #001B33 100%);
      border-right: 3px solid {SECONDARY};
  }}
  section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
  section[data-testid="stSidebar"] h1 {{
      color: #FFFFFF !important; font-size: 1.6rem; margin-bottom: 0.2rem;
      border-bottom: 2px solid {SECONDARY}; padding-bottom: 0.5rem;
  }}
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
      color: #D9DEE5 !important; font-size: 0.86rem; line-height: 1.5;
  }}
  section[data-testid="stSidebar"] hr {{ border-color: rgba(141,113,80,0.5); }}

  /* 메뉴 버튼: 골드 바탕 + 흰 글씨.
     선택자를 .stButton 안쪽으로 한정한다. 그러지 않으면 사이드바 접기 버튼까지
     골드가 칠해지고, 특이도가 더 높아 개별 예외 규칙이 먹지 않는다. */
  section[data-testid="stSidebar"] .stButton button {{
      background-color: {SECONDARY} !important;
      border: 1px solid {SECONDARY} !important;
      color: #FFFFFF !important;
      font-weight: 600; justify-content: flex-start;
      padding: 0.55rem 0.9rem; margin-bottom: 4px; border-radius: 8px;
  }}
  section[data-testid="stSidebar"] .stButton button:hover {{
      background-color: #A0855F !important; border-color: #A0855F !important;
  }}
  /* 선택된 메뉴: 좌우 양쪽에 흰 띠 */
  section[data-testid="stSidebar"] .stButton button[kind="primary"],
  section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {{
      background-color: #6F573B !important; border-color: #6F573B !important;
      box-shadow: inset 4px 0 0 0 #FFFFFF, inset -4px 0 0 0 #FFFFFF;
  }}

  /* 사이드바 접기/펼치기 버튼: 배경 없이 흰 아이콘만 */
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button,
  [data-testid="stSidebarCollapseButton"] button,
  [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="collapsedControl"] button,
  [data-testid="stExpandSidebarButton"] button {{
      background: none !important;
      background-color: transparent !important;
      border: none !important;
      box-shadow: none !important;
      color: #FFFFFF !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button:hover,
  [data-testid="stSidebarCollapseButton"] button:hover,
  [data-testid="stSidebarCollapsedControl"] button:hover,
  [data-testid="collapsedControl"] button:hover,
  [data-testid="stExpandSidebarButton"] button:hover {{
      background: none !important;
      background-color: transparent !important;
  }}

  /* 선택한 역의 [출발역/도착역으로 설정] 버튼: 네이비 바탕 + 금색 글씨 */
  .st-key-btn_set_origin button, .st-key-btn_set_dest button,
  div[class*="st-key-btn_set_"] button {{
      background-color: {SECONDARY} !important;
      border: 1px solid {SECONDARY} !important;
      color: #FFFFFF !important;
      font-weight: 700;
  }}
  .st-key-btn_set_origin button:hover, .st-key-btn_set_dest button:hover,
  div[class*="st-key-btn_set_"] button:hover {{
      background-color: #A0855F !important; border-color: #A0855F !important;
      color: #FFFFFF !important;
  }}
  .st-key-btn_set_origin button p, .st-key-btn_set_dest button p,
  div[class*="st-key-btn_set_"] button p {{ color: #FFFFFF !important; }}

  /* ---------- 본문 버튼 ---------- */
  div.stButton > button[kind="primary"] {{
      background-color: {PRIMARY}; border-color: {PRIMARY};
  }}
  div.stButton > button[kind="secondary"] {{
      border-color: {SECONDARY}; color: {SECONDARY};
  }}

  /* ---------- 탭 강조를 골드로 (기본 빨강 대체) ---------- */
  button[data-baseweb="tab"] {{ color: #4B5563; }}
  button[data-baseweb="tab"][aria-selected="true"] {{
      color: {SECONDARY} !important; font-weight: 700;
  }}
  .stTabs [data-baseweb="tab-highlight"],
  [data-baseweb="tab-highlight"] {{ background-color: {SECONDARY} !important; }}
  .stTabs [data-baseweb="tab-border"] {{ background-color: #E5E7EB; }}
  .stTabs button[role="tab"][aria-selected="true"] {{
      color: {SECONDARY} !important;
  }}
  .stTabs button[role="tab"][aria-selected="true"] p {{
      color: {SECONDARY} !important; font-weight: 700;
  }}
</style>
""", unsafe_allow_html=True)



EVENT_TYPE_KO = {
    "fireworks": "불꽃축제", "cherry_blossom": "봄꽃축제",
    "new_year_bell": "제야의 종", "halloween_proxy": "할로윈 상권",
    "essay": "대학 논술", "transfer_exam": "대학 편입",
}
EVENT_NAME_FIX = {
    "Halloween commercial crowd period": "할로윈 상권 혼잡",
    " essay": " 논술", " transfer_exam": " 편입",
}


def ko_event_name(v) -> str:
    """이벤트명을 한글화하고 **연도를 맨 앞으로** 옮긴다.

    '서울세계불꽃축제 2022' / '동국대 2023 논술' 처럼 연도 위치가 제각각이면
    표에서 읽기 어렵다. '2022 서울세계불꽃축제' / '2023 동국대 논술' 로 통일한다.
    """
    t = str(v)
    for a, b in EVENT_NAME_FIX.items():
        t = t.replace(a, b)
    m = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", t)
    if m:
        year = m.group(0)
        rest = (t[:m.start()] + t[m.end():]).strip()
        rest = re.sub(r"\s{2,}", " ", rest)
        if rest:
            t = "%s %s" % (year, rest)
    return t


def round1(df: pd.DataFrame) -> pd.DataFrame:
    """표에 보여줄 실수 값을 **소수 첫째 자리 문자열**로 확정한다.

    st.table / st.dataframe 은 float 컬럼을 기본 4자리로 렌더링한다(1.5 -> 1.5000).
    round(1) 만으로는 표시가 바뀌지 않으므로 문자열로 포맷해야 한다.
    정수형 컬럼과 문자열이 섞인 컬럼은 건드리지 않는다.
    """
    out = df.copy()
    for c in out.columns:
        col = out[c]
        if pd.api.types.is_bool_dtype(col) or pd.api.types.is_integer_dtype(col):
            continue
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().sum() == 0:
            continue
        if num.isna().sum() > col.isna().sum():      # 문자열이 섞인 컬럼
            continue
        if (num.dropna() % 1 == 0).all():
            out[c] = num.map(lambda v: "-" if pd.isna(v) else "{:,.0f}".format(v))
        else:
            out[c] = num.map(lambda v: "-" if pd.isna(v) else "{:,.1f}".format(v))
    return out


# --------------------------------------------------------------------------
# 로더
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_mart(name: str, sub: str = "data/marts", stamp: float = 0.0):
    """앱 데이터 폴더(1~9호선)를 먼저, 없으면 기존 data/ 를 읽는다.
    stamp: 캐시 구분용 파일 시각. (이름이 '_' 로 시작하면 Streamlit 이 캐시 구분에 쓰지 않으므로 밑줄 없이 둔다)"""
    for base in dict.fromkeys((DATA_ROOT / sub, ROOT / sub)):
        for ext in (".parquet", ".csv.gz", ".csv"):
            p = base / (name + ext)
            if p.exists():
                try:
                    return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
                except Exception:
                    continue
    return None


def ensure_map_fresh() -> float:
    """기본 좌표 엑셀이 앱 노선도보다 새로우면 노선도를 다시 가져온다. 반환: 노선도 파일 시각(캐시 키)."""
    target = DATA_ROOT / "data" / "master" / "map_visual_nodes.csv"
    if DATA_ROOT == ROOT or not MAP_WORKBOOK.exists():
        return target.stat().st_mtime if target.exists() else 0.0
    if not target.exists() or MAP_WORKBOOK.stat().st_mtime > target.stat().st_mtime:
        import importlib.util
        spec = importlib.util.spec_from_file_location("app_build30", ROOT / "scripts" / "v2" / "30_build_app_data.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with st.spinner("엑셀 좌표가 바뀌어 노선도를 다시 가져오는 중…"):
            ok, msg = mod.import_map(DATA_ROOT, MAP_WORKBOOK)
        if not ok:
            st.error("노선도 가져오기 실패: " + msg)
        else:
            st.toast("엑셀 좌표를 노선도에 반영했습니다")
    return target.stat().st_mtime if target.exists() else 0.0


def load_display_master():
    """캐시 함수 밖에서 엑셀 변경을 먼저 확인한다 (캐시 함수 안에서 화면 요소를 만들면 Streamlit 오류)."""
    return _load_display_master_cached(ensure_map_fresh())


@st.cache_data(show_spinner=False)
def _load_display_master_cached(stamp: float):
    return load_mart("station_display_master", "data/master", stamp)


@st.cache_data(show_spinner=False)
def load_map_bundle(stamp: float = 0.0):
    """벡터 노선도 좌표 묶음. source of truth 는 좌표 워크북(5120x2880)이다. stamp = 노선도 파일 시각 (캐시 구분)."""
    b = {n: load_mart(n, "data/master", stamp) for n in
         ("map_visual_nodes", "map_line_segments", "map_transfer_links",
          "map_click_areas", "map_labels")}
    return b if b["map_visual_nodes"] is not None else None


@st.cache_data(show_spinner=False)
def precomputed_tags():
    return sorted({p.stem.replace("route_evaluation_mart_", "")
                   for p in MARTS.glob("route_evaluation_mart_*")} - {""})


@st.cache_resource(show_spinner=False)
def get_scorer(day_type: str, depart: str, query_date):
    import station_routing as sr
    mod = sr.load_scorer_class(ROOT)
    from yeoyuro_v2.line9 import register_v1_directions
    register_v1_directions(mod)                      # 9L/9X 방향 등록 (1~8호선 영향 없음)
    return mod.RouteScorer(DATA_ROOT, day_type, depart, query_date)


def sr_module():
    import station_routing as sr
    return sr


# --------------------------------------------------------------------------
# v2.1 시간 진행형 평가 (yeoyuro_v2). 실패하면 None -> v1 값 그대로 표시한다.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_td_evaluator(day_type: str, depart: str, query_date):
    try:
        from yeoyuro_v2.time_dependent import TimeDependentEvaluator
        scorer = get_scorer(day_type, depart, query_date)
        return TimeDependentEvaluator(scorer, sr_module().load_headway(DATA_ROOT), "step")
    except Exception:
        return None


def _query_args():
    dow = pd.Timestamp(st.session_state["q_date"]).dayofweek
    day_type = "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")
    hhmm = "%02d:%02d" % (st.session_state["q_time"].hour, st.session_state["q_time"].minute)
    return day_type, hhmm, str(st.session_state["q_date"])


def retime_with_v2(result):
    """find_route_by_station 결과를 구간 진입 시각 기준으로 다시 판정한다."""
    try:
        from yeoyuro_v2.app_bridge import retime_result
        from yeoyuro_v2.time_dependent import parse_hhmm
        day_type, hhmm, qd = _query_args()
        ev = get_td_evaluator(day_type, hhmm, qd)
        if ev is None:
            return result
        return retime_result(result, lambda b: ev, parse_hhmm(hhmm), st.session_state["q_mode"])
    except Exception:
        return result


# Streamlit selectbox 는 옵션 텍스트를 평문으로만 렌더링한다(HTML/CSS 미적용).
# 그래서 리스트 안에서는 색을 쓸 수 없고, 유니코드 문자로만 노선을 구분한다.
# 컬러가 꼭 필요하면 아래 플래그를 True 로 바꿔 이모지 방식을 쓸 수 있다.
# 다만 4호선(하늘)이 1호선(남색)과 구분되지 않고, 7호선(올리브)·8호선(분홍)의
# 색 대응이 나빠 숫자 방식보다 읽기 어려울 수 있다. 직접 보고 선택할 것.
USE_EMOJI_LINE_MARKS = False

CIRCLED = {"1": "①", "2": "②", "3": "③", "4": "④",
           "5": "⑤", "6": "⑥", "7": "⑦", "8": "⑧", "9": "⑨"}

# 실제 노선색에 가장 가까운 컬러 이모지 (근사값)
EMOJI_MARK = {"1": "🔵", "2": "🟢", "3": "🟠", "4": "🔹",
              "5": "🟣", "6": "🟤", "7": "🟡", "8": "🔴", "9": "🟨"}


def short_label(station_key: str, lines: str) -> str:
    """선택 리스트용 라벨. '③⑧ 가락시장' 형태. '역'은 생략하되 서울역은 예외."""
    table = EMOJI_MARK if USE_EMOJI_LINE_MARKS else CIRCLED
    marks = "".join(table.get(l.strip(), "") for l in str(lines).split(","))
    name = station_key if station_key == "서울역" else (station_key.rstrip("역") or station_key)
    return ("%s %s" % (marks, name)).strip()


def line_badges(lines: str) -> str:
    out = []
    for ln in str(lines).split(","):
        ln = ln.strip()
        if ln:
            out.append('<span class="badge" style="background:%s">%s</span>'
                       % (LINE_COLORS.get(ln, PRIMARY), ln))
    return "".join(out)


# --------------------------------------------------------------------------
# 상태
# --------------------------------------------------------------------------
KST = ZoneInfo("Asia/Seoul")


def _now_defaults():
    """초기 접속·새로고침 시 날짜/시각을 현재 시각으로 맞춘다(30분 단위 내림).

    배포 컨테이너의 TZ 는 UTC 다. 고정하지 않으면 서울 기준 접속 시각과
    기본값이 9시간 어긋난다.
    """
    now = datetime.now(KST)
    return now.date(), dtime(now.hour, 0 if now.minute < 30 else 30)


_D_DATE, _D_TIME = _now_defaults()
DEFAULTS = {
    "selected_station_key": None,
    "origin_station_key": None,
    "destination_station_key": None,
    "menu": None,
    "q_date": _D_DATE,
    "q_time": _D_TIME,
    "q_mode": "calm",
    "q_access": False,          # 공모전판: 이동편의 안내(엘리베이터 하차 칸). 경로 계산에는 관여하지 않는다
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------------------------------
# 정보 구조 (v2 IA 리팩토링, D-057)
#   사이드바 1차 메뉴 4개 + 각 메뉴 안의 subview.
#   subview 는 st.tabs 가 아니라 session_state + st.segmented_control 로 제어한다
#   (st.tabs 는 모든 탭을 매번 실행하고, 코드에서 특정 탭으로 전환할 수 없다).
# --------------------------------------------------------------------------
NAV = {"경로 추천": ["경로", "시간 바꿔 보기"],
       "혼잡 탐색": ["역·노선별", "네트워크 지도"],
       "이벤트 분석": ["시나리오", "과거 데이터"],
       "역사 안내": ["엘리베이터 하차 칸", "역사 도면"],     # 공모전판 (Facility + 공간정보)
       "프로젝트 & 검증": []}
PAGES = list(NAV)
VIEW_KEY = {"경로 추천": "view_route", "혼잡 탐색": "view_explore", "이벤트 분석": "view_event",
            "역사 안내": "view_facility"}
# 예전 메뉴 이름(북마크·테스트·이전 세션) → (페이지, subview)
LEGACY_MENU = {"쾌적 경로 찾기": ("경로 추천", "경로"),
               "시간 분산": ("경로 추천", "시간 바꿔 보기"),
               "혼잡 분산 지도": ("혼잡 탐색", "네트워크 지도"),
               "시간대별 혼잡 조회": ("혼잡 탐색", "역·노선별"),
               "이벤트 시나리오": ("이벤트 분석", "시나리오"),
               "이벤트 혼잡 경보": ("이벤트 분석", "과거 데이터"),
               "빠른 환승 안내": ("경로 추천", "경로"),        # 독립 메뉴 제거 → 경로 결과의 환승 구간에 포함
               "프로젝트 소개 및 검증 리포트": ("프로젝트 & 검증", None),
               "노선도 편집": ("노선도 편집", None)}
MENUS = PAGES                     # 하위 호환 (예전 코드가 MENUS 를 참조)

RESULT_NOTE = "ⓘ 과거 패턴 기반 기대값 · 실시간 정보 아님"
STRUCTURAL_LABEL = "회피 어려움"
STRUCTURAL_HELP = "정해진 경로·시간 대안 범위에서 개선안이 발견되지 않은 이동"

# 공유 입력: canonical(일반 key, source of truth) ↔ widget key.
#   widget → canonical : 각 위젯의 on_change
#   canonical → widget : 페이지/subview 이동 시점(_sync_widgets_from_canonical)과,
#                        위젯 state 가 없을 때의 setdefault 시드
#   매 렌더마다 위젯 값을 덮어쓰지 않는다 (사용자가 막 바꾼 값을 지우지 않기 위해).
WIDGET_CANON = {"sel_origin": "origin_station_key", "sel_dest": "destination_station_key",
                "w_date": "q_date", "w_time": "q_time", "w_mode": "q_mode", "w_access": "q_access",
                "ts_origin": "origin_station_key", "ts_dest": "destination_station_key",
                "ts_date": "q_date", "ts_time": "q_time"}


def _w2c(widget_key: str):
    """on_change: 위젯 값 → canonical."""
    st.session_state[WIDGET_CANON[widget_key]] = st.session_state.get(widget_key)


def _sync_widgets_from_canonical():
    """이동 시점에만 canonical → 위젯. 콜백 안에서 부르므로 위젯 생성 전이다."""
    for w, c in WIDGET_CANON.items():
        v = st.session_state.get(c)
        if v is None and w in ("ts_origin", "ts_dest", "w_date", "w_time", "w_mode", "ts_date", "ts_time"):
            st.session_state.pop(w, None)          # 다음 렌더에서 기본값으로 시드
        else:
            st.session_state[w] = v


def _seed(widget_key: str, default=None):
    """위젯 state 가 없을 때만 canonical 값으로 채운다."""
    if widget_key not in st.session_state:
        v = st.session_state.get(WIDGET_CANON[widget_key])
        st.session_state[widget_key] = default if v is None else v


def _go(page: str, view: str | None = None):
    """사이드바·CTA·링크 버튼 공통 이동 (on_click 콜백)."""
    st.session_state["nav_page"] = page
    if view and page in VIEW_KEY:
        st.session_state[VIEW_KEY[page]] = view
    _sync_widgets_from_canonical()


def _query_sig():
    """경로 결과가 어떤 입력으로 계산됐는지 구분하는 서명. 하나라도 바뀌면 결과는 stale."""
    s = st.session_state
    t = s.get("q_time")
    return (s.get("origin_station_key"), s.get("destination_station_key"), str(s.get("q_date")),
            None if t is None else "%02d:%02d" % (t.hour, t.minute), s.get("q_mode"))


# --------------------------------------------------------------------------
# 노선도
# --------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 5120, 2880

# 역명은 yeoyuro_v2.map_labels 가 캔버스 좌표 단위 SVG 한 장으로 그린다 (D-067).
# 글자 크기·간격: map_labels.LABEL_FONT_UNITS / LABEL_DY_NORMAL / LABEL_DY_MAJOR.
# 역명이 확대 배율만큼 커지므로 확대 수단을 열어 둔다: 마우스 휠 확대, 오른쪽 위 +/− · 원래대로 버튼, 더블클릭 = 전체 보기.
MAP_ZOOM_CONFIG = {"scrollZoom": True, "displayModeBar": True, "displaylogo": False,
                   "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage", "autoScale2d", "zoom2d"]}
# 노선도 표시 범위의 기본값 (데이터가 없을 때만 사용). 실제 범위는 map_bounds() 가
# 역 점·역명 좌표에서 계산한다 → 좌표를 옮겨도 잘리지 않는다 (v2.6: 9호선 개화 x=640 이 잘리던 문제).
X0, X1, Y0, Y1 = 700, 4450, 150, 2750
MAP_PAD = dict(left=110, right=110, top=90, bottom=130)   # 역명 글자·마커 반지름 여유 (역명은 점 아래)


def map_bounds(bundle: dict) -> tuple[float, float, float, float]:
    """역 점과 역명 위치를 모두 담는 표시 범위 (x0, x1, y0, y1)."""
    xs, ys = [], []
    for name, xc, yc in (("map_visual_nodes", "x_px", "y_px"), ("map_labels", "label_x_px", "label_y_px")):
        df = bundle.get(name)
        if df is not None and len(df) and xc in df and yc in df:
            xs += pd.to_numeric(df[xc], errors="coerce").dropna().tolist()
            ys += pd.to_numeric(df[yc], errors="coerce").dropna().tolist()
    if not xs:
        return X0, X1, Y0, Y1
    return (min(xs) - MAP_PAD["left"], max(xs) + MAP_PAD["right"],
            min(ys) - MAP_PAD["top"], max(ys) + MAP_PAD["bottom"])
# 참고: Plotly 에는 '최대 확대 배율' 을 막는 속성이 없다(minallowed/maxallowed 는
# 바깥 범위만 제한한다). 확대 깊이 제한은 JS 커스텀 컴포넌트가 필요하다.


def path_to_visual_nodes(path, vn: pd.DataFrame):
    """station_uid path -> station_key -> visual_node_id -> 좌표."""
    have = set(vn["visual_node_id"])
    pos = {r.visual_node_id: (r.x_px, r.y_px) for r in vn.itertuples()}
    bykey = {}
    for r in vn.itertuples():
        bykey.setdefault(r.station_key, []).append(r.visual_node_id)
    xs, ys = [], []
    for node in path:
        line = node.split("_", 1)[0]
        line = "9" if line in ("9L", "9X") else line          # 9호선 일반/급행은 노선도에서 한 줄
        key = node.split("_", 1)[1].split("@")[0]
        vid = "%s_L%s" % (key, line)
        if vid not in have:
            cand = bykey.get(key)
            if not cand:
                continue
            vid = cand[0]
        x, y = pos[vid]
        if xs and xs[-1] == x and ys[-1] == y:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def draw_map(bundle: dict, highlight_path=None, show_endpoints: bool = True):
    """벡터 노선도. 레이어 순서:
    노선 → 환승 connector → 역 marker → 역명 label → 경로 highlight → 출도착 halo → 클릭영역
    """
    import plotly.graph_objects as go
    vn = bundle["map_visual_nodes"]
    seg = bundle["map_line_segments"]
    tl = bundle["map_transfer_links"]
    ca = bundle["map_click_areas"]
    lb = bundle["map_labels"]
    fig = go.Figure()

    # 경로가 선택되면 나머지 노선망을 옅게 해 추천 경로로 시선이 가게 한다 (선택 경로는 5에서 원래 노선색으로 다시 그림)
    dim = 0.3 if highlight_path else 1.0
    # 1. 노선
    if seg is not None:
        for (line, _), g in seg.groupby(["line_id", "color"]):
            xs, ys = [], []
            for r in g.itertuples():
                xs += [r.x0, r.x1, None]
                ys += [r.y0, r.y1, None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line=dict(color=LINE_COLORS.get(str(line), "#888"),
                                               width=6),
                                     opacity=dim, showlegend=False))

    # 2. 환승 connector
    if tl is not None and "x0" in tl.columns:
        xs, ys = [], []
        for r in tl.itertuples():
            if pd.isna(r.x0) or not bool(r.connector_visible):
                continue
            xs += [r.x0, r.x1, None]
            ys += [r.y0, r.y1, None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line=dict(color="#9CA3AF", width=2.5),
                                     opacity=dim, showlegend=False))

    # 3. 역 marker
    tr = vn[vn["is_transfer_station"] == True]
    nm = vn[vn["is_transfer_station"] != True]
    fig.add_trace(go.Scatter(x=nm["x_px"], y=nm["y_px"], mode="markers",
                             marker=dict(size=7, color="white",
                                         line=dict(width=1.3, color="#4B5563")),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=tr["x_px"], y=tr["y_px"], mode="markers",
                             marker=dict(size=13, color="white",
                                         line=dict(width=2.4, color="#111827")),
                             hoverinfo="skip", showlegend=False))

    # 4. 역명 label — 캔버스 좌표에 묶인 SVG 이미지 (D-067)
    #    Plotly text 는 화면 픽셀 고정이라 확대해도 커지지 않는다. 이미지로 얹으면 확대 배율만큼 글자도 커져
    #    지도 앱처럼 전체 보기에서는 작게, 확대하면 읽기 좋게 보인다. layer="above" 라 노선·경로 띠 위에 그려진다.
    if lb is not None and len(lb):
        fig.add_layout_image(**_map_labels.layout_image(lb, CANVAS_W, CANVAS_H))

    # 5. 추천 경로 highlight (반투명 띠 + 구간별 원래 노선색)
    if highlight_path:
        xs, ys = path_to_visual_nodes(highlight_path, vn)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line=dict(color=PRIMARY, width=14),
                                     opacity=0.38, showlegend=False))
        run = []
        for node in list(highlight_path) + [None]:
            ln = node.split("_", 1)[0] if node else None
            if run and (ln != run[0][0]):
                px, py = path_to_visual_nodes([n for _, n in run], vn)
                if len(px) >= 2:
                    fig.add_trace(go.Scatter(x=px, y=py, mode="lines", hoverinfo="skip",
                                             line=dict(color=LINE_COLORS.get(run[0][0], "#888"), width=6),
                                             showlegend=False))
                run = []
            if node:
                run.append((ln, node))

    # 6. 출발/도착 halo
    if ca is not None and show_endpoints:
        for key, color in ((st.session_state["origin_station_key"], "#1B7F3B"),
                           (st.session_state["destination_station_key"], DANGER)):
            if not key:
                continue
            row = ca[ca["station_key"] == key]
            if row.empty:
                continue
            fig.add_trace(go.Scatter(
                x=row["click_x_px"], y=row["click_y_px"], mode="markers",
                marker=dict(size=30, color="rgba(0,0,0,0)",
                            line=dict(width=4, color=color)),
                hoverinfo="skip", showlegend=False))

    # 7. 투명 클릭 영역 (가장 위). 역 중심을 정확히 누르지 않아도 선택된다.
    if ca is not None:
        # 워크북 click_radius_px 를 화면 크기로 환산 (일반 24 / 환승 48 / 복합 56)
        csize = np.clip(pd.to_numeric(ca["click_radius_px"], errors="coerce")
                        .fillna(24) * 0.55, 16, 34)
        fig.add_trace(go.Scatter(
            x=ca["click_x_px"], y=ca["click_y_px"], mode="markers",
            marker=dict(size=csize, color="rgba(0,0,0,0)",
                        line=dict(width=0, color="rgba(0,0,0,0)")),
            customdata=ca[["station_key", "display_name", "available_lines"]].values,
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]}호선<extra></extra>",
            showlegend=False))

    bx0, bx1, by0, by1 = map_bounds(bundle)
    fig.update_layout(
        height=820, margin=dict(l=2, r=2, t=2, b=2),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        # minallowed / maxallowed 로 팬·줌 범위를 초기 화면 밖으로 나가지 못하게 묶는다.
        # (plotly.js 2.24+ 지원. 구버전에서는 무시되며 동작에는 지장 없다)
        # minallowed/maxallowed 로 초기 화면 밖으로 나가지 못하게 묶는다.
        xaxis=dict(visible=False, range=[bx0, bx1], fixedrange=False,
                   minallowed=bx0, maxallowed=bx1),
        yaxis=dict(visible=False, range=[by1, by0],   # 이미지 좌표계: y 아래로 증가
                   minallowed=by0, maxallowed=by1,
                   scaleanchor="x", scaleratio=1),
        clickmode="event+select", dragmode="pan")
    return fig


from html import escape as _esc

from yeoyuro_v2 import map_labels as _map_labels


def _set_endpoint(kind: str, key: str):
    """지도에서 고른 역을 출발/도착으로 지정한다.

    selectbox 위젯 state 는 위젯 생성 뒤에는 수정할 수 없으므로 on_click 콜백에서 바꾼다.
    """
    if kind == "origin":
        st.session_state["origin_station_key"] = key
        st.session_state["sel_origin"] = key
    else:
        st.session_state["destination_station_key"] = key
        st.session_state["sel_dest"] = key
    st.session_state["selected_station_key"] = None


def _reset_all():
    for k in ("origin_station_key", "destination_station_key",
              "selected_station_key", "route_result", "route_fallback", "route_query", "ts_handoff"):
        st.session_state[k] = None
    for k in ("sel_origin", "sel_dest"):
        st.session_state[k] = None


def congestion_line_chart(piv: pd.DataFrame, height: int = 300):
    """시간대별 혼잡도 라인차트.

    st.line_chart(Vega-Lite)는 휠 스크롤로 축이 무한히 밀린다.
    fixedrange 를 건 Plotly 로 대체해 이동/확대를 막는다.
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    palette = [PRIMARY, SECONDARY, "#2E7D32", "#B23A48"]
    for i, c in enumerate(piv.columns):
        fig.add_trace(go.Scatter(x=list(piv.index), y=piv[c], mode="lines",
                                 name=str(c), line=dict(width=2.4,
                                                        color=palette[i % len(palette)])))
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        xaxis=dict(fixedrange=True, tickangle=-45, gridcolor="#EEF0F2"),
        yaxis=dict(fixedrange=True, title="기대 혼잡도(%)", gridcolor="#EEF0F2",
                   rangemode="tozero"),
        dragmode=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False,
                                                  "scrollZoom": False})


def congestion_tone(v):
    """혼잡 위험 색. 노선색과 섞이지 않게 주황/빨강 계열로만 제한한다."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#6B7280", ""
    if v >= 130:
        return DANGER, "⚠️ "
    if v >= 100:
        return WARN, "⚠️ "
    return "#6B7280", ""


@st.cache_data(show_spinner=False)
def transfer_tip(station: str, from_line: str, to_line: str,
                 prev_station: str | None = None, next_station: str | None = None):
    """환승 동선상 유리한 호차/문.

    같은 환승역이라도 **진행 방향에 따라 위치가 정반대**다.
    예) 잠실 8→2 : 몽촌토성 방면 하차 6-4 / 석촌 방면 하차 1-1

    그래서 소요시간만 보고 고르면 안 되고, 실제 경로의 앞뒤 역으로 방면을 좁힌다.
      arrive_toward  : 타고 온 열차가 향하는 방면 = 직전 역이 **아닌** 쪽
      depart_toward  : 갈아탄 열차가 향하는 방면 = 다음 역과 일치하는 쪽
    원본에 없는 조합은 추정하지 않는다.
    """
    tip = load_mart("transfer_tip_mart")
    if tip is None:
        return None
    sub = tip[(tip["station_name"] == station)
              & (tip["from_line"].astype(str) == str(from_line))
              & (tip["to_line"].astype(str) == str(to_line))]
    if sub.empty:
        return None

    matched = sub
    if next_station:
        m = matched[matched["depart_toward"].astype(str).str.contains(
            next_station, regex=False, na=False)]
        if not m.empty:
            matched = m
    if prev_station:
        m = matched[~matched["arrive_toward"].astype(str).str.contains(
            prev_station, regex=False, na=False)]
        if not m.empty:
            matched = m

    exact = bool(next_station or prev_station) and len(matched) < len(sub)
    r = matched.sort_values("transfer_time_min").iloc[0]

    def _pos(car, door):
        if pd.isna(car) or pd.isna(door):
            return None
        def _n(v):
            try:
                return str(int(float(v)))
            except (TypeError, ValueError):
                return str(v).strip()
        # 원본에 '모든 호차'/'모든 문' 으로 적힌 조합이 있다(예: 성수 본선-성수지선).
        # 접미사를 그대로 붙이면 '모든 호차호차' 가 되므로 원문을 그대로 쓴다.
        c, d = _n(car), _n(door)
        c = "모든 칸" if c.startswith("모든") else "%s번째 칸" % c     # 화면 용어는 'n번째 칸'으로 통일 (D-066)
        d = d if d.startswith("모든") else "%s번 문" % d
        return "%s %s" % (c, d)

    return {"alight": _pos(r["alight_car"], r["alight_door"]),
            "board": _pos(r["board_car"], r["board_door"]),
            "arrive_toward": str(r["arrive_toward"]),
            "depart_toward": str(r["depart_toward"]),
            "direction_matched": exact}


TIMELINE_CSS = """
<style>
.mc-tl { margin: 2px 0 6px 0; }
.mc-seg { border-left: 6px solid var(--mc-line); background: #FAFBFC;
          border-radius: 0 10px 10px 0; padding: 10px 14px; margin: 6px 0;
          border-top: 1px solid #E5E7EB; border-right: 1px solid #E5E7EB;
          border-bottom: 1px solid #E5E7EB; }
.mc-badge { display:inline-block; background: var(--mc-line); color:#fff;
            font-size:0.78rem; font-weight:700; padding:2px 9px;
            border-radius:11px; margin-right:8px; }
.mc-dir { color:#4B5563; font-size:0.82rem; }
.mc-od { font-size:1.02rem; font-weight:700; color:#111827; margin-top:5px; }
.mc-meta { color:#4B5563; font-size:0.84rem; margin-top:4px; }
.mc-tf { background:#FFFFFF; border:1px dashed #9CA3AF; border-radius:10px;
         padding:9px 14px; margin:6px 0 6px 22px; }
.mc-tf-h { font-weight:700; color:#002D56; font-size:0.94rem; }
.mc-tf-b { color:#4B5563; font-size:0.84rem; margin-top:3px; }
.mc-note { color:#6B7280; font-size:0.76rem; margin-top:4px; }
.mc-acc { background:#EEF5FF; border:1px solid #BFD7F5; border-radius:8px; padding:7px 10px; margin-top:8px; }
.mc-acc-h { color:#0B3D75; font-size:0.86rem; }
.mc-acc-t { color:#0B3D75; font-size:1.0rem; font-weight:800; }
.mc-acc-s { color:#4B5563; font-size:0.82rem; font-weight:500; }
.mc-acc-b { color:#1F2937; font-size:0.86rem; margin-top:3px; }
.mc-acc-w { color:#9A3412; font-size:0.82rem; margin-top:4px; font-weight:600; }
@media (prefers-color-scheme: dark) {
  .mc-acc { background:#14263A; border-color:#2F4A6B; } .mc-acc-h, .mc-acc-t { color:#9EC5F0; } .mc-acc-b { color:#E5E7EB; }
  .mc-acc-w { color:#FDBA74; }
  .mc-seg { background:#1B1F24; border-color:#374151; }
  .mc-od { color:#F3F4F6; } .mc-meta, .mc-dir, .mc-tf-b { color:#C7CBD1; }
  .mc-tf { background:#15181C; } .mc-tf-h { color:#9EC5F0; }
}
</style>
"""


def _tip_pos(car, door):
    """호차-문 을 '2-1' 형태로 합친다 (환승 상세 표)."""
    if pd.isna(car) or pd.isna(door):
        return "-"

    def _n(v):
        try:
            return str(int(float(v)))
        except (TypeError, ValueError):
            return str(v).strip()
    c, d = _n(car), _n(door)
    if c.startswith("모든") or d.startswith("모든"):
        return "%s %s" % (c, d)
    return "%s-%s" % (c, d)


def transfer_detail(stn: str, fl: str, tl: str):
    """환승 구간 '자세히 보기' 내용 (예전 '빠른 환승 안내' 화면의 표 + 환승 소요·인원).

    metric 카드를 쓰지 않는다: 경로 결과 화면의 KPI 수를 늘리지 않기 위해서다.
    """
    # 순서: 요약 한 줄(소요·인원·페널티) → 방면별 위치 표 → 읽는 법 두 줄
    tip = load_mart("transfer_tip_mart")
    sub3 = None if tip is None else tip[(tip["station_name"] == stn)
                                        & (tip["from_line"].astype(str) == str(fl))
                                        & (tip["to_line"].astype(str) == str(tl))]
    has_tip = sub3 is not None and not sub3.empty
    te = load_mart("transfer_edge_mart")
    row = None if te is None else te[(te["station_name"] == stn) & (te["from_line"].astype(str) == str(fl))
                                     & (te["to_line"].astype(str) == str(tl))]
    has_edge = row is not None and not row.empty
    if has_edge:
        r = row.iloc[0]
        parts = ["환승 소요시간 %.1f분" % r["transfer_time_min"]]
        if pd.notna(r.get("weekday_volume")):
            parts.append("평일 환승인원 %s명" % f'{int(r["weekday_volume"]):,}')
        parts.append("환승 페널티 %.1f분" % r["transfer_penalty_min"])
        st.markdown("  ·  ".join(parts))
    if has_tip:
        view = pd.DataFrame({
            "타고 온 방면": sub3["arrive_toward"].values,
            "하차 위치": [_tip_pos(a, b) for a, b in zip(sub3["alight_car"], sub3["alight_door"])],
            "갈아탈 방면": sub3["depart_toward"].values,
            "승차 위치": [_tip_pos(a, b) for a, b in zip(sub3["board_car"], sub3["board_door"])],
            "환승 소요(분)": np.round(pd.to_numeric(sub3["transfer_time_min"], errors="coerce"), 1).values,
        })
        st.table(round1(view).reset_index(drop=True))
    else:
        st.caption("이 환승 조합은 원본 자료에 빠른 환승 위치가 없어 안내하지 않습니다.")
    notes = []
    if has_tip:
        notes.append("승하차 위치는 칸-문 입니다, (2-1 = 2번째 칸 1번 문)")
    if has_edge:
        notes.append("환승 페널티 = 도보 소요시간 + 환승인원 기반 가산(5만 명당 1분, 최대 3분)")
    if notes:
        st.caption("  \n".join(notes))


def car_hints(ev) -> dict:
    """여유 칸 안내(시범, 공모전 P0-2). 평일 + 패턴 파일이 있을 때만. 경로 계산에는 관여하지 않는다."""
    try:
        if _query_args()[0] != "weekday":
            return {}
        f = DATA_ROOT / "data" / "marts" / "car_pattern_sample.parquet"
        if not f.exists():
            return {}
        pat = load_mart("car_pattern_sample", stamp=f.stat().st_mtime)
        if pat is None:
            return {}
        from yeoyuro_v2.car_pattern import route_hints
        dep = ev.get("depart_min")
        if dep is None:
            t = st.session_state["q_time"]
            dep = t.hour * 60 + t.minute
        return route_hints(pat, ev.get("segments") or [], float(dep))
    except Exception:
        return {}


# 하차 칸 화면 문구는 Streamlit 밖 순수 모듈에서 만든다 (모든 상태를 단위 테스트로 훑기 위해, 2026-10-17 KeyError 이후)
from yeoyuro_v2.facility_view import (CAR_HINT_NOTE, car_door_text as _car_door_text,  # noqa: E402
                                      decision as facility_decision, evidence as view_evidence)
ACCESS_LABEL = "♿ 이동편의 우선"
ACCESS_HELP = ("휠체어·유모차·큰 짐 이용자를 위한 하차 칸 안내를 켭니다. 도착역 엘리베이터와 가까운 칸을 기준으로, "
               "4호선 평일에는 주변 칸의 평소 혼잡도까지 비교해 추천합니다. 경로 계산 자체는 바뀌지 않습니다.")

# ---- 공개 배포판 / 시연판 (공모전판)
#   공모전 제공 샘플(칸별 혼잡도·역사 도면)과 이용허락범위를 확인하지 않은 파생 자료(빠른하차정보 엘리베이터 표)는
#   git 에 올리지 않는다. 그래서 공개 배포판에서는 해당 기능이 없다 — '없는 것처럼' 숨기지 않고 이유를 보여준다.
PUBLIC_SAMPLE_NOTE = ("공모전 제공 샘플을 활용한 기능으로, 공개 배포판에서는 원본 데이터 이용조건 확인 전 "
                      "비활성화되어 있습니다. 발표·시연은 전체 데이터로 진행합니다.")
PUBLIC_ELEVATOR_NOTE = ("서울교통공사 빠른하차정보(공공데이터포털)를 활용한 기능으로, 공개 배포판에서는 "
                        "이용허락범위 확인 전 비활성화되어 있습니다. 발표·시연은 전체 데이터로 진행합니다.")
FAST_EXIT_NOTE = "서울교통공사 빠른하차정보"


def _pattern_file() -> Path:
    return DATA_ROOT / "data" / "marts" / "car_pattern_sample.parquet"


def _load_car_pattern():
    f = _pattern_file()
    return load_mart("car_pattern_sample", stamp=f.stat().st_mtime) if f.exists() else None


def data_mode() -> dict:
    """어떤 공모전 자료가 이 실행 환경에 있는지. 사이드바 표기와 비활성 안내에 쓴다."""
    spatial = ROOT / "data" / "raw" / "spatial_dongmyo"
    return {"elevator": load_mart("fast_exit_elevator") is not None,
            "car_sample": _pattern_file().exists(),
            "spatial": spatial.exists() and any(spatial.glob("*.shp"))}


def _fmt_ymd(s) -> str:
    s = str(s or "")
    return "%s.%s.%s." % (s[:4], s[4:6], s[6:8]) if len(s) == 8 and s.isdigit() else s


# ---- 칸 혼잡도 색 (배경 = 절대 혼잡도). 역할(엘리베이터·추천)은 테두리·배지로만 표시한다.
#      '추천'은 후보 가운데 상대적으로 덜 붐빈다는 뜻이지 쾌적하다는 뜻이 아니므로 추천 칸을 초록으로 칠하지 않는다.
CONG_BANDS = ((80.0, "#E6F4EA", "여유 80% 미만"), (100.0, "#FFF3CD", "보통 80~100%"),
              (130.0, "#FFDDBA", "혼잡 100~130%"), (float("inf"), "#FFC4C4", "매우 혼잡 130% 이상"))
EV_BORDER, PICK_BORDER = "#1D4ED8", "#15803D"


def _cong_bg(v) -> str:
    if v is None:
        return "#F3F4F6"
    for hi, bg, _ in CONG_BANDS:
        if v < hi:
            return bg
    return CONG_BANDS[-1][1]


def car_strip_html(n: int, pct: list[int] | None, ev_labels: dict[int, str], pick: int | None = None,
                   gain_pct: int | None = None, toward: str = "", compact: bool = False) -> str:
    """가로형 열차 칸 그림 (한 줄, 좁은 화면에서는 가로 스크롤).

    배경 = 평소 혼잡도(pct 가 있을 때), 파란 테두리·♿ = 엘리베이터 칸, 초록 테두리·★ = 추천 칸."""
    w = 46 if compact else 66
    cells = []
    for c in range(1, n + 1):
        v = pct[c - 1] if pct and c - 1 < len(pct) else None
        el, pk = c in ev_labels, (pick == c and c not in ev_labels)
        bd = ("3px solid %s" % EV_BORDER) if el else (("3px solid %s" % PICK_BORDER) if pk else "1px solid #CBD5E1")
        badge = ""
        if el:
            badge = '<div style="color:%s;font-weight:700;font-size:0.72rem">♿ EV %s</div>' % (EV_BORDER, _esc(ev_labels[c]))
        elif pk:
            badge = ('<div style="color:%s;font-weight:700;font-size:0.72rem">★ 추천</div>' % PICK_BORDER
                     + ('<div style="color:%s;font-size:0.72rem">−%d%%p</div>' % (PICK_BORDER, gain_pct)
                        if gain_pct else ""))
        val = '<div style="font-size:%s;color:#111827">%s</div>' % (
            "0.78rem" if compact else "0.9rem", ("%d%%" % v) if v is not None else "&nbsp;")
        cells.append('<div style="flex:1 0 %dpx;min-width:%dpx;background:%s;border:%s;border-radius:7px;'
                     'padding:%s;text-align:center;line-height:1.25">'
                     '<div style="font-size:0.74rem;color:#4B5563">%d</div>%s%s</div>'
                     % (w, w, _cong_bg(v), bd, "4px 2px" if compact else "7px 2px", c, val, badge))
    head = ('<div style="color:#4B5563;font-size:0.8rem;margin:2px 0 3px 0">%s방면 열차 · %d칸</div>'
            % ((_esc(toward) + " ") if toward else "", n))
    return ('%s<div style="overflow-x:auto;padding-bottom:2px"><div style="display:flex;gap:4px;min-width:%dpx">'
            '%s</div></div>' % (head, n * (w + 4), "".join(cells)))


def strip_legend_html(with_cong: bool) -> str:
    sw = ('<span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:%s;'
          'border:1px solid #CBD5E1;margin:0 3px -1px 8px"></span>%s')
    parts = [sw % (bg, lab) for _, bg, lab in CONG_BANDS] if with_cong else []
    parts.append('<span style="color:%s;font-weight:700;margin-left:10px">▢ 파랑</span> 엘리베이터 칸' % EV_BORDER)
    parts.append('<span style="color:%s;font-weight:700;margin-left:10px">▢ 초록</span> 추천 칸 '
                 '(후보 중 상대적으로 덜 붐빔 · 쾌적하다는 뜻은 아님)' % PICK_BORDER)
    return '<div style="color:#6B7280;font-size:0.74rem;margin-top:4px">%s</div>' % "".join(parts)


def facility_hint(ev) -> dict:
    """교통약자 하차 칸 (공모전판). '이동편의 우선'이 켜졌을 때만. 경로 계산에는 관여하지 않는다.

    반환: {"seg": 마지막 승차 구간 번호, "station", "line", "toward", "prev", "elevators", "crowd", "public"}
    엘리베이터 자료가 없거나 진행 방향을 하나로 정할 수 없으면 elevators 가 빈 목록이다 (틀린 칸을 안내하지 않는다)."""
    if not st.session_state.get("q_access"):
        return {}
    try:
        from yeoyuro_v2.facility_match import alight_crowd_match, alight_elevators
        segs = ev.get("segments") or []
        rides = [i for i, s in enumerate(segs) if s.get("kind") == "ride"]
        if not rides:
            return {}
        i = rides[-1]
        sg = segs[i]
        stns = list(sg.get("stations") or [])
        if len(stns) < 2:
            return {}
        line = str(sg.get("line"))
        out = {"seg": i, "station": stns[-1], "prev": stns[-2], "line": line, "elevators": [], "crowd": None,
               "toward": None, "public": None, "supported": line in {str(n) for n in range(1, 9)}}
        mart = load_mart("fast_exit_elevator")
        if mart is None:
            out["public"] = PUBLIC_ELEVATOR_NOTE
            return out
        out["elevators"] = alight_elevators(mart, line, stns[-1], stns[-2])
        if out["elevators"]:
            out["toward"] = out["elevators"][0]["toward"]
        if out["elevators"] and line == "4" and _query_args()[0] == "weekday":
            pat = _load_car_pattern()
            if pat is None:
                out["public"] = PUBLIC_SAMPLE_NOTE
            else:
                dep = ev.get("depart_min")
                if dep is None:
                    t = st.session_state["q_time"]
                    dep = t.hour * 60 + t.minute
                t_arr = float(dep)
                for s in segs[:i]:
                    t_arr += float(s.get("minutes") or 0) + float(s.get("wait_min") or 0)
                t_arr += float(sg.get("minutes") or 0)
                # 하차역 직전 역간을 지나는 시각 = 도착 1분 전
                out["crowd"] = alight_crowd_match(pat, line, stns[-2], stns[-1], t_arr - 1, out["elevators"])
        return out
    except Exception:
        return {}


def facility_html(fac: dict) -> str:
    """마지막 승차 구간 카드 안의 '하차 칸 결론' (결론 → 근거 → 출처 순)."""
    if not fac:
        return ""
    els = fac.get("elevators") or []
    if not els:
        if fac.get("public"):
            why = fac["public"]
        elif fac.get("supported"):
            why = "이 방향의 엘리베이터 칸 정보가 없습니다 · 역 안내를 확인해 주세요"
        else:
            why = "%s은 빠른하차정보 제공 범위(1~8호선) 밖입니다 · 역 안내를 확인해 주세요" % line_label(fac.get("line"))
        return ('<div class="mc-acc"><div class="mc-acc-h">♿ %s 도착</div><div class="mc-note">%s</div></div>'
                % (_esc(fac["station"]), _esc(why)))
    d = facility_decision(fac["station"], fac["line"], els, fac.get("crowd"), fac.get("public"))
    body = ['<div class="mc-acc"><div class="mc-acc-t">%s <span class="mc-acc-s">· %s</span></div>'
            % (_esc(d["title"]), _esc(fac["station"]))]
    body += ['<div class="mc-acc-b">%s</div>' % ln for ln in d["lines"]]
    if d["warn"]:
        body.append('<div class="mc-acc-w">⚠️ %s 아래 출발 시각 대안을 함께 확인해 보세요.</div>' % _esc(d["warn"]))
    src = "%s · %s 기준" % (FAST_EXIT_NOTE, _fmt_ymd(els[0].get("crtr_ymd")))
    if d["note"]:
        src += " · " + d["note"]
    body.append('<div class="mc-note">%s</div></div>' % _esc(src))
    return "".join(body)


def _open_facility(line: str, station: str, toward: str | None, time_bin: str | None):
    """경로 결과 → 역사 안내(같은 역·방면·시간대)."""
    st.session_state["fac_line"] = str(line)
    st.session_state["fac_station"] = station
    if toward:
        st.session_state["fac_toward"] = toward
    if time_bin:
        st.session_state["fac_bin"] = time_bin
    _go("역사 안내", "엘리베이터 하차 칸")


def render_timeline(segs, hints: dict | None = None, facility: dict | None = None):
    """노선형 타임라인 카드. 사용자에게 내부 id 를 노출하지 않는다.

    환승 구간마다 빠른 환승 위치·예상 환승 시간을 바로 보여주고, [자세히 보기] 에 방면별 표를 둔다
    (D-057: '빠른 환승 안내' 독립 메뉴를 경로 결과 안으로 옮김).
    """
    html = [TIMELINE_CSS, '<div class="mc-tl">']

    def flush():
        if html:
            st.markdown("".join(html) + "</div>", unsafe_allow_html=True)
            html.clear()
            html.append('<div class="mc-tl">')

    for i, sg in enumerate(segs):
        if sg["kind"] == "ride":
            color = LINE_COLORS.get(str(sg["line"]), PRIMARY)
            # 방면은 경로상 다음 역 기준. 1정거장 구간은 생략한다.
            dirlab = sg.get("direction_label") or ""
            ctone, cicon = congestion_tone(sg.get("max_congestion"))
            # 구버전 segment 나 precomputed fallback 에는 일부 키가 없을 수 있다.
            mins, cong = sg.get("minutes"), sg.get("max_congestion")
            meta = "🚉 %d개 역 이동" % sg.get("n_stops", 0)
            if mins is not None:
                meta += " · ⏱️ 약 %d분" % round(mins)
            if cong is not None:
                meta += (' · <span style="color:%s">%s최대 기대 혼잡도 %.0f%%</span>'
                         % (ctone, cicon, cong))
            hint = (hints or {}).get(i)
            fac_here = bool(facility) and facility.get("seg") == i
            if fac_here:
                hint = None            # 교통약자 안내가 켜지면 마지막 구간은 엘리베이터 기준 안내로 대신한다
            hint_html = ""
            if hint:
                hint_html = ('<div class="mc-meta">🚃 여유 칸(시범): <b>%s번째 칸</b> — 평소 이 시간 열차 평균보다 '
                             '약 %.0f%%p 덜 붐빔</div><div class="mc-note">%s</div>'
                             % ("·".join(str(c) for c in hint["cars"]), hint["gain"], _esc(CAR_HINT_NOTE)))
                if hint.get("all_crowded"):
                    hint_html += ('<div class="mc-note">이 칸도 평소 정원 100%% 이상입니다 · '
                                  '출발 시각을 바꾸는 대안을 함께 확인해 보세요</div>')
            if fac_here:
                hint_html += facility_html(facility)
            html.append(
                '<div class="mc-seg" style="--mc-line:%s">'
                '<span class="mc-badge">%s</span>'
                '<span class="mc-dir">%s</span>'
                '<div class="mc-od">%s → %s</div>'
                '<div class="mc-meta">%s</div>%s'
                '</div>'
                % (color, _esc(line_label(sg["line"])), _esc(dirlab),
                   _esc(sg["from"]), _esc(sg["to"]), meta, hint_html))
        else:
            # 진행 방향을 좁히기 위해 직전/다음 역을 넘긴다.
            prev_st = next_st = None
            if i > 0 and segs[i - 1].get("kind") == "ride":
                st_list = segs[i - 1].get("stations") or []
                prev_st = st_list[-2] if len(st_list) >= 2 else None
            if i + 1 < len(segs) and segs[i + 1].get("kind") == "ride":
                st_list = segs[i + 1].get("stations") or []
                next_st = st_list[1] if len(st_list) >= 2 else None
            tip = transfer_tip(sg["at"], sg["from_line"], sg["to_line"], prev_st, next_st)
            tmin, wmin = sg.get("minutes"), sg.get("wait_min")
            # 1줄: 빠른 하차/승차 위치 (없으면 '미제공')
            if tip and (tip["alight"] or tip["board"]):
                rows = []
                if tip["alight"]:
                    rows.append("빠른 하차 %s" % _esc(tip["alight"]))
                if tip["board"]:
                    rows.append("승차 %s" % _esc(tip["board"]))
                pos = " · ".join(rows)
                # 원본에 자기 역명이 방면으로 적힌 이상값이 있다(강동 5→5).
                # 해당 방면 문구만 생략하고 호차/문 안내는 그대로 유지한다.
                from station_routing import transfer_basis_text
                note = (transfer_basis_text(sg["at"], tip["arrive_toward"],
                                            tip["depart_toward"])
                        if tip.get("direction_matched") else "")
            else:
                pos, note = "빠른 환승 위치 정보 미제공", ""
            # 2줄: 예상 환승 시간 (도보 + 대기)
            if tmin is not None or wmin:
                walk = max(1, round(tmin)) if tmin is not None else 0
                wait = max(1, round(wmin)) if wmin else 0
                det = []
                if tmin is not None:
                    det.append("도보 약 %d분" % walk)
                if wmin:
                    det.append("대기 약 %d분" % wait)
                eta = "예상 환승 약 %d분 (%s)" % (walk + wait, " · ".join(det))
            else:
                eta = ""
            html.append(
                '<div class="mc-tf"><div class="mc-tf-h">↓ %s · %s → %s</div>'
                '<div class="mc-tf-b">🚶 %s%s</div>%s</div>'
                % (_esc(sg["at"]), _esc(line_label(sg["from_line"])), _esc(line_label(sg["to_line"])),
                   pos, (" · " + eta) if eta else "",
                   ('<div class="mc-note">%s</div>' % _esc(note)) if note else ""))
            flush()
            with st.expander("자세히 보기 · %s 환승" % sg["at"]):
                transfer_detail(sg["at"], str(sg["from_line"]), str(sg["to_line"]))
    st.markdown("".join(html) + "</div>", unsafe_allow_html=True)


def time_breakdown(ev):
    """예상 소요시간을 승차·중간정차·환승도보·환승대기로 분해한다.

    합계는 항상 actual_time_min 과 같아야 한다.
    최초 승차 전 대기시간은 어느 항목에도 들어가지 않는다.
    """
    wait = ev.get("transfer_wait_min") or 0
    walk = sum(sg.get("minutes") or 0 for sg in ev.get("segments") or []
               if sg.get("kind") == "transfer")
    dwell = ev.get("dwell_time_min") or 0
    # running = (승차+정차+도보) - 정차 - 도보. 구버전 결과에도 안전하다.
    run = (ev.get("ride_time_min") or 0) - dwell - walk
    return run, dwell, walk, wait


def route_summary_line(ev, o, d_):
    lines = []
    for sg in ev["segments"]:
        if sg["kind"] == "ride" and sg["line"] not in lines:
            lines.append(sg["line"])
    run, dwell, walk, wait = time_breakdown(ev)
    tail = (" (승차 %d분 + 환승·대기 %d분)"
            % (round(run + dwell), round(walk + wait))
            ) if (wait or walk) else ""
    return "%s → %s · %s · 환승 %d회 · 예상 %d분%s · 최대 기대 혼잡도 %.0f%%" % (
        o, d_, " + ".join(line_label(x) for x in lines),
        ev.get("transfer_count", 0), round(ev.get("actual_time_min") or 0), tail,
        ev.get("max_congestion") or 0)


CONG_REF_LINES = ((80.0, "체감 가중 시작 80%", "#9CA3AF"),
                  (100.0, "정원 100% (1칸 160명)", "#E8A33D"),
                  (130.0, "혼잡 주의 130%", "#D9534F"))

PROFILE_COLORS = {"current": "#1F4E79", "recommended": "#2E7D5B"}


def _current_scorer():
    """경로 검색에 쓴 것과 같은 인자로 scorer 를 얻는다.

    get_scorer 는 @st.cache_resource 라 같은 인자면 이미 만들어진 객체를 준다.
    여기서 새로 계산하는 것은 없다.
    """
    dow = pd.Timestamp(st.session_state["q_date"]).dayofweek
    day_type = "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")
    hhmm = "%02d:%02d" % (st.session_state["q_time"].hour,
                          st.session_state["q_time"].minute)
    return get_scorer(day_type, hhmm, str(st.session_state["q_date"]))


def build_route_congestion_profile(scorer, path, time_bin=None, edge_congestion=None):
    """경로를 승차 구간(segment)별로 나눠 기대 혼잡도 계열을 만든다.

    혼잡도는 역이 아니라 **구간**의 값이다. 재차율(X→Y)은 X 와 Y 사이 열차의
    혼잡도이므로, 역에 값을 붙이려면 규칙이 필요하다.

        구간의 중간역   그 역을 떠나는 엣지의 값
        구간의 마지막역 그 역에 도착하는 엣지의 값

    이렇게 하면 두 가지가 해결된다.
      - 도착역까지 선이 이어진다(마지막 역도 값을 갖는다)
      - 환승역에서 두 호선의 값이 각각 표시된다
        (예: 4호선 사당 = 이수→사당 구간, 2호선 사당 = 사당→방배 구간)

    직결 분기 통과(5호선 강동)는 환승이 아니므로 구간을 끊지 않는다.

    time_bin 을 주면 그 시간대로 조회한다. 경로가 같으므로 x축이 그대로 맞는다.
    이벤트 배수는 time_alternative() 와 같은 방식으로 적용해, 화면 문구의
    최대값과 그래프 최대값이 어긋나지 않게 한다.

    반환: [{"line_id", "stations": [...], "values": [...]}, ...]
    """
    sr = sr_module()
    ride_idx = {(r.from_node, r.to_node): (str(r.line_id), r.direction)
                for r in scorer.ride.itertuples()}

    tb = time_bin or scorer.time_bin
    lk = scorer.lookup[(scorer.lookup["day_type"] == scorer.day_type)
                       & (scorer.lookup["time_bin"] == tb)]
    key = lk.set_index(["station_uid", "direction"])["congestion_median"].to_dict()

    eff = {}
    if getattr(scorer, "event_effect_by_hour", None):
        eff = scorer.event_effect_by_hour.get(int(str(tb)[:2]), {}) or {}

    def cong(u, v):
        if edge_congestion is not None:
            c = edge_congestion.get((u, v))
            return None if c is None or pd.isna(c) else float(c)
        _, direction = ride_idx.get((u, v), ("", None))
        c = key.get((u, direction))
        if c is None or pd.isna(c):
            return None
        return float(c) * float(eff.get(sr.station_of(u), {}).get("mult", 1.0))

    segments, cur = [], None
    for i, (u, v) in enumerate(zip(path, path[1:])):
        e = next((x for x in scorer.adj.get(u, []) if x["to"] == v), None)
        if e is None:
            continue
        if e["kind"] != "ride":
            # 직결 분기 통과는 같은 열차이므로 구간을 끊지 않는다.
            if e["kind"] == "transfer" and not sr._is_through_pass(scorer, path, i, u, v):
                cur = None
            continue
        line_id = ride_idx.get((u, v), ("", None))[0]
        c = cong(u, v)
        if cur is None:
            cur = {"line_id": line_id, "stations": [], "values": []}
            segments.append(cur)
            cur["stations"].append(sr.station_of(u))
            cur["values"].append(c)
        else:
            # 직전 역은 '떠나는 구간' 값으로 갱신한다.
            cur["values"][-1] = c
        cur["stations"].append(sr.station_of(v))
        cur["values"].append(c)      # 마지막 역은 '도착하는 구간' 값

    return [s for s in segments if len(s["stations"]) >= 2]


def render_route_congestion_profile(ev, time_alt=None, key=None):
    """추천 경로의 기대 혼잡도 흐름을 하나의 그래프로 그린다."""
    import plotly.graph_objects as go

    try:
        scorer = _current_scorer()
        cur_segs = build_route_congestion_profile(scorer, ev["path"],
                                                  edge_congestion=ev.get("edge_congestion"))
    except Exception:
        return
    if not cur_segs:
        return

    names = []
    for sg in cur_segs:
        for s in sg["stations"]:
            if not names or names[-1] != s:
                names.append(s)

    fig = go.Figure()

    def add_series(segs, label, color, dash, width, marker):
        first = True
        for sg in segs:
            fig.add_trace(go.Scatter(
                x=sg["stations"], y=sg["values"], mode="lines+markers",
                name=label, legendgroup=label, showlegend=first,
                line=dict(color=color, width=width, dash=dash),
                marker=dict(size=marker),
                customdata=[[sg["line_id"], label]] * len(sg["stations"]),
                hovertemplate=("역: %{x}<br>호선: %{customdata[0]}호선"
                               "<br>기대 혼잡도: %{y:.0f}%"
                               "<br>시간: %{customdata[1]}<extra></extra>")))
            first = False

    # 추천 시간대 곡선. 경로가 같을 때만 겹친다.
    alt_label = None
    if time_alt and time_alt.get("best_time_bin"):
        try:
            if time_alt.get("td") and ev.get("edge_congestion") is not None:
                day_type, hhmm, qd = _query_args()
                tde = get_td_evaluator(day_type, hhmm, qd)
                r = tde.evaluate(ev["path"], time_alt["best_depart_min"], keep_trace=True)
                alt_segs = build_route_congestion_profile(
                    scorer, ev["path"],
                    edge_congestion={(t.u, t.v): t.congestion for t in r.trace if t.kind == "ride"})
            else:
                alt_segs = build_route_congestion_profile(
                    scorer, ev["path"], time_bin=time_alt["best_time_bin"])
        except Exception:
            alt_segs = []
        same = (len(alt_segs) == len(cur_segs)
                and all(a["stations"] == b["stations"]
                        for a, b in zip(alt_segs, cur_segs)))
        if same:
            alt_label = "추천 출발 %s" % (time_alt.get("best_depart")
                                          or str(time_alt["best_time_bin"])[:5])
            add_series(alt_segs, alt_label, PROFILE_COLORS["recommended"],
                       "dot", 2, 5)

    cur_label = "현재 출발 %s" % (scorer.depart if ev.get("edge_congestion") is not None
                                  else str(scorer.time_bin)[:5])
    add_series(cur_segs, cur_label, PROFILE_COLORS["current"], None, 2.4, 6)

    for val, label, color in CONG_REF_LINES:
        fig.add_hline(y=val, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=10, color=color))

    # 환승 지점 수직 점선. segments 에서 뽑으므로 강동 직결 분기처럼
    # 환승으로 세지 않는 통과 지점은 자동으로 빠진다.
    for sg in ev.get("segments") or []:
        if sg.get("kind") != "transfer" or sg.get("at") not in names:
            continue
        fig.add_vline(x=names.index(sg["at"]),
                      line=dict(color="#8D7150", width=1, dash="dash"),
                      annotation_text="환승 %s→%s" % (line_label(sg.get("from_line", "")),
                                                   line_label(sg.get("to_line", ""))),
                      annotation_position="top",
                      annotation_font=dict(size=10, color="#8D7150"))

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=46, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", dragmode=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        showlegend=bool(alt_label),
        xaxis=dict(fixedrange=True, tickangle=-45, automargin=True,
                   tickfont=dict(size=10), gridcolor="#F3F4F6",
                   categoryorder="array", categoryarray=names),
        yaxis=dict(fixedrange=True, title="기대 혼잡도(%)",
                   rangemode="tozero", gridcolor="#EEF0F2"))
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displayModeBar": False, "scrollZoom": False})



def route_card(title, ev, o, d_, note=None, tone="normal",
               show_profile=False, time_alt=None, profile_key=None):
    with st.container(border=True):
        st.markdown("**%s**" % title)
        st.caption(route_summary_line(ev, o, d_))
        c = st.columns(5)
        c[0].metric("예상 소요시간", "%.0f분" % ev["actual_time_min"])
        c[1].metric("쾌적 체감시간", "%.0f분" % ev["perceived_time_min"])
        c[2].metric("최대 기대 혼잡도", "%.0f%%" % (ev["max_congestion"] or 0))
        c[3].metric("환승", "%d회" % ev["transfer_count"])
        if ev.get("exposure_100_min") is not None:
            c[4].metric("100%+ 혼잡 노출", "%.0f분" % ev["exposure_100_min"],
                        help="차내에 있는 시간 중 기대 혼잡도 100%% 이상 구간을 지나는 시간. "
                             "과거 패턴의 중앙값 기준이며, 붐비는 날(p90) 기준으로는 %.0f분."
                             % ev.get("exposure_100_min_p90", 0))
        else:
            c[4].metric("혼잡 주의 구간", "%d개" % ev.get("p95_exposure_count", 0))

        fac = facility_hint(ev)
        render_timeline(ev["segments"], car_hints(ev), fac)
        if fac.get("elevators"):
            cr = fac.get("crowd") or {}
            st.button("♿ 역사 안내에서 %s 하차 칸 보기 →" % fac["station"], key="fac_cta_%s" % (profile_key or title),
                      on_click=_open_facility,
                      args=(fac["line"], fac["station"], fac.get("toward"), cr.get("time_bin")))
        if ev.get("td_policy"):
            st.caption("출발 %s → 도착 약 %02d:%02d / 혼잡도는 각 구간에 실제로 도착하는 시각의 과거 패턴으로 "
                       "계산한 것으로 실시간 정보가 아닙니다."
                       % (_query_args()[1], (int(round(ev["arrive_min"])) // 60) % 24,
                          int(round(ev["arrive_min"])) % 60))
        if ev.get("event_risk_min"):
            st.caption("이벤트 영향으로 쾌적 체감시간 +%.1f분 가산" % ev["event_risk_min"])
        if show_profile:
            st.markdown("**📊 구간별 기대 혼잡도**")
            render_route_congestion_profile(ev, time_alt, key=profile_key)
        if note:
            (st.success if tone == "good" else st.caption)(note)


def page_route():
    disp = load_display_master()
    bundle = load_map_bundle(ensure_map_fresh())
    if disp is None:
        st.warning("`station_display_master.csv` 가 없습니다. "
                   "`12_build_display_masters.py` 를 먼저 실행하세요.")
        return

    # 좌측 입력 패널은 좁게, 노선도는 넓게. 16:9 비율이라 세로도 함께 키운다.
    left, right = st.columns([0.78, 2.35], gap="small")

    # ---------- 왼쪽 패널 ----------
    with left:
        keys = sorted(disp["station_key"].tolist())   # 가나다순
        labels = {r.station_key: short_label(r.station_key, r.available_lines)
                  for r in disp.itertuples()}

        # key 를 쓰는 위젯에 index 를 함께 주면 Streamlit 이 경고를 낸다.
        #   "created with a default value but also had its value set via Session State"
        # 값은 session_state 로만 관리하고 index 는 주지 않는다.
        #   canonical(origin_station_key 등) 이 source of truth. 위젯 state 가 없을 때만 시드하고,
        #   위젯 → canonical 은 on_change 로만 옮긴다 (D-057).
        for w in ("sel_origin", "sel_dest", "w_date", "w_time", "w_mode"):
            _seed(w)

        o = st.selectbox("출발역", keys,
                         format_func=lambda k: labels.get(k, k),
                         placeholder="역 이름을 입력하세요", key="sel_origin",
                         on_change=_w2c, args=("sel_origin",))

        d_ = st.selectbox("도착역", keys,
                          format_func=lambda k: labels.get(k, k),
                          placeholder="역 이름을 입력하세요", key="sel_dest",
                          on_change=_w2c, args=("sel_dest",))

        # 노선 색은 selectbox 안에서 표현할 수 없으므로, 선택 결과를 한 줄로 요약해 보여준다.
        if o or d_:
            def _one(key, tag):
                if not key:
                    return ""
                lines = disp.loc[disp.station_key == key, "available_lines"].iloc[0]
                return "<span style='color:#6B7280;font-size:0.8rem'>%s</span> %s <b>%s</b>" % (
                    tag, line_badges(lines), key)
            parts = [x for x in (_one(o, "출발"), _one(d_, "도착")) if x]
            st.markdown(
                "<div style='margin:-6px 0 10px 0'>" + " &nbsp;→&nbsp; ".join(parts) + "</div>",
                unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.date_input("날짜", key="w_date", on_change=_w2c, args=("w_date",))
        c2.time_input("출발 시각", key="w_time", on_change=_w2c, args=("w_time",))
        st.selectbox("선호 모드", list(MODE_LABEL), format_func=lambda x: MODE_LABEL[x],
                     key="w_mode", on_change=_w2c, args=("w_mode",))
        _seed("w_access")
        st.toggle(ACCESS_LABEL, key="w_access", on_change=_w2c, args=("w_access",), help=ACCESS_HELP)

        do_search = st.button("경로 검색", width="stretch", type="primary")
        st.button("초기화", width="stretch", on_click=_reset_all)

        sel = st.session_state.get("selected_station_key")
        if sel:
            row = disp[disp.station_key == sel]
            if not row.empty:
                r = row.iloc[0]
                with st.container(border=True):
                    st.markdown(
                        "<div style='font-weight:700;margin-bottom:6px'>"
                        "선택한 역 · %s <span style='margin-left:4px'>%s</span></div>"
                        % (r["display_name"], line_badges(r["available_lines"])),
                        unsafe_allow_html=True)
                    b1, b2 = st.columns(2)
                    # selectbox 위젯 state(sel_origin/sel_dest)까지 같이 갱신해야
                    # 위쪽 입력창이 지도 선택과 동기화된다.
                    b1.button("출발역으로 설정", width="stretch",
                              key="btn_set_origin",
                              on_click=_set_endpoint, args=("origin", sel))
                    b2.button("도착역으로 설정", width="stretch",
                              key="btn_set_dest",
                              on_click=_set_endpoint, args=("dest", sel))
                    with st.expander("역 혼잡 정보"):
                        show_station_info(sel)

    # 출발/도착이 바뀌면 이전 결과와 노선도 하이라이트를 즉시 지운다.
    # 지도를 그리기 **전에** 실행해야 이전 경로가 한 번 더 그려지지 않는다.
    if (st.session_state.get("last_od")
            != (st.session_state["origin_station_key"],
                st.session_state["destination_station_key"])):
        st.session_state["last_od"] = (st.session_state["origin_station_key"],
                                       st.session_state["destination_station_key"])
        st.session_state["route_result"] = None
        st.session_state["route_fallback"] = None

    # ---------- 경로 계산 ----------
    #  지도를 그리기 **전에** 계산해야 [경로 검색] 한 번으로 하이라이트가 뜬다.
    #  계산 뒤에 그리면 그 실행에서는 이전 결과(또는 없음)로 그려져 두 번 눌러야 한다.
    o = st.session_state["origin_station_key"]
    d_ = st.session_state["destination_station_key"]
    ready = bool(o and d_ and o != d_)

    if ready and do_search:
        dow = pd.Timestamp(st.session_state["q_date"]).dayofweek
        day_type = "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")
        hhmm = "%02d:%02d" % (st.session_state["q_time"].hour,
                              st.session_state["q_time"].minute)
        sr = sr_module()
        result, fallback = None, False
        with st.spinner("과거 혼잡도 패턴을 분석하여 최적의 대안 경로를 탐색 중입니다..."):
            try:
                rs = get_scorer(day_type, hhmm, str(st.session_state["q_date"]))
                result = sr.find_route_by_station(rs, load_display_master(), o, d_,
                                                  st.session_state["q_mode"])
                if not result.get("ok"):
                    result = None
                else:
                    result = retime_with_v2(result)
            except Exception:
                result = None
            if result is None:
                result = fallback_route(o, d_)
                fallback = bool(result)
        st.session_state["route_result"] = result
        st.session_state["route_fallback"] = fallback
        st.session_state["route_query"] = _query_sig()
        st.session_state["ts_handoff"] = None          # 새 검색이면 예전 시간 비교 인계는 무효

    # 결과가 지금 입력(출발·도착·날짜·시각·모드)으로 계산된 것인지. 다르면 결과·CTA·하이라이트를 숨긴다.
    stale = (st.session_state.get("route_result") is not None
             and st.session_state.get("route_query") != _query_sig())

    # ---------- 오른쪽 노선도 ----------
    with right:
        if bundle is None:
            st.info("노선도 좌표가 없습니다. `14_import_map_workbook.py` 를 먼저 실행하세요.")
        else:
            path = None
            res = st.session_state.get("route_result")
            if res and res.get("ok") and not stale:
                path = res["recommended"]["path"]
            ev = st.plotly_chart(draw_map(bundle, path), width="stretch",
                                 on_select="rerun", key="subway_map",
                                 selection_mode="points", config=MAP_ZOOM_CONFIG)
            try:
                pts = ev.selection["points"] if ev and ev.selection else []
                if pts:
                    cd = pts[0].get("customdata")
                    if cd:
                        st.session_state["selected_station_key"] = cd[0]
            except Exception:
                pass

    # ---------- 결과 ----------
    st.divider()
    if not ready:
        if o and d_ and o == d_:
            st.info("출발역과 도착역이 같습니다. 다른 역을 선택해 주세요.")
        return

    result = st.session_state.get("route_result")
    if result is None:
        if do_search:
            st.info("현재 조건에서는 프로젝트 범위 내 경로를 찾지 못했습니다. "
                    "다른 구간이나 시각을 선택해 주세요.")
        else:
            st.caption("[경로 검색] 을 누르면 결과가 표시됩니다.")
        return
    if stale:
        st.warning("검색 조건(날짜·시각·선호 모드 등)이 바뀌었습니다. **[경로 검색]** 을 다시 눌러 주세요. "
                   "이전 결과는 지금 조건과 맞지 않아 숨겼습니다.")
        return
    _render_route_result(result, bool(st.session_state.get("route_fallback")), o, d_)


SHIFT_REASON_KO = {
    "insufficient_reduction": "혼잡 노출이 충분히 줄지 않음",
    "time_increase": "소요시간이 5분 넘게 늘어남",
    "route_changed": "그 시각에는 다른 경로가 최속",
    "out_of_window": "운행 시간대 밖",
}


def _shift_label(m):
    m = int(m)
    return "%d분 %s" % (abs(m), "일찍" if m < 0 else "늦게")


def render_time_shift_advice(ts):
    """v2.2 Time Shift 결과 안내. 경로 대안이 없을 때 쓴다."""
    if ts["type"] == "calm":
        st.success("현재 경로는 이미 혼잡 노출이 적은 이동입니다 "
                   "(100%%+ 구간 %.0f분, 최대 기대 혼잡도 %.0f%%). 경로나 시간을 바꿀 필요가 크지 않습니다."
                   % (ts["base_exposure_100_min"], ts["base_max_congestion"]))
        return
    if ts["type"] == "not_time_shiftable":
        st.warning("최단 경로 외에 **유의미한 쾌적 대안 경로가 없고**, 출발을 ±60분 안에서 옮겨도 "
                   "혼잡 노출이 의미 있게 줄지 않습니다 (%s). "
                   "경로·시간 조정만으로는 피하기 어려운 이동일 수 있습니다."
                   % SHIFT_REASON_KO.get(ts["binding_constraint"], ts["binding_constraint"]))
        return
    top = ts["top"]
    b = top[0]
    st.warning("최단 경로 외에 **유의미한 쾌적 대안 경로가 없습니다.** 대신 출발 시간을 조정해 보세요.")
    parts = []
    if b["exposure_100_drop_min"] >= 0.5:
        parts.append("100%%+ 혼잡 구간 **%.0f분 → %.0f분**"
                     % (ts["base_exposure_100_min"], b["exposure_100_min"]))
    parts.append("최대 기대 혼잡도 **%.0f%% → %.0f%%**" % (ts["base_max_congestion"], b["max_congestion"]))
    dt = b["actual_delta_min"]
    dt_txt = "소요시간 변화 거의 없음" if abs(dt) < 0.5 else "소요시간 %+.0f분" % dt
    st.info("**%s(%s 출발)** 하면 같은 경로에서 %s · %s."
            % (_shift_label(b["shift_min"]), b["depart"], ", ".join(parts), dt_txt))
    if len(top) > 1:
        st.caption("다른 출발 시각 후보")
        st.dataframe(pd.DataFrame([{
            "출발": x["depart"], "조정": _shift_label(x["shift_min"]),
            "100%+ 노출(분)": x["exposure_100_min"], "최대 기대 혼잡도(%)": x["max_congestion"],
            "소요시간 변화(분)": x["actual_delta_min"],
            "경로": "동일" if x["same_route"] else "유사 경로"} for x in top]),
            hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# v2 Counterfactual card — "지금 / 경로 변경 / 시간 조정 / 둘 다"
#   판정은 v2.3 사전 등록 기준(T1)과 같은 classify() 를 쓴다 (yeoyuro_v2/counterfactual.py).
#   기준은 시간 진행형 최속 경로라, 위 '추천 경로'(모드에 따라 다를 수 있음)와 다를 수 있다.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_td_router(day_type: str, query_date):
    import station_routing as sr
    from yeoyuro_v2.time_dependent import TDRouter
    mod = sr.load_scorer_class(ROOT)
    from yeoyuro_v2.line9 import register_v1_directions
    register_v1_directions(mod)
    return TDRouter(DATA_ROOT, mod, sr, load_display_master(), day_type, policy="step", k=5, multi_bin=True,
                    query_date=query_date)


@st.cache_data(show_spinner=False, max_entries=200)
def counterfactual_card_data(o: str, d: str, day_type: str, hhmm: str, query_date: str, stamp: float):
    """캐시 가능한 dict 로 돌려준다. stamp = 노선도·데이터 갱신 시각 (캐시 구분)."""
    from dataclasses import asdict
    from yeoyuro_v2.counterfactual import counterfactual
    from yeoyuro_v2.time_dependent import parse_hhmm
    c = counterfactual(get_td_router(day_type, query_date), o, d, parse_hhmm(hhmm))
    if c is None:
        return None
    for k in ("now", "route", "time", "joint"):
        c[k] = asdict(c[k])
    return c


CF_TYPE_LABEL = {"structural": STRUCTURAL_LABEL}     # 화면 표시명만 바꾼다. 내부 타입명은 그대로.
CF_TONE = {"calm": "success", "route_shiftable": "info", "time_shiftable": "info",
           "dual": "info", "structural": "warning", "invalid": "warning"}
CF_ADVICE = {
    "calm": "지금 출발해도 기대 혼잡도 100% 이상 구간이 거의 없습니다. 경로나 시간을 바꿀 필요가 크지 않습니다.",
    "route_shiftable": "**다른 경로**로 가면 혼잡 노출이 의미 있게 줄어듭니다. 출발 시각을 바꾸는 것보다 효과적입니다.",
    "time_shiftable": "같은 경로라도 **출발 시각을 조금 옮기면** 혼잡 노출이 의미 있게 줄어듭니다.",
    "dual": "경로를 바꾸거나 시각을 옮기거나, **둘 중 어느 쪽으로도** 혼잡 노출을 줄일 수 있습니다.",
    # D-057: 사용자 화면에서는 '구조적' 대신 '회피 어려움'. 탐색 범위 안에서 못 찾은 것이지 증명이 아니다.
    "structural": "현재 탐색한 경로·시간 대안 범위에서는 뚜렷한 개선안을 찾지 못했습니다.",
    "invalid": "출발·도착 시각이 혼잡 자료가 있는 시간대(05:30~01:00) 밖이라 판정하지 않습니다.",
}


def _path_lines(path: list) -> str:
    """노드 경로 → '7호선 상봉→건대입구 · 2호선 건대입구→잠실' 같은 구간 요약."""
    segs = []
    for n in path:
        line, rest = n.split("_", 1)
        st_ = rest.split("@")[0]
        if not segs or segs[-1][0] != line:
            segs.append([line, st_, st_])
        else:
            segs[-1][2] = st_
    return " · ".join("%s %s→%s" % (line_label(l), a, b) for l, a, b in segs if a != b) or "—"


def _cf_cell(col, title, opt, base, highlight=False):
    with col.container(border=True):
        col_title = ("✅ " if opt["ok"] and opt["kind"] != "now" else ("" if opt["kind"] == "now" else "✖ ")) + title
        st.markdown("**%s**" % col_title)
        if opt["actual"] is None:
            st.caption(opt["reason"] or "대안 없음")
            return
        if opt["kind"] != "now":
            sub = []
            if opt["shift_min"]:
                sub.append("**%s 출발** (%s)" % (opt["depart"], _shift_label(opt["shift_min"])))
            elif opt["kind"] in ("route", "joint"):
                sub.append("%s 출발" % opt["depart"])
            st.markdown(" · ".join(sub) if sub else "같은 시각")
        else:
            st.markdown("**%s 출발**" % opt["depart"])
        same = opt["kind"] == "now"
        d = lambda k: None if same else opt[k] - base[k]

        def _dl(k, fmt):
            # 반올림해 0 이 되는 변화는 표시하지 않는다 ('+0분' 이 빨간 ↑ 로 보여 늘어난 것처럼 읽히는 문제)
            if same:
                return None
            v = d(k)
            return None if round(v) == 0 else fmt % v
        st.metric("100%+ 혼잡 노출", "%.0f분" % opt["exp100"],
                  _dl("exp100", "%+.0f분"), delta_color="inverse")
        st.metric("예상 소요시간", "%.0f분" % opt["actual"],
                  _dl("actual", "%+.0f분"), delta_color="inverse")
        st.metric("최대 기대 혼잡도", "%.0f%%" % opt["max_cong"],
                  _dl("max_cong", "%+.0f%%p"), delta_color="inverse")
        st.caption("환승 %d회 · %s" % (opt["transfers"], _path_lines(opt["path"])))
        if not opt["ok"] and opt["reason"]:
            st.caption("가장 가까운 후보도 기준 미달: **%s**" % opt["reason"])


def _cf_pill(c):
    """혼잡 진단의 결론 한 줄 (4개 카드를 읽기 전에 3초 안에 보이도록). 판정은 카드와 같은 c 를 그대로 쓴다."""
    b = c.get("now") or {}
    for kind, icon in (("time", "🕒"), ("route", "🔀")):
        o = c.get(kind) or {}
        if not o.get("ok") or o.get("exp100") is None or b.get("exp100") is None:
            continue
        if kind == "time":
            head = "추천: %s 출발 · %s" % (o["depart"], _shift_label(o["shift_min"]))
        else:
            head = "추천: 다른 경로 · %s" % _path_lines(o["path"])
        st.markdown('<div style="display:inline-block;background:#E8F1FB;border:1px solid #9CC3EA;border-radius:22px;'
                    'padding:7px 16px;margin:2px 0 10px 0"><span style="font-weight:800;color:#0B3D75">%s %s</span>'
                    '<span style="color:#1F2937;margin-left:12px">정원 초과(100%%+) 혼잡 노출 <b>%.0f분 → %.0f분</b></span>'
                    '</div>' % (icon, _esc(head), b["exp100"], o["exp100"]), unsafe_allow_html=True)
        return


def render_counterfactual(o, d_):
    day_type, hhmm, qd = _query_args()
    try:
        with st.spinner("경로를 바꾸면, 시간을 바꾸면 어떻게 되는지 계산하는 중…"):
            c = counterfactual_card_data(o, d_, day_type, hhmm, qd, ensure_map_fresh())
    except Exception as e:                                   # noqa: BLE001
        st.caption("혼잡 진단을 계산하지 못했습니다 (%s)." % type(e).__name__)
        return
    if c is None:
        return
    st.subheader("혼잡 진단 — 경로를 바꿀까, 시간을 바꿀까?")
    label = CF_TYPE_LABEL.get(c["type"], c["type_ko"])
    getattr(st, CF_TONE[c["type"]])("**%s** / %s" % (label, CF_ADVICE[c["type"]]))
    _cf_pill(c)
    if c["type"] == "structural":
        st.caption("ⓘ %s = %s" % (STRUCTURAL_LABEL, STRUCTURAL_HELP), help=STRUCTURAL_HELP)
    if c["type"] in ("calm", "invalid"):
        return c
    base = c["now"]
    cols = st.columns(4)
    _cf_cell(cols[0], "지금 (최속 경로)", base, base)
    _cf_cell(cols[1], "경로 변경", c["route"], base)
    _cf_cell(cols[2], "시간 조정 (±60분)", c["time"], base)
    _cf_cell(cols[3], "경로 + 시간 함께 (참고)", c["joint"], base)
    st.caption(
        "판정 기준: 100%+ 혼잡 노출이 **5분 이상 또는 절반 이상** 줄어야 '효과 있음'. "
        "경로 변경은 소요 +15분·체감 +5분·환승 +1회 이내, 시간 조정은 같은(또는 거의 같은) 경로로 ±60분 안·소요 +5분 이내. "
        "기준은 위 추천 경로가 아니라 같은 시각의 최속 경로입니다.",
        help="판정 기준은 분석 전에 고정했습니다 (사전 등록 %s). '경로 + 시간 함께'는 주 판정에 쓰지 않는 참고 선택지입니다."
             % c["threshold_set_id"])
    return c


def render_time_cta(c):
    """v2 Time Shift 판정(T1, Counterfactual 의 '시간 조정')이 성공했을 때만 보이는 CTA.

    임의의 혼잡도 임계값이 아니라 c["time"]["ok"] 를 그대로 쓴다.
    """
    if not c or not c.get("time") or not c["time"]["ok"]:
        return False
    b, t = c["now"], c["time"]
    drop = (b["exp100"] or 0) - (t["exp100"] or 0)
    if drop >= 0.5:
        effect = "100%%+ 혼잡 노출이 %.0f분 줄어듭니다 (%.0f분 → %.0f분)" % (drop, b["exp100"], t["exp100"])
    else:
        effect = "최대 기대 혼잡도가 %.0f%% → %.0f%% 로 낮아집니다" % (b["max_cong"], t["max_cong"])
    with st.container(border=True):
        # 결론(🕒 추천 한 줄)은 위에 이미 있으므로 여기서는 다음 행동만 안내한다. effect 문장은 버튼 도움말로.
        st.markdown("**다른 출발 시각도 비교해 보세요**")
        handoff = {"sig": st.session_state.get("route_query"), "path": list(b["path"]),
                   "shift_min": int(t["shift_min"])}
        st.button("시간대별 비교 보기 →", key="cta_time", type="primary", on_click=_cta_go, args=(handoff,),
                  help="%s(%s) 출발하면 %s." % (_shift_label(t["shift_min"]), t["depart"], effect))
    return True


def _cta_go(handoff: dict):
    st.session_state["ts_handoff"] = handoff
    _go("경로 추천", "시간 바꿔 보기")


def _render_route_result(result, fallback, o, d_):

    if fallback:
        st.info("현재 입력하신 조건의 실시간 연산이 제한되어, "
                "가장 유사한 **대표 시나리오(미리 계산된 결과)** 를 제공합니다.")

    rec, alt = result["recommended"], result.get("alternative")
    st.subheader("추천 경로")
    st.caption(RESULT_NOTE, help=DISCLAIMER)
    route_card("%s 모드 추천" % MODE_LABEL[st.session_state["q_mode"]], rec, o, d_,
               show_profile=True, time_alt=result.get("time_alternative"),
               profile_key="profile_recommended")

    has_cta = False
    if not fallback:
        has_cta = render_time_cta(render_counterfactual(o, d_))

    if alt:
        st.subheader("최단경로 대비 쾌적 대안")
        route_card("대안 경로", alt, o, d_,
                   "예상 소요시간은 %.1f분 늘어나지만, 최대 기대 혼잡도는 %.1f%%p 낮습니다."
                   % (alt["time_loss_vs_fastest"], alt["comfort_gain_vs_fastest"]), "good",
                   show_profile=True, profile_key="profile_alternative")
    else:
        ts = result.get("time_shift")
        if ts is not None and has_cta:
            # 시간 조정 안내는 위 CTA 와 '시간 바꿔 보기' 로 합쳤다 (D-057). 경로 대안 없음만 알린다.
            st.caption("최단 경로 외에 유의미한 쾌적 대안 경로는 없습니다.")
        elif ts is not None:
            render_time_shift_advice(ts)
        else:
            # v1 경로 (precomputed fallback 등)
            st.warning("현재 최단 경로 외에 **유의미한 쾌적 대안 경로가 없습니다.** "
                       "대신, 같은 경로에서 더 여유로운 출발 시간을 추천합니다.")
            ta = result.get("time_alternative")
            if ta:
                st.info("대신 **%s 출발**을 권장합니다. 같은 경로의 최대 기대 혼잡도가 "
                        "**%.0f%% → %.0f%%** 로 낮아집니다."
                        % (ta["best_time_bin"][:5], ta["current_max_congestion"],
                           ta["best_max_congestion"]))

    with st.expander("후보 경로 비교"):
        rows = []
        for c in result["candidates"]:
            _run, _dwell, _walk, _wait = time_breakdown(c)
            rows.append({"예상 소요시간": c["actual_time_min"],
                         "승차": round(_run, 1),
                         "중간역 정차": round(_dwell, 1),
                         "환승 도보": round(_walk, 1),
                         "환승 대기": c.get("transfer_wait_min"),
                         "쾌적 체감시간": c["perceived_time_min"],
                         "평균 기대 혼잡도": c["avg_congestion"],
                         "최대 기대 혼잡도": c["max_congestion"],
                         "환승": c["transfer_count"],
                         "100%+ 노출(분)": c.get("exposure_100_min"),
                         "경로": " → ".join(
                             s["to"] for s in c["segments"] if s["kind"] == "ride")})
        st.dataframe(round1(pd.DataFrame(rows)), width="stretch", hide_index=True)



@st.cache_data(show_spinner=False)
def fallback_route(o: str, d: str):
    """precomputed demo case. 실시간 연산 실패 시 자연스럽게 대체."""
    for tag in ["0830"] + precomputed_tags():
        cases = load_mart("route_evaluation_mart_%s" % tag)
        if cases is None:
            continue
        sub = cases[cases["request_id"].str.startswith("%s->%s|calm" % (o, d))]
        if sub.empty:
            continue
        sub = sub.sort_values("route_rank")

        def to_ev(r):
            nodes = r["path_nodes"].split(" -> ")
            segs, cur, start, stops = [], nodes[0].split("_")[0], \
                nodes[0].split("_", 1)[1].split("@")[0], 0
            for u, v in zip(nodes, nodes[1:]):
                un = u.split("_", 1)[1].split("@")[0]
                vn = v.split("_", 1)[1].split("@")[0]
                if un == vn:
                    segs.append({"kind": "ride", "line": cur, "from": start,
                                 "to": un, "n_stops": stops, "minutes": None,
                                 "max_congestion": None, "stations": [],
                                 "direction_label": ""})
                    segs.append({"kind": "transfer", "at": un, "from_line": cur,
                                 "to_line": v.split("_")[0], "minutes": None})
                    cur, start, stops = v.split("_")[0], vn, 0
                else:
                    stops += 1
            segs.append({"kind": "ride", "line": cur, "from": start,
                         "to": nodes[-1].split("_", 1)[1].split("@")[0],
                         "n_stops": stops, "minutes": None,
                         "max_congestion": None, "stations": [],
                         "direction_label": ""})
            return {"path": nodes, "segments": segs,
                    "actual_time_min": r["actual_time_min"],
                    "perceived_time_min": r["perceived_time_min"],
                    "avg_congestion": r["avg_congestion"],
                    "max_congestion": r["max_congestion"],
                    "transfer_count": int(r["transfer_count"]),
                    "p95_exposure_count": int(r["high_congestion_exposure_count"]),
                    "event_risk_min": 0.0}

        cands = [to_ev(r) for _, r in sub.iterrows()]
        altrows = sub[sub["recommendation_policy"] == "meaningful_alternative"]
        alt = None
        if not altrows.empty:
            r = altrows.iloc[0]
            alt = to_ev(r)
            alt["time_loss_vs_fastest"] = float(r["time_loss_vs_fastest"])
            alt["comfort_gain_vs_fastest"] = float(r["max_congestion_reduction_vs_fastest"])
        return {"ok": True, "recommended": cands[0], "fastest": cands[0],
                "alternative": alt, "time_alternative": None,
                "candidates": cands, "n_origin_nodes": 1}
    return None


# --------------------------------------------------------------------------
# 페이지. 시간 분산 (v2.2 Time Shift)
# --------------------------------------------------------------------------
def _ts_inputs_to_canonical():
    """[비교하기] 클릭 시 시간 비교 입력을 공유 입력(canonical)으로 옮긴다."""
    for w in ("ts_origin", "ts_dest", "ts_date", "ts_time"):
        _w2c(w)


def _ts_reset_handoff():
    st.session_state["ts_handoff"] = None


def page_time_shift():
    """경로 추천 > 시간 바꿔 보기.

    - 경로 CTA 에서 넘어왔고, 그 경로 결과(route_query)가 지금 공유 입력과 같으며 기준 path 가 있으면
      → 자동 계산 (Counterfactual 과 같은 엔진·같은 기준 path·같은 T1 → CTA 숫자와 일치)
    - 직접 진입했거나 조건이 바뀌었으면 → 입력부 + [비교하기] (기존 계산 그대로)
    """
    from yeoyuro_v2.temporal_shift import evaluate_time_shift, shift_curve
    from yeoyuro_v2.time_dependent import parse_hhmm, bin_index_of
    from yeoyuro_v2.shift_rules import T1

    st.caption("같은 이동을 조금 일찍 또는 늦게 출발하면 혼잡 노출이 얼마나 줄어드는지 비교합니다.")
    disp = load_display_master()
    if disp is None:
        st.warning("`station_display_master.csv` 가 없습니다.")
        return
    sig = _query_sig()
    ho = st.session_state.get("ts_handoff")
    auto = bool(ho and ho.get("path") and ho.get("sig") is not None
                and ho["sig"] == sig == st.session_state.get("route_query"))

    if auto:
        o, d_, qd, hhmm = sig[0], sig[1], sig[2], sig[3]
        dow = pd.Timestamp(qd).dayofweek
        day_type = "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")
        b1, b2, b3 = st.columns([3, 1, 1])
        b1.markdown("**%s → %s** · %s %s 기준 · 경로 추천 결과의 최속 경로" % (o, d_, qd, hhmm))
        b2.button("← 경로로 돌아가기", key="ts_back", width="stretch",
                  on_click=_go, args=("경로 추천", "경로"))
        b3.button("조건 바꾸기", key="ts_edit", width="stretch", on_click=_ts_reset_handoff)
        with st.spinner("시각별로 같은 경로를 다시 계산하는 중입니다..."):
            try:
                router = get_td_router(day_type, qd)
                t0 = parse_hhmm(hhmm)
                ev = router.evaluator(bin_index_of(t0))
                path = list(ho["path"])
                ts = evaluate_time_shift(ev, path, t0, T1,
                                         reopt=lambda t: (router.evaluate_od(o, d_, t) or [None])[0])
                curve = pd.DataFrame(shift_curve(ev, path, t0))
            except Exception as e:  # noqa: BLE001
                st.warning("계산 중 문제가 발생했습니다: %s" % e)
                return
        _render_time_shift_result(ts, curve, "%s → %s · %s · 기준 출발 %s" % (o, d_, _path_lines(path), hhmm))
        return

    if ho and ho.get("sig") is not None and ho["sig"] != sig:
        st.caption("경로 검색 이후 조건이 바뀌어 자동 비교를 하지 않았습니다. 아래에서 다시 비교해 주세요.")
    keys = sorted(disp["station_key"].tolist())
    lm = dict(zip(disp["station_key"], disp["available_lines"]))
    _seed("ts_origin", "노원" if "노원" in keys else keys[0])
    _seed("ts_dest", "양천구청" if "양천구청" in keys else keys[1])
    _seed("ts_date")
    _seed("ts_time")
    c1, c2 = st.columns(2)
    o = c1.selectbox("출발역", keys, format_func=lambda k: short_label(k, lm.get(k, "")), key="ts_origin",
                     on_change=_w2c, args=("ts_origin",))
    d_ = c2.selectbox("도착역", keys, format_func=lambda k: short_label(k, lm.get(k, "")), key="ts_dest",
                      on_change=_w2c, args=("ts_dest",))
    c3, c4 = st.columns(2)
    q_date = c3.date_input("날짜", key="ts_date", on_change=_w2c, args=("ts_date",))
    q_time = c4.time_input("기준 출발 시각", key="ts_time", on_change=_w2c, args=("ts_time",))
    if not st.button("비교하기", type="primary", key="ts_go", on_click=_ts_inputs_to_canonical):
        st.caption("[비교하기] 를 누르면 ±90분 곡선과 ±60분 안의 추천 시각을 계산합니다.")
        return
    if o is None or d_ is None:
        st.info("출발역과 도착역을 선택해 주세요.")
        return
    if o == d_:
        st.info("출발역과 도착역이 같습니다.")
        return

    dow = pd.Timestamp(q_date).dayofweek
    day_type = "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")
    hhmm = "%02d:%02d" % (q_time.hour, q_time.minute)
    with st.spinner("시각별로 같은 경로를 다시 계산하는 중입니다..."):
        try:
            rs = get_scorer(day_type, hhmm, str(q_date))
            res = sr_module().find_route_by_station(rs, disp, o, d_, "fast")
            tde = get_td_evaluator(day_type, hhmm, str(q_date))
            if not res.get("ok") or tde is None:
                st.warning("경로를 계산하지 못했습니다.")
                return
            t0 = parse_hhmm(hhmm)
            from yeoyuro_v2.app_bridge import retime_result
            res = retime_result(res, lambda b: tde, t0, "fast")
            path = res["fastest"]["path"]
            ts = evaluate_time_shift(tde, path, t0, T1)
            curve = pd.DataFrame(shift_curve(tde, path, t0))
        except Exception as e:  # noqa: BLE001
            st.warning("계산 중 문제가 발생했습니다: %s" % e)
            return
    _render_time_shift_result(ts, curve, route_summary_line(res["fastest"], o, d_) + " · 기준 출발 " + hhmm)


def _render_time_shift_result(ts, curve, summary: str):
    """시간 비교 결과 (KPI 4 + 곡선 + 추천 시각 표 + 판정 기준). 자동/직접 두 경로가 함께 쓴다."""
    import plotly.graph_objects as go
    base = ts["base"]
    st.caption(summary)
    m = st.columns(4)
    m[0].metric("예상 소요시간", "%.0f분" % base.actual_time_min)
    m[1].metric("100%+ 혼잡 노출", "%.0f분" % base.exposure[100])
    m[2].metric("최대 기대 혼잡도", "%.0f%%" % base.max_congestion)
    verdict = {"time_shiftable": "시간 조정 효과 있음", "not_time_shiftable": "시간 조정 효과 작음",
               "calm": "이미 여유로운 이동"}[ts["type"]]
    m[3].metric("판정", verdict)
    st.caption(RESULT_NOTE, help=DISCLAIMER)

    fig = go.Figure()
    fig.add_bar(x=curve["depart"], y=curve["exposure_100_min"], name="100%+ 혼잡 노출(분)",
                marker_color=["#1F4E79" if s == 0 else "#9DB4CC" for s in curve["shift_min"]])
    fig.add_scatter(x=curve["depart"], y=curve["max_congestion"], name="최대 기대 혼잡도(%)",
                    yaxis="y2", mode="lines+markers", line=dict(color="#D9534F"))
    fig.add_scatter(x=curve["depart"], y=curve["actual_time_min"], name="예상 소요시간(분)",
                    mode="lines", line=dict(color="#6B7280", dash="dot"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), dragmode=False,
                      plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                      legend=dict(orientation="h", y=1.1, x=0),
                      xaxis=dict(title="출발 시각 (진한 막대 = 기준)", fixedrange=True),
                      yaxis=dict(title="분", rangemode="tozero", fixedrange=True),
                      yaxis2=dict(title="%", overlaying="y", side="right", rangemode="tozero",
                                  fixedrange=True, showgrid=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    if ts["type"] == "time_shiftable":
        st.subheader("추천 출발 시각")
        st.dataframe(pd.DataFrame([{
            "순위": i + 1, "출발": x.depart, "조정": _shift_label(x.shift_min),
            "100%+ 노출(분)": x.exposure_100_min,
            "노출 감소(분)": x.exposure_100_drop_min,
            "최대 기대 혼잡도(%)": x.max_congestion,
            "소요시간 변화(분)": x.actual_delta_min} for i, x in enumerate(ts["top"])]),
            hide_index=True, width="stretch")
        st.caption("순위는 점수가 아니라 규칙입니다: ① 출발을 덜 옮기는 쪽 ② 노출 감소가 큰 쪽 "
                   "③ 같은 폭이면 늦게 출발하는 쪽.")
    elif ts["type"] == "not_time_shiftable":
        st.warning("±30·60분 어느 쪽으로 옮겨도 판정 기준(100%%+ 노출 5분 이상 또는 절반 이상 감소, "
                   "소요시간 증가 5분 이하)을 만족하지 않습니다. 주된 이유: %s."
                   % SHIFT_REASON_KO.get(ts["binding_constraint"], ts["binding_constraint"]))
    with st.expander("판정 기준"):
        st.markdown(
            "- 후보: 기준 출발 ±30·60분, 같은 경로를 **각 구간 도착 시각** 기준으로 다시 계산\n"
            "- 성공: 100%+ 노출이 5분 이상 **또는** 절반 이상 줄고, 소요시간 증가 5분 이하 "
            "(v2.3 사전 등록 기준 T1)\n"
            "- 기준 이동의 100%+ 노출이 3분 미만이고 최대 기대 혼잡도가 100% 미만이면 '이미 여유로운 이동'\n"
            "- 그래프는 15분 간격 참고값이며, 판정에는 30분 간격 후보만 씁니다.\n"
            "- 실시간 정보가 아니라 과거 혼잡 패턴의 중앙값입니다.")


# --------------------------------------------------------------------------
# 페이지. 혼잡 분산 지도 (v2.4 Mobility Atlas)
# --------------------------------------------------------------------------
ATLAS_DIMS = {
    "congestion_exposure": ("통과 1회당 100%+ 노출(분)", "bad",
                            "표본 이동이 이 구간을 지날 때 기대 혼잡도 100% 이상인 시간의 평균"),
    "exposed_traversal_share": ("100%+ 를 겪은 통과 비율", "bad",
                                "이 구간을 지난 표본 이동 중, 이 구간에서 100% 이상을 겪은 비율"),
    "structural_share": ("회피 어려움 비율", "bad",
                         "이 구간에서 100%+ 를 겪은 이동 중, 정해진 경로·시간 대안 범위에서 개선안이 발견되지 않은 "
                         "이동의 비율"),
    "alternative_opportunity": ("경로 변경으로 피할 수 있는 비율", "good",
                                "이 구간에서 100%+ 를 겪은 이동 중, 기준을 만족하는 우회 경로가 있는 비율"),
    "temporal_shift_benefit": ("시간 조정으로 피할 수 있는 비율", "good",
                               "이 구간에서 100%+ 를 겪은 이동 중, ±60분 출발 조정으로 피할 수 있는 비율"),
    "demand_weighted_exposure": ("수요 proxy 가중 노출(분)", "bad",
                                 "통과 1회당 노출을 승하차 기반 수요 proxy 로 가중한 값 (실제 이동량 아님)"),
}
ATLAS_BAD = ["#FDE7D7", "#F9B98F", "#F0844F", "#D9481F", "#9E2A0E"]
ATLAS_GOOD = ["#E3F0E6", "#B5D9BE", "#7CBC8E", "#3F9A5E", "#1B6B3A"]
TYPE_KO_APP = {"calm": "여유 (Calm)", "time_shiftable": "시간 조정으로 해결", "dual": "경로·시간 모두 가능",
               "route_shiftable": "경로 변경으로만 해결", "structural": STRUCTURAL_LABEL}


def page_atlas():
    import plotly.graph_objects as go
    st.caption("평일 출퇴근 시간대의 혼잡이 어디에서 경로로, 시간으로 피할 수 있고 어디는 둘 다 어려운지 보여줍니다. "
               "무작위로 뽑은 역 쌍 620개 × 평일 기준 출발 6개(3,720건) 표본의 집계이며, "
               "구간을 지나는 모든 승객의 값이 아닙니다.")
    st.info("**1~8호선 분석 표본 기준** · 9호선은 별도 재분석으로 다루며 이 지도에는 표시하지 않습니다.", icon="ℹ️")
    atlas = load_mart("mobility_atlas_mart", "data/marts/v2")
    odm = load_mart("od_shiftability_mart", "data/marts/v2")
    bundle = load_map_bundle(ensure_map_fresh())
    if atlas is None or odm is None or bundle is None:
        st.warning("v2 mart 가 없습니다. `scripts/v2/23_build_od_shiftability.py` 와 "
                   "`scripts/v2/24_build_mobility_atlas.py` 를 먼저 실행하세요.")
        return

    c1, c2, c3 = st.columns([1.1, 1.6, 1.1])
    grp = c1.radio("시간대", ["am", "pm", "all"], horizontal=True, key="atlas_grp",
                   format_func=lambda x: {"am": "오전 피크", "pm": "오후 피크", "all": "전체"}[x])
    dim = c2.selectbox("지표", list(ATLAS_DIMS), key="atlas_dim",
                       format_func=lambda k: ATLAS_DIMS[k][0])
    weight = c3.radio("유형 비율 기준", ["count", "proxy"], horizontal=True, key="atlas_w",
                      format_func=lambda x: {"count": "역 쌍 수", "proxy": "수요 proxy"}[x])

    # ---- 유형 분포
    times = {"am": ("07:30", "08:00", "08:30"), "pm": ("17:30", "18:00", "18:30")}
    sub = odm if grp == "all" else odm[odm.base_time.isin(times[grp])]
    w = sub.demand_proxy_w if weight == "proxy" else pd.Series(1.0, index=sub.index)
    tot_hot = w[sub.final_shiftability_type != "calm"].sum()
    m = st.columns(5)
    for i, k in enumerate(["calm", "time_shiftable", "dual", "route_shiftable", "structural"]):
        if k == "calm":
            m[i].metric(TYPE_KO_APP[k], "%.1f%%" % (100 * w[sub.final_shiftability_type == k].sum() / w.sum()),
                        help="전체 이동 대비. 기준 경로가 100% 이상 구간을 지나지 않음")
        else:
            m[i].metric(TYPE_KO_APP[k], "%.1f%%" % (100 * w[sub.final_shiftability_type == k].sum()
                                                    / max(tot_hot, 1e-9)),
                        help=("%s. 혼잡 이동(여유 제외) 대비" % STRUCTURAL_HELP) if k == "structural"
                        else ("혼잡 이동(여유 제외) 대비. 한 사람이 옮길 때 기준이며, 많은 사람이 함께 옮기는 효과는 "
                              "반영하지 않습니다. 유형 판정 기준은 분석 전에 고정했습니다."))

    # ---- 지도
    a = atlas[(atlas.time_group == grp) & (atlas.atlas_dimension == dim)].copy()
    label, tone, desc = ATLAS_DIMS[dim]
    a = a[a.value.notna()]
    if dim in ("congestion_exposure", "demand_weighted_exposure", "exposed_traversal_share"):
        a = a[a.value > 0]
    a["pair"] = a.apply(lambda r: tuple(sorted([r.from_node, r.to_node])), axis=1)
    worst = a.sort_values("value", ascending=(tone == "good")).drop_duplicates("pair")
    palette = ATLAS_BAD if tone == "bad" else ATLAS_GOOD
    if dim.endswith("_share") or dim in ("alternative_opportunity", "temporal_shift_benefit"):
        edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        fmt = lambda v: "%.0f%%" % (100 * v)
    else:
        q = worst.value.quantile([0, .2, .4, .6, .8]).tolist() + [worst.value.max() + 1e-9]
        edges = sorted(set(round(x, 3) for x in q))
        fmt = lambda v: "%.2f분" % v
    fig = draw_map(bundle, None, show_endpoints=False)
    vn = bundle["map_visual_nodes"]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        part = worst[(worst.value >= lo) & (worst.value < hi)]
        if part.empty:
            continue
        xs, ys, hx, hy, ht = [], [], [], [], []
        for r in part.itertuples():
            px, py = path_to_visual_nodes([r.from_node, r.to_node], vn)
            if len(px) < 2:
                continue
            xs += px + [None]
            ys += py + [None]
            hx.append(sum(px) / len(px))
            hy.append(sum(py) / len(py))
            ht.append("<b>%s</b><br>%s: %s<br>통과 %d · 이 구간 100%%+ %d"
                      % (r.section, label, fmt(r.value), r.n_traversals, r.n_hot_traversals))
        col = palette[min(i, len(palette) - 1)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=True,
                                 name="%s ~ %s" % (fmt(lo), fmt(min(hi, worst.value.max()))),
                                 line=dict(color=col, width=11), opacity=0.92))
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"),
                                 hovertext=ht, hoverinfo="text", showlegend=False))
    # 역명은 layout image(layer="above") 라 혼잡 띠 위에 그려진다 (D-067)
    fig.update_layout(height=760, showlegend=True,
                      legend=dict(orientation="h", y=1.02, x=0, title=dict(text=label + "  ")))
    st.plotly_chart(fig, width="stretch", config=MAP_ZOOM_CONFIG, key="atlas_map")
    st.caption("%s. 양방향 중 %s 쪽 값을 칠했습니다. 비율 지표는 그 구간에서 100%%+ 를 겪은 표본이 5건 이상인 구간만 "
               "표시합니다." % (desc, "더 나쁜" if tone == "bad" else "더 낮은"))

    # ---- 표
    st.subheader("상위 구간")
    top = a.sort_values("value", ascending=(tone == "good")).head(10)
    st.dataframe(pd.DataFrame({
        "구간": top.section, label: [fmt(v) for v in top.value],
        "통과(표본)": top.n_traversals, "이 구간 100%+ 통과": top.n_hot_traversals}),
        hide_index=True, width="stretch")
    with st.expander("이 화면을 읽는 법"):
        st.markdown(
            "- **종합 점수는 없습니다.** 피로도 정답 데이터가 없어 지표를 하나로 합치면 가중치를 정당화할 수 "
            "없기 때문에, 지표를 따로 보여줍니다.\n"
            "- 회피 어려움 비율(%s)은 시간 조정 창(±60분) 선택에 민감합니다. ±30분이면 커지고 ±90분이면 거의 사라집니다. "
            "숫자의 크기보다 **어느 구간에 몰려 있는가**를 보세요.\n"
            "- 수요 proxy 는 역별 승하차로 만든 추정이며 실제 출발-도착 이동량이 아닙니다." % STRUCTURAL_HELP)


# --------------------------------------------------------------------------
# 페이지. 이벤트 시나리오 (v2.5 Event Stress Test)
# --------------------------------------------------------------------------
EVENT_SCENARIOS = {"FW_2025": "서울세계불꽃축제 2025 (9/27 토, 17~19시 출발)",
                   "NYB_2025": "제야의 종 2025 (12/31 수, 21~23시 출발)",
                   "HW_2025": "핼러윈 2025 (10/31 금, 19~21시 출발)",
                   "CB_2025": "여의도 벚꽃 2025 (4/11 금, 12·15·18시 출발)"}
TYPE_SHORT = {"calm": "여유", "time_shiftable": "시간 조정", "dual": "경로·시간",
              "route_shiftable": "경로 변경", "structural": STRUCTURAL_LABEL}


def page_event_scenario():
    import plotly.graph_objects as go
    st.caption("**What-if 분석** · 평소에는 경로나 시간 조정으로 피할 수 있던 이동이, 대형 이벤트 날에는 "
               "어떻게 바뀌는지 봅니다.")
    st.warning("이벤트 날의 차내 혼잡은 관측 자료가 없습니다. 실제 이벤트 당일 차내 혼잡 관측치가 아니라, "
               "이벤트 당일 승하차 급증(실측)을 혼잡 배수로 바꾸는 가정(λ=0.3)으로 만든 **가정된 시나리오**입니다.")
    l1, l2 = st.columns([4, 1.3])
    l1.caption("가정의 근거: **과거 데이터** 탭의 이벤트 날 승하차 급증(전후 비교·DiD 순효과)입니다.")
    l2.button("근거 데이터 보기 →", key="evs_to_past", width="stretch",
              on_click=_go, args=("이벤트 분석", "과거 데이터"))
    df = load_mart("event_scenario_mart", "data/marts/v2")
    if df is None:
        st.warning("event_scenario_mart 가 없습니다. `scripts/v2/25_event_stress_test.py` 를 먼저 실행하세요.")
        return
    eid = st.selectbox("이벤트", list(EVENT_SCENARIOS), format_func=lambda k: EVENT_SCENARIOS[k],
                       key="evs_event")
    g = df[df.event_id == eid]
    t = g[g.treated]
    c = st.columns(4)
    c[0].metric("표본 이동", "%d" % len(g), help="무작위 이동 150쌍 + 이벤트 방문 이동 약 100쌍 × 출발 3개")
    c[1].metric("이벤트 구간을 지나는 이동", "%d" % len(t))
    c[2].metric("그중 유형이 바뀐 이동", "%.0f%%" % (100 * (t.baseline_type != t.event_shiftability_type).mean())
                if len(t) else "-")
    c[3].metric("이벤트 날 %s" % STRUCTURAL_LABEL, "%d건" % int((t.event_shiftability_type == "structural").sum()),
                help="%s. 평시 %d건" % (STRUCTURAL_HELP, int((t.baseline_type == "structural").sum())))

    if len(t):
        order = ["calm", "time_shiftable", "dual", "route_shiftable", "structural"]
        ct = pd.crosstab(t.baseline_type, t.event_shiftability_type).reindex(
            index=order, columns=order, fill_value=0)
        fig = go.Figure(go.Heatmap(
            z=ct.values, x=[TYPE_SHORT[k] for k in order], y=[TYPE_SHORT[k] for k in order],
            colorscale="Oranges", text=ct.values, texttemplate="%{text}", showscale=False,
            hovertemplate="평시 %{y} → 이벤트 %{x}: %{z}건<extra></extra>"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis=dict(title="이벤트 날 유형", side="top"),
                          yaxis=dict(title="평시 유형", autorange="reversed"))
        st.subheader("유형 전이 (이벤트 구간을 지나는 이동)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="evs_heat")
        # 고정 예시는 이벤트에 따라 0건일 수 있어(예: 불꽃축제는 평시가 모두 '여유'), 이 이벤트에서 가장 많이 바뀐 칸을 예로 든다
        meaning = {"calm": "혼잡 구간을 지나지 않음", "time_shiftable": "출발 시각을 옮겨야 혼잡을 줄일 수 있음",
                   "dual": "경로나 출발 시각을 바꿔야 혼잡을 줄일 수 있음",
                   "route_shiftable": "다른 경로로 가야 혼잡을 줄일 수 있음",
                   "structural": "탐색한 대안 범위에서 개선안을 찾지 못함"}
        off = [(int(ct.loc[a_, b_]), a_, b_) for a_ in order for b_ in order if a_ != b_ and ct.loc[a_, b_] > 0]
        ex = ("평시 '%s' → 이벤트 '%s' %d건 = 이벤트 날엔 %s."
              % (TYPE_SHORT[max(off)[1]], TYPE_SHORT[max(off)[2]], max(off)[0], meaning[max(off)[2]])) if off else \
            "이 이벤트에서는 대응 방법이 바뀐 이동이 없습니다."
        st.caption("대각선 밖의 칸이 이벤트 때문에 대응 방법이 바뀐 이동입니다.  \n" + ex)
        changed = t[t.baseline_type != t.event_shiftability_type]
        if len(changed):
            st.subheader("대응 방법이 바뀐 이동 예시")
            ex = changed.sort_values("event_max_congestion_l03", ascending=False).head(10)
            st.dataframe(pd.DataFrame({
                "출발": ex.origin, "도착": ex.destination, "출발 시각": ex.base_time,
                "평시": ex.baseline_type.map(TYPE_SHORT), "이벤트": ex.event_shiftability_type.map(TYPE_SHORT),
                "최대 기대 혼잡도 평시→이벤트": ["%.0f%% → %.0f%%" % (a, b) for a, b in
                                         zip(ex.baseline_max_congestion, ex.event_max_congestion_l03)]}),
                hide_index=True, width="stretch")
    with st.expander("가정과 한계"):
        st.markdown(
            "- 혼잡 배수 = 1 + 검증 가중치 × (승하차 급증 배율 − 1) × λ. λ=0.3 은 v1 에서 정한 값이며 관측으로 "
            "보정된 값이 아닙니다. 리포트에 λ=0.15·0.6 결과도 함께 있습니다.\n"
            "- 배수는 이벤트 역에서 출발하는 구간에만, 이벤트 시간대에만 적용합니다.\n"
            "- 판정 기준과 가설은 계산 전에 고정했습니다 (docs/v2/preregistration_v25.md).")


# --------------------------------------------------------------------------
# 페이지 2. 혼잡 조회
# --------------------------------------------------------------------------
def show_station_info(station_key: str):
    """선택한 역의 혼잡 정보. 환승역은 노선을 골라 하나씩 본다.

    환승역에서 방향만으로 묶으면 서로 다른 노선의 상·하행이 한 그래프에 섞여
    (이수 4호선/7호선처럼) 어느 노선인지 알 수 없다. 노선을 먼저 고르게 한다.
    """
    prof = load_mart("congestion_station_profile")
    lookup = load_mart("congestion_edge_lookup")
    if prof is None:
        st.caption("혼잡도 프로파일이 없습니다.")
        return
    sub = prof[(prof["station_name"] == station_key) & (prof["day_type"] == "weekday")]
    if sub.empty:
        st.caption("해당 역의 관측 데이터가 없습니다.")
        return

    lines = sorted(sub["line_id"].astype(str).unique(), key=line_sort_key)
    if len(lines) > 1:
        line = st.radio("노선", lines, horizontal=True,
                        format_func=line_label,
                        key="info_line_%s" % station_key)
    else:
        line = lines[0]

    one = sub[sub["line_id"].astype(str) == line].copy()
    one["방향"] = one["direction"].map(DIRECTION_KO).fillna(one["direction"])
    st.table(round1(one[["방향", "mean_congestion", "max_congestion",
                         "peak_time_bin", "peak_duration_min"]]
                    .rename(columns={"mean_congestion": "평균(%)",
                                     "max_congestion": "관측 최대(%)",
                                     "peak_time_bin": "피크",
                                     "peak_duration_min": "지속(분)"}))
             .reset_index(drop=True))

    if lookup is not None:
        lk = lookup[(lookup["station_name"] == station_key)
                    & (lookup["day_type"] == "weekday")
                    & (lookup["line_id"].astype(str) == line)]
        if not lk.empty:
            piv = lk.pivot_table(index="time_bin", columns="direction",
                                 values="congestion_median").sort_index()
            piv.columns = [DIRECTION_KO.get(c, c) for c in piv.columns]
            congestion_line_chart(piv, 220)


def page_congestion():
    tab_st, tab_line = st.tabs(["역별 혼잡도", "노선별 혼잡도"])
    with tab_st:
        page_congestion_station()
    with tab_line:
        page_congestion_line()


def page_congestion_station():
    prof = load_mart("congestion_station_profile")
    if prof is None:
        st.warning("`04_build_congestion_mart.py` 실행이 필요합니다.")
        return
    disp = load_display_master()
    keys = sorted(disp["station_key"].tolist()) if disp is not None \
        else sorted(prof["station_name"].unique())
    labels = {}
    if disp is not None:
        lm = dict(zip(disp["station_key"], disp["available_lines"]))
        labels = {k: short_label(k, lm.get(k, "")) for k in keys}

    c1, c2 = st.columns([2, 1])
    key = c1.selectbox("역", keys, index=keys.index("서울역") if "서울역" in keys else 0,
                       format_func=lambda k: labels.get(k, k))
    day_type = c2.selectbox("요일유형", ["weekday", "saturday", "sunday"],
                            format_func=lambda x: DAY_TYPE_KO[x])

    sub = prof[(prof["station_name"] == key) & (prof["day_type"] == day_type)]
    if sub.empty:
        st.info("해당 조합의 관측 데이터가 없습니다.")
        return

    # 환승역은 노선을 골라 하나씩 본다. 섞으면 어느 노선의 값인지 알 수 없다.
    lines = sorted(sub["line_id"].astype(str).unique(), key=line_sort_key)
    line = (st.radio("노선", lines, horizontal=True,
                     format_func=line_label, key="cong_line_%s" % key)
            if len(lines) > 1 else lines[0])

    one = sub[sub["line_id"].astype(str) == line].copy()
    one["방향"] = one["direction"].map(DIRECTION_KO).fillna(one["direction"])
    st.table(round1(one[["방향", "mean_congestion", "max_congestion",
                         "p95_congestion", "peak_time_bin", "peak_duration_min",
                         "am_peak_mean", "pm_peak_mean"]]
                    .rename(columns={"mean_congestion": "평균(%)",
                                     "max_congestion": "관측 최대(%)",
                                     "p95_congestion": "상위5%(%)",
                                     "peak_time_bin": "피크 시간대",
                                     "peak_duration_min": "피크 지속(분)",
                                     "am_peak_mean": "오전피크",
                                     "pm_peak_mean": "오후피크"}))
             .reset_index(drop=True))

    lookup = load_mart("congestion_edge_lookup")
    if lookup is not None:
        lk = lookup[(lookup["station_name"] == key)
                    & (lookup["day_type"] == day_type)
                    & (lookup["line_id"].astype(str) == line)]
        if not lk.empty:
            st.subheader("%s 시간대별 기대 혼잡도" % line_label(line))
            piv = lk.pivot_table(index="time_bin", columns="direction",
                                 values="congestion_median").sort_index()
            piv.columns = [DIRECTION_KO.get(c, c) for c in piv.columns]
            congestion_line_chart(piv, 320)

    st.divider()
    st.subheader("평일 최혼잡 구간 Top 10")
    st.caption("관측 최대 = 11개 시점(2022.12~2026.6) 가운데 가장 높았던 값입니다. 경로 추천과 위 그래프의 "
               "'기대 혼잡도'는 시점별 중앙값이라 더 낮게 나옵니다 (예: 성신여대입구 하행 08:00대 관측 최대 185.5% · 기대 134.0%).")
    top = prof[prof["day_type"] == "weekday"].nlargest(10, "max_congestion").copy()
    top["호선"] = top["line_id"].map(line_label)
    top["방향"] = top["direction"].map(DIRECTION_KO).fillna(top["direction"])
    st.table(round1(top[["호선", "station_name", "방향", "peak_time_bin",
                         "mean_congestion", "max_congestion", "peak_duration_min"]]
                    .rename(columns={"station_name": "역명", "peak_time_bin": "피크 시간대",
                                     "mean_congestion": "평균(%)",
                                     "max_congestion": "관측 최대(%)",
                                     "peak_duration_min": "피크 지속(분)"}))
             .reset_index(drop=True))
    st.caption("최댓값이 가장 큰 역과 피크가 가장 오래 지속되는 역은 서로 다릅니다.")


@st.cache_data(show_spinner=False)
def line_node_sequence(line: str, scope: str, start: str, end: str):
    """그래프를 걸어서 노선 계통의 **운행 순서** 노드 목록을 만든다.

    station_code 정렬은 쓸 수 없다. 성수지선이 성수(211) -> 용답(244) -> 신답(245)
    -> 용두(250) -> 신설동(246) 처럼 번호와 운행 순서가 다르기 때문이다.
    분기역은 본선/지선 노드가 나뉘어 있으므로 같은 역 안의 계통 환승도 이어준다.
    """
    ride = load_mart("route_edge_mart")
    if ride is None:
        return []
    r = ride[ride["line_id"].astype(str) == str(line)]

    def sname(n):
        return str(n).split("_", 1)[1].split("@")[0]

    def allowed(n):
        nm, has_branch = sname(n), "@" in str(n)
        if scope == "line2_main":
            return not has_branch and nm not in (SEONGSU_ONLY | SINJEONG_ONLY)
        if scope == "seongsu":
            return ("seongsu" in str(n)) or nm in SEONGSU_ONLY
        if scope == "sinjeong":
            return ("sinjeong" in str(n)) or nm in SINJEONG_ONLY
        if scope == "hanam":
            return "macheon" not in str(n) and nm not in MACHEON_ONLY
        if scope == "macheon":
            return nm not in HANAM_ONLY
        return True

    adj = {}
    for row in r.itertuples():
        a, b = row.from_node, row.to_node
        if allowed(a) and allowed(b):
            adj.setdefault(a, []).append(b)

    # 같은 역 안의 계통 환승(강동/성수/신도림/응암)도 이어준다.
    tr = load_mart("transfer_edge_mart")
    if tr is not None:
        t = tr[(tr["from_line"].astype(str) == str(line))
               & (tr["to_line"].astype(str) == str(line))]
        for row in t.itertuples():
            a, b = row.from_node, row.to_node
            if allowed(a) and allowed(b):
                adj.setdefault(a, []).append(b)
                if scope != "main" or str(line) != "6":
                    adj.setdefault(b, []).append(a)

    nodes = set(adj) | {v for vs in adj.values() for v in vs}
    starts = sorted(n for n in nodes if sname(n) == start)
    ends = {n for n in nodes if sname(n) == end}
    if not starts or not ends:
        return []
    cur = starts[0]
    seq, seen = [cur], {cur}
    while True:
        succ = [v for v in adj.get(cur, []) if v not in seen]
        if not succ:
            break
        # 순환선에서 시작하자마자 끝점으로 붙지 않도록 끝점은 마지막에 고른다.
        pick = next((v for v in succ if v not in ends), succ[0])
        seq.append(pick)
        seen.add(pick)
        cur = pick
        if cur in ends:
            break
    return seq


def page_congestion_line():
    """노선/방향/요일유형/시간대를 고르면 노선 전체 기대 혼잡도를 역 순서대로 본다."""
    lookup = load_mart("congestion_edge_lookup")
    if lookup is None:
        st.warning("`04_build_congestion_mart.py` 실행이 필요합니다.")
        return

    labels = [p["label"] for p in SERVICE_PATTERNS]
    c1, c2 = st.columns([1.2, 1])
    label = c1.selectbox("노선", labels, key="cs_line")
    pat = next(p for p in SERVICE_PATTERNS if p["label"] == label)
    dir_labels = [d[0] for d in pat["directions"]]
    dir_label = c2.selectbox("방향", dir_labels, key="cs_dir_%s" % label)
    _, direction, start, end = next(d for d in pat["directions"] if d[0] == dir_label)

    c3, c4 = st.columns([1, 1.4])
    day_type = c3.selectbox("요일유형", ["weekday", "saturday", "sunday"],
                            format_func=lambda x: DAY_TYPE_KO[x], key="cs_day")
    bins = sorted(lookup["time_bin"].dropna().unique())
    default = bins.index("08:00~08:30") if "08:00~08:30" in bins else 0
    time_bin = c4.selectbox("시간대", bins, index=default, key="cs_bin")

    seq = line_node_sequence(pat["line"], pat["scope"], start, end)
    if not seq:
        st.info("노선 순서를 만들지 못했습니다. `06_build_route_graph.py` 실행을 확인하세요.")
        return

    d = lookup[(lookup["station_uid"].isin(seq))
               & (lookup["direction"] == direction)
               & (lookup["day_type"] == day_type)
               & (lookup["time_bin"] == time_bin)].copy()
    if d.empty:
        st.info("해당 조건의 관측 데이터가 없습니다. 다른 시간대를 선택해 보세요.")
        return

    order = {n: i for i, n in enumerate(seq)}
    d["_ord"] = d["station_uid"].map(order)
    d = d.sort_values("_ord").drop_duplicates("station_uid").reset_index(drop=True)
    d["표시역명"] = d["station_name"].map(lambda x: DISPLAY_STATION_NAME.get(x, x))
    # 같은 역이 본선/지선 노드로 두 번 나오면 계통을 덧붙여 구분한다.
    dup = d["표시역명"].duplicated(keep=False)
    d.loc[dup, "표시역명"] = d.loc[dup].apply(
        lambda r: "%s(순환)" % r["표시역명"] if "eungam" in str(r["station_uid"])
        else r["표시역명"], axis=1)

    import plotly.graph_objects as go
    color = LINE_COLORS.get(pat["line"], PRIMARY)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(d))), y=d["congestion_median"], mode="lines+markers",
        line=dict(color=color, width=2.6), marker=dict(size=7, color=color),
        customdata=np.stack([d["표시역명"], [label] * len(d), [dir_label] * len(d),
                             [DAY_TYPE_KO[day_type]] * len(d),
                             [time_bin] * len(d)], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}"
                       "<br>%{customdata[3]} %{customdata[4]}"
                       "<br>기대 혼잡도 %{y:.1f}%<extra></extra>"),
        showlegend=False))
    for y, c, t in ((80, "#9CA3AF", "체감 가중 시작 80%"),
                    (100, WARN, "정원 100% (1칸 160명)"), (130, DANGER, "혼잡 주의 130%")):
        if d["congestion_median"].max() >= y * 0.75:
            fig.add_hline(y=y, line_dash="dot", line_color=c, opacity=0.55,
                          annotation_text=t, annotation_position="top left",
                          annotation_font_size=10)
    fig.update_layout(
        height=440, margin=dict(l=8, r=8, t=8, b=8),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", dragmode=False,
        xaxis=dict(fixedrange=True, tickangle=-60, gridcolor="#F3F4F6",
                   tickmode="array", tickvals=list(range(len(d))),
                   ticktext=d["표시역명"].tolist()),
        yaxis=dict(fixedrange=True, title="기대 혼잡도(%)",
                   gridcolor="#EEF0F2", rangemode="tozero"))
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False})

    c = st.columns(3)
    top = d.loc[d["congestion_median"].idxmax()]
    c[0].metric("최대 기대 혼잡도", "%.1f%%" % top["congestion_median"])
    c[0].caption("최대 지점: %s" % top["표시역명"])     # delta 칸에 넣으면 '증가(↑)'로 읽혀 글로 표시
    c[1].metric("구간 평균", "%.1f%%" % d["congestion_median"].mean())
    c[2].metric("역 수", "%d개" % len(d))

    with st.expander("역별 값 보기"):
        v = d[["표시역명", "congestion_median", "congestion_p90", "n_snapshots"]].copy()
        v.insert(0, "순서", range(1, len(v) + 1))
        st.table(round1(v.rename(columns={"표시역명": "역명",
                                          "congestion_median": "기대 혼잡도(%)",
                                          "congestion_p90": "상위10%(%)",
                                          "n_snapshots": "관측 스냅샷"}))
                 .reset_index(drop=True))


def page_event():
    st.caption("**과거 데이터** · 지난 이벤트 날의 실제 승하차 기록으로 본 이벤트 효과입니다 (사후 분석, 실시간 알림 아님).")
    l1, l2 = st.columns([4, 1.3])
    l1.caption("이 효과(승하차 급증)가 **시나리오** 탭의 혼잡 배수 가정의 근거입니다.")
    l2.button("이 근거로 만든 시나리오 →", key="evp_to_scn", width="stretch",
              on_click=_go, args=("이벤트 분석", "시나리오"))
    did = load_mart("event_did_mart")
    spike = load_mart("event_spike_mart")
    if did is None and spike is None:
        st.warning("`05_build_event_spike_mart.py` 실행이 필요합니다.")
        return

    if did is not None:
        peak = (did.sort_values("spike_ratio", ascending=False)
                .groupby(["event_id", "event_name", "station_name"], as_index=False).first())
        st.subheader("이벤트 유형별 순효과")
        st.caption("`spike_ratio` 는 대조군 없는 전후 비교입니다. 그날 도시 전체가 붐볐다면 "
                   "배율이 부풀려집니다. 대조군을 둔 이중차분(DiD)으로 순효과를 분리했습니다.")
        g = (peak.groupby("event_type")
             .agg(건수=("event_id", "size"), 전후비교=("spike_ratio", "median"),
                  DiD순효과=("did_ratio_normalized", "median"),
                  절대증가=("did_absolute_lift", "median")))
        g.index = [EVENT_TYPE_KO.get(x, x) for x in g.index]
        g.index.name = "이벤트 유형"
        g = g.rename(columns={"전후비교": "전후비교(배)", "DiD순효과": "DiD 순효과(배)",
                              "절대증가": "절대증가(명)"})
        st.table(round1(g))
        st.caption("숫자가 작아진 것이 정확해진 것입니다. 비율만 보면 왜곡되므로 "
                   "절대 증가 인원을 함께 봅니다.")

        st.subheader("이벤트별 상세")
        det = peak.nlargest(15, "spike_ratio").sort_values("date")[
            ["event_name", "station_name", "date", "spike_ratio", "control_ratio",
             "did_ratio_normalized", "did_absolute_lift", "parallel_trend_ok"]].copy()
        det["event_name"] = det["event_name"].map(ko_event_name)
        det["parallel_trend_ok"] = det["parallel_trend_ok"].map(
            {True: "충족", False: "위반"}).fillna("판정불가")
        st.table(round1(det.rename(
            columns={"event_name": "이벤트", "station_name": "역", "date": "날짜",
                     "spike_ratio": "전후비교(배)", "control_ratio": "대조군(배)",
                     "did_ratio_normalized": "DiD 순효과(배)",
                     "did_absolute_lift": "절대증가(명)",
                     "parallel_trend_ok": "평행추세"})).reset_index(drop=True))
        n_bad = int((did["parallel_trend_ok"] == False).sum())
        st.caption("평행추세 위반 %d행. 위반 이벤트는 DiD 가정이 깨진 것이므로 "
                   "효과 추정을 신뢰하지 않습니다." % n_bad)

    if spike is not None:
        nb = spike[(spike["event_type"] == "new_year_bell")
                   & (spike["station_name"] == "종각")]
        if not nb.empty:
            st.subheader("일 총량으로는 보이지 않는 이벤트 — 제야의 종")
            cmp = nb.groupby(["event_id", "scope"])["spike_ratio"].max().unstack("scope")
            cmp.columns = [{"daily": "일 총량(배)",
                            "peak_hours": "야간 시간대(배)"}.get(c, c) for c in cmp.columns]
            # NYB_2022 같은 내부 id 대신 연도 표기로 바꾼다.
            cmp.index = ["%s 제야의 종" % str(x).split("_")[-1] for x in cmp.index]
            cmp.index.name = "이벤트"
            st.table(round1(cmp))
            st.caption("일 총량은 1.0배(효과 없음)인데 야간만 보면 4~12배입니다. "
                       "일 총량 피처로 쓰면 이벤트가 사라집니다.")


# --------------------------------------------------------------------------
# 페이지 5. 프로젝트 소개 및 검증 리포트
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def graph_scope(stamp: float = 0.0) -> dict:
    """현재 앱이 실제로 쓰는 경로 그래프의 범위 (프로젝트 페이지 표기를 하드코딩하지 않기 위해)."""
    ride = load_mart("route_edge_mart")
    te = load_mart("transfer_edge_mart")
    disp = load_display_master()
    out = {"nodes": None, "ride": None, "transfer": None, "stations": None, "lines": []}
    if ride is not None:
        out["nodes"] = len(set(ride["from_node"]) | set(ride["to_node"]))
        out["ride"] = len(ride)
        out["lines"] = sorted(ride["line_id"].astype(str).unique(), key=line_sort_key)
    if te is not None:
        out["transfer"] = len(te)
    if disp is not None:
        out["stations"] = len(disp)
    return out


def contest_validation():
    """공모전 데이터 검증 요약. 숫자는 reports/v2/*.json(스크립트 32·35·36·37 산출물)에서 읽는다 — 화면에 직접 적지 않는다."""
    import json
    rep = ROOT / "reports" / "v2"

    def _j(name):
        try:
            return json.loads((rep / name).read_text(encoding="utf-8"))
        except Exception:
            return None
    car, fac, sp, tr = (_j("car_sample_validation.json"), _j("facility_poc.json"),
                        _j("spatial_poc.json"), _j("transfer_car.json"))
    st.markdown("##### 공모전 데이터 검증")
    rows = []
    if car:
        r = car["treatments"]["raw"]
        ex = r["external"]["by_day"]["weekday"]
        th = r["external"]["threshold100_weekday_cell"]
        rows.append(["기대 혼잡도 외부 정합성", "평일 구간×방향×30분 %d칸 (샘플 실측 vs 앱)" % ex["cells"],
                     "Pearson %.2f · Spearman %.2f · 평균 절대 차이 %.1f%%p" % (ex["pearson"], ex["spearman"], ex["mae"])])
        rows.append(["정원 100% 초과 구간 판정", "같은 %d칸 중 실측 100%% 초과 %d칸" % (ex["cells"], th["tp"] + th["fn"]),
                     "재현율 %.0f%% · 정밀도 %.2f (놓친 칸 중 기대값이 100%%에서 20%%p 이내인 비율 %.0f%%)"
                     % (100 * th["recall"], th["precision"], 100 * th["fn_exp_gap_below_100"]["share_within_20"])])
        for k, lab, smp in (("lodo", "하루씩 빼고 학습", "출퇴근 열차 평균 60%% 이상 통과 %d건"),
                            ("lofo", "편성 하나씩 빼고 학습", "출퇴근 열차 평균 60%% 이상 통과 %d건"),
                            ("deploy", "배포 방식 (혼잡 구간·시간도 학습 자료로만 선정)",
                             "학습 자료의 혼잡 구간·30분 통과 전부 %d건")):
            v = r[k]
            rows.append(["여유 칸 추천 · %s" % lab, smp % v["n"],
                         "추천 칸이 열차 중앙값 이하 %.1f%% · 열차 평균보다 %.1f%%p 덜 붐빔 (무작위 칸 %+.1f%%p)"
                         % (100 * v["hit_rate"], v["gain_mean"], v["random_gain_mean"])])
        g = car.get("grade_only_fallback_lodo")
        if g:
            rows.append(["등급(1~4)만 개방될 때", "같은 %d건" % g["n_eval"],
                         "칸 구분 가능 %.0f%% · 그중 %.1f%% · %.1f%%p (같은 사례를 %%로 안내하면 %.1f%% · %.1f%%p)"
                         % (100 * g["coverage"], 100 * g["hit_rate"], g["gain_mean"],
                            100 * g["pct_on_same_cases"]["hit_rate"], g["pct_on_same_cases"]["gain_mean"])])
    if fac:
        s = fac["survey"]
        rows.append(["교통약자 하차 칸", "4호선 엘리베이터 역 도착 구간×30분 %d개 (열차 평균 60%% 이상)" % s["combos"],
                     "옆 칸이 5%%p 이상 덜 붐빔 %.0f%% · 평균 %.1f%%p · 칸 번호 기준을 반대로 봐도 같은 칸 %.0f%%"
                     % (100 * s["neighbor_better_share"], s["gain_when_moved_mean"], 100 * s["orientation_same_pick_share"])])
    if sp:
        vc = sp.get("vertical_check") or []
        rows.append(["역사 도면 2D·3D", "동묘앞역 SHP(B1~B3) · GLB 3개",
                     "층 바닥 높이 2D = 3D: " + " · ".join("%s %.2f m" % (x["floor"][:2], x["shp_floor_z"]) for x in vc)
                     + " (차이 %.1f m)" % max((abs(x["diff_m"]) for x in vc), default=0)])
    if tr:
        p = tr["primary_alight"]
        rows.append(["환승동선 인접 칸 (보조분석, 사전 등록)", "열차 통과 %d회 · %d일" % (p["n_passages"], p["n_dates"]),
                     "사전등록 기준 미충족 · 주장에 미사용 (효과 %+.2f%%p, 95%% CI [%+.2f, %+.2f])"
                     % (p["pooled"], p["ci95"][0], p["ci95"][1])])
    if not rows:
        st.caption("검증 결과 파일(reports/v2/*.json)이 없습니다.")
        return
    st.dataframe(pd.DataFrame(rows, columns=["항목", "표본", "결과"]), hide_index=True, width="stretch",
                 column_config={"항목": st.column_config.TextColumn(width="medium"),
                                "표본": st.column_config.TextColumn(width="medium"),
                                "결과": st.column_config.TextColumn(width="large")})
    n = (car or {}).get("sample", {})
    st.caption("칸별 혼잡도 샘플: 공모전 제공 4호선 편성 %s개 (%s) · 결과는 이 표본 기준이며 다른 노선·계절로 일반화하지 않습니다. "
               "재현: scripts/v2/32·35·36·37." % (n.get("trains", "-"), " ~ ".join(str(x) for x in n.get("dates", []))))


def page_project():
    tabs = st.tabs(["프로젝트 개요", "데이터", "방법론", "검증 결과", "한계"])
    gs = graph_scope(ensure_map_fresh())
    has9 = any(x.startswith("9") for x in gs["lines"])
    line_txt = "1~9호선 (9호선 일반·급행)" if has9 else "1~8호선"

    with tabs[0]:
        st.markdown("#### 서울 지하철 %s 혼잡도 기반 쾌적 경로 추천 · v2 공모전판" % line_txt.split(" ")[0])
        st.dataframe(pd.DataFrame([
            ["버전", "v2 (시간 진행형 경로 판정 · Counterfactual · 혼잡 분산 지도 · 이벤트 시나리오)"],
            ["지원 노선", line_txt],
            ["경로 그래프", "%s 노드 · 승차 엣지 %s + 환승 엣지 %s · 역 %s곳"
             % (gs["nodes"], gs["ride"], gs["transfer"], gs["stations"])],
            ["Time-dependent 적용", "적용 — 각 구간에 실제로 도착하는 시각의 30분 bin 혼잡도·배차로 다시 판정 "
                                     "(step policy). v1 은 출발 시각 bin 을 경로 전체에 고정"],
            ["경로/시간 대안 판정", "v2.3 사전 등록 기준 T1 (100%+ 노출 5분 이상 또는 절반 이상 감소)"],
            ["혼잡도 의미", "과거 패턴 기반 기대값 (정원 대비 승차 인원, 100% = 1칸 160명) · 실시간 정보 아님"],
            ["공모전판 추가", "여유 칸 안내(칸별 혼잡도 샘플, 4호선 시범) · 교통약자 하차 칸(빠른하차정보 1~8호선 + 4호선 칸 비교) · "
                             "역사 도면 2D·3D(동묘앞, 공간정보 PoC) — 검증은 '검증 결과' 탭"],
        ], columns=["항목", "현재 앱"]), hide_index=True, width="stretch")
        st.caption("그래프 수치는 앱 데이터 폴더(route_edge_mart · transfer_edge_mart · station_display_master)에서 "
                   "실행 시 직접 셉니다. v2 상세 결과는 README / 리포트(docs/v2)에 정리합니다.")
        c = st.columns(3)
        for col, (a, b, d) in zip(c, [("승하차 원본", "797,946행", "48개월"),
                                      ("혼잡도 셀", "715,065", "11 스냅샷"),
                                      ("검증 이벤트", "87건", "DiD 순효과")]):
            col.metric(a, b)
            col.caption(d)                                # delta 칸에 넣으면 '증가(↑)'로 읽혀 글로 표시
        st.caption("위 세 수치는 v1 데이터 구축 단계(서울교통공사 1~8호선 자료) 기준입니다.")
        st.markdown("""
##### 핵심 발견
| # | 발견 |
|---|---|
| 1 | 쾌적 대안은 일부 OD 에서만 나타난다 (v1 평가: 08:30 3/13, 18:00 2/13). 저녁의 병목은 혼잡 감소폭 단일 조건 |
| 2 | 이벤트는 대안의 개수가 아니라 구성을 바꾼다 (불꽃축제일 18:00 은 선호 모드별로 결과가 갈린다) |
| 3 | LightGBM 이 groupby 평균 baseline 을 이기지 못했고, 사전 등록한 조건대로 baseline 채택 |
| 4 | 전후 비교는 이벤트 효과를 과대추정한다 (불꽃축제 3.674 → DiD 3.403) |
""")

    with tabs[2]:
        st.markdown("""
##### 경로 점수
경로 추천을 **시간 · 혼잡 · 환승 피로 · 이벤트 위험 · 착석 가능성** 을 함께 고려하는
multi-objective scoring 문제로 재정의했습니다. 혼잡도(%)와 시간(분)은 단위가 달라
그대로 더할 수 없으므로 모든 비용을 **체감 이동시간(분)** 으로 환산했습니다.
""")
        st.code("perceived_time = travel_time × (1 + 0.5 × max(0, 혼잡도 − 80) / 100)")
        st.markdown("""
##### 사용자 UI 와 내부 그래프의 분리
여유로 서울 은 사용자에게 **역 단위 입력**을 제공하지만, 내부 그래프와 혼잡도 계산은
**호선별 station_uid 단위**로 유지합니다. 환승역에서 어떤 노선을 처음 탈지는 사용자가
고정하지 않는 한 알고리즘이 후보로 비교하며, **최초 승차 노선 선택은 환승으로
계산하지 않습니다.**

##### v2 방법
| 기능 | 방법 |
|---|---|
| 시간 진행형 판정 | 각 구간 진입 시각의 30분 bin 혼잡도·배차 간격으로 경로를 다시 평가 (step policy) |
| 혼잡 진단 카드 | 지금 / 경로 변경 / 시간 조정(±60분) / 둘 다 — v2.3 사전 등록 기준 T1 과 같은 classify() |
| 시간 바꿔 보기 | 같은 경로를 ±30·60분 옮겨 재평가, 판정은 T1. 곡선은 15분 간격 참고값 |
| 혼잡 분산 지도 | 무작위 OD 620쌍 × 6시각 표본의 구간별 집계, 종합 점수 없음 |
| 이벤트 시나리오 | 승하차 급증(실측) → 혼잡 배수 가정(λ=0.3), 사전 등록(preregistration_v25) |
""")

    with tabs[1]:
        sv = load_mart("schema_validation", "reports/data_quality")
        cc = load_mart("cross_check", "reports/data_quality")
        if sv is None:
            st.info("`11_validate_schemas.py` 실행이 필요합니다.")
        else:
            c = st.columns(2)
            c[0].metric("테이블 스키마", "PASS %d / FAIL %d"
                        % (int((sv.status == "PASS").sum()), int((sv.status == "FAIL").sum())))
            if cc is not None:
                c[1].metric("교차 검증", "PASS %d / FAIL %d"
                            % (int((cc.status == "PASS").sum()),
                               int((cc.status == "FAIL").sum())))
            st.dataframe(sv, width="stretch", hide_index=True)
            if cc is not None:
                st.dataframe(cc, width="stretch", hide_index=True)
            st.caption("리포트는 읽지 않으면 아무 일도 일어나지 않습니다. "
                       "pandera 스키마 검증은 깨지면 파이프라인이 멈추는 테스트입니다.")

    with tabs[2]:                       # 방법론 (계속) — 탭 컨테이너는 여러 번 들어갈 수 있다
        st.markdown("##### 혼잡도 모델")
        st.warning("**사전에 정한 통과 조건**: LightGBM 이 `역×방향×요일×시간대 평균` "
                   "baseline 을 valid 에서 이기지 못하면 모델을 폐기하고 baseline 을 "
                   "사용한다. → **판정: baseline 채택**")
        for name, sub, cap in [("baseline_ladder", "reports/model",
                                "L1→L2 에서 MAE 11.1→2.4. 혼잡도를 결정하는 것은 '어느 역인가'입니다."),
                               ("ablation_study", "reports/model",
                                "B→C 가 48개월 승하차 데이터를 쓰는 이유입니다. R² 0.450 → 0.734."),
                               ("shap_summary_model2", "reports/model",
                                "폐기한 Model-1 이 아니라 미관측 역 보간에 쓰는 Model-2 를 설명합니다."),
                               ("error_analysis_by_line", "reports/model", "")]:
            df = load_mart(name, sub)
            if df is not None:
                st.subheader(name)
                st.dataframe(df, width="stretch", hide_index=True)
                if cap:
                    st.caption(cap)
        st.error("**혼잡도가 높을수록 부정확합니다.** ~30% 구간 MAE 1.474 → 130%+ 구간 9.227. "
                 "그래서 고혼잡 경고는 절대값이 아니라 **상대 순위(상위 5% 초과)** 로 제공합니다.")

        st.subheader("평가 체계")
        st.dataframe(pd.DataFrame([
            ["데이터 품질", "스키마 검증", "pandera + 교차검증"],
            ["이벤트 효과", "인과 효과 추정", "DiD + 평행추세 검증, absolute_lift"],
            ["혼잡도 예측", "회귀 + 불균형 분류", "Baseline ladder, Skill Score, PR-AUC, Ablation, SHAP"],
            ["경로 추천", "다목적 최적화", "3축 Pareto, Stretch Factor, directed-edge Jaccard"],
            ["경로·시간 대안 (v2)", "사전 등록 판정", "T1 기준 classify(), 강건성 검증"],
            ["착석 가능성", "검증 불가", "정답 데이터 없음 — 한계로 명시"],
        ], columns=["레이어", "문제 유형", "표준 지표"]), width="stretch", hide_index=True)

    with tabs[3]:
        contest_validation()
        st.markdown("##### v1 경로 추천 평가")
        st.caption("아래는 v1 경로 추천 평가(1~8호선, 미리 계산한 대표 OD)입니다. "
                   "v2 상세 검증 결과는 README / docs/v2 리포트에서 보강합니다.")
        tags = precomputed_tags()
        if not tags:
            st.info("`10_evaluate_routes.py` 실행이 필요합니다.")
        else:
            tag = st.selectbox("평가 조건", tags)
            cases = load_mart("route_evaluation_mart_%s" % tag)
            summ = load_mart("route_evaluation_summary_%s" % tag)
            if summ is not None:
                calm = summ[summ["preference_mode"] == "calm"]
                c = st.columns(4)
                c[0].metric("OD쌍", len(calm))
                c[1].metric("유의미한 대안", int(calm["has_meaningful_alternative"].sum()))
                c[2].metric("Pareto front 평균", "%.2f" % calm["n_pareto_routes_3d"].mean())
                c[3].metric("제약 위반", int((cases["constraint_violation"] != "").sum())
                            if cases is not None else 0)
                png = REPORTS / "route" / ("pareto_front_scatter_%s.png" % tag)
                if png.exists():
                    st.image(str(png), width="stretch")
                st.markdown("""
목적함수는 셋이며 모두 작을수록 좋습니다: `actual_time` · `max_congestion` ·
**`transfer_penalty`**. 2축만 쓰면 *31분/122%/환승 0회* 직통이
*30분/120%/환승 2회* 에 지배당해 사라집니다. **Pareto 는 후보 보존 장치이지
최종 순위가 아닙니다.**
""")
                show = calm.copy()
                show["OD"] = (show["origin"].str.split("_").str[1] + " → "
                              + show["destination"].str.split("_").str[1])
                st.dataframe(show[["OD", "n_candidate_routes", "n_pareto_routes_3d",
                                   "best_max_congestion_reduction",
                                   "avg_jaccard_similarity_top3",
                                   "has_meaningful_alternative"]],
                             width="stretch", hide_index=True)
                st.caption("`Pareto front = 1` 인 OD 는 trade-off 자체가 존재하지 않습니다. "
                           "임의 임계값이 아니라 수학적으로 대안이 없음을 보입니다.")

    with tabs[4]:
        st.subheader("데이터가 없어서 못 하는 것")
        st.dataframe(pd.DataFrame([
            ["실시간 혼잡도 예측", "실시간 위치·재차 데이터 없음", "과거 패턴 기반 기대 혼잡도"],
            ["전 노선 칸별 혼잡도", "칸별 혼잡도는 공모전 샘플(4호선 편성 6개)만 확보", "4호선 여유 칸 안내(시범) · 환승 동선상 유리한 칸·문"],
            ["착석 확률", "좌석 점유 데이터 없음", "착석 가능성 proxy 점수"],
            ["이벤트 당일 실제 혼잡도", "혼잡도가 분기 평균 패턴", "이벤트성 spike 기반 위험 보정"],
            ["신분당선·경의중앙선 등 우회", "1~9호선 외 운영 노선 자료 미포함", "1~9호선 내 대안 경로"],
        ], columns=["못 하는 것", "사유", "대신 제공하는 것"]), width="stretch", hide_index=True)

        st.subheader("알려진 약점")
        st.markdown("""
1. **고혼잡 구간에서 오차가 6배** — 고혼잡 경고는 절대값이 아니라 상대 순위로 제공합니다.
2. **`label_high` 기준 불일치** — 미관측 역은 전역 p95 로 대체되어 분류 지표에 편향이 있습니다.
3. **Pareto front 는 K개 후보 안에서의 front** — Yen's K-shortest 집합에 한정됩니다.
4. **출발 시각의 time_bin 을 경로 전체에 고정** — 시간 전진을 반영하지 않았습니다. (v2 에서 해결)
5. **확인됨 (v2)**: 혼잡도 = 정원 대비 승차 인원 비율, 100% = 1칸 160명.
   **미해결**: 스냅샷 날짜의 의미, 8호선 오차 원인.
""")
        for name, rel in [("평가 전략", "docs/evaluation_strategy.md"),
                          ("모델 카드", "docs/model_card_congestion.md")]:
            p = ROOT / rel
            if p.exists():
                with st.expander("%s 전문" % name):
                    st.markdown(p.read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------
# 페이지. 노선도 편집 (개발용, 주소에 ?edit=1 일 때만 메뉴에 보임)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _map_checker():
    """scripts/v2/29_check_map_workbook.py 를 모듈로 불러온다 (검증 + 1~9호선 경로 그래프 기준 가져오기)."""
    import importlib.util
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("chk29_app", ROOT / "scripts" / "v2" / "29_check_map_workbook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _map_editor_view():
    import plotly.graph_objects as go
    from datetime import datetime as _dt
    from yeoyuro_v2.map_workbook import diff as _diff
    stt = st.session_state.setdefault("map_editor", {"mtime": None, "n": 0})
    mt = MAP_WORKBOOK.stat().st_mtime
    if stt["mtime"] != mt:
        chk = _map_checker()
        with st.spinner("저장된 엑셀을 읽어 다시 그리는 중…"):
            res = chk.check(MAP_WORKBOOK)
        if res.get("preview_ok", True) and (chk.CHECK_ROOT / "data" / "master" / "map_visual_nodes.csv").exists():
            bundle = {n: pd.read_csv(chk.CHECK_ROOT / "data" / "master" / f"{n}.csv") for n in
                      ("map_visual_nodes", "map_line_segments", "map_transfer_links", "map_click_areas", "map_labels")}
        else:
            bundle = stt.get("bundle")
        stt.update(changes=_diff(stt.get("snap"), res["snap"]) if stt.get("snap") else None,
                   snap=res["snap"], res=res, bundle=bundle, mtime=mt, n=stt["n"] + 1,
                   saved=_dt.fromtimestamp(mt).strftime("%H:%M:%S"), checked=_dt.now().strftime("%H:%M:%S"))
    res, bundle = stt["res"], stt["bundle"]
    ok = res["ok"] and not res["clash"]
    st.markdown(f"**{'✅ 통과' if ok else ('⚠️ 통과 (겹침 있음)' if res['ok'] else '❌ 확인 필요')}** · "
                f"그림 #{stt['n']} · 엑셀 저장 {stt['saved']} · 확인 {_dt.now():%H:%M:%S} (2초마다 자동 확인)")
    if stt.get("changes"):
        st.info("**이번 저장에서 고친 칸**\n\n" + "\n".join("- " + c for c in stt["changes"][:30]))
    elif stt.get("changes") == []:
        st.warning("저장은 감지했지만 좌표 관련 칸은 바뀌지 않았습니다 (다른 칸을 고쳤거나 같은 값으로 저장).")
    if res["clash"]:
        st.warning("다른 역끼리 같은 좌표: " + " / ".join(f"({x}, {y}) {', '.join(ks)}" for x, y, ks in res["clash"]))
    if not res.get("preview_ok", True):
        st.error("엑셀 내용을 노선도로 바꾸는 단계에서 오류가 났습니다:\n\n" + res.get("stderr", "")[-800:])
    if bundle is None:
        return
    fig = draw_map(bundle, None, show_endpoints=False)
    vn = bundle["map_visual_nodes"]
    fig.add_trace(go.Scatter(x=vn.x_px, y=vn.y_px, mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"),
                             hovertext=[f"{r.visual_node_id}  ({r.x_px:.0f}, {r.y_px:.0f})" for r in vn.itertuples()],
                             hoverinfo="text", showlegend=False))
    fig.update_layout(height=880, uirevision="map_editor", dragmode="pan")
    st.plotly_chart(fig, width="stretch", key="map_editor_chart",
                    config={"scrollZoom": True, "displayModeBar": True})


def page_map_editor():
    st.title("노선도 편집")
    st.caption("엑셀 기본 파일을 직접 읽어 앱 노선도로 그립니다. 엑셀에서 고치고 Ctrl+S 하면 2초 안에 바뀝니다. "
               "마우스 휠 = 확대, 드래그 = 이동, 점에 마우스를 올리면 노드 이름과 좌표가 보입니다.")
    st.code(str(MAP_WORKBOOK), language=None)
    if not MAP_WORKBOOK.exists():
        st.error("기본 좌표 파일이 없습니다.")
        return
    if hasattr(st, "fragment"):
        st.fragment(run_every=2)(_map_editor_view)()
    else:
        st.button("다시 그리기")
        _map_editor_view()


# --------------------------------------------------------------------------
# 공모전판: 역사 안내 (Facility + 공간정보)
#   - 엘리베이터 하차 칸: 빠른하차정보(1~8호선) × 칸별 혼잡도 샘플(4호선)
#   - 역사 도면: 공모전 제공 동묘앞역 2D SHP·3D GLB. 안전시설 위치는 원본이 있을 때만 그린다 (임의로 만들지 않는다)
# --------------------------------------------------------------------------
SPATIAL_DIR = ROOT / "data" / "raw" / "spatial_dongmyo"
SAFETY_DIR = ROOT / "data" / "raw" / "safety_facility"     # 안전시설 위치정보 원본(명세 열: 위도·경도·층번호·시설명)을 두는 곳
SPATIAL_STATION = "동묘앞"
SPATIAL_FLOOR_LABEL = {"B1F": "B1", "B2F": "B2", "B3F": "B3"}
FLOOR_ETYPE = {"unclassified": ("공간", "rgba(203,213,225,0.55)", "#94A3B8"),
               "stair": ("계단", "rgba(234,88,12,0.55)", "#C2410C"),
               "column": ("기둥", "rgba(107,114,128,0.8)", "#4B5563"),
               "inner_wall": ("내벽", "rgba(55,65,81,0.8)", "#374151")}
POC_BADGE = ('<span style="display:inline-block;background:#EEF2FF;color:#3730A3;border:1px solid #C7D2FE;'
             'border-radius:10px;padding:1px 9px;font-size:0.74rem;font-weight:700;margin-right:6px">%s</span>')


def _line_neighbors(line: str, station: str) -> set[str]:
    e = load_mart("route_edge_mart")
    if e is None:
        return set()
    e = e[(e.edge_type == "ride") & (e.line_id.astype(str) == str(line)) & (e.from_station == station)]
    return set(e.to_station)


def _fac_reset(*keys):
    for k in keys:
        st.session_state.pop(k, None)


def _public_card(text: str):
    st.markdown('<div style="border:1px dashed #94A3B8;border-radius:10px;padding:10px 14px;color:#334155;'
                'font-size:0.86rem;background:#F8FAFC">🔒 %s</div>' % _esc(text), unsafe_allow_html=True)


def page_facility_elevator():
    """역사 안내 › 엘리베이터 하차 칸 — 결론 → 칸 그림 → 근거(접힘) 순서."""
    import plotly.graph_objects as go
    from yeoyuro_v2.facility_match import LINE_CARS, alight_crowd_match
    mart = load_mart("fast_exit_elevator")
    if mart is None:
        _public_card(PUBLIC_ELEVATOR_NOTE)
        return
    left, right = st.columns([0.8, 2.4], gap="medium")
    with left:
        lines = sorted(mart.line.astype(str).unique(), key=int)
        st.session_state.setdefault("fac_line", "4" if "4" in lines else lines[0])
        line = st.selectbox("노선", lines, format_func=line_label, key="fac_line",
                            on_change=_fac_reset, args=("fac_station", "fac_toward", "fac_bin"))
        sub = mart[mart.line.astype(str) == line]
        stations = sorted(sub.station.unique())
        if st.session_state.get("fac_station") not in stations:
            st.session_state["fac_station"] = ("동대문역사문화공원" if "동대문역사문화공원" in stations else stations[0])
        stn = st.selectbox("하차역", stations, key="fac_station", on_change=_fac_reset, args=("fac_toward", "fac_bin"))
        towards = sorted(sub[sub.station == stn].toward.unique())
        if st.session_state.get("fac_toward") not in towards:
            # 처음 열 때는 기획서 장면 3 의 예(동대문 → 동대문역사문화공원, 충무로 방면)를 보여준다
            st.session_state["fac_toward"] = "충무로" if (stn == "동대문역사문화공원" and "충무로" in towards) else towards[0]
        tw = st.selectbox("타고 가는 열차", towards, format_func=lambda x: "%s 방면" % x, key="fac_toward",
                          on_change=_fac_reset, args=("fac_bin",))
        els_df = sub[(sub.station == stn) & (sub.toward == tw)].sort_values(["car", "door"])
        elevators = [{"car": int(r.car), "door": None if pd.isna(r.door) else int(r.door), "label": r.label,
                      "toward": r.toward, "crtr_ymd": r.crtr_ymd} for r in els_df.itertuples()]
        crowd, public, prev_st, bin_note = None, None, None, None
        prev = _line_neighbors(line, stn) - {tw}
        if line == "4" and elevators:
            pat = _load_car_pattern()
            if pat is None:
                public = PUBLIC_SAMPLE_NOTE
            elif len(prev) == 1:
                prev_st = next(iter(prev))
                bins = sorted(pat[(pat.line.astype(str) == "4") & (pat.from_station == prev_st)
                                  & (pat.to_station == stn)].time_bin.unique())
                if bins:
                    req = st.session_state.get("fac_bin")
                    if req not in bins:
                        # 경로 결과에서 넘어온 시간대에 표본이 없으면 가장 가까운 시간대를 고르고, 바꿨다는 사실을 화면에 밝힌다
                        def _m(x):
                            h, mm = map(int, str(x).split(":"))
                            return h * 60 + mm
                        if req and re.fullmatch(r"\d\d:\d\d", str(req)):
                            near = min(bins, key=lambda b: (abs(_m(b) - _m(req)), _m(b)))
                            bin_note = ("평일 %s대는 칸별 혼잡도 샘플 표본이 부족해, 가장 가까운 %s대 결과를 보여줍니다."
                                        % (req, near))
                        else:
                            near = "07:30" if "07:30" in bins else bins[0]
                        st.session_state["fac_bin"] = near
                    tb = st.selectbox("평일 도착 시간대", bins, key="fac_bin",
                                      help="칸별 혼잡도 샘플에 표본(통과 3회·2일 이상)이 있는 30분 구간만 고를 수 있습니다.")
                    crowd = alight_crowd_match(pat, "4", prev_st, stn, None, elevators, time_bin=tb)
                else:
                    crowd = {"no_sample": True}

    with right:
        ctx = "%s · %s · %s 방면" % (line_label(line), stn, tw)
        if crowd and crowd.get("time_bin") and not crowd.get("no_sample"):
            ctx += " · 평일 %s" % crowd["time_bin"]
        st.markdown('<div style="color:#475569;font-size:0.9rem;font-weight:600;margin-bottom:4px">%s</div>'
                    % _esc(ctx), unsafe_allow_html=True)
        if bin_note:
            st.markdown('<div style="color:#9A3412;font-size:0.82rem;margin:-2px 0 6px 0">ⓘ %s</div>' % _esc(bin_note),
                        unsafe_allow_html=True)
        if not elevators:
            st.info("이 역·방면은 빠른하차정보에 엘리베이터 칸이 없습니다. 역 안내를 확인해 주세요.")
            return
        d = facility_decision(stn, line, elevators, crowd, public)
        with st.container(border=True):
            html = ['<div style="font-size:1.45rem;font-weight:800;color:#0B3D75;margin-bottom:4px">%s</div>'
                    % _esc(d["title"])]
            html += ['<div style="font-size:0.98rem;color:#1F2937;margin-top:2px">%s</div>' % ln for ln in d["lines"]]
            if d["warn"]:
                html.append('<div style="margin-top:8px;color:#9A3412;font-weight:700">⚠️ %s</div>' % _esc(d["warn"]))
            st.markdown("".join(html), unsafe_allow_html=True)
            b1, b2, _sp = st.columns([1, 1, 0.6])
            b1.button("🕒 출발 시각 바꿔 보기", key="fac_to_time", width="stretch",
                      on_click=_go, args=("경로 추천", "시간 바꿔 보기"),
                      help="경로 추천의 '시간 바꿔 보기'에서 ±60분 출발 시각별 혼잡을 비교합니다.")
            if stn == SPATIAL_STATION:
                b2.button("🗺️ 역사 도면에서 보기", key="fac_to_map", width="stretch",
                          on_click=_go, args=("역사 안내", "역사 도면"))
            else:
                b2.button("🗺️ 역사 도면에서 보기", key="fac_to_map", width="stretch", disabled=True,
                          help="공모전 제공 역사 도면 샘플은 동묘앞역만 있습니다.")

        n = LINE_CARS.get(line, int(sub.car.max()))
        st.markdown(car_strip_html(n, d["pct"], d["ev_labels"], d["pick"], d["gain_pct"], tw),
                    unsafe_allow_html=True)
        st.markdown(strip_legend_html(bool(d["pct"])), unsafe_allow_html=True)
        if public:
            st.write("")
            _public_card(public)

        ev = view_evidence(crowd, d) if prev_st else None
        if ev:
            pct = crowd["pct"]
            ev_cars, pk = ev["ev_cars"], ev["pick"]
            with st.expander(ev["expander"]):
                st.markdown("**칸별 평소 혼잡도 근거**")
                st.caption("%s → %s, 평일 %s대 · 열차 평균 %.0f%% · 통과 %d회 평균"
                           % (prev_st, stn, crowd["time_bin"], crowd["train_mean"], crowd["n_passages"]))
                lw = [3 if (c in ev_cars or c == pk) else 0 for c in range(1, 11)]
                lc = [EV_BORDER if c in ev_cars else (PICK_BORDER if c == pk else "#FFFFFF") for c in range(1, 11)]
                fig = go.Figure(go.Bar(x=["%d" % c for c in range(1, 11)], y=pct,
                                       marker=dict(color=[_cong_bg(v) for v in pct], line=dict(width=lw, color=lc)),
                                       text=["%d%%" % v for v in pct], textposition="outside",
                                       hovertemplate="%{x}번째 칸 · 평소 %{y}%<extra></extra>"))
                fig.add_hline(y=100, line=dict(color="#E8A33D", dash="dash", width=1),
                              annotation_text="정원 100%", annotation_position="top right")
                fig.update_layout(height=290, margin=dict(l=10, r=10, t=24, b=10), plot_bgcolor="#FFFFFF",
                                  paper_bgcolor="#FFFFFF", dragmode=False, showlegend=False,
                                  xaxis=dict(title="칸 (진행 방향 기준)", fixedrange=True, type="category"),
                                  yaxis=dict(title="평소 혼잡도(%)", fixedrange=True, rangemode="tozero",
                                             range=[0, max(130.0, max(pct) * 1.15)]))
                st.plotly_chart(fig, width="stretch", key="fac_bar", config={"displayModeBar": False})
                st.caption(ev["caption"], help=ev["help"])
        st.caption("%s · %s 기준%s" % (FAST_EXIT_NOTE, _fmt_ymd(elevators[0]["crtr_ymd"]),
                                       (" · " + d["note"]) if d["note"] else ""),
                   help="칸 번호는 빠른하차정보 표기(진행 방향 기준)를 따릅니다.")


@st.cache_data(show_spinner=False)
def _spatial_floors(folder: str, stamp: float):
    from yeoyuro_v2.spatial_poc import read_floors
    return read_floors(Path(folder))


def _load_safety_facilities() -> pd.DataFrame | None:
    """안전시설 위치정보 원본(CSV). 없으면 None — 위치를 만들지 않는다."""
    if not SAFETY_DIR.exists():
        return None
    frames = []
    for p in sorted(SAFETY_DIR.glob("*.csv")):
        for enc in ("utf-8-sig", "cp949"):
            try:
                frames.append(pd.read_csv(p, encoding=enc))
                break
            except Exception:
                continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return df if {"위도", "경도", "층번호"} <= set(df.columns) else None


def _floor_figure(floors: dict, fl: str, facilities: pd.DataFrame | None = None):
    """한 층 도면 (단위 m, 세 층 공통 원점). 계단은 주황. 시설은 원본 좌표가 있을 때만 그린다."""
    import plotly.graph_objects as go
    x0 = min(f["bbox"][0] for f in floors.values())
    y0 = min(f["bbox"][1] for f in floors.values())
    fig = go.Figure()
    for et, (name, fill, line) in FLOOR_ETYPE.items():
        xs, ys = [], []
        for p in floors[fl]["polygons"]:
            if (p["etype"] or "unclassified") != et:
                continue
            for ring in p["rings"]:
                xs += list(ring[:, 0] - x0) + [None]
                ys += list(ring[:, 1] - y0) + [None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=fill,
                                     line=dict(color=line, width=1), name=name, hoverinfo="name"))
    if facilities is not None and len(facilities):
        f = facilities[(facilities.floor == fl) & facilities.in_bbox]
        if len(f):
            fig.add_trace(go.Scatter(x=f.x - x0, y=f.y - y0, mode="markers+text", text=f.name,
                                     textposition="top center", name="안전시설",
                                     marker=dict(size=11, color="#DC2626", symbol="diamond")))
    fig.update_layout(height=500, margin=dict(l=4, r=4, t=4, b=4), plot_bgcolor="#F8FAFC",
                      paper_bgcolor="#F8FAFC", legend=dict(orientation="h", y=1.0, x=0.01),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    return fig


@st.cache_resource(show_spinner="3D 모델을 불러오는 중…")
def _spatial_meshes(folder: str, stamp: float, max_faces: int = 20000) -> dict:
    """GLB 층별 메시 (면이 많으면 균등 표본). {'B1': (x, y, z, i, j, k, n_faces_원본), ...} — glTF Y-up → z 가 높이."""
    import trimesh
    rng = np.random.default_rng(0)
    out = {}
    for p in sorted(Path(folder).glob("*.glb")):
        m = trimesh.load(str(p), force="mesh")
        faces = m.faces
        n0 = len(faces)
        if len(faces) > max_faces:
            faces = faces[rng.choice(len(faces), max_faces, replace=False)]
        used, inv = np.unique(faces.ravel(), return_inverse=True)
        v = m.vertices[used]
        f = inv.reshape(-1, 3)
        key = re.search(r"B\d", p.stem)
        out[key.group(0) if key else p.stem] = (v[:, 0], -v[:, 2], v[:, 1], f[:, 0], f[:, 1], f[:, 2], n0)
    return out


def _mesh_figure(meshes: dict, focus: str, cam_rev: int):
    """같은 자리 3D 보기. focus 층은 진하게, 나머지는 옅게. 쿼터뷰·연회색 배경."""
    import plotly.graph_objects as go
    colors = {"B1": "#3B82F6", "B2": "#F59E0B", "B3": "#10B981"}
    fig = go.Figure()
    for k, (x, y, z, i, j, kk, _n) in meshes.items():
        op = 1.0 if focus in ("전체", k) else 0.22      # 선택 층 100%, 나머지 20~30%
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=kk, color=colors.get(k, "#9CA3AF"), opacity=op,
                                name=k, showlegend=True, flatshading=True, hoverinfo="name",
                                lighting=dict(ambient=0.55, diffuse=0.8)))
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#EEF1F4",
                      uirevision="cam%d" % cam_rev,
                      scene=dict(bgcolor="#EEF1F4", aspectmode="manual", aspectratio=dict(x=2.0, y=1.25, z=0.75),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                                 camera=dict(eye=dict(x=1.15, y=-1.15, z=0.95), center=dict(x=0, y=0, z=-0.1))),
                      legend=dict(orientation="h", y=0.99, x=0.01, bgcolor="rgba(255,255,255,0.6)"))
    return fig


def _station_elevator_card(mart, station: str):
    """도면 옆 '이동편의 정보' — 칸-문 위치는 도면 좌표가 아니므로 글로만 보여준다."""
    st.markdown("**♿ 이동편의 정보**")
    if mart is None:
        st.caption(PUBLIC_ELEVATOR_NOTE)
        return
    s = mart[mart.station == station].sort_values(["line", "toward", "car", "door"])
    if s.empty:
        st.caption("이 역의 엘리베이터 칸 정보가 없습니다.")
        return
    rows = []
    for (ln, tw), g in s.groupby(["line", "toward"], sort=False):
        rows.append("<div style='margin:2px 0'><span style='color:%s;font-weight:700'>%s</span> %s 방면 · "
                    "엘리베이터 <b>%s</b></div>" % (LINE_COLORS.get(str(ln), PRIMARY), _esc(line_label(ln)), _esc(tw),
                                                  " · ".join("<span style='white-space:nowrap'>%s</span>" % _esc(x) for x in g.label)))
    st.markdown("<div style='font-size:0.86rem'>%s</div>" % "".join(rows), unsafe_allow_html=True)
    st.caption("승강장 칸-문 위치와 이 도면 좌표를 잇는 공식 연결정보가 없어, 지도 위에는 정확한 위치를 표시하지 않습니다.")


def page_spatial():
    """역사 안내 › 역사 도면 — 2D 도면과 3D 모델을 같은 자리에서 전환. 기술 정보는 접어 둔다."""
    st.markdown('<div style="font-size:1.25rem;font-weight:800;color:#002D56">%s역 역사 구조 '
                '<span style="font-size:0.85rem;color:#6B7280;font-weight:500;margin-right:8px">1·6호선</span>%s</div>'
                % (SPATIAL_STATION, POC_BADGE % "공간정보 PoC"), unsafe_allow_html=True)
    mode = st.session_state.get("sp_mode") or "2D 도면"
    st.markdown('<div style="color:#6B7280;font-size:0.82rem;margin:2px 0 6px 0">%s</div>'
                % ("구조 확인용 · 실내 길찾기 기능 아님 · 안전시설 좌표는 정식 데이터 개방 후 연동됩니다" if mode == "3D 모델"
                   else "현재는 공모전 제공 역사 도면을 표시하며, 안전시설 좌표는 정식 데이터 개방 후 연동됩니다"),
                unsafe_allow_html=True)
    mode = st.segmented_control("보기", ["2D 도면", "3D 모델"], key="sp_mode", required=True,
                                default="2D 도면", label_visibility="collapsed") or "2D 도면"
    shp = sorted(SPATIAL_DIR.glob("*.shp")) if SPATIAL_DIR.exists() else []
    glbs = sorted(SPATIAL_DIR.glob("*.glb")) if SPATIAL_DIR.exists() else []
    mart = load_mart("fast_exit_elevator")
    raw_fac = _load_safety_facilities()
    left, right = st.columns([2.7, 1], gap="medium")
    floors = None
    with left:
        if not shp:
            _public_card(PUBLIC_SAMPLE_NOTE)
            img = ROOT / "reports" / "v2" / ("fig_spatial_2d.png" if mode == "2D 도면" else "fig_spatial_3d.png")
            if img.exists():
                st.image(str(img), caption="로컬 시연판에서 만든 그림 (공모전 제공 동묘앞역 도면)")
        elif mode == "2D 도면":
            floors = _spatial_floors(str(SPATIAL_DIR), max(p.stat().st_mtime for p in shp))
            names = [f for f in ("B1F", "B2F", "B3F") if f in floors]
            st.session_state.setdefault("sp_floor", "B2F" if "B2F" in names else names[0])  # 승강장·계단이 가장 많은 층
            fl = st.segmented_control("층", names, key="sp_floor", required=True,
                                      format_func=lambda x: SPATIAL_FLOOR_LABEL.get(x, x),
                                      label_visibility="collapsed") or names[0]
            fac_df = None
            if raw_fac is not None and st.session_state.get("sp_fac"):
                from yeoyuro_v2.spatial_poc import place_facilities
                fac_df = place_facilities(raw_fac, floors)
            st.plotly_chart(_floor_figure(floors, fl, fac_df), width="stretch", key="sp_floor_fig",
                            config={"displayModeBar": False, "scrollZoom": True})
        else:
            try:
                import trimesh  # noqa: F401
                ok3d = bool(glbs)
            except Exception:
                ok3d = False
            c1, c2 = st.columns([3, 1])
            focus = c1.segmented_control("층 강조", ["전체", "B1", "B2", "B3"], key="sp_floor3d", required=True,
                                         default="전체", label_visibility="collapsed") or "전체"
            if c2.button("시점 초기화", key="sp_cam_reset", width="stretch"):
                st.session_state["sp_cam"] = st.session_state.get("sp_cam", 0) + 1
            if ok3d:
                meshes = _spatial_meshes(str(SPATIAL_DIR), max(p.stat().st_mtime for p in glbs))
                st.plotly_chart(_mesh_figure(meshes, focus, st.session_state.get("sp_cam", 0)),
                                width="stretch", key="sp_3d_fig")
            else:
                img = ROOT / "reports" / "v2" / "fig_spatial_3d.png"
                if img.exists():
                    st.image(str(img))
            st.caption("마우스로 돌리고 휠로 확대합니다.")

    with right:
        with st.container(border=True):
            _station_elevator_card(mart, SPATIAL_STATION)
            st.divider()
            if raw_fac is None:
                st.toggle("🔒 안전시설 위치", key="sp_fac", value=False, disabled=True,
                          help="안전시설 위치정보 원본이 개방되면 위도·경도·층번호로 이 도면에 겹쳐 표시합니다.")
                st.caption("정식 데이터 개방 후 활성화")
            else:
                st.toggle("안전시설 위치", key="sp_fac", help="원본의 위도·경도·층번호를 도면 좌표로 바꿔 겹칩니다.")
            st.button("♿ 엘리베이터 하차 칸 보기 →", key="sp_to_fac", width="stretch",
                      on_click=_open_facility, args=("6", SPATIAL_STATION, None, None))

    with st.expander("데이터 범위 및 한계"):
        st.markdown(
            "- 도면: 공모전 제공 동묘앞역 2D 도면(SHP, B1~B3)과 3D 모델(GLB).\n"
            "- 안전시설 위치정보 원본은 아직 받지 못했습니다. **위치를 임의로 만들어 그리지 않습니다.** "
            "원본이 개방되면 위도·경도는 좌표 변환으로, 층번호(−1 = B1)는 층 배정으로 같은 도면에 겹칩니다.\n"
            "- 2D 도면의 층 바닥 높이와 3D 모델의 층 바닥 높이는 세 층 모두 일치했습니다(차이 0.0 m).\n"
            "- 확인이 필요한 것: 3D 모델의 수평 좌표와 2D 도면의 정합 규칙, 안전시설 X·Y·Z 좌표의 기준, "
            "승강장 칸-문 위치와 도면 좌표의 연결.\n"
            "- 3D 길찾기·동선 최적화는 이번 범위가 아닙니다.")
    with st.expander("개발자 정보 보기"):
        if shp:
            floors = floors or _spatial_floors(str(SPATIAL_DIR), max(p.stat().st_mtime for p in shp))
            rows = []
            for k, f in floors.items():
                et = pd.Series([p["etype"] for p in f["polygons"]]).value_counts().to_dict()
                rows.append({"층": SPATIAL_FLOOR_LABEL.get(k, k), "좌표계": f["crs"], "폴리곤": len(f["polygons"]),
                             "계단": et.get("stair", 0), "크기(m)": "%.0f × %.0f" % (f["bbox"][2] - f["bbox"][0],
                                                                                  f["bbox"][3] - f["bbox"][1])})
            st.table(pd.DataFrame(rows))
            st.caption("원본: " + ", ".join(p.name for p in shp + glbs))
        else:
            st.caption("이 실행 환경에는 공모전 제공 도면 원본이 없습니다.")


PAGE_CAPTION = {"경로 추천": "",
                "혼잡 탐색": "",
                "이벤트 분석": "과거 이벤트 날의 실제 기록(근거)과, 그 근거로 만든 what-if 시나리오를 봅니다.",
                "역사 안내": "교통약자가 이용할 칸을 추천하고, 역 안 구조를 도면으로 확인합니다.",
                "프로젝트 & 검증": ""}

VIEW_FN = {("경로 추천", "경로"): "page_route",
           ("경로 추천", "시간 바꿔 보기"): "page_time_shift",
           ("혼잡 탐색", "역·노선별"): "page_congestion",
           ("혼잡 탐색", "네트워크 지도"): "page_atlas",
           ("이벤트 분석", "시나리오"): "page_event_scenario",
           ("이벤트 분석", "과거 데이터"): "page_event",
           ("역사 안내", "엘리베이터 하차 칸"): "page_facility_elevator",
           ("역사 안내", "역사 도면"): "page_spatial",
           ("프로젝트 & 검증", None): "page_project",
           ("노선도 편집", None): "page_map_editor"}


def _apply_legacy_menu():
    """예전 menu 값(이전 세션·테스트·북마크)을 새 (페이지, subview)로 옮긴다. 한 번 옮기면 지운다."""
    old = st.session_state.get("menu")
    if old in LEGACY_MENU:
        page, view = LEGACY_MENU[old]
        st.session_state["nav_page"] = page
        if view and page in VIEW_KEY:
            st.session_state[VIEW_KEY[page]] = view
    st.session_state["menu"] = None


def main():
    _apply_legacy_menu()
    edit = st.query_params.get("edit") == "1"
    page = st.session_state.get("nav_page")
    if page not in NAV and not (page == "노선도 편집" and edit):
        page = st.session_state["nav_page"] = PAGES[0]

    st.sidebar.title("🚇 여유로 서울")
    st.sidebar.caption("서울 지하철 혼잡도 기반  \n쾌적 경로 추천 시스템")
    _dm = data_mode()
    _have = {"elevator": "빠른하차정보 엘리베이터 칸", "car_sample": "칸별 혼잡도 샘플(4호선)", "spatial": "동묘앞역 2D·3D 도면"}
    st.sidebar.markdown('<span style="display:inline-block;background:rgba(255,255,255,0.12);border:1px solid '
                        'rgba(255,255,255,0.35);border-radius:10px;padding:1px 9px;font-size:0.74rem">공모전 시연 버전</span>',
                        unsafe_allow_html=True)
    _data_help = (DISCLAIMER + "\n\n**이 실행 환경의 공모전 자료**: " + " · ".join(
        "%s %s" % (v, "있음" if _dm[k] else "없음(비활성)") for k, v in _have.items())
        + ". 공모전 제공 샘플과 이용허락범위를 확인하지 않은 자료는 공개 배포판에 넣지 않으며, "
          "해당 기능은 비활성 사유를 화면에 표시합니다.")
    for m in PAGES + (["노선도 편집"] if edit else []):
        st.sidebar.button(m, width="stretch", key="nav_%s" % m,
                          type="primary" if page == m else "secondary",
                          on_click=_go, args=(m,))
    st.sidebar.divider()
    st.sidebar.caption("ⓘ 데이터 기준  \n과거 패턴 기반 기대값", help=_data_help)

    if page == "노선도 편집":
        page_map_editor()
        return
    st.title(page)
    view = None
    if NAV[page]:
        vk = VIEW_KEY[page]
        if st.session_state.get(vk) not in NAV[page]:
            st.session_state[vk] = NAV[page][0]
        view = st.segmented_control("보기", NAV[page], key=vk, required=True,
                                    label_visibility="collapsed", on_change=_sync_widgets_from_canonical)
        view = view if view in NAV[page] else NAV[page][0]
    if page != "경로 추천" or view == "경로":
        if PAGE_CAPTION.get(page):
            st.caption(PAGE_CAPTION[page])
    globals()[VIEW_FN[(page, view)]]()


main()
