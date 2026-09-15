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
}

DISCLAIMER = "표시된 혼잡도는 실시간 측정값이 아닌 **과거 패턴 기반 기대 혼잡도**입니다."

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
def load_mart(name: str, sub: str = "data/marts"):
    base = ROOT / sub
    for ext in (".parquet", ".csv.gz", ".csv"):
        p = base / (name + ext)
        if p.exists():
            try:
                return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_display_master():
    return load_mart("station_display_master", "data/master")


@st.cache_data(show_spinner=False)
def load_map_bundle():
    """벡터 노선도 좌표 묶음. source of truth 는 좌표 워크북(5120x2880)이다."""
    b = {n: load_mart(n, "data/master") for n in
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
    return mod.RouteScorer(ROOT, day_type, depart, query_date)


def sr_module():
    import station_routing as sr
    return sr


# Streamlit selectbox 는 옵션 텍스트를 평문으로만 렌더링한다(HTML/CSS 미적용).
# 그래서 리스트 안에서는 색을 쓸 수 없고, 유니코드 문자로만 노선을 구분한다.
# 컬러가 꼭 필요하면 아래 플래그를 True 로 바꿔 이모지 방식을 쓸 수 있다.
# 다만 4호선(하늘)이 1호선(남색)과 구분되지 않고, 7호선(올리브)·8호선(분홍)의
# 색 대응이 나빠 숫자 방식보다 읽기 어려울 수 있다. 직접 보고 선택할 것.
USE_EMOJI_LINE_MARKS = False

CIRCLED = {"1": "①", "2": "②", "3": "③", "4": "④",
           "5": "⑤", "6": "⑥", "7": "⑦", "8": "⑧"}

# 실제 노선색에 가장 가까운 컬러 이모지 (근사값)
EMOJI_MARK = {"1": "🔵", "2": "🟢", "3": "🟠", "4": "🔹",
              "5": "🟣", "6": "🟤", "7": "🟡", "8": "🔴"}


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
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

MENUS = ["쾌적 경로 찾기", "시간대별 혼잡 조회", "빠른 환승 안내",
         "이벤트 혼잡 경보", "프로젝트 소개 및 검증 리포트"]


# --------------------------------------------------------------------------
# 노선도
# --------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 5120, 2880

# 역명 라벨 미세 조정 파라미터.
#   LABEL_FONT_SCALE : 워크북 label_size_px 에 곱하는 배율(약 5~7% 축소)
#   LABEL_GAP_*      : 마커와 라벨 사이 간격. **데이터 좌표 단위** 라 배율에 따라 함께
#                      변한다. 화면 픽셀 고정으로 주면 전체보기에서 멀어 보이고
#                      확대하면 붙어 보이는 문제가 생긴다.
LABEL_FONT_SCALE = 0.95
#   간격 = 데이터 좌표 몫(확대 시 커짐) + 화면 픽셀 몫(배율 무관 고정).
#   데이터 몫을 줄이고 고정 몫을 두면 초기 화면 간격은 유지되면서
#   확대했을 때 지나치게 벌어지지 않는다.
LABEL_GAP_NORMAL = 2       # 일반역 데이터 몫
LABEL_GAP_TRANSFER = 3     # 환승역 데이터 몫
LABEL_GAP_FIXED_PX = 1     # 화면 고정 몫(빈 줄 높이)

# 역명 라벨을 마커 아래로 내리는 거리(캔버스 좌표 단위).
# 화면 픽셀이 아니라 데이터 좌표라, 축소하면 간격이 좁아지고 확대하면 넓어진다.
# 전체보기에서 라벨이 마커에 붙어 보이고, 확대하면 겹치지 않는 여유가 생긴다.
LABEL_DY_NORMAL = 40      # 일반역
LABEL_DY_MAJOR = 52       # 환승역(마커가 커서 조금 더 띄운다)
LABEL_SIZE_DELTA = -0.3   # 워크북 label_size_px 대비 축소폭(약 7%)
# 노선도 표시 범위. 팬/줌이 이 밖으로 나가지 못하게 고정한다.
X0, X1, Y0, Y1 = 700, 4450, 150, 2750
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


def draw_map(bundle: dict, highlight_path=None):
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
                                     showlegend=False))

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
                                     showlegend=False))

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

    # 4. 역명 label
    #    간격을 데이터 좌표로 주면 확대/축소에 따라 자연스럽게 함께 변한다.
    #    전체보기에서는 마커에 가깝게, 확대하면 여유가 생긴다.
    if lb is not None:
        big_mask = pd.to_numeric(lb["label_size_px"], errors="coerce").fillna(10.5) > 10.5
        for size, t in lb.groupby("label_size_px"):
            gap = LABEL_GAP_TRANSFER if float(size) > 10.5 else LABEL_GAP_NORMAL
            fig.add_trace(go.Scatter(
                x=t["label_x_px"], y=t["label_y_px"] + gap, mode="text",
                text=["<span style='font-size:%dpx'><br></span><b>%s</b>"
                      % (LABEL_GAP_FIXED_PX, k) for k in t["station_key"]],
                textposition="bottom center",
                textfont=dict(size=round(float(size) * LABEL_FONT_SCALE, 1),
                              color="#111827"),
                hoverinfo="skip", showlegend=False))

    # 5. 추천 경로 highlight
    if highlight_path:
        xs, ys = path_to_visual_nodes(highlight_path, vn)
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line=dict(color=PRIMARY, width=14),
                                     opacity=0.38, showlegend=False))

    # 6. 출발/도착 halo
    if ca is not None:
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

    fig.update_layout(
        height=820, margin=dict(l=2, r=2, t=2, b=2),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        # minallowed / maxallowed 로 팬·줌 범위를 초기 화면 밖으로 나가지 못하게 묶는다.
        # (plotly.js 2.24+ 지원. 구버전에서는 무시되며 동작에는 지장 없다)
        # minallowed/maxallowed 로 초기 화면 밖으로 나가지 못하게 묶는다.
        xaxis=dict(visible=False, range=[X0, X1], fixedrange=False,
                   minallowed=X0, maxallowed=X1),
        yaxis=dict(visible=False, range=[Y1, Y0],   # 이미지 좌표계: y 아래로 증가
                   minallowed=Y0, maxallowed=Y1,
                   scaleanchor="x", scaleratio=1),
        clickmode="event+select", dragmode="pan")
    return fig


from html import escape as _esc


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
              "selected_station_key", "route_result", "route_fallback"):
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
        c = c if c.startswith("모든") else "%s호차" % c
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
@media (prefers-color-scheme: dark) {
  .mc-seg { background:#1B1F24; border-color:#374151; }
  .mc-od { color:#F3F4F6; } .mc-meta, .mc-dir, .mc-tf-b { color:#C7CBD1; }
  .mc-tf { background:#15181C; } .mc-tf-h { color:#9EC5F0; }
}
</style>
"""


def render_timeline(segs):
    """노선형 타임라인 카드. 사용자에게 내부 id 를 노출하지 않는다."""
    html = [TIMELINE_CSS, '<div class="mc-tl">']
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
            html.append(
                '<div class="mc-seg" style="--mc-line:%s">'
                '<span class="mc-badge">%s호선</span>'
                '<span class="mc-dir">%s</span>'
                '<div class="mc-od">%s → %s</div>'
                '<div class="mc-meta">%s</div>'
                '</div>'
                % (color, _esc(str(sg["line"])), _esc(dirlab),
                   _esc(sg["from"]), _esc(sg["to"]), meta))
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
            parts = ["🚶 %s호선 → %s호선"
                     % (_esc(str(sg["from_line"])), _esc(str(sg["to_line"])))]
            if tmin is not None:
                parts.append("도보 약 %d분" % max(1, round(tmin)))
            if wmin:
                parts.append("대기 약 %.0f분" % max(1, round(wmin)))
            body = " · ".join(parts)
            extra = ""
            if tip and (tip["alight"] or tip["board"]):
                rows = []
                if tip["alight"]:
                    rows.append("하차 위치: %s" % _esc(tip["alight"]))
                if tip["board"]:
                    rows.append("승차 위치: %s" % _esc(tip["board"]))
                # 원본에 자기 역명이 방면으로 적힌 이상값이 있다(강동 5→5).
                # 해당 방면 문구만 생략하고 호차/문 안내는 그대로 유지한다.
                from station_routing import transfer_basis_text
                note = (transfer_basis_text(sg["at"], tip["arrive_toward"],
                                            tip["depart_toward"])
                        if tip.get("direction_matched") else "")
                extra = ('<div class="mc-tf-b">%s</div>' % " · ".join(rows))
                if note:
                    extra += '<div class="mc-note">%s</div>' % _esc(note)
            html.append(
                '<div class="mc-tf"><div class="mc-tf-h">↓ %s역 환승</div>'
                '<div class="mc-tf-b">%s</div>%s</div>'
                % (_esc(sg["at"]), body, extra))
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


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
        o, d_, " + ".join("%s호선" % x for x in lines),
        ev.get("transfer_count", 0), round(ev.get("actual_time_min") or 0), tail,
        ev.get("max_congestion") or 0)


CONG_REF_LINES = ((80.0, "체감 가중 시작 80%", "#9CA3AF"),
                  (100.0, "정원 100%", "#E8A33D"),
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


def build_route_congestion_profile(scorer, path, time_bin=None):
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
        cur_segs = build_route_congestion_profile(scorer, ev["path"])
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
            alt_segs = build_route_congestion_profile(
                scorer, ev["path"], time_bin=time_alt["best_time_bin"])
        except Exception:
            alt_segs = []
        same = (len(alt_segs) == len(cur_segs)
                and all(a["stations"] == b["stations"]
                        for a, b in zip(alt_segs, cur_segs)))
        if same:
            alt_label = "추천 출발 %s" % str(time_alt["best_time_bin"])[:5]
            add_series(alt_segs, alt_label, PROFILE_COLORS["recommended"],
                       "dot", 2, 5)

    cur_label = "현재 출발 %s" % str(scorer.time_bin)[:5]
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
                      annotation_text="환승 %s→%s호선" % (sg.get("from_line", ""),
                                                     sg.get("to_line", "")),
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
        c[4].metric("혼잡 주의 구간", "%d개" % ev.get("p95_exposure_count", 0))

        render_timeline(ev["segments"])
        if ev.get("event_risk_min"):
            st.caption("이벤트 영향으로 쾌적 체감시간 +%.1f분 가산" % ev["event_risk_min"])
        if show_profile:
            st.markdown("**📊 구간별 기대 혼잡도**")
            render_route_congestion_profile(ev, time_alt, key=profile_key)
        if note:
            (st.success if tone == "good" else st.caption)(note)


def page_route():
    st.title("쾌적 경로 찾기")
    st.info(DISCLAIMER)

    disp = load_display_master()
    bundle = load_map_bundle()
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
        st.session_state.setdefault("sel_origin", st.session_state["origin_station_key"])
        st.session_state.setdefault("sel_dest", st.session_state["destination_station_key"])

        o = st.selectbox("출발역", keys,
                         format_func=lambda k: labels.get(k, k),
                         placeholder="역 이름을 입력하세요", key="sel_origin")
        if o:
            st.session_state["origin_station_key"] = o

        d_ = st.selectbox("도착역", keys,
                          format_func=lambda k: labels.get(k, k),
                          placeholder="역 이름을 입력하세요", key="sel_dest")
        if d_:
            st.session_state["destination_station_key"] = d_

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
        q_date = c1.date_input("날짜", st.session_state["q_date"])
        q_time = c2.time_input("출발 시각", st.session_state["q_time"])
        mode = st.selectbox("선호 모드", list(MODE_LABEL),
                            index=list(MODE_LABEL).index(st.session_state["q_mode"]),
                            format_func=lambda x: MODE_LABEL[x])
        st.session_state.update(q_date=q_date, q_time=q_time, q_mode=mode)

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
            except Exception:
                result = None
            if result is None:
                result = fallback_route(o, d_)
                fallback = bool(result)
        st.session_state["route_result"] = result
        st.session_state["route_fallback"] = fallback

    # ---------- 오른쪽 노선도 ----------
    with right:
        if bundle is None:
            st.info("노선도 좌표가 없습니다. `14_import_map_workbook.py` 를 먼저 실행하세요.")
        else:
            path = None
            res = st.session_state.get("route_result")
            if res and res.get("ok"):
                path = res["recommended"]["path"]
            ev = st.plotly_chart(draw_map(bundle, path), width="stretch",
                                 on_select="rerun", key="subway_map",
                                 selection_mode="points")
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
    _render_route_result(result, bool(st.session_state.get("route_fallback")), o, d_)


def _render_route_result(result, fallback, o, d_):

    if fallback:
        st.info("현재 입력하신 조건의 실시간 연산이 제한되어, "
                "가장 유사한 **대표 시나리오(미리 계산된 결과)** 를 제공합니다.")

    rec, alt = result["recommended"], result.get("alternative")
    st.subheader("추천 경로")
    route_card("%s 모드 추천" % MODE_LABEL[st.session_state["q_mode"]], rec, o, d_,
               show_profile=True, time_alt=result.get("time_alternative"),
               profile_key="profile_recommended")

    if alt:
        st.subheader("최단경로 대비 쾌적 대안")
        route_card("대안 경로", alt, o, d_,
                   "예상 소요시간은 %.1f분 늘어나지만, 최대 기대 혼잡도는 %.1f%%p 낮습니다."
                   % (alt["time_loss_vs_fastest"], alt["comfort_gain_vs_fastest"]), "good",
                   show_profile=True, profile_key="profile_alternative")
    else:
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
                         "혼잡 주의 구간": c.get("p95_exposure_count", 0),
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

    lines = sorted(sub["line_id"].astype(str).unique(), key=lambda x: int(x))
    if len(lines) > 1:
        line = st.radio("노선", lines, horizontal=True,
                        format_func=lambda x: "%s호선" % x,
                        key="info_line_%s" % station_key)
    else:
        line = lines[0]

    one = sub[sub["line_id"].astype(str) == line].copy()
    one["방향"] = one["direction"].map(DIRECTION_KO).fillna(one["direction"])
    st.table(round1(one[["방향", "mean_congestion", "max_congestion",
                         "peak_time_bin", "peak_duration_min"]]
                    .rename(columns={"mean_congestion": "평균(%)",
                                     "max_congestion": "최대(%)",
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
    st.title("시간대별 혼잡 조회")
    st.info(DISCLAIMER)
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
    lines = sorted(sub["line_id"].astype(str).unique(), key=lambda x: int(x))
    line = (st.radio("노선", lines, horizontal=True,
                     format_func=lambda x: "%s호선" % x, key="cong_line_%s" % key)
            if len(lines) > 1 else lines[0])

    one = sub[sub["line_id"].astype(str) == line].copy()
    one["방향"] = one["direction"].map(DIRECTION_KO).fillna(one["direction"])
    st.table(round1(one[["방향", "mean_congestion", "max_congestion",
                         "p95_congestion", "peak_time_bin", "peak_duration_min",
                         "am_peak_mean", "pm_peak_mean"]]
                    .rename(columns={"mean_congestion": "평균(%)",
                                     "max_congestion": "최대(%)",
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
            st.subheader("%s호선 시간대별 기대 혼잡도" % line)
            piv = lk.pivot_table(index="time_bin", columns="direction",
                                 values="congestion_median").sort_index()
            piv.columns = [DIRECTION_KO.get(c, c) for c in piv.columns]
            congestion_line_chart(piv, 320)

    st.divider()
    st.subheader("평일 최혼잡 구간 Top 10")
    top = prof[prof["day_type"] == "weekday"].nlargest(10, "max_congestion").copy()
    top["호선"] = top["line_id"].astype(str) + "호선"
    top["방향"] = top["direction"].map(DIRECTION_KO).fillna(top["direction"])
    st.table(round1(top[["호선", "station_name", "방향", "peak_time_bin",
                         "mean_congestion", "max_congestion", "peak_duration_min"]]
                    .rename(columns={"station_name": "역명", "peak_time_bin": "피크 시간대",
                                     "mean_congestion": "평균(%)",
                                     "max_congestion": "최대(%)",
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
                    (100, WARN, "정원 100%"), (130, DANGER, "혼잡 주의 130%")):
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
    c[0].metric("최대 기대 혼잡도", "%.1f%%" % top["congestion_median"],
                top["표시역명"], delta_color="off")
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
    st.title("이벤트 혼잡 경보")
    st.info(DISCLAIMER)
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
# 페이지 4. 환승 안내
# --------------------------------------------------------------------------
def page_transfer():
    st.title("빠른 환승 안내")
    st.warning("아직 객차별 혼잡도 데이터가 없습니다. "
               "따라서 **환승 동선상 유리한 칸**만 안내합니다.")
    tip = load_mart("transfer_tip_mart")
    if tip is None:
        st.info("`06_build_route_graph.py` 실행이 필요합니다.")
        return
    c1, c2, c3 = st.columns(3)
    disp = load_display_master()
    tkeys = sorted(tip["station_name"].unique())
    tlabels = {}
    if disp is not None:
        lm = dict(zip(disp["station_key"], disp["available_lines"]))
        tlabels = {k: short_label(k, lm.get(k, "")) for k in tkeys}
    stn = c1.selectbox("환승역", tkeys,
                       format_func=lambda k: tlabels.get(k, k))
    sub = tip[tip["station_name"] == stn]
    fl = c2.selectbox("타고 온 호선", sorted(sub["from_line"].astype(str).unique()))
    sub2 = sub[sub["from_line"].astype(str) == fl]
    tl = c3.selectbox("갈아탈 호선", sorted(sub2["to_line"].astype(str).unique()))
    sub3 = sub2[sub2["to_line"].astype(str) == tl]
    if sub3.empty:
        st.info("원본 데이터에 해당 환승 조합이 없어 안내하지 않습니다.")
        return
    def _pos(car, door):
        """호차-문 을 '2-1' 형태로 합친다."""
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

    view = pd.DataFrame({
        "타고 온 방면": sub3["arrive_toward"].values,
        "하차 위치": [_pos(a, b) for a, b in zip(sub3["alight_car"], sub3["alight_door"])],
        "갈아탈 방면": sub3["depart_toward"].values,
        "승차 위치": [_pos(a, b) for a, b in zip(sub3["board_car"], sub3["board_door"])],
        "환승 소요(분)": np.round(pd.to_numeric(sub3["transfer_time_min"],
                                              errors="coerce"), 1).values,
    })
    st.table(round1(view).reset_index(drop=True))
    st.caption("승하차 위치는 호차-문 입니다. (2-1 = 2번째 칸 1번 문)")
    te = load_mart("transfer_edge_mart")
    if te is not None:
        row = te[(te["station_name"] == stn) & (te["from_line"].astype(str) == fl)
                 & (te["to_line"].astype(str) == tl)]
        if not row.empty:
            r = row.iloc[0]
            c = st.columns(3)
            c[0].metric("환승 소요시간", "%.1f분" % r["transfer_time_min"])
            if pd.notna(r.get("weekday_volume")):
                c[1].metric("평일 환승인원", f'{int(r["weekday_volume"]):,}명')
            c[2].metric("환승 페널티", "%.1f분" % r["transfer_penalty_min"])
            st.caption("환승 페널티 = 도보 소요시간 + 환승인원 기반 가산(5만 명당 1분, 최대 3분)")


# --------------------------------------------------------------------------
# 페이지 5. 프로젝트 소개 및 검증 리포트
# --------------------------------------------------------------------------
def page_project():
    st.title("프로젝트 소개 및 검증 리포트")
    tabs = st.tabs(["프로젝트 개요", "데이터 품질", "혼잡도 모델",
                    "경로 추천 평가", "한계와 개선 방향"])

    with tabs[0]:
        st.markdown("#### 서울 지하철 1~8호선 혼잡도 기반 쾌적 경로 추천")
        c = st.columns(4)
        for col, (a, b, d) in zip(c, [("승하차 원본", "797,946행", "48개월"),
                                      ("혼잡도 셀", "715,065", "11 스냅샷"),
                                      ("그래프", "280 노드", "536+81 엣지"),
                                      ("검증 이벤트", "87건", "DiD 순효과")]):
            col.metric(a, b, d, delta_color="off")
        st.markdown("""
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

##### 핵심 발견
| # | 발견 |
|---|---|
| 1 | 쾌적 대안은 일부 OD 에서만 나타난다 (08:30 3/13, 18:00 2/13). 저녁의 병목은 혼잡 감소폭 단일 조건 |
| 2 | 이벤트는 대안의 개수가 아니라 구성을 바꾼다 (불꽃축제일 18:00 은 선호 모드별로 결과가 갈린다) |
| 3 | LightGBM 이 groupby 평균 baseline 을 이기지 못했고, 사전 등록한 조건대로 baseline 채택 |
| 4 | 전후 비교는 이벤트 효과를 과대추정한다 (불꽃축제 3.674 → DiD 3.403) |
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

    with tabs[2]:
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

    with tabs[3]:
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
            ["객차별 혼잡도 예측", "객차 단위 재차 데이터 없음", "환승 동선상 유리한 호차/문"],
            ["착석 확률", "좌석 점유 데이터 없음", "착석 가능성 proxy 점수"],
            ["이벤트 당일 실제 혼잡도", "혼잡도가 분기 평균 패턴", "이벤트성 spike 기반 위험 보정"],
            ["9호선·신분당선 우회", "서울교통공사 관할 밖", "1~8호선 내 대안 경로"],
        ], columns=["못 하는 것", "사유", "대신 제공하는 것"]), width="stretch", hide_index=True)

        st.subheader("평가 체계")
        st.dataframe(pd.DataFrame([
            ["데이터 품질", "스키마 검증", "pandera + 교차검증"],
            ["이벤트 효과", "인과 효과 추정", "DiD + 평행추세 검증, absolute_lift"],
            ["혼잡도 예측", "회귀 + 불균형 분류", "Baseline ladder, Skill Score, PR-AUC, Ablation, SHAP"],
            ["경로 추천", "다목적 최적화", "3축 Pareto, Stretch Factor, directed-edge Jaccard"],
            ["착석 가능성", "검증 불가", "정답 데이터 없음 — 한계로 명시"],
        ], columns=["레이어", "문제 유형", "표준 지표"]), width="stretch", hide_index=True)

        st.subheader("알려진 약점")
        st.markdown("""
1. **고혼잡 구간에서 오차가 6배** — 고혼잡 경고는 절대값이 아니라 상대 순위로 제공합니다.
2. **`label_high` 기준 불일치** — 미관측 역은 전역 p95 로 대체되어 분류 지표에 편향이 있습니다.
3. **Pareto front 는 K개 후보 안에서의 front** — Yen's K-shortest 집합에 한정됩니다.
4. **출발 시각의 time_bin 을 경로 전체에 고정** — 시간 전진을 반영하지 않았습니다.
5. **미해결**: 혼잡도 산식, 스냅샷 날짜의 의미, 8호선 오차 원인.
""")
        for name, rel in [("평가 전략", "docs/evaluation_strategy.md"),
                          ("모델 카드", "docs/model_card_congestion.md")]:
            p = ROOT / rel
            if p.exists():
                with st.expander("%s 전문" % name):
                    st.markdown(p.read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------
def main():
    st.sidebar.title("🚇 여유로 서울")
    st.sidebar.caption("서울 지하철 혼잡도 기반  \n쾌적 경로 추천 시스템")
    if not st.session_state.get("menu"):
        st.session_state["menu"] = MENUS[0]
    for m in MENUS:
        st.sidebar.button(
            m, width="stretch", key="menu_%s" % m,
            type="primary" if st.session_state["menu"] == m else "secondary",
            on_click=lambda x=m: st.session_state.update(menu=x))
    menu = st.session_state["menu"]
    st.sidebar.divider()
    st.sidebar.caption("실시간 정보가 아닌  \n과거 패턴 기반 예측입니다.")

    # 이름 기반 dispatch. 메뉴 순서를 바꿔도 연결이 어긋나지 않는다.
    {"쾌적 경로 찾기": page_route,
     "시간대별 혼잡 조회": page_congestion,
     "빠른 환승 안내": page_transfer,
     "이벤트 혼잡 경보": page_event,
     "프로젝트 소개 및 검증 리포트": page_project}.get(menu, page_route)()


main()
