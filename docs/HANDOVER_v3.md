# 여유로 서울 v2 인수인계 프롬프트

> 새 대화 첫 메시지에 아래 전체를 붙여넣으면 된다.
> `---` 아래부터가 붙여넣을 내용이다.
>
> 이 문서는 v1 배포 완료 시점(2026-09) 기준이다.
> 이전 판(MetroCalm 시절 v1 작업 기록)은 `docs/HANDOVER_v2.md` 로 남아 있다.

---

# 여유로 서울 v2 작업 시작

서울 지하철 1~8호선 혼잡도 기반 쾌적 경로 추천 시스템 **여유로 서울 (Yeoyuro Seoul)** 의 v2 를 진행합니다.

- 저장소: https://github.com/FlightChamp/yeoyuro-seoul
- 배포: https://yeoyuro-seoul.streamlit.app/
- v1 은 배포까지 완료된 상태입니다.

## 0. 나에 대한 정보

- 개인 포트폴리오 프로젝트이며, 1금융권 데이터 직군 지원용입니다.
- 목표는 "화려한 모델"이 아니라 **데이터 엔지니어링 → 분석 → 모델링 → 서비스 → 배포**
  전 과정을 정직하게 끝까지 돌린 것을 보여주는 것입니다.
- 저는 구현이 실제로 동작했는지를 출력값으로 확인하고 넘어갑니다.
  "적용했다"가 아니라 **"실행해서 이런 값이 나왔다"** 로 보고해 주세요.
- 작업 환경은 Windows / PowerShell / **conda `metrocalm` 환경 / Python 3.12** 입니다.
  (환경 이름은 리브랜딩 전에 만든 것이라 그대로 `metrocalm` 입니다.)
  프로젝트 경로는 `C:\Programming\MyProject\yeoyuro_seoul_project` 입니다.

## 1. v1 에서 확정된 것 (바꾸지 말 것)

### 표현 규칙 — 이건 절대 어기면 안 됩니다

| 금지 | 대체 |
|---|---|
| 실시간 혼잡도 예측 | 과거 패턴 기반 **기대 혼잡도** |
| 객차별 혼잡도 예측 | 환승 동선상 유리한 호차/문 |
| 착석 확률 | 착석 가능성 **proxy 점수** |
| "이 칸이 덜 붐빈다" | (사용 금지) |
| 정확한 다음 열차 | 계획 시각표 기준 다음 열차 |

혼잡도 원본은 **분기별 요일유형 평균 패턴**이며 특정 날짜 실측값이 아닙니다.
격자 하나당 관측치가 최대 11개뿐입니다.

### 프로젝트 범위

```
1호선 서울역~청량리    2호선 전구간(성수·신정지선 포함)
3호선 지축~오금        4호선 불암산~남태령
5호선 방화~하남검단산 / 방화~마천
6호선 응암~신내        7호선 장암~온수        8호선 암사역사공원~모란
```
9호선·신분당선 등은 v1 데이터·그래프 구축 범위 밖입니다.

### 핵심 아키텍처

- 사용자 입력은 **역 단위**, 내부 그래프·혼잡도는 **호선별 station_uid** 단위.
  환승역에서 첫 승차 노선은 알고리즘이 후보로 비교하며, **최초 승차는 환승으로 세지 않습니다**
  (가상 노드 `V_ORIGIN_*` / `V_DEST_*`, origin_access·destination_exit 엣지).
  이 가상 노드는 `evaluate()` 에 들어가기 전에 제거됩니다.
- 노선도 좌표의 source of truth 는
  `data/master/yeoyuro_seoul_vector_map_coordinate_workbook.xlsx` (5120×2880).
  **자동 graph layout(spring/kamada_kawai) 사용 금지.**
  좌표를 고칠 때는 이 워크북을 직접 열어 수정하고
  `python scripts\14_import_map_workbook.py --root . --preview` 로 재생성합니다.
  Downloads 를 경유하지 않습니다.
- 방면 표기는 **경로상 다음 역** 기준입니다(예: "아차산 방면").
  종점 기준은 지선·순환에서 예외가 많아 폐기했습니다(`bound_label` 에 참고용으로만 보관).

### 그래프 모델링 — 과거에 깨진 지점들

| 문제 | 대응 |
|---|---|
| 분기역 단일 노드 → 지선 승객이 통과 | 성수·신도림·강동 본선/지선 노드 분리 |
| 응암순환 `구산 → 응암 → 역촌` 환승 0회 | `6_응암@eungam_loop` 분리 + 재승차 환승 엣지 |
| `구산 → 응암 → 새절 → 응암 → 역촌` U턴이 재승차 우회 | **역 재방문 금지** 제약 |
| 환승 호차/문이 진행 방향 무시 | 경로 앞뒤 역으로 방면 매칭 |

**U턴은 배차가 긴 낮 시간대에만 나타나 08:30 검증에서 놓쳤습니다.**
그래서 회귀 테스트는 5개 시간대(08:30/11:30/13:30/19:00/22:00)로 돌립니다.

### 사전 등록한 모델 판정 (뒤집지 말 것)

> LightGBM 이 `역×방향×요일×시간대 평균` baseline 을 valid 에서 이기지 못하면
> 모델을 폐기하고 baseline 을 서비스에 사용한다. → **판정: baseline 채택**

Skill Score: Regime A −0.464 / Regime B −0.012.
모델(Model-2, 역 식별자 제외)은 **미관측 역 보간용**으로만 유지합니다.

### 중간역 정차시간 — v1 후반에 추가된 확정 사항

원본 `역간거리 및 소요시간` 의 소요시간은 **순수 주행시간**이며 정차시간이 없습니다.
`열차운행현황` 의 공표 소요시간·표정속도와 교차검증해 확인했습니다.

| 호선 | 역간 합계 | 공표 소요 | 중간역당 |
|---|---|---|---|
| 1 | 14.0분 | 18.0분 | **30.0초** |
| 3 | 51.5분 | 67.5분 | **30.0초** |
| 6 | 55.5분 | 75.0분 | 30.8초 |
| 7 | 69.0분 | 87.0분 | 27.0초 |

→ `DEFAULT_DWELL_TIME_MIN = 0.5` (scripts/09).
**이 값은 κ=0.5, C0=80 같은 "초기값"과 성격이 다릅니다.** 공표 데이터에서 유도된 값입니다.

계산 규칙: 연속한 ride 엣지 묶음마다 중간 정차역 수 = `max(k-1, 0)`.
출발역·도착역·환승 하차역·환승 승차역은 정의상 제외됩니다.
`actual` 과 `perceived` 에 **각각 1회, 혼잡도 보정 없이** 더합니다.

**엣지 가중치에 미리 섞지 마세요.** 같은 엣지도 경로에 따라 중간역일 수도 종점일 수도 있습니다.

## 2. v1 확인값 (재현 시 이 값이 나와야 정상)

| 항목 | 값 |
|---|---|
| 승하차 원본 / mart | 797,946행 / 15,957,680행 |
| 혼잡도 long / 범위 내 | 715,299 / 715,065 (11 스냅샷, 스키마 3종) |
| 호선 오기입 교정 | 156행 (5호선 둔촌동·올림픽공원이 2호선으로 기재) |
| 그래프 | 281 노드 / 승차 536 / 환승 82 / 강연결 True |
| smoke test | 9/9 PASS |
| 학습 마트 | 706,678행, label_high 5.25% |
| baseline L2 | MAE 2.364 / R² 0.947 / PR-AUC 0.857 |
| Ablation B→C | R² 0.450 → 0.734 (승하차 흐름 피처 기여) |
| 고혼잡 130%+ 구간 MAE | 9.227 (전체 평균의 약 4배) |
| DiD | 처치역 28 / 대조역 248 / 평행추세 위반 28행 |
| 불꽃축제 | 전후비교 3.674 → DiD 순효과 3.403 |
| 경로 평가 | 08:30 대안 3/13, 18:00 대안 2/13, 불꽃축제일 calm 2/13 · 그 외 3/13 |
| 스키마 검증 | 테이블 PASS 7 / 교차 PASS 10 |
| **pytest** | **116 passed** |

검산용 예시: 2호선 신촌→잠실(20역 19엣지) = 주행 25.5분 + 정차 18×0.5 = **34.5분**.

## 3. 테스트 구성 (116건)

| 파일 | 건수 | 검증 대상 |
|---|---|---|
| `test_station_routing.py` | 41 | 경로 탐색 regression (지선·순환·U턴·재방문 금지) |
| `test_transfer_direction.py` | 27 | 환승 방면 표기, 자기역명 이상값, substring 오탐 방지 |
| `test_dwell_time.py` | 18 | 중간역 정차시간 개수·합산·이중보정 방지 |
| `test_timeline_consistency.py` | 13 | 화면 표시값과 카드 총계의 일치 |

**`test_timeline_consistency.py` 가 있는 이유를 기억해 주세요.**
v1 후반에 dwell 을 `evaluate()` 에만 반영하고 `describe_path()` 의 segment `minutes` 를
빠뜨려, 총계 38분 / segment 29분 으로 어긋난 적이 있습니다.
그걸 고치면서 `sys.modules` 로 scorer 모듈을 찾았는데, `load_scorer_class()` 는
`module_from_spec` + `exec_module` 로 모듈을 만들고 **`sys.modules` 에 등록하지 않아**
조회가 `None` 을 반환하고 상수가 조용히 0.0 으로 떨어졌습니다.
현재는 `station_routing._dwell_min()` 이 클래스의 `__globals__` 를 직접 읽습니다.

**`evaluate()` 반환값만 검사하는 테스트로는 두 결함 모두 잡히지 않았습니다.**
표시값을 검증하는 테스트를 유지해 주세요.

### 기각한 기법과 이유 (다시 제안하지 말 것)

| 후보 | 기각 사유 |
|---|---|
| LSTM / Transformer / GNN | 혼잡도 스냅샷이 11개. 격자당 관측치 11개에 딥러닝은 과적합 |
| NDCG / MAP | 정답 선호도 라벨(사용자 피드백)이 없음 |
| MASE | 시계열 naive forecast 를 분모로 쓰는 지표라 격자 회귀에 정의 불일치 → Skill Score |
| MLflow | 학습이 사실상 1회 |
| Great Expectations | pandera 로 충분 |
| A/B 테스트 | 사용자 없음 |
| Hugging Face Spaces | Streamlit SDK 생성 불가, 컴퓨트 Space 는 PRO 유료 |

## 4. 파일 구조

```
yeoyuro_seoul_project/
├── README.md, LICENSE, requirements.txt, requirements-dev.txt
├── .github/workflows/tests.yml     # CI: Python 3.12, pytest
├── .streamlit/config.toml          # primaryColor #8D7150 (BOM 없이 저장할 것)
├── bootstrap_yeoyuro_seoul.py
├── station_routing.py              # app/streamlit 와 동일본 유지
├── scripts/
│   ├── 01_build_ridership_mart.py      03_build_stg_congestion.py
│   ├── 04_build_congestion_mart.py     05_build_event_spike_mart.py
│   ├── 05b_did_event_effect.py         06_build_route_graph.py
│   ├── 06b_smoke_test_graph.py         07_build_training_mart.py
│   ├── 08_train_congestion_model.py    08b_evaluate_congestion_model.py
│   ├── 09_route_scoring_prototype.py   10_evaluate_routes.py
│   ├── 11_validate_schemas.py          12_build_display_masters.py
│   ├── 14_import_map_workbook.py       15_build_headway_mart.py
├── app/streamlit/yeoyuro_seoul_app.py, station_routing.py
├── tests/ (4개 파일, 99 tests)
├── data/master/**                  # 커밋됨 (프로젝트 정의)
├── data/marts/                     # 앱 런타임 mart 15개만 커밋 (약 0.85MB)
├── data/{raw,staging,interim}/     # .gitignore
├── models/                         # .gitignore (baseline 채택으로 런타임 불필요)
├── docs/{DATA,evaluation_strategy,model_card_congestion,PROJECT_PLAN_v2,
│         deployment_plan,known_data_issues,project_naming,
│         dwell_time_adjustment_report,HANDOVER_v2}.md
├── docs/images/*.png               # README 스크린샷 7장
└── reports/{data_quality,model,route,figures}/
```

`13_build_map_layout.py` 는 좌표 워크북으로 대체되어 삭제했습니다. 되살리지 마세요.

## 5. 배포 구성

| 항목 | 값 |
|---|---|
| 플랫폼 | Streamlit Community Cloud |
| Main file path | `app/streamlit/yeoyuro_seoul_app.py` (인자 없이 실행) |
| Python | 3.12 |
| Secrets | 없음 (외부 API 미사용) |

`requirements.txt` 는 **앱 실행 전용 6개**(streamlit, pandas, numpy, plotly, pyarrow, tzdata),
`requirements-dev.txt` 는 재현용입니다. Streamlit Cloud 가 루트 `requirements.txt` 만
인식하기 때문에 역할을 이렇게 나눴습니다.

앱이 읽는 것은 커밋된 `data/master`, `data/marts`(경량 15개), `reports` 뿐입니다.
**CI 도 이 경량 데이터만으로 116 passed 가 나옵니다.** `data/raw` 나 `models` 에
의존하는 테스트를 새로 추가하지 마세요. CI 가 깨집니다.

배포 컨테이너 TZ 는 UTC 입니다. `_now_defaults()` 가 `ZoneInfo("Asia/Seoul")` 로 고정하고
있으니 시각 관련 코드를 건드릴 때 주의해 주세요.

## 6. v1 의 알려진 한계 (v2 후보)

1. **출발 시각의 time_bin 을 경로 전체에 고정** — 시간 전진 미반영.
   정차시간을 반영해 총 소요시간이 늘어난 만큼 이 한계의 영향도 커졌습니다.
2. **환승 대기시간이 30분 bin 평균** — 특정 시각의 실제 다음 열차가 아님.
   균등 도착 가정(기댓값 = 배차/2), 최초 승차 전 대기는 미반영.
3. **고혼잡 구간 오차** — ~30% 구간 MAE 1.474 vs 130%+ 구간 9.227.
   현재는 절대값 대신 상대 순위(p95 초과)로 우회.
4. **`label_high` 기준 불일치** — station holdout 역은 train 에 없어 전역 p95 로 대체.
   Regime B 분류 지표에 편향이 있음.
5. **Pareto front 는 K개 후보 안에서의 front** — Yen's K-shortest 집합에 한정.
6. **정차시간이 단일 상수** — 노선·역·시간대별 실제 정차시간은 다름.
7. **미해결** — 혼잡도 산식(정원 대비 %인지), 스냅샷 날짜의 의미(관측 종료일/발행일),
   8호선 오차가 큰 원인(MAE 3.600, p95 오차 13.9).
8. **노선도 도심부 라벨 밀집** — 서울역·시청·충정로 주변.
9. **Plotly 최대 확대 배율 제한 불가** — JS 커스텀 컴포넌트 필요.
10. **원본 데이터 이상값** — `docs/known_data_issues.md` 참고.
    강동 5→5 방면 표기는 UI sanitizer 로 방어 중이며,
    경로 기준 방면 유도 표시는 v1.1 과제로 남겨뒀습니다.

## 7. v2 후보 (우선순위는 함께 정하고 싶습니다)

1. **경로 탐색 시 시간 전진 반영** — 구간마다 도착 예상 시각의 time_bin 사용.
   한계 1번을 직접 닫습니다. 새 데이터가 필요 없고 회귀 테스트로 검증됩니다.
   먼저 정할 것: 탐색까지 시간 전진할지(Yen's K-shortest 후보 집합이 달라짐),
   아니면 탐색은 출발 bin 정적 가중치로 두고 재점수화만 시간 전진할지.
2. **정밀 시각표 기반 next departure** — 원본
   `서울교통공사_서울 도시철도 열차운행시각표_20260616.csv` 가 `data/raw/train_operation/` 에 있음.
   1번이 "환승역 도착 시각"을 만들어야 이 기능의 조회 시점이 생깁니다.
   표현은 "계획 시각표 기준 다음 열차"로 씁니다(실적이 아님).
3. **고혼잡 전용 분류 모델** — 회귀 대신 `label_high` 직접 학습.
   한계 4번(라벨 기준 불일치)을 먼저 정리해야 개선치를 해석할 수 있습니다.
4. **역 유사도 기반 보간** — 미관측 역에 노선 평균 대신 유사 역 가중 평균.
5. **9호선 확장** — 6개년 30분 단위 혼잡도 자료는 확보했으나, 9호선은 서울교통공사
   관할이 아니라 **승하차·역간거리·소요시간·배차·환승 호차/문 데이터가 없습니다.**
   혼잡도만으로는 경로 그래프에 올릴 수 없습니다. 급행/완행 구분도 별도 과제입니다.
   확장 전에 `docs/known_data_issues.md` 3번(개화역 상선 0값) 재검증이 필요합니다.
6. FastAPI / Dockerfile.
7. SVG 커스텀 노선도 컴포넌트.

## 8. 작업 방식 요청

- 스크립트를 수정하면 **실제로 실행해서 출력값을 보여주세요.** 확인값과 대조하겠습니다.
- 파일을 여러 곳 고칠 때는 **한 번에 묶어 패치하지 말고** 단계별로 저장·검증해 주세요.
- PowerShell 패치 스크립트를 주실 때는 다음을 지켜주세요.
  - 파일의 줄바꿈(CRLF/LF)을 **감지**해서 패턴에 맞출 것. 파일마다 다릅니다.
  - 치환 전에 **매칭 건수를 출력**하고, 1건이 아니면 중단할 것.
  - 백업 → 문법 검사 → 실패 시 자동 복원.
  - here-string(`@'`)은 **줄 맨 앞에서 시작**해야 합니다. 배열 안에 들여쓰면 파싱 오류가 납니다.
  - 블록 전체를 한 번에 붙여넣는 것을 전제로 작성해 주세요(한 줄씩 실행하면 가드가 작동하지 않습니다).
- 파일을 주실 때는 Downloads 에 남지 않도록 `Move-Item` 기준 명령으로 안내해 주세요.
- `data/master` 등 프로젝트가 생성한 실데이터를 덮어쓰지 마세요.
- 그래프·라우팅을 건드리면 반드시 `pytest tests/ -q` 로 **116개 통과**를 확인해 주세요.
- 새 기능을 추가하면 회귀 테스트도 함께 추가해 주세요.
  **테스트가 실제로 버그를 잡는지** 일시적으로 결함을 주입해 확인하는 것까지 포함합니다.
- 문서에 숫자를 적을 때는 다른 문서와 충돌하지 않는지 확인해 주세요.
  (v1 에서 41/99/15 passed 가 문서마다 달라 정리한 적이 있습니다.)

먼저 위 내용을 이해했는지 확인하고, v2 후보 중 무엇부터 할지 의견을 주세요.
