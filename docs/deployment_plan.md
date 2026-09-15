# 여유로 서울 (Yeoyuro Seoul) v1 배포 계획

> 대상: 여유로 서울 v1 포트폴리오 데모 배포
> 원칙: 기능·알고리즘·모델을 바꾸지 않고, 로컬과 **동일한** 동작을 외부에 공개한다.
> 상태: 로컬 검증 완료 / 배포 대기. Live Demo URL 은 배포 후 README 와 함께 갱신한다.

---

## 1. 진단 결과

GitHub 레포(`FlightChamp/yeoyuro-seoul`)를 클론해 확인한 값이다.

| 항목 | 값 |
|---|---|
| 클론 크기 | 5.5MB (`.git` 2.2MB 포함) |
| 추적 파일 | 113개 |
| 민감 파일 추적 | 없음 (`.env`/`secret`/`credential`/`key`/`pem`/`token` 검색 0건) |
| 절대경로 하드코딩 | 없음 (`bootstrap_yeoyuro_seoul.py` docstring 예시 1건, 실행 무관) |
| 최대 추적 파일 | `docs/images/08_model_report.png` |

### 경로 처리

`resolve_root()` 는 `--root` 인자가 없으면 `data/marts` 디렉터리가 존재하는 최상위를 찾는다.
`data/marts/.gitkeep` 이 커밋돼 있어 **클론 직후에도 레포 루트를 정확히 해석한다.**

→ 배포 실행 명령은 인자 없는 형태를 쓴다.

```
streamlit run app/streamlit/yeoyuro_seoul_app.py
```

`-- --root .` 는 Streamlit Community Cloud 의 Main file path 설정에서 인자를 넘길 수 없으므로 사용하지 않는다.

### 앱이 실제로 읽는 파일

`yeoyuro_seoul_app.py` → `station_routing.py` → `scripts/09_route_scoring_prototype.py` 체인을 전수 확인했다.

| 위치 | 상태 | 파일 |
|---|---|---|
| `data/master/` | **이미 커밋됨** | 20개 (station/map/event 마스터) |
| `reports/data_quality`, `reports/model`, `reports/route` | **이미 커밋됨** | 리포트 탭이 읽는 md/csv/png |
| `data/marts/` | **미커밋 — 이번에 추가** | 아래 목록 |
| `models/` | **불필요** | 런타임 참조 0건 |

`data/marts/` 중 앱이 읽는 파일:

```
route_edge_mart                 경로 탐색 (필수)
transfer_edge_mart              경로 탐색 (필수)
congestion_edge_lookup          경로 탐색 (필수)
headway_station_30min           환승 대기시간
headway_line_30min              환승 대기시간 폴백
congestion_station_profile      시간대별 혼잡 조회
transfer_tip_mart               빠른 환승 안내
event_did_mart                  이벤트 혼잡 경보
event_spike_mart                이벤트 혼잡 경보
route_evaluation_mart_*         경로 평가 리포트 / 계산 실패 시 fallback
route_evaluation_summary_*      경로 평가 요약
```

### 제외하는 데이터와 이유

| 제외 | 이유 |
|---|---|
| `data/raw/**` | 서울교통공사 원본(약 143MB). 재배포 제약이 있고 앱이 읽지 않는다 |
| `data/staging/**`, `data/interim/**` | 중간 산출물. 스크립트로 재생성 |
| `congestion_training_mart` (706,678행) | 학습 전용. 앱 참조 0건 |
| 승하차 마트 (15,957,680행) | 파이프라인 중간 결과. 앱 참조 0건 |
| `models/**` (약 33MB) | 사전 등록 판정에 따라 baseline 을 서비스에 쓰므로 런타임 불필요 |

> baseline 채택 판정 덕분에 배포 산출물에 모델 파일이 빠졌다.
> 모델을 서비스에 넣었다면 33MB 와 lightgbm 런타임 의존이 그대로 따라왔을 것이다.

### 데이터 구조 선택

| 선택지 | 장점 | 단점 | 판정 |
|---|---|---|---|
| A. 현 구조 유지 + 필요한 마트만 Git 포함 | 앱 코드 무수정, 로컬과 100% 동일 | `.gitignore` 예외 관리 필요 | **채택** |
| B. `data/app/` 경량 전용 디렉터리 | 배포 산출물이 한눈에 보임 | 로더 경로 전면 수정 → 기능 재작성 금지 원칙 위반 | 기각 |
| C. 배포 환경 sample/demo mode | 용량 최소 | 로컬과 다른 것을 시연하게 됨 | 기각 |

---

## 2. 배포 플랫폼 결정

### Hugging Face Spaces — 현재 무료 배포 불가

Hugging Face 공식 문서(Spaces Overview) 기준:

- Space 생성 시 선택 가능한 SDK 는 **Gradio / Docker / static HTML** 세 가지다. Streamlit SDK 는 생성 옵션에서 제외됐다.
- **Static Space 만 누구나 무료**이며, 컴퓨트에서 실행되는 Gradio·Docker Space 생성은 개인 계정 PRO(월 $9), 조직 Team/Enterprise 플랜을 요구한다.

여유로 서울 은 Streamlit 앱이므로 Docker SDK 를 써야 하고 → 유료 플랜이 필요하다.
CPU Basic 의 16GB RAM 은 매력적이지만 **Space 자체를 만들 수 없으므로 무의미하다.**

### 비교

| 기준 | Streamlit Community Cloud | HF Spaces |
|---|---|---|
| 무료 배포 가능성 | 가능 | Streamlit 앱은 불가 (PRO 필요) |
| 배포 난이도 | 낮음 — GitHub 연결 후 파일 경로 지정 | Docker 작성 + 결제 |
| 데이터/메모리 여유 | 메모리 최대 2.7GB / 스토리지 50GB | 16GB (사용 불가) |
| cold start / sleep | 무트래픽 12시간 후 hibernate, 방문 시 기동 | — |
| GitHub 연동 | 네이티브 (push 시 자동 재배포) | 미러링 필요 |
| AI 포트폴리오 브랜딩 | 보통 | 좋음 |
| 디버깅 난이도 | 낮음 (웹 로그 확인) | 중간 |
| 여유로 서울 적합도 | **높음** | 낮음 |

### 최종 선택: Streamlit Community Cloud

선택 이유:

1. HF 무료 생성이 막혀 실질 대안이 하나뿐이다.
2. 앱이 읽는 마트가 전부 소형이라 메모리가 병목이 아니다. HF 의 유일한 우위가 무의미하다.
3. GitHub push → 자동 재배포라 v2 재배포 비용이 0 에 가깝다.
4. 지원 대상이 1금융권 데이터 직군이므로, ML 데모 브랜딩보다 "열리는 링크"의 가치가 크다.

sleep 은 포트폴리오 데모에서 치명적이지 않다. README 에 명시하고, 면접 전 한 번 방문해 깨워두면 된다.

---

## 3. requirements 정리

앱 런타임 import 체인 전수 조사 결과, 실제 필요 패키지는 5개다.

| 패키지 | 런타임 필요 | 비고 |
|---|---|---|
| streamlit, pandas, numpy | O | |
| plotly | O | 벡터 노선도 |
| pyarrow | O | parquet 마트 |
| lightgbm, shap, scikit-learn | **X** | 학습·평가 전용. import 0건 |
| pandera, pytest | **X** | 검증·테스트 전용 |
| matplotlib, openpyxl | **X** | 리포트 생성·원본 읽기 전용 |

lightgbm·shap 은 컴파일 wheel 이라 Python 버전이 맞지 않으면 빌드가 통째로 실패한다.
**배포 실패 확률이 가장 높은 지점이며, 제거만으로 해소된다.**

Streamlit Community Cloud 는 루트 `requirements.txt` 만 자동 인식하므로 다음과 같이 역할을 바꾼다.

- `requirements.txt` → 앱 배포용 (5개 + tzdata)
- `requirements-dev.txt` → 전체 파이프라인 재현용

Python 버전은 **3.12 또는 3.13** 으로 지정한다. 로컬은 3.14 지만, Community Cloud 는
EOL·프리릴리스·피처 버전을 허용하지 않으므로 보수적으로 고정한다.

---

## 4. 코드 수정 (2건, 기능 변경 없음)

### 4.1 타임존

`_now_defaults()` 가 `datetime.now()` 를 쓴다. Community Cloud 는 미국 호스팅이고 컨테이너 TZ 가 UTC 라,
서울 기준 접속 시각과 기본값이 9시간 어긋난다. 퇴근시간 혼잡 시연에 직접적인 문제다.

```python
from zoneinfo import ZoneInfo
KST = ZoneInfo("Asia/Seoul")

def _now_defaults():
    """초기 접속·새로고침 시 날짜/시각을 현재 시각으로 맞춘다(30분 단위 내림).
    배포 컨테이너 TZ 가 UTC 이므로 KST 로 고정한다."""
    now = datetime.now(KST)
    return now.date(), dtime(now.hour, 0 if now.minute < 30 else 30)
```

### 4.2 `.gitignore`

`data/marts/**` 제외는 유지하고, 앱이 읽는 마트만 예외로 되살린다(§1 목록).

이 두 건 외에 앱 로직·알고리즘·모델은 수정하지 않는다.

---

## 5. 배포 절차

1. `.gitignore` 수정 → `git status` 로 의도한 마트만 추가 대상인지 확인
2. `requirements.txt` / `requirements-dev.txt` 교체
3. 타임존 수정 (`app/streamlit/yeoyuro_seoul_app.py`, `station_routing.py` 동일본 유지 확인)
4. `pytest tests/ -q` → 116 passed 확인
5. 로컬 실행 검증 (§7)
6. 마트 커밋 → push, 레포 크기 확인
7. share.streamlit.io 에서 New app
   - Repository: `FlightChamp/yeoyuro-seoul`
   - Branch: `main`
   - Main file path: `app/streamlit/yeoyuro_seoul_app.py`
   - Advanced settings → Python version: 3.12
   - Secrets: **불필요** (외부 API·인증 없음)
8. 빌드 로그에서 설치 실패·FileNotFoundError 확인
9. 배포 URL 로 §7 검증 재수행
10. README 에 Live Demo / Deployment 섹션 추가

---

## 6. 데이터 크기 점검 결과

- 마트 합계: **0.85MB** (15개 파일. 최대 `congestion_edge_lookup.parquet` 279KB)
- 커밋 후 레포 총 크기: 약 6.4MB
- 50MB 초과 파일: 없음

---

## 7. 로컬 검증 결과

```
pytest tests/ -q                          → 116 passed
streamlit run app/streamlit/yeoyuro_seoul_app.py   (인자 없이)
```

검증 OD (8건): 서울역→강남 / 상일동→강남 / 광나루→한양대 / 왕십리→한양대 /
군자→여의나루 / 응암→연신내 / 방화→마천 / 신설동→까치산

화면: 쾌적 경로 찾기, 추천 경로 카드, 최단경로 대비 대안, 대안 없음 시 시간대 추천,
시간대별 혼잡 조회, 노선별 혼잡 단면, 빠른 환승 안내, 이벤트 혼잡 경보, 프로젝트 소개·검증 리포트

확인 항목: 시작 오류 없음 / FileNotFoundError 없음 / 지도 표시 / 역 선택 / 경로 검색 /
환승 대기시간 표시 / 최초 승차 전 대기 미포함 / 혼잡도 문구가 실시간으로 오해되지 않음 / 리포트 탭

---

## 8. 예상 리스크

| 리스크 | 가능성 | 대응 |
|---|---|---|
| requirements 빌드 실패 | 중 → **낮음** | lightgbm·shap 제거로 해소 |
| 기본 시각 9시간 오차 | **높음 → 해소** | KST 고정 |
| 마트 누락 FileNotFoundError | 낮음 | 앱이 `None` 가드로 우아하게 degrade 하지만, 탭이 비면 데모가 무의미하므로 커밋 목록 대조 필수 |
| 메모리 초과 | 낮음 | 마트가 소형이고 `@st.cache_data` 적용 완료 |
| cold start 지연 | 중 | README 에 명시. 시연 전 사전 방문 |
| 공공데이터 재배포 범위 | **미확인** | 파생 집계 마트 공개가 서울교통공사 이용허락범위에 부합하는지 확인 필요 |

---

## 9. README 반영 사항

Live Demo 섹션(배포 링크 + 데이터 해석 주의), Deployment 섹션(플랫폼, 배포용 데이터 구성,
실행 명령, sleep 안내, 재현용 파이프라인과 배포용 앱의 차이).

---

## 10. 향후 개선

- sample mode: 마트 없이도 화면을 보여주는 축소 모드
- Dockerfile: HF Spaces PRO 또는 타 플랫폼 이전 시
- GitHub Actions CI: push 시 `pytest tests/` 자동 실행
- HF Spaces 미러: PRO 사용 시점에 재검토
- 9호선 v2 확장 후 재배포
