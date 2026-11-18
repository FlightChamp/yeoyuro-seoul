# v2 로드맵

> 2026-11-18 기준. 공모전 대응(contest 브랜치, 1fa17ab 에서 동결) 이후 v2 를 다시 이어가는 계획이다.
> 작업 브랜치는 `v2-next`, 시험 배포는 v2-demo 앱(Streamlit Community Cloud), 공개 배포(main)는 v1 이다.
> 결정의 근거는 `decision_log.md` 의 ID 로 적는다.

## 지금까지 (v2.0 ~ Release, 2026-09 ~ 10)

| 단계 | 내용 | 문서 |
|---|---|---|
| v2.0 | 시간 진행형 평가 엔진 (구간 진입 시각의 혼잡·배차로 판정) | [time_dependent_routing.md](time_dependent_routing.md) |
| v2.1 | 혼잡 노출 시간 지표 | [exposure_metrics.md](exposure_metrics.md) |
| v2.2 | 출발 시각 이동(Time Shift) | [temporal_shift.md](temporal_shift.md) |
| v2.3 / v2.3b | OD 분류(경로·시간·둘 다·구조적) + 강건성 | [od_shiftability.md](od_shiftability.md), [robustness.md](robustness.md) |
| v2.4 | 혼잡 분산 지도 | [mobility_atlas.md](mobility_atlas.md) |
| v2.5 | 이벤트 스트레스 테스트 | [event_stress_test.md](event_stress_test.md) |
| v2.6 / v2.6c | 9호선 통합 + 정원 보정 검사 | [line9.md](line9.md) |
| Release | 혼잡 진단 카드, 앱 IA 정리 | [counterfactual.md](counterfactual.md), [app_ia.md](app_ia.md) |

## 공모전 이후 (2026-11 ~)

| Phase | 목표 | 상태 |
|---|---|---|
| 0 | 저장소 정리: 브랜치 점검, `v2-next` 분기, 공모전 자료 비공개 유지 | 완료 |
| 1 | 엔진·앱 신뢰성: contest 공통 노선도 이식(D-074), 계산 출처 기록(D-075·076), 1~9호선 통합 테스트(D-077), Python 3.14·pandas 3 호환과 CI 두 버전(D-078·079), v2-demo 배포 확인 | 완료 |
| 2 | 분석 마감: 수치 정합성 정정(D-080), 핵심 수치 재현성 고정(D-082), 결과·평가 문서([v2_results.md](v2_results.md), [v2_evaluation_strategy.md](v2_evaluation_strategy.md)), 앱에 v2 결과 반영 | 진행 중 |
| 3 | 코레일·공항철도 제한 통합: 서울 구간만, **혼잡도 없이 소요시간만**. 혼잡도 없는 구간은 '혼잡도 미확인'으로 따로 처리(0% 취급 금지). 시각표 자료 확보·검증부터 | 예정 |
| 4 | 모바일 최적화 → v2 정식 출시: `v2-next` → `main` 병합, README 전면 개편, 배포 Python 버전 하나로 고정(D-078) | 예정 |

### Phase 3 후보 구간 (사용자 검토안, 확정 전)

- 1호선 온수–서울역·금천구청–구로·청량리–도봉산, 경의중앙선 양원–수색, 수인분당선 청량리–복정
- 공항철도 서울역–김포공항 (인천공항 연장은 검토 중)
- 수인분당선 청량리 구간·경의중앙선 서울역 구간은 배차를 정밀하게 만들 수 있을 때만, 아니면 본선만

## 하지 않기로 한 것

| 항목 | 이유 | 근거 |
|---|---|---|
| 단일 스트레스 점수 | 지표마다 의미가 달라 뭉치면 해석이 불가능 | v2 기획 |
| 9호선 환승 지도 앱 대조 | 이미 공식 환승거리 기반, 민감도로 결론 유지, 지도 앱 값도 추정치 | D-081 |
| 9호선 혼잡 p90 채우기 | 3개 스냅샷으로 꼬리 분포를 낼 수 없음 | D-045 |
| 공모전 자료(4호선 칸별 샘플·동묘앞 도면·엘리베이터 표) 공개 | 공모전 제공·공모전판 산출물 | Phase 0 |
| Git 기록 재작성 | 공개 금지 자료가 확인된 경우에만 별도 검토 | Phase 0 |

## 남은 결정

- 공모전 파생 그림·집계(reports/v2 의 fig_spatial_*, fig_car_*, spatial_poc.json 등) 처리 — 공모전 요강의 제공 자료 이용 조건 확인 후
- 정식 출시 시 배포 Python 버전(3.12 또는 3.14)
