# Phase 3A-2f 공휴일 시각표 요일 검증 — D-104 (재계산 b)

- 규칙: [사전 등록 문서 11절](../../docs/v2/preregistration_3a2.md) D-104 (구현 전 고정) · 구현 기록 D-104a · 사후 수정 D-104b (11-6절)
- 재현: `python scripts/v2/51_check_holidays.py --workers 8 --tag b`
- 첫 결과 [holidays_3a2f.md](holidays_3a2f.md) 는 그대로 둔다 (평가기 시각표 요일 결함이 있던 계산)

| H | 검증 | 결과 | 기준 |
|---|---|---|---|
| 자료 | 2025~2027 공휴일 날짜 수 (평일·토요일 공휴일) | 2025: 19일 (평일 17·토 1), 2026: 22일 (평일 16·토 4), 2027: 24일 (평일 15·토 5) | 사전 등록 표와 같음 (테스트) |
| H6 | 비공휴일 앱 화면 = 구현 전 | 같음 **14/14** | 모두 같음 |
| H7 | 공휴일 앱 화면 | [('신림 강남 2026-05-01 08:00 calm', ['captions'], '안내'), ('홍대입구 잠실 2025-06-03 18:00 calm', ['captions'], '안내'), ('상봉 건대입구 2026-05-02 00:30 calm', ['captions'], '안내')] | 기존 카드는 안내 문구만 다름 |
| H9 | 확장 카드 전·후 (2026-05-01 × 412쌍 × 3) | 비교 1236 · 상태 바뀜 213 · 열차 바뀜 1136 · 경로 바뀜 247 · 기대 소요 바뀜 552 · 확정 추천 연결 불가 0 · 오류 0 | 보고 · 연결 불가 0 |

## H9 상세

- 상태 전 {'ok': 872, 'no_benefit': 277, 'none': 87} → 후 {'ok': 756, 'none': 276, 'no_benefit': 204}
- 시각별: {'08:00': {'status_changed': 3, 'trains_changed': 397}, '18:00': {'status_changed': 1, 'trains_changed': 412}, '23:30': {'status_changed': 209, 'trains_changed': 327}}
- 엔진 단계 제외 사유 합계: 전 {'운행 종료': 949} → 후 {'운행 종료': 1282}
- 시각표 확인 소요 차이(둘 다 연결 확인): 중앙 2.0 · 최대 29.0분
- 열차가 바뀐 예: [{'o': '혜화', 'd': '용산', 'hhmm': '08:00', 'status_b': 'ok', 'status_a': 'ok', 'best_trains_b': '4031|K49', 'best_trains_a': 'K4527|S39', 'train_total_b': 29.5, 'train_total_a': 25.0}, {'o': '혜화', 'd': '용산', 'hhmm': '18:00', 'status_b': 'ok', 'status_a': 'ok', 'best_trains_b': '4661|K165', 'best_trains_a': 'K4637|K155', 'train_total_b': 24.0, 'train_total_a': 21.5}, {'o': '혜화', 'd': '용산', 'hhmm': '23:30', 'status_b': 'ok', 'status_a': 'ok', 'best_trains_b': 'K4335|K825', 'best_trains_a': 'K4335|K823', 'train_total_b': 31.5, 'train_total_a': 29.0}, {'o': '신이문', 'd': '목동', 'hhmm': '08:00', 'status_b': 'ok', 'status_a': 'ok', 'best_trains_b': 'K427|5530', 'best_trains_a': 'K425|5032', 'train_total_b': 52.333, 'train_total_a': 50.833}, {'o': '신이문', 'd': '목동', 'hhmm': '18:00', 'status_b': 'ok', 'status_a': 'ok', 'best_trains_b': 'K167|5638', 'best_trains_a': 'K481|5614', 'train_total_b': 53.5, 'train_total_a': 55.333}]

## H6 차이

- 없음

## 한계

- 혼잡·배차 통계는 공휴일 관측값이 아니라 요일 기준 (P3·P4) — 기존 카드의 배차 기반 대기도 평일 공휴일에 평일 배차
- 토요일과 겹친 공휴일 = 휴일 시각표는 설계 가정 (미확인), 선거일 특별 운행 미반영
- 공휴일 판정 범위(2025~2027)와 시각표 유효기간(2026-09-01 시각표)은 다름

