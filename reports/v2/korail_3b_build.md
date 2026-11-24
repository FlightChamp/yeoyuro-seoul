# Phase 3B-1 경의중앙·경춘·수인분당·공항철도 정차표 — D-105

- 규칙: [사전 등록](../../docs/v2/preregistration_3b.md) 3절(T1~T5)·4-3절(C1)
- 재현: `python scripts/v2/52_build_korail_3b.py` (원본은 저장소에 없음, 상세 정차표도 gitignore — P4)

## 원본

| 파일 | SHA-256 |
|---|---|
| 202608281a045c7905b490.xlsx | `5cee7c8d3409d02bcde2c412b6ab6bc5d5fa5335d07e9331187d89bed6473a70` |
| 202608281a045c7cc64640.xlsx | `a6e95ad13a77fd6a39ee42eb869b7a2238bed6e4a1050514b27f871fd0d3527a` |
| 수인분당선 시각표 2026.8.22 적용.xlsx | `44543f0280afb17acbcdb12aefcd40ec79ff28fc28b6c1e1f80d9743ee3d14d0` |
| 공항철도 일반열차 운행시각표 2025.12.29.적용.xlsx | `2fd72bee9b1709e0c7a88c3a09f3363ea1955a29695a2a51f984f3c0821ef638` |
| 서울교통공사_환승역거리 소요시간 정보_20260831.csv | `434216e2c26fa7149ae7566badfa9938f87e83f28deb542aab68e1b10a9dc822` |

## 데이터 검사 (T4·T5, 운행일 경계)

| 항목 | 결과 |
|---|---|
| T4 범위 안 열차 수 = 3B-0 감사 | 불일치 **0** |
| T5 경의중앙 평일 > 휴일 | 183 > 149 → 예 |
| 시각 역행·같은 역 중복·정차 순서 뒤섞임 | 0·0·0 |
| 열차번호 중복 | 0 |
| 운행일 04:30~다음 날 04:29 밖 시각 | 0 |
| 범위 안 두 역 미만이라 뺀 열차 | 298 |
| 정차표 행 수 | 30,575 |

| 구간·요일·방향 | 열차 수 |
|---|---|
| AR · holiday · * | 318 |
| AR · weekday · * | 366 |
| GC · holiday · DOWN | 43 |
| GC · holiday · UP | 43 |
| GC · weekday · DOWN | 61 |
| GC · weekday · UP | 60 |
| GJ_A · holiday · DOWN | 73 |
| GJ_A · holiday · UP | 72 |
| GJ_A · weekday · DOWN | 90 |
| GJ_A · weekday · UP | 89 |
| GJ_B · holiday · DOWN | 72 |
| GJ_B · holiday · UP | 72 |
| GJ_B · weekday · DOWN | 85 |
| GJ_B · weekday · UP | 86 |
| SB · holiday · DOWN | 130 |
| SB · holiday · UP | 130 |
| SB · weekday · DOWN | 168 |
| SB · weekday · UP | 168 |

그룹별 열차 수 (휴일 시각표 = 토·일 두 벌):

| 요일·그룹 | 열차 |
|---|---|
| saturday · AR | 318 |
| saturday · GC | 86 |
| saturday · GJ | 149 |
| saturday · SB | 260 |
| sunday · AR | 318 |
| sunday · GC | 86 |
| sunday · GJ | 149 |
| sunday · SB | 260 |
| weekday · AR | 366 |
| weekday · GC | 121 |
| weekday · GJ | 183 |
| weekday · SB | 336 |

## B 등급 환승 계수 C1 (사전 등록 4-3)

- 규칙: ①∪② 절대 오차 중앙값 최소, 1.81 과 차이 0.10분 미만이면 1.81 (D-105 4-3 C1)

| 검증 자료 | n | ×1.00 | ×1.60 | ×1.81 | ×2.78 |
|---|---|---|---|---|---|
| ①∪② (판정) | 91 | 1.383 | 0.720 | 0.656 | 1.607 |
| ① 3-B | 12 | 0.775 | 0.455 | 0.958 | 2.143 |
| ② 1~8호선 | 79 | 1.450 | 0.827 | 0.632 | 1.557 |

- 값 = 절대 오차 중앙값(분). 오차 최소 계수 ×1.81 → **기본값 ×1.81**
- 관측 ÷ 공식 시간 비 중앙값: {'1-8': 1.814, '3B': 1.55}
- 이 기본값은 3B-4 결과를 본 뒤 바꾸지 않는다 (C4). 경로 영향은 3B-4 B5 에서 네 계수로 보고
- 한계: 관측값 자체도 실측이 아님 (① 수도권 환승 데이터·지도 앱, ② v1 환승시간)

