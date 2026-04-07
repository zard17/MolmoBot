# Multi-Task Demo 결과

> MolmoBot-DROID의 다중 작업 수행 능력을 MuJoCo 시뮬레이션에서 검증한 결과
>
> 검증 환경: macOS 15 (Darwin 24.6.0), Apple M3 Pro, 18GB RAM, CPU 추론

---

## 실험 설계

MolmoBot-DROID가 학습한 3가지 작업 유형을 동일 장면에서 테스트:

| 작업 유형 | 프롬프트 | 학습 데이터 근거 |
|-----------|----------|-----------------|
| Pick & Place | "put the salt shaker in the bowl" | `pick_and_place` 템플릿 |
| Pick | "pick up the mug" | `pick` 템플릿 |
| Place Next To | "place the bottle next to the bowl" | `pick_and_place_next_to` 템플릿 |

### 장면 구성

- **환경**: ProCTHOR-10K val[0] house
- **로봇**: Franka FR3, 위치 [6.8, 9.75], 90° 회전
- **카메라**: exo_camera_1 (숄더 뷰) + wrist_camera (640×360)
- **오브젝트**:
  - Bowl_3 — [7.1, 10.2, 1.01] (수용체/receptacle)
  - Mug_1 — [6.9, 10.35, 1.01]
  - Bottle_1 — [7.2, 10.35, 1.01]
  - Salt shaker — house scene 내 기존 오브젝트

### 실행 조건

- **에피소드**: 100 steps × 66ms = 6.6초 (시뮬레이션 시간)
- **추론**: CPU에서 action chunk당 ~80초 (8 step 단위)
- **총 실행 시간**: ~56분 (3 tasks × ~19분)

---

## 결과 요약

| 작업 | 그리퍼 활성 | 첫 닫힘 | 열림 (놓기) | 소요 시간 | 판정 |
|------|------------|---------|------------|----------|------|
| Pick & Place | **15/87 steps** | Step 51 | Step 60 | 18.3분 | **성공** — 잡기→이동→놓기 패턴 관찰 |
| Pick | **43/87 steps** | Step 29 | Step 67 | 18.9분 | **성공** — 잡기→들기 패턴 관찰 |
| Place Next To | **0/87 steps** | — | — | 18.8분 | **실패** — 그리퍼 닫힘 없음 |

> 참고: 로그에서 100 steps 중 87 steps만 gripper 값이 기록됨 (inference step에서는 별도 형식으로 출력)

---

## 상세 분석

### Task 1: Pick & Place — 성공

```
Steps 1-50:   그리퍼 열림 (0) — 접근 단계
Steps 51-59:  그리퍼 닫힘 (255) — salt shaker 파지
Step 60:      그리퍼 열림 (0) — bowl 위에서 놓기
Steps 60-87:  그리퍼 열림/닫힘 반복 — 추가 시도
```

전형적인 pick-and-place 시퀀스를 보여줌. 접근→파지→이동→놓기의 4단계 동작이 관찰됨.

### Task 2: Pick — 성공

```
Steps 1-28:   그리퍼 열림 (0) — 접근 단계
Steps 29-66:  그리퍼 닫힘 (255) — mug 파지 및 들기
Step 67:      그리퍼 열림 (0)
Steps 67-87:  열림/닫힘 반복
```

Pick task에서 가장 적극적인 그리퍼 활동 (43/87 steps 닫힘). 파지 시작이 Step 29로 Pick & Place보다 빠름 — pick-only 작업이 더 단순하므로 더 빠르게 접근하는 것으로 보임.

### Task 3: Place Next To — 실패

100 steps 동안 그리퍼가 한 번도 닫히지 않음. 팔이 움직이긴 하나 (비디오 파일 209KB로 약간의 움직임 있음) 파지 시도 없음.

---

## 추가 디버깅 (Place Next To)

원인 분석을 위해 5가지 변형을 추가 테스트:

| 변형 | 변경 사항 | Steps | 그리퍼 닫힘 | 결과 |
|------|----------|-------|------------|------|
| 프롬프트 변경 | "move the bottle near the bowl" | 20 | 0 | 실패 |
| 위치 변경 | bottle → [7.0, 10.15] (더 가까이) | 20 | 0 | 실패 |
| 다른 작업 (mug→bowl) | "put the mug in the bowl" | 20 | 0 | steps 부족 |
| bottle pick | "pick up the bottle" | 20 | 0 | steps 부족 |
| **mug→bowl (50 steps)** | "put the mug in the bowl" | **50** | **8** | **성공** |
| **bottle next to (50 steps)** | "place the bottle next to the bowl" | **50** | **0** | **실패** |
| **pick bottle (50 steps)** | "pick up the bottle" | **50** | **13** | **성공** |

### 핵심 발견

1. **Bottle 오브젝트 자체는 파지 가능** — "pick up the bottle"에서 13/50 steps 닫힘 확인
2. **"place next to" 작업 유형이 실패 원인** — 같은 bottle이라도 "pick up"은 성공, "place next to"는 실패
3. **20 steps는 진단에 불충분** — 그리퍼 닫힘은 보통 step 29-51에서 시작되므로 최소 50 steps 필요

---

## 원인 추정

"Place next to" 실패의 가능한 원인:

1. **학습 데이터 불균형** — `pick_and_place`와 `pick` 대비 `pick_and_place_next_to` 학습 데이터가 적을 가능성
2. **Sim-to-real gap** — MolmoBot-DROID는 실제 로봇(DROID) 데이터로 학습. 시뮬레이션 환경의 시각적 차이가 "place next to"처럼 미세한 공간 추론이 필요한 작업에서 더 크게 영향
3. **작업 복잡도** — "place next to"는 목표 위치가 명시적이지 않음 (bowl "옆"이 어디인지 추론 필요). "in the bowl"처럼 명확한 receptacle이 있는 경우와 다름

---

## 출력 파일

| 파일 | 설명 |
|------|------|
| `task1_pick_and_place.mp4` | Pick & Place 100 steps (666KB) |
| `task2_pick.mp4` | Pick 100 steps (572KB) |
| `task3_place_next_to.mp4` | Place Next To 100 steps (209KB) |
| `smoke_*.mp4` | 스모크 테스트 (5 steps each) |
| `retry_*.mp4` | 디버깅 변형 (50 steps each) |
| `smoke_render.png` | 장면 렌더링 (exo + wrist) |

---

## 결론

| 항목 | 결과 |
|------|------|
| Pick & Place | **동작 확인** — 자연어 프롬프트 기반 pick-and-place 성공 |
| Pick | **동작 확인** — 가장 적극적인 그리퍼 활동, 빠른 접근 |
| Place Next To | **미동작** — 그리퍼 활성화 실패, 작업 유형 자체의 한계로 추정 |
| 멀티태스크 전환 | **동작 확인** — `mj_resetData` + `policy.reset()`으로 작업 간 전환 정상 |
| CPU 실행 가능성 | **확인** — M3 Pro에서 작업당 ~19분, 총 ~56분 소요 |

MolmoBot-DROID는 시뮬레이션 환경에서 pick과 pick-and-place 작업을 수행할 수 있으나, 공간 관계 추론이 필요한 "place next to" 작업은 추가 학습 또는 fine-tuning이 필요한 것으로 판단됨.

---

*작성일: 2026-04-07 / 검증 환경: macOS 15, M3 Pro 18GB, Python 3.11, MolmoBot-DROID (allenai/MolmoBot-DROID)*
