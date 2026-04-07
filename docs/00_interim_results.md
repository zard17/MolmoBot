# 가상 데이터 기반 로봇 학습 타당성 검토 — 중간결과

> 2026-04-07 기준 | 검증 수단: MolmoBot (Allen AI) | 환경: macOS CPU + Ubuntu GPU

---

## 1. 검토 목표

**시뮬레이션에서 생성한 가상 데이터만으로 실물 로봇에서 유의미한 조작 성능을 얻을 수 있는지** 검증한다. MolmoBot은 이 가설을 가장 직접적으로 시험할 수 있는 공개 프레임워크로서 평가 대상이다.

구체적으로:

1. 가상 데이터만으로 학습한 정책이 실물 로봇에서 **zero-shot으로 동작하는가?**
2. zero-shot이 부족하면, 가상 데이터 사전학습 + 소량 실물 데이터 파인튜닝으로 **실물 데이터 단독 학습보다 나은 결과를 얻을 수 있는가?**

---

## 2. 검증 환경

| 항목 | macOS (CPU) | Ubuntu (GPU) |
|------|-------------|--------------|
| 하드웨어 | Apple M3 Pro, 18GB RAM | NVIDIA A100 SXM 80GB |
| OS | macOS 15 (Darwin 24.6.0) | Ubuntu 22.04 (headless) |
| Python | 3.11 | 3.11 |
| 렌더링 | MuJoCo 기본 | EGL (`MUJOCO_GL=egl`) |
| JAX | CPU only | CPU only (CUDA 전환 가능 확인) |
| 모델 | MolmoBot-DROID (Molmo2-4B, bfloat16) | 동일 |

---

## 3. 중간결과

### 환경 재현성

| 검증 항목 | macOS CPU | Ubuntu GPU |
|-----------|-----------|------------|
| `uv sync --extra eval` | 성공 | 성공 |
| 모델 로드 | 10.8초 | 정상 |
| 렌더링 | 기본 동작 | EGL 라이브러리 설치 후 동작 |
| 노트북 sanity check | 통과 | 통과 (스크립트 동등물) |

**결론:** 두 환경 모두 설치/실행 재현 가능.

### 추론 성능

| 측정 항목 | macOS CPU |
|-----------|-----------|
| Action chunk 추론 (8 steps) | ~80초 |
| 버퍼 step 실행 | ~2초 |
| 평균 step 시간 | 12.45초 |
| 200 steps 총 시간 | ~40분 |
| 목표 실시간 (Franka) | 66ms/step |
| CPU vs 실시간 비율 | **~190배 느림** |

**결론:** CPU에서는 검증 가능하지만 실시간 제어 불가. 실제 배포에는 GPU 필수.

### 시뮬레이션 벤치마크 (Ubuntu GPU)

| 벤치마크 | 에피소드 | Horizon | 성공률 | 비고 |
|----------|---------|---------|--------|------|
| Smoke (built-in 1ep) | 1 | 600 | **1/1 (100%)** | 기본 동작 확인 |
| Pick-only DROID mini | 10 | 200 | **7/10 (70%)** | 현재 최강 Franka sim 시그널 |
| Pick-and-place DROID mini | 10 | 300 | **미실행** | config mismatch로 차단 |
| RBY1 door smoke (normalized 1ep) | 1 | 300 | **1/1 (100%)** | `MolmoBot-RBY1Multitask`, benchmark metadata normalize 후 end-to-end 완료 |
| RBY1 door small slice (reusable workflow) | 3 | 600 | **1/3 (33.3%)** | `run_feasibility.py` + normalized first-3 door slice로 재현 가능 |
| RBY1 pnp small slice (reusable workflow) | 3 | 400 | **0/0 (미실행)** | task sampling blocked by `max_place_receptacle_pos_displacement` mismatch (`0.15` vs `0.1`) |

Pick-and-place 실패 원인: `Expected max_place_receptacle_pos_displacement=0.15, got 0.05` (벤치마크/config 호환성 이슈)
RBY1 pnp 실패 원인: `Expected max_place_receptacle_pos_displacement=0.15, got 0.1` (released benchmark vs current config mismatch)

**결론:** Franka/DROID에서 pick 70% 성공. RBY1에서는 1ep smoke를 넘어 reusable workflow로 3ep door slice를 측정했고 `1/3`까지 확인했다. 즉, 타겟 embodiment 기준의 반복 가능한 sim 검증 경로는 확보되었다. 반면 pnp는 정책 문제가 아니라 benchmark/config 호환성 이슈로 아직 rollout 단계에 진입하지 못했다.

---

## 4. 핵심 발견 사항

### 가상 데이터 접근법의 특성

- **데이터 규모:** 170만 에피소드, 2.95억 프레임, 5,704시간 분량. 94,200개 절차적 생성 환경, 11,400+ 오브젝트
- **데이터 생성 방식:** 기존 approach와 동일하게 scripted policy 기반이지만, 6-DoF grasp sampling + IK/CuRobo motion planning + retry logic으로 구성된 고도화된 planner. 이 규모에서 검증 완료
- **도메인 랜덤화:** 조명/텍스처/물리/액션 노이즈 + 카메라를 360도 전방위로 랜덤 배치하여 임의 시점에서 동작하도록 학습. 단순히 양을 키운 것이 아니라 랜덤화의 범위와 체계성이 핵심
- **파이프라인 전체 공개:** datagen 코드, 환경, 에셋이 molmospaces 레포에 오픈소스(Apache 2.0). 환경/랜덤화/녹화 인프라를 재사용 가능
- **RBY1 planner 존재:** CuRobo 기반 motion planning이 이미 구현되어 있음. 단, 우리 RBY1의 커스텀 부분(하드웨어 변경, 센서 구성 등)에 따라 그대로 사용 못할 가능성 있음
- 상세 구조 및 적용 경로는 [01_technical_analysis.md](./01_technical_analysis.md) 참조

### 리스크

| 리스크 | 수준 | 설명 |
|--------|------|------|
| 파이프라인 적용 비용 | 중간 | 환경/랜덤화 인프라는 재사용 가능하나, 우리 RBY1 커스텀에 맞는 planner 수정 및 MJCF 모델 구축 필요 |
| GPU 요구 (학습) | 높음 | 가상 데이터로 학습하더라도 최소 4-8x A100/H100 필요 |
| 기존 approach와의 직접 비교 불가 | 중간 | 기존 scripted policy + OpenPI도 미완성이라 현 시점에서 정량 비교가 어려움. MolmoBot 단독 결과로 방향을 판단해야 함 |
| 외부 의존성 | 중간 | molmo_spaces 등 Allen AI 패키지에 의존 (업데이트 불확실). 가상 데이터 접근법 자체를 내재화하려면 이 의존성을 넘어야 함 |
| 라이선스 | 낮음 | Apache 2.0 (상업적 사용 가능) |

---

## 5. 현재 상태 판정

현재 근거가 **지지하는** 결론:
> 가상 데이터 학습이 sim에서 non-trivial한 성능을 낸다 (Franka pick 70%). 더 깊이 검증할 가치가 있다.

현재 근거가 **아직 지지하지 못하는** 결론:
> 타겟 로봇(RBY1)에서 broad benchmark 수준으로 유효하다.

남은 질문:
1. **RBY1 door 성능이 3ep beyond에서 유지되는가?** → 즉시 확장 가능
2. **RBY1 pnp config mismatch (`0.15` vs `0.1`)를 정렬하면 rollout이 실제로 실행되는가?** → 단기 수정 가능
3. **sim 성능이 실물 RBY1으로 전이되는가?** → 하드웨어 확보 후 검증

---

## 6. 도입 경로 선택지

MolmoBot의 학습 코드는 MuJoCo에 의존하지 않는다 (HDF5 데이터만 읽음). 단, 데이터 생성 파이프라인(molmospaces)은 MuJoCo 전용. 기존 Isaac Sim 파이프라인과의 관계에 따라 세 가지 경로가 있다.

| 경로 | 설명 | 예상 effort | 핵심 trade-off |
|------|------|------------|---------------|
| **A. MolmoBot 풀 스택** | MuJoCo로 전환, molmospaces 파이프라인 사용 | 중간 | 로봇 MJCF 변환 + RBY1 커스텀 반영 필요. 대신 94,200 환경 + RBY1 planner + 전방위 랜덤화를 즉시 사용 가능. 물리 엔진(PhysX→MuJoCo) 차이는 도메인 랜덤화가 설계상 커버 |
| **B. 학습 코드만 도입** | Isaac Sim 유지, HDF5 exporter만 추가 | 낮음 | MolmoBot 성과의 핵심(환경 다양성, 랜덤화 범위, 데이터 규모)을 못 씀. 기존 파이프라인의 다양성/규모가 충분한지에 달림 |
| **C. Isaac Sim에서 재구현** | Isaac Sim 위에 molmospaces급 환경 다양성/랜덤화 구축 | 높음 | 가장 이상적이지만 사실상 molmospaces를 Isaac Sim용으로 다시 만드는 것 |

기존 Isaac Sim의 리소스 파일(메시, 텍스처, 로봇 모델)은 MuJoCo로 변환 가능. 전환 시 실질적으로 버리는 것은 Isaac Sim API에 종속된 코드(scripted policy, 랜덤화 로직)이며, 이는 molmospaces의 기존 구현이 대체한다.

RBY1 sim 벤치마크 결과에 따라 경로를 결정한다.

---

## 7. Next Steps

### 단기 — 타겟 로봇(RBY1) 기준으로 sim 시그널 확보

1. **RBY1 reusable workflow를 benchmark coverage로 확장**
   - 완료: `run_feasibility.py`가 normalized RBY1 small-slice workflow를 지원
   - 완료: `MolmoBotRBY1DoorPlusOpenEvalConfig` first-3 door slice `1/3`
   - 다음: door slice를 더 넓히거나 seed/episode count를 늘려 분산 확인

2. **RBY1 pnp config mismatch 해결**
   - 현재 blocker: `Expected max_place_receptacle_pos_displacement=0.15, got 0.1`
   - released pnp benchmark와 현재 molmo_spaces config를 정렬한 뒤 `MolmoBotRBY1PickPnPEvalConfig`를 재실행

3. **Franka pick-and-place config mismatch 해결** (선택, RBY1 다음)
   - `max_place_receptacle_pos_displacement` 파라미터 불일치 수정

4. **학습 데이터 파이프라인 분석**
   - 가상 데이터 규모/다양성/생성 비용, RBY1용 vs Franka용 차이점
   - 우리가 직접 데이터를 만들 수 있는지 판단하는 근거 확보

5. **GPU JAX 활성화** (선택, 벤치마크 처리량이 병목일 때만)

### 중기 — 실물 RBY1에서의 전이 검증

6. **RBY1 하드웨어 확보 후 real-robot zero-shot trial**
   - 카메라 구성 (wrist_r + head + wrist_l), observation/action contract 정의
   - `run_feasibility_trials.py`를 RBY1에 맞게 구성
   - **핵심 질문 "가상 데이터가 실물 RBY1에서도 도움이 되는가"의 직접적 답변**

7. **결과에 따른 전략 판단**
   - **성공 시:** 가상 데이터 파이프라인 투자 정당화. RBY1용 데이터 생성 경로 구체화
   - **실패 시:** 원인 분류 (perception / camera / action / embodiment / task distribution) → 가상 데이터 자체의 한계인지 도메인 gap인지 구분 → sim+real 혼합 가능성 판단

---

## 8. 관련 문서

| 문서 | 설명 |
|------|------|
| [01_technical_analysis.md](./01_technical_analysis.md) | 가상 데이터 파이프라인 구조, 학습 인프라 요구사항, 우리 로봇 적용 경로 상세 |
| [02_gpu_benchmark_log.md](./02_gpu_benchmark_log.md) | GPU 환경 벤치마크 실행 기록 및 평가 워크플로우 재현 가이드 |
| [03_cpu_demo_guide.md](./03_cpu_demo_guide.md) | macOS CPU 환경 데모 실행 가이드 (빠른 체험용) |

### 산출물 (artifacts)

| 파일 | 설명 |
|------|------|
| [smoke_success_exo.mp4](./artifacts/molmobot_feasibility/smoke_success_exo.mp4) | Smoke 벤치마크 성공 (Ubuntu GPU, exo view) |
| [pick_subset_success_exo.mp4](./artifacts/molmobot_feasibility/pick_subset_success_exo.mp4) | Pick-only 성공 사례 (Ubuntu GPU) |
| [pick_subset_failure_exo.mp4](./artifacts/molmobot_feasibility/pick_subset_failure_exo.mp4) | Pick-only 실패 사례 (Ubuntu GPU) |
| [rollout_macos_cpu.mp4](./artifacts/molmobot_feasibility/rollout_macos_cpu.mp4) | 전체 롤아웃 (macOS CPU) |
| [rollout_200_macos_cpu.mp4](./artifacts/molmobot_feasibility/rollout_200_macos_cpu.mp4) | 200-step 롤아웃 (macOS CPU) |
| [sample_render_macos_cpu.png](./artifacts/molmobot_feasibility/sample_render_macos_cpu.png) | 시뮬레이션 렌더링 샘플 (macOS CPU) |
| [rby1_door_smoke_20260407/README.md](./artifacts/molmobot_feasibility/rby1_door_smoke_20260407/README.md) | RBY1 door smoke 1ep 결과 요약 및 검증용 영상/log 링크 |
| [rby1_doorplusopen_small_20260407/README.md](./artifacts/molmobot_feasibility/rby1_doorplusopen_small_20260407/README.md) | reusable workflow 기반 RBY1 door 3ep 결과 (`1/3`) 및 대표 영상/log |
| [rby1_pnp_blocker_20260407/README.md](./artifacts/molmobot_feasibility/rby1_pnp_blocker_20260407/README.md) | reusable workflow 기반 RBY1 pnp blocker 로그 및 manifest |

---

*본 문서는 두 검증 환경(macOS CPU / Ubuntu GPU)의 결과를 종합하여 2026-04-07 작성되었습니다.*
