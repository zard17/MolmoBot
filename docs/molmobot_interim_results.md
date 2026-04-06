# MolmoBot 타당성 검토 중간결과

> 2026-04-07 기준, 두 환경(macOS CPU / Ubuntu GPU)에서의 검증 결과 종합

---

## 1. 검증 환경 요약

| 항목 | macOS (CPU) | Ubuntu (GPU) |
|------|-------------|--------------|
| 하드웨어 | Apple M3 Pro, 18GB RAM | NVIDIA A100 |
| OS | macOS 15 (Darwin 24.6.0) | Ubuntu (headless) |
| Python | 3.11 | 3.11 |
| 렌더링 | MuJoCo 기본 | EGL (`MUJOCO_GL=egl`) |
| JAX | CPU only | CPU only (CUDA 전환 가능 확인) |
| 모델 | MolmoBot-DROID (Molmo2-4B, bfloat16) | 동일 |

---

## 2. 중간결과

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
| Pick-only DROID mini | 10 | 200 | **7/10 (70%)** | 현재 최강 sim 시그널 |
| Pick-and-place DROID mini | 10 | 300 | **미실행** | config mismatch로 차단 |

Pick-and-place 실패 원인: `Expected max_place_receptacle_pos_displacement=0.15, got 0.05` (벤치마크/config 호환성 이슈)

**결론:** Pick 태스크에서 70% 성공은 모델이 non-trivial한 조작 능력을 보유함을 시사. 다만 horizon 200으로 단축했으므로 논문 재현 수치는 아님.

---

## 3. 핵심 발견 사항

### 기술적 특성

- **아키텍처:** Vision-Language-Action (VLA) 모델. Molmo2-4B (Qwen3 LLM 백본) + Flow Matching ActionExpert
- **추론 메커니즘:** 16 steps 예측, 8 steps 실행 후 재추론 (action chunking)
- **멀티뷰 지원:** 랜덤 카메라 배치로 학습 → 임의 시점 대응
- **지원 로봇:** Franka (8-DOF), RBY1 (19-29 DOF)

### 확장성

- 새 로봇 추가 시: `synthmanip_presets.py`에 ACTION_SPECS + CAMERA_PRESETS 정의 → 비교적 구조화된 확장 경로
- 학습 인프라: FSDP2 기반 분산 학습, `device_batch_size` 조절로 소규모 GPU 적응 가능
- 커스텀 로봇 적용 시 필요: MuJoCo MJCF 모델 + 시뮬레이션 데이터 생성 파이프라인 자체 구축

### 리스크

| 리스크 | 수준 | 설명 |
|--------|------|------|
| GPU 요구 (학습) | 높음 | 최소 4-8x A100/H100 |
| GPU 요구 (추론) | 중간 | 1x GPU (~8GB VRAM), CPU는 비실시간 |
| 외부 의존성 | 중간 | molmo_spaces, molmospaces-resources (Allen AI 관리, 업데이트 불확실) |
| 데이터 생성 | 중간 | 커스텀 로봇용 시뮬 데이터 파이프라인 자체 구축 필요 |
| Objaverse 에셋 버전 | 낮음 | 버전 불일치 경고 발생 (기능에는 영향 없음, 정식 재현 시 주의) |
| 라이선스 | 낮음 | Apache 2.0 (상업적 사용 가능) |

---

## 4. 현재 상태 판정

> **MolmoBot은 시뮬레이션에서의 인프라 및 기초 성능 검증을 통과했다.**
>
> 그러나 **sim2real 전이는 아직 완전히 미검증** 상태이며, 진정한 채택 의사결정은 real-robot zero-shot trial 이후에만 가능하다.

---

## 5. Next Steps

### 단기 (sim 시그널 강화)

1. **Pick-and-place config mismatch 해결**
   - `max_place_receptacle_pos_displacement` 파라미터 불일치 조사 및 수정
   - 수정 후 10ep pick-and-place 벤치마크 재실행

2. **더 큰 규모의 sim 벤치마크**
   - Pick-only 50-100 에피소드 → 현재 70%의 통계적 유의미성 확인
   - 다양한 태스크 구성으로 일반화 성능 확인

3. **GPU JAX 활성화** (선택)
   - `jax-cuda12-plugin[with-cuda]==0.6.2` 설치 (dry-run 호환 확인됨)
   - 벤치마크 속도 개선이 필요할 때만

### 중기 (real-robot gate)

4. **하드웨어 확보 후 real-robot zero-shot trial**
   - 타겟 로봇 embodiment 정의
   - 카메라 및 observation/action contract 정의
   - `run_feasibility_trials.py`로 소규모 태스크 실행
   - **이것이 진정한 채택 의사결정 게이트**

5. **Zero-shot 실패 시 gap 분류**
   - perception / camera placement / action semantics / timing / embodiment / task distribution 중 어디가 병목인지 진단
   - 진단 결과에 따라: 모델만 재활용 / 학습 레시피 재활용 / 데이터 구조 재활용 / 커스텀 적응 경로 중 결정

---

## 6. 관련 문서 인덱스

| 문서 | 설명 |
|------|------|
| [molmobot_feasibility_report.md](./molmobot_feasibility_report.md) | 기술 타당성 상세 분석 (아키텍처, 학습, 데이터, 커스터마이징) |
| [molmobot_demo_guide_macos_cpu.md](./molmobot_demo_guide_macos_cpu.md) | macOS CPU 데모 실행 가이드 |
| [franka_droid_feasibility_status_20260406.md](./franka_droid_feasibility_status_20260406.md) | Ubuntu GPU 벤치마크 상세 로그 |
| [franka_droid_feasibility.md](./franka_droid_feasibility.md) | 평가 워크플로우 (Phase 0-3) |
| [molmobot_adoption_status.md](./molmobot_adoption_status.md) | 채택 의사결정 프레임워크 및 remaining steps |

### 산출물 (artifacts)

| 파일 | 설명 |
|------|------|
| [smoke_success_exo.mp4](./artifacts/molmobot_feasibility/smoke_success_exo.mp4) | Smoke 벤치마크 성공 (Ubuntu GPU, exo view) |
| [pick_subset_success_exo.mp4](./artifacts/molmobot_feasibility/pick_subset_success_exo.mp4) | Pick-only 성공 사례 (Ubuntu GPU) |
| [pick_subset_failure_exo.mp4](./artifacts/molmobot_feasibility/pick_subset_failure_exo.mp4) | Pick-only 실패 사례 (Ubuntu GPU) |
| [rollout_macos_cpu.mp4](./artifacts/molmobot_feasibility/rollout_macos_cpu.mp4) | 전체 롤아웃 (macOS CPU) |
| [rollout_200_macos_cpu.mp4](./artifacts/molmobot_feasibility/rollout_200_macos_cpu.mp4) | 200-step 롤아웃 (macOS CPU) |
| [sample_render_macos_cpu.png](./artifacts/molmobot_feasibility/sample_render_macos_cpu.png) | 시뮬레이션 렌더링 샘플 (macOS CPU) |

---

*본 문서는 두 검증 환경(macOS CPU / Ubuntu GPU)의 결과를 종합하여 2026-04-07 작성되었습니다.*
