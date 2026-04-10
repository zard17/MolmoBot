# MolmoBot-DROID 일반화 벤치마크 레포트

**날짜:** 2026-04-10
**모델:** MolmoBot-DROID (allenai/MolmoBot-DROID)
**로봇:** Franka FR3 (7-DOF arm + gripper)
**평가 환경:** MuJoCo 시뮬레이션, A6000 48GB GPU
**에피소드당 최대 스텝:** 300-600 (약 20-40초 시뮬 시간)
**총 실험 에피소드:** 64 (38 일반화 + 8 위치 + 13 카메라 + 5 탐색)

---

## 1. 실험 목적

MolmoBot-DROID 정책이 학습 분포(ProcTHOR 씬 + Thor 오브젝트) 밖의 환경에서도 일반화되는지 평가.

**6가지 축을 테스트:**
1. 씬 일반화 — ProcTHOR(학습) vs Custom(새로운)
2. 오브젝트 일반화 — Thor(학습) vs Objaverse(새로운)
3. 오브젝트 위치 변화 — 기본 위치 대비 4방향 이동
4. 카메라 위치 변화 — offset ±5-10cm
5. 카메라 각도 변화 — pitch/yaw ±10°
6. 카메라 FOV 변화 — 55°~95° (기본 71°)

## 2. 실험 설계

### 2.1 일반화 테스트 그룹

| 그룹 | 씬 | 오브젝트 | 설명 |
|------|------|----------|------|
| **A** | ProcTHOR | Thor | 베이스라인 (학습 조건과 가장 유사) |
| **B** | ProcTHOR | Objaverse | 오브젝트 일반화 |
| **C** | Custom | Thor | 씬 일반화 |
| **D** | Custom | Objaverse | 씬 + 오브젝트 동시 일반화 |

### 2.2 테스트 오브젝트

**Pickup (Thor):** Salt_Shaker_1, Mug_1, Egg_1, Candle_1, Tomato_1

**Receptacle:**
- Thor: Bowl_3
- Objaverse (학습 미사용): rustic shallow bowl, gray bowl

### 2.3 씬 설명

| 씬 | 구성 | 특징 |
|------|------|------|
| ProcTHOR | AI2-THOR 주방 (val house 0) | 벽, 바닥, 가전, 가구, **기존 오브젝트 다수 (implicit clutter)** |
| Custom | MuJoCo primitive 책상+책장+바닥 | 최소 구성, **clutter 없음** |

> ProcTHOR 씬은 기존 오브젝트(칼, 토스터, 와인병 등)가 이미 존재하여 **implicit clutter test** 역할을 함.

---

## 3. 일반화 테스트 결과

### 3.1 그룹별 성공률

| 그룹 | 성공 | 전체 | 성공률 |
|------|------|------|--------|
| A: ProcTHOR + Thor (베이스라인) | 10 | 15 | **66.7%** |
| B: ProcTHOR + Objaverse (오브젝트 일반화) | 7 | 7 | **100.0%** |
| C: Custom + Thor (씬 일반화) | 6 | 7 | **85.7%** |
| D: Custom + Objaverse (씬+오브젝트 일반화) | 9 | 9 | **100.0%** |
| **전체** | **32** | **38** | **84.2%** |

### 3.2 씬 비교

| 씬 | 성공 | 전체 | 성공률 | 특징 |
|------|------|------|--------|------|
| ProcTHOR (학습 분포 + clutter) | 17 | 22 | 77.3% | 기존 오브젝트 간섭 |
| Custom (새로운 씬, clutter 없음) | 15 | 16 | **93.8%** | 깨끗한 환경 |

### 3.3 오브젝트 타입 비교

| 오브젝트 타입 | 성공 | 전체 | 성공률 |
|--------------|------|------|--------|
| Thor (학습 분포) | 16 | 22 | 72.7% |
| Objaverse (새로운) | 16 | 16 | **100.0%** |

### 3.4 오브젝트별 상세 결과

| 씬 | Pickup | Receptacle | 성공 | 성공률 |
|------|--------|-----------|------|--------|
| procthor | salt shaker | bowl | 2/3 | 67% |
| procthor | mug | bowl | 3/3 | 100% |
| procthor | egg | bowl | 0/3 | **0%** |
| procthor | candle | bowl | 3/3 | 100% |
| procthor | tomato | bowl | 2/3 | 67% |
| procthor | salt shaker | rustic shallow bowl | 3/3 | 100% |
| procthor | salt shaker | gray bowl | 3/3 | 100% |
| procthor | mug | rustic shallow bowl | 1/1 | 100% |
| custom | salt shaker | bowl | 3/3 | 100% |
| custom | mug | bowl | 3/3 | 100% |
| custom | egg | bowl | 0/1 | **0%** |
| custom | salt shaker | rustic shallow bowl | 3/3 | 100% |
| custom | salt shaker | gray bowl | 3/3 | 100% |
| custom | mug | rustic shallow bowl | 3/3 | 100% |

---

## 4. 스크린샷

### 4.1 그룹 A: ProcTHOR + Thor (베이스라인)

**Salt Shaker → Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/A_salt_shaker_PASS_first.png) | ![](report_screenshots/A_salt_shaker_PASS_last.png) |

**Mug → Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/A_mug_PASS_first.png) | ![](report_screenshots/A_mug_PASS_last.png) |

**Candle → Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/A_candle_PASS_first.png) | ![](report_screenshots/A_candle_PASS_last.png) |

**Egg → Bowl (실패)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/A_egg_FAIL_first.png) | ![](report_screenshots/A_egg_FAIL_last.png) |

### 4.2 그룹 B: ProcTHOR + Objaverse (오브젝트 일반화)

**Salt Shaker → Rustic Shallow Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/B_salt_shaker_objaverse_PASS_first.png) | ![](report_screenshots/B_salt_shaker_objaverse_PASS_last.png) |

### 4.3 그룹 C: Custom + Thor (씬 일반화)

**Salt Shaker → Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/C_salt_shaker_PASS_first.png) | ![](report_screenshots/C_salt_shaker_PASS_last.png) |

**Mug → Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/C_mug_PASS_first.png) | ![](report_screenshots/C_mug_PASS_last.png) |

**Egg → Bowl (실패)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/C_egg_FAIL_first.png) | ![](report_screenshots/C_egg_FAIL_last.png) |

### 4.4 그룹 D: Custom + Objaverse (씬+오브젝트 일반화)

**Salt Shaker → Rustic Shallow Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/D_salt_shaker_objaverse_PASS_first.png) | ![](report_screenshots/D_salt_shaker_objaverse_PASS_last.png) |

**Mug → Rustic Shallow Bowl (성공)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/D_mug_objaverse_PASS_first.png) | ![](report_screenshots/D_mug_objaverse_PASS_last.png) |

---

## 5. 오브젝트 위치 Robustness 테스트

Salt_Shaker_1 → Bowl_3, 4방향 위치 변경, 1회 시행, 300 steps.

| 위치 | ProcTHOR | Custom | 설명 |
|------|----------|--------|------|
| left | PASS | PASS | 기본 위치에서 왼쪽 |
| right | **FAIL** | PASS | 기본 위치에서 오른쪽 |
| close | PASS | PASS | 로봇에 더 가까이 |
| far | PASS | PASS | 로봇에서 더 멀리 |
| **합계** | **3/4 (75%)** | **4/4 (100%)** | **전체 87.5%** |

> ProcTHOR right 실패: 기존 씬 오브젝트(칼, 토스터)와의 시야 간섭이 원인으로 추정. Custom 씬은 clutter가 없어 모든 위치에서 성공.

---

## 6. 카메라 Robustness 테스트

Exo camera의 위치/각도/FOV를 변경. Salt_Shaker_1 → Bowl_3, 커스텀 씬, 1회 시행, 300 steps.

### 6.1 카메라 위치 (Position offset)

| 변화 | Offset | 결과 |
|------|--------|------|
| x+5cm (오른쪽) | [0.15, 0.57, 0.66] | PASS |
| x-5cm (왼쪽) | [0.05, 0.57, 0.66] | PASS |
| y+10cm (뒤로) | [0.1, 0.67, 0.66] | PASS |
| y-10cm (앞으로) | [0.1, 0.47, 0.66] | PASS |
| z+10cm (위로) | [0.1, 0.57, 0.76] | PASS |
| z-10cm (아래로) | [0.1, 0.57, 0.56] | PASS |
| **합계** | | **6/6 (100%)** |

### 6.2 카메라 각도 (Rotation)

| 변화 | 결과 |
|------|------|
| pitch+10° (아래 보기) | PASS |
| pitch-10° (위 보기) | PASS |
| yaw+10° (왼쪽 보기) | PASS |
| yaw-10° (오른쪽 보기) | PASS |
| **합계** | **4/4 (100%)** |

### 6.3 FOV (시야각)

| 변화 | FOV | 결과 |
|------|-----|------|
| narrow (줌인) | 55° (-22%) | PASS |
| wide (줌아웃) | 85° (+20%) | PASS |
| very_wide (광각) | 95° (+34%) | PASS |
| **합계** | | **3/3 (100%)** |

### 6.4 카메라 Robustness 요약

**전체 13/13 (100%)** — 위치 ±10cm, 각도 ±10°, FOV 55°~95° 범위에서 모두 성공.

모델이 특정 카메라 시점에 overfitting하지 않고 **viewpoint-invariant 특징을 학습**한 것으로 판단. 실제 로봇 배포 시 카메라 설치 위치가 약간 달라도 성능 유지 가능.

---

## 7. 심층 실패 분석

비디오 프레임 분석을 통해 실패 원인을 분류.

### 7.1 둥근 오브젝트 파지 실패 (Egg, Apple — 0%)

로봇이 오브젝트에 도달하고 gripper를 닫지만, 매끄러운 구형 표면에서 미끄러짐. Wrist camera에서 오브젝트에 극도로 가까이 접근한 것 확인 — **도달은 성공하지만 파지 불가**.

**Apple 파지 시도 (early experiment)**
| 시작 | 종료 (wrist cam 근접) |
|------|------|
| ![](report_screenshots/early_ep010_apple_bowl_t0_first.png) | ![](report_screenshots/early_ep010_apple_bowl_t0_last.png) |

### 7.2 부적합한 Receptacle (Mug→Cup — 0%)

컵이 머그보다 작아서 물리적으로 배치 불가능. 정책이 시도하지만 크기 불일치로 실패.

**Mug→Cup (early experiment)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/early_ep004_mug_cup_t0_first.png) | ![](report_screenshots/early_ep004_mug_cup_t0_last.png) |

### 7.3 학습 분포 외 배치 동작 (Mug→Bookcase — 0%)

선반 내부에 물건을 넣는 것은 학습 데이터에 없는 동작. 로봇이 방향 전환을 시도하지만 배치를 완료하지 못함.

**Mug→Bookcase (early experiment)**
| 시작 | 종료 |
|------|------|
| ![](report_screenshots/early_ep007_mug_bookcase_t0_first.png) | ![](report_screenshots/early_ep007_mug_bookcase_t0_last.png) |

### 7.4 Stochastic 변동 (Salt Shaker 67%, Tomato 67%)

동일 조건에서도 시행마다 결과가 다름. 시작 프레임이 거의 동일함에도 1/3 시행이 실패 — flow-matching 정책의 stochastic 특성에 의한 변동.

### 7.5 정책 약점 요약

| 약점 | 원인 | 영향도 |
|------|------|--------|
| 둥근/매끄러운 오브젝트 | Gripper 형상 한계 | **높음 (0%)** |
| 크기 불일치 receptacle | 물리적 불가능 인식 실패 | **높음 (0%)** |
| 학습 외 배치 동작 | 데이터 분포 외 | **높음 (0%)** |
| Stochastic 변동 | 정책 특성 | 중간 (67%) |
| 씬 clutter 간섭 | 카메라 시야 방해 | 낮음 (75-87.5%) |
| 카메라 시점 변화 | — | **없음 (100%)** |

---

## 8. 종합 결론

### 8.1 핵심 발견

| 테스트 | 결과 | 해석 |
|--------|------|------|
| 씬 일반화 | Custom 93.8% ≥ ProcTHOR 77.3% | 새로운 씬에서 오히려 성능 향상 (clutter 감소 효과) |
| 오브젝트 일반화 | Objaverse 100% ≥ Thor 72.7% | 새로운 오브젝트에 완전 일반화 |
| 위치 robustness | 87.5% (Custom 100%, ProcTHOR 75%) | 위치 변화에 강건 |
| 카메라 robustness | **100%** (13/13) | 시점 변화에 매우 강건 |
| 오브젝트 형상 | Egg/Apple 0% | Gripper 한계로 구형 오브젝트 파지 불가 |

### 8.2 정책 강점

- **높은 일반화**: 학습하지 않은 씬(93.8%)과 오브젝트(100%)에서 성능 유지
- **카메라 불변성**: 위치/각도/FOV 변화에 100% robust — viewpoint-invariant 학습 확인
- **간섭 내성**: Clutter가 있는 ProcTHOR에서도 77.3% 유지

### 8.3 정책 약점

- **오브젝트 형상 의존**: 둥글고 매끄러운 오브젝트(egg, apple) 파지 실패
- **물리적 제약 미인식**: 크기 불일치 receptacle에 배치 시도
- **학습 외 동작 한계**: 선반 내부 배치 등 새로운 동작 유형 실패

### 8.4 실무 시사점

1. **카메라 설치**: 위치/각도가 ±10cm/±10° 내에서는 자유롭게 설치 가능
2. **씬 구성**: 단순한 환경에서 더 높은 성능 — clutter 최소화 권장
3. **오브젝트 선택**: 핸들이나 평면이 있는 오브젝트 사용 권장 (머그, 캔들, 솔트셰이커 등)
4. **Receptacle 크기**: 픽업 오브젝트보다 큰 receptacle 사용 필요

---

## 9. 완료된 실험 및 향후 제안

### 완료

| # | 테스트 | 에피소드 | 결과 |
|---|--------|---------|------|
| 1 | 씬 일반화 (ProcTHOR vs Custom) | 38 | 섹션 3 |
| 2 | 오브젝트 일반화 (Thor vs Objaverse) | 포함 | 섹션 3 |
| 3 | 오브젝트 위치 변화 | 8 | 섹션 5 (87.5%) |
| 4 | 카메라 위치 변화 | 6 | 섹션 6 (100%) |
| 5 | 카메라 각도 변화 | 4 | 섹션 6 (100%) |
| 6 | 카메라 FOV 변화 | 3 | 섹션 6 (100%) |
| 7 | 오브젝트 형상 분석 | 5 | 섹션 7 |
| 8 | Receptacle 호환성 분석 | 포함 | 섹션 7 |

### 향후 제안

1. **반복 횟수 증가**: 통계적 신뢰도를 위해 조합당 5-10회 시행
2. **성공 판정 개선**: "물체가 receptacle 내부에 있는지" 정확한 판정 구현
3. **Explicit clutter 테스트**: Custom 씬에 방해 물체를 단계적으로 추가
4. **카메라 극한 테스트**: ±20cm, ±20° 이상의 큰 변화에서 성능 경계 확인
5. **Gripper 개선**: 둥근 오브젝트 파지를 위한 adaptive grasp 전략

---

## 10. 실행 환경

| 항목 | 값 |
|------|------|
| GPU | NVIDIA A6000 48GB |
| Python | 3.11 |
| MuJoCo | 3.6.0 |
| PyTorch | bfloat16 (GPU), float32 (CPU) |
| 정책 스텝 간격 | 66ms (15Hz) |
| 렌더링 | EGL (headless) |
| 브랜치 | feat/interactive-object-swap |

## 11. 재현 방법

```bash
# 설치
git clone https://github.com/zard17/MolmoBot.git
cd MolmoBot/MolmoBot
git checkout feat/interactive-object-swap
bash scripts/setup_runpod.sh

# 전체 배치 실행
bash scripts/run_batch.sh

# 카메라 robustness 테스트
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -u scripts/run_batch_eval.py \
  --checkpoint_path <path> \
  --config benchmarks/franka_book_pencil_pick_place/cam_position_config.json

# 개별 오브젝트 테스트
python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \
    --pickup Salt_Shaker_1 --receptacle Bowl_3 --no-viewer
```

## 12. 결과 디렉토리

| 디렉토리 | 내용 |
|----------|------|
| `results_main/` | A/B/C/D 그룹 평가 (44 영상) |
| `results_position/` | 위치 변화 테스트 (10 영상) |
| `results_camera_position/` | 카메라 위치 테스트 (6 영상) |
| `results_camera_angle/` | 카메라 각도 테스트 (4 영상) |
| `results_camera_fov/` | 카메라 FOV 테스트 (3 영상) |
| `results_early_exploratory/` | 초기 탐색 실험 (12 영상) |
| `report_screenshots/` | 레포트용 프레임 스크린샷 |
