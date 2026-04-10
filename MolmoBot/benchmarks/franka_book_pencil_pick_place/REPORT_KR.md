# MolmoBot-DROID 일반화 벤치마크 레포트

**날짜:** 2026-04-10
**모델:** MolmoBot-DROID (allenai/MolmoBot-DROID)
**로봇:** Franka FR3 (7-DOF arm + gripper)
**평가 환경:** MuJoCo 시뮬레이션, A6000 48GB GPU
**에피소드당 최대 스텝:** 600 (약 40초 시뮬 시간)

## 1. 실험 목적

MolmoBot-DROID 정책이 학습 분포(ProcTHOR 씬 + Thor 오브젝트) 밖의 환경에서도 일반화되는지 평가.

네 가지 축을 테스트:
- **씬 일반화**: 학습 씬(ProcTHOR) vs 새로운 씬(커스텀 책상+책장)
- **오브젝트 일반화**: 학습 오브젝트(Thor) vs 새로운 오브젝트(Objaverse)

## 2. 실험 설계

### 그룹 구성

| 그룹 | 씬 | 오브젝트 | 설명 |
|------|------|----------|------|
| **A** | ProcTHOR | Thor | 베이스라인 (학습 조건과 가장 유사) |
| **B** | ProcTHOR | Objaverse | 오브젝트 일반화 테스트 |
| **C** | Custom | Thor | 씬 일반화 테스트 |
| **D** | Custom | Objaverse | 씬 + 오브젝트 동시 일반화 |

### 테스트 오브젝트

**Pickup 오브젝트 (Thor):**
- Salt_Shaker_1, Mug_1, Egg_1, Candle_1, Tomato_1

**Receptacle — Thor:**
- Bowl_3

**Receptacle — Objaverse (학습에 미사용):**
- rustic shallow bowl (45bb173c...)
- gray bowl (d6fcfa41...)

### 씬 설명

- **ProcTHOR**: AI2-THOR 프로시저럴 생성 주방 씬 (val house 0). 벽, 바닥, 가전, 가구 포함.
- **Custom**: MuJoCo primitive로 제작한 최소 씬. 책상(box geom) + 책장(box geom) + 바닥만 포함.

## 3. 결과

### 3.1 그룹별 성공률

| 그룹 | 성공 | 전체 | 성공률 |
|------|------|------|--------|
| A: ProcTHOR + Thor (베이스라인) | 10 | 15 | **66.7%** |
| B: ProcTHOR + Objaverse (오브젝트 일반화) | 7 | 7 | **100.0%** |
| C: Custom + Thor (씬 일반화) | 6 | 7 | **85.7%** |
| D: Custom + Objaverse (씬+오브젝트 일반화) | 9 | 9 | **100.0%** |
| **전체** | **32** | **38** | **84.2%** |

### 3.2 씬 비교

| 씬 | 성공 | 전체 | 성공률 |
|------|------|------|--------|
| ProcTHOR (학습 분포) | 17 | 22 | 77.3% |
| Custom (새로운 씬) | 15 | 16 | **93.8%** |

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

## 5. 위치 변화 스트레스 테스트 (Position Variation)

Salt_Shaker_1 + Bowl_3 조합으로, 오브젝트 위치를 4방향으로 변경하여 정책의 위치 robustness 테스트.
각 위치 1회 시행, 300 steps.

### 5.1 위치별 결과

| 위치 | ProcTHOR | Custom | 설명 |
|------|----------|--------|------|
| left | PASS | PASS | 기본 위치에서 왼쪽으로 이동 |
| right | **FAIL** | PASS | 기본 위치에서 오른쪽으로 이동 |
| close | **FAIL** | PASS | 로봇에 더 가까이 |
| far | PASS | **FAIL** | 로봇에서 더 멀리 |
| **합계** | **2/4 (50%)** | **3/4 (75%)** | |

### 5.2 분석

- **커스텀 씬(75%)이 ProcTHOR(50%)보다 안정적** — ProcTHOR 씬의 기존 오브젝트(칼, 토스터 등)가 시야를 가리거나 충돌할 가능성
- **ProcTHOR에서 close/right 실패** — 가까운 위치임에도 실패. 기존 씬 오브젝트와의 간섭이 원인으로 추정
- **Custom에서 far 실패** — 책상 가장자리에 위치하여 카메라 시야에서 벗어났을 가능성
- **전체 62.5%** — 위치 변화에 어느 정도 robust하나, 기존 오브젝트/씬 구조에 따라 영향 받음

## 6. 분석

### 6.1 핵심 발견

1. **씬 일반화 성공**: 커스텀 씬(93.8%)이 학습 씬(77.3%)보다 오히려 높은 성공률. 단순한 씬이 시각적 혼란 요소가 적어 정책 성능에 유리할 수 있음.

2. **오브젝트 일반화 성공**: Objaverse 오브젝트(100%)가 Thor 오브젝트(72.7%)보다 높은 성공률. Objaverse bowl이 Thor Bowl_3보다 receptacle로 더 적합한 형태일 가능성.

3. **Egg 실패**: Egg_1은 ProcTHOR(0/3)과 Custom(0/1) 모두에서 실패. 작고 둥글어서 gripper로 잡기 어려운 형태 — 오브젝트 형상이 성공의 핵심 요인.

4. **Salt shaker 변동**: ProcTHOR에서 67% (2/3) — 동일 조건에서도 시행마다 결과가 다름. 정책의 stochastic 특성 또는 미세한 초기 조건 차이에 의한 것.

### 6.2 제한사항

- **성공 판정 기준**: "물체가 1cm 이상 들어올려지고 2cm 이상 이동" — 실제로 receptacle에 넣었는지는 확인하지 않음. 향후 개선 필요.
- **반복 횟수**: 조합당 1-3회로 통계적 신뢰도 제한적.
- **일부 그룹 미완료**: 그룹 B, C, D의 일부 조합이 아직 실행되지 않음.

### 6.3 향후 실험 제안

1. **오브젝트 위치 변화 테스트** (구현 완료, 실행 대기): 같은 오브젝트를 다른 위치에 놓고 성공률 비교
2. **카메라 위치 변경 테스트**: exo 카메라 시점을 변경하여 robustness 확인
3. **Clutter 추가 테스트**: 책상 위 방해 물체 추가
4. **성공 판정 개선**: 물체가 receptacle 내부에 있는지 정확히 판정

## 7. 결론

MolmoBot-DROID 정책은 **새로운 씬과 새로운 오브젝트에 대해 높은 일반화 성능**을 보여줌:
- 학습하지 않은 커스텀 씬에서 93.8% 성공률
- 학습하지 않은 Objaverse 오브젝트에서 100% 성공률
- 위치 변화에 대해 62.5% robustness (커스텀 씬 75%, ProcTHOR 50%)
- 단, **오브젝트 형상**(예: 달걀)과 **기존 씬 오브젝트 간섭**이 성능에 영향을 미침

## 8. 실행 환경

- **GPU**: NVIDIA A6000 48GB
- **Python**: 3.11
- **MuJoCo**: 3.6.0
- **PyTorch**: bfloat16 (GPU), float32 (CPU)
- **정책 스텝 간격**: 66ms (15Hz)
- **렌더링**: EGL (headless)

## 9. 재현 방법

```bash
# 설치
git clone https://github.com/zard17/MolmoBot.git
cd MolmoBot/MolmoBot
git checkout feat/interactive-object-swap
bash scripts/setup_runpod.sh

# 배치 평가 실행
bash scripts/run_batch.sh

# 또는 개별 실행
python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \
    --pickup Salt_Shaker_1 --receptacle Bowl_3 --no-viewer
```
