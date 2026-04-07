# 최신 기술 동향: 로봇 조작 학습을 위한 시뮬레이션 데이터 생성

> 2026년 4월 기준 문헌 조사 | MolmoBot 접근법과의 비교
> MolmoBot 참조: 170만 에피소드, MuJoCo, CuRobo 플래너, 6-DoF 그래스프 샘플링, 공격적 도메인 랜덤화

---

## 1. 데이터 생성을 위한 모션 플래닝

### 플래너 비교

| 플래너 | 백엔드 | 속도 | 강점 | 약점 |
|--------|--------|------|------|------|
| **CuRobo** (NVIDIA) | GPU (CUDA) | 평균 ~45ms, CPU 대비 60배 | 50ms 미만 플래닝; 병렬 배치 (512 솔루션); 낮은 저크 (2.1 vs 5.8 rad/s^3); 메시 기반 충돌 | NVIDIA GPU 필수; 지원 로봇 모델 제한; 코어 비공개 |
| **OMPL** (MoveIt) | CPU | 평균 ~1200ms | 성숙한 생태계; 다양한 샘플링 알고리즘 (RRT*, PRM 등); 문서화 잘됨 | CPU 전용; 가변 사이클 (4-16초); 궤적 후처리 필요 (CHOMP/STOMP) |
| **MoveIt 2** | CPU (OMPL/CHOMP/STOMP) | 4-16초 사이클 | ROS2 통합; 산업 표준; 대규모 커뮤니티 | 대규모 데이터 생성에 느림; 자기 충돌 모델이 메시 기반보다 덜 신뢰 |
| **cuMotion** (NVIDIA, 2024+) | GPU | 실시간 가능 | Isaac에서 CuRobo 후속 통합 경로; MoveIt2 플러그인 | 신규, 검증 부족 |

### 데이터 생성 모범 사례

대규모 **고처리량 궤적 생성**에는 CuRobo가 확실한 최선입니다. CPU 플래너 대비 60배 속도 이점이 MolmoBot의 170만 에피소드 생성을 가능하게 합니다. MoveIt/OMPL은 실시간 배포와 ROS 통합에는 유효하나, 100만 에피소드 규모의 데이터 생성 백본으로는 비현실적입니다.

**MolmoBot의 접근법 (CuRobo + IK + 재시도 로직)은 현재 모범 사례에 부합합니다.** 재시도 로직이 CuRobo의 간헐적 플래닝 실패를 보상하며, IK 기반 그래스프 샘플링과의 조합이 견고한 파이프라인을 제공합니다.

### MolmoBot의 갭

CuRobo는 네이티브로 지원하는 로봇 모델이 제한적입니다. 커스텀 로봇 (예: 수정된 RBY1)으로 확장하려면 CuRobo 설정 파일 (로봇 YAML, 충돌 구체)을 구축해야 하며, 이는 상당한 작업입니다.

---

## 2. 그래스프 생성

### 방법 비교

| 방법 | 연도 | 접근법 | 학습 데이터 | 핵심 결과 | 한계 |
|------|------|--------|------------|----------|------|
| **Contact-GraspNet** | 2021 | 포인트 클라우드 기반 4-DoF | 1,700만 시뮬 그래스프 | 미지 물체에서 90%+ | 평행 조 전용; 테이블탑 편향 |
| **AnyGrasp** | 2023 | 장면 중심, 실제 포인트 클라우드 | 실세계 데이터 | 빠른 추론; 클러터에서 견고 | 제한적 라이선스 (머신 락); 테이블탑 외 일반화 부족 |
| **GraspGen** (NVIDIA) | 2025 | DiffusionTransformer + 판별기 | 5,300만 시뮬 그래스프 | 실세계 81.3% (M2T2 대비 28%↑); 시뮬에서 Contact-GraspNet 대비 17%↑ | 대규모 학습 데이터 필요; 컴퓨팅 부담 |
| **6-DoF GraspNet** | 2019 | VAE 기반 | 시뮬 그래스프 | 기초 연구 | 최신 방법에 의해 대체 |
| **AnyDexGrasp** | 2024 | 다지 손 확장 | 실제 + 시뮬 | 다지 그래스프 합성 | 다지 전용; 평행 조 비대상 |

### 현재 모범 사례

**데이터 생성 파이프라인** (실시간 배포가 아닌)에서의 주류 접근법:
1. 물체 표면에서 후보 그래스프 포즈 샘플링 (접촉점 또는 표면 법선 기반)
2. IK로 충돌/도달 가능성 필터링
3. 학습된 판별기 또는 분석적 지표로 점수 산정
4. CuRobo로 접근 궤적 플래닝

**MolmoBot은 6-DoF 그래스프 샘플링 + IK 필터링**을 사용하며, 견고한 베이스라인입니다. GraspGen (2025)이 학습 기반 그래스프 생성의 새로운 SOTA이나, 주로 런타임용으로 설계되었습니다. 오프라인 배치 생성에서는 MolmoBot의 샘플링 + 필터링이 더 실용적입니다.

### 갭

MolmoBot은 학습된 그래스프 품질 예측기 (GraspGen의 판별기 같은)를 사용하지 않습니다. 이를 추가하면 생성된 그래스프의 품질이 향상되고 재시도율이 감소할 수 있습니다.

---

## 3. 도메인 랜덤화 모범 사례

### 랜덤화 대상 (2024-2025 합의)

| 범주 | 파라미터 | 중요도 | 비고 |
|------|---------|--------|------|
| **시각** | 텍스처, 조명 (색상, 강도, 방향), 그림자, 카메라 포즈, 카메라 내부 파라미터 (FoV, 왜곡) | 핵심 | 가장 성숙한 영역; MolmoBot이 360도 카메라 랜덤화 포함하여 광범위하게 수행 |
| **객체 다양성** | 형상, 크기, 색상, 재질, 물체 수, 배치 | 핵심 | ICLR 2025 스케일링 법칙 논문에서 가장 영향력 있는 요인으로 확인 |
| **물리/동역학** | 마찰 계수, 질량, 관성, 감쇠, 관절 강성 | 중요 | 측정 가능한 파라미터에 대해서는 SysID가 맹목적 랜덤화보다 효과적 |
| **구동** | 모터 토크 한계, 제어 지연, 액션 노이즈, 그리퍼 힘 | 중요 | CDR에 따르면 순서가 중요: 토크 먼저, 그 다음 노이즈 |
| **장면 레이아웃** | 클러터, 배경 물체, 테이블 높이, 방해 물체 | 중요 | RoboTwin 2.0이 5가지 핵심 차원 중 하나로 식별 |
| **언어** | 다양한 태스크 설명, 의역 | 보통 | VLA 모델 일반화에 도움 |

### 고급 기법

- **Continual Domain Randomization (CDR, IROS 2024)**: 모든 랜덤화 파라미터를 한꺼번에가 아닌 순차적으로 도입. 토크부터 시작하고 노이즈를 추가. 지속 학습으로 이전 랜덤화 효과를 유지. 일괄 랜덤화와 동등하거나 우수한 성능 + 더 나은 학습 안정성.

- **DROPO (2023)**: 소량의 실세계 데이터에서 우도 기반 최적화로 분포를 추정하는 오프라인 도메인 랜덤화. 파라미터 불확실성을 명시적으로 모델링.

- **Automatic Domain Randomization (ADR)**: 정규화 플로우를 사용한 엔트로피 정규화 보상 최대화로 최적 랜덤화 분포를 자동 발견.

### MolmoBot의 접근법

MolmoBot은 시각 (조명, 텍스처, 전방향 카메라 포즈), 물리 (액션 노이즈), 장면 (94,200개 절차적 환경, 11,400+ 객체) 전반에 걸쳐 공격적 랜덤화를 적용합니다. 이는 **대부분의 공개 파이프라인보다 공격적**이며, 특히 전방향 카메라 랜덤화가 이례적입니다.

### 일반 파이프라인의 갭 (MolmoBot 포함)

1. **접촉 물리 랜덤화**가 부족 — 마찰, 반발 계수, 접촉 강성이 접촉 중심 태스크에 필요한 수준으로 랜덤화되지 않음
2. **구동기 동역학** (모터 곡선, 백래시, 케이블 구동 로봇의 신축)이 보통 단순화됨
3. 가우시안 이외의 **센서 노이즈 모델** (구조적 노이즈, 가림 패턴, 깊이 센서 아티팩트)이 거의 모델링되지 않음
4. **SysID vs DR 트레이드오프**: 최근 연구에 따르면 측정 *가능한* 파라미터 (질량, 관성)에 대해서는 시스템 식별이 도메인 랜덤화를 능가. 측정 가능한 파라미터의 맹목적 랜덤화는 오히려 역효과

---

## 4. 접촉 중심 조작

### 현황

| 태스크 유형 | 시뮬 성숙도 | 최적 접근법 | 핵심 참고 |
|------------|-----------|-----------|----------|
| Pick-and-place | 높음 | 스크립트 + DR + IL | MolmoBot, RoboCasa 등 다수 |
| 페그 삽입 (<0.5mm 클리어런스) | 중상 | RL + SDF 보상 + 커리큘럼 | IndustReal (NVIDIA), 실세계 83-99% |
| 기어 맞물림, 너트-볼트 | 중간 | RL + 힘 기반 보상 | Factory/IndustReal (Isaac Lab) |
| 도구 사용 | 중하 | RL + 시연 | 공개된 sim-to-real 제한적 |
| 변형 가능 물체 | 낮음 | MPM/FEM + 촉각 시뮬 | DiffTactile (ICLR 2024), Genesis |
| 다중 팔 조립 | 중하 | 양팔 IL + 힘/토크 | BIP 정책 + 멀티모달 센싱 |

### 핵심 발견

- **IndustReal (NVIDIA)**: <0.6mm 클리어런스의 실제 산업 조립에서 83-99% 성공률 달성. Isaac Sim에서 RL + SDF 보상 + 샘플링 기반 커리큘럼으로 전체 학습.

- **변형 가능 물체 조작**은 최난제. MLS-MPM (Material Point Method)이 선도적 시뮬레이션 접근법. DiffTactile이 접촉 중심 태스크를 위한 미분 가능 촉각 시뮬레이션 제공. Genesis가 미분 가능 MPM 솔버로 변형 시뮬레이션 지원.

- **촉각 센싱**이 접촉 중심 태스크에 필수적이라는 인식 증가. 시뮬레이션의 Vision-Based Tactile Sensors (VBTSs)가 전이 가능한 관측 (접촉 면적, 윤곽) 생성.

### MolmoBot의 위치

MolmoBot은 pick-and-place와 관절 물체 조작 (문 열기)에 초점. 정밀 공차 조립, 변형 물체, 도구 사용은 **다루지 않음**. 이는 분야 전체와 일관 — 모든 접촉 중심 태스크를 처리하는 단일 파이프라인은 없음. MolmoBot 규모 (170만 에피소드, 8개 태스크 유형)의 pick-and-place가 가장 성숙한 범주.

---

## 5. Sim-to-Real 갭

### 알려진 잔여 갭 (2025-2026)

| 갭 | 심각도 | 설명 |
|----|--------|------|
| **접촉 동역학** | 높음 | 단순화된 충돌 검출 (볼록 분해, 구 근사); 정적 마찰 이력 현상 미모델링; 비선형 재료 변형 부재 |
| **변형체** | 높음 | 대부분 시뮬레이터가 강체 가정; 실제 로봇은 유연한 링크, 순응 관절 보유; 물체가 하중 하에 변형 |
| **센서 충실도** | 중상 | 깊이 센서 노이즈 모델이 근사적; 실제 카메라는 롤링 셔터, 자동 노출, 화이트 밸런스 변동 |
| **구동기 동역학** | 중간 | 실제 모터는 비선형 토크 곡선, 열 효과, 백래시; 케이블 구동 로봇은 신축과 이력 현상 |
| **환경 확률성** | 중간 | 공기 흐름, 테이블 진동, 조명 변화, 전자기 간섭 — 체계적 모델링 어려움 |
| **물체 물리** | 중간 | 재료 특성 (마찰, 무게 분포)이 "동일" 물체의 인스턴스 간 변이 |
| **시각 갭** | 중하 | 공격적 DR + 실제 이미지 증강으로 대부분 해결; 투명/반사 물체에서 잔여 갭 |

### 새로운 해결책

- **Real-is-Sim / 디지털 트윈**: 실세계 스캔에서 고충실도 디지털 트윈을 구축한 후 트윈 주변을 랜덤화. 현실에서 시작하여 갭을 브릿지.
- **미분 가능 시뮬레이션**: Genesis, DiffTactile — 실제 동작과 매칭하기 위한 그래디언트 기반 파라미터 최적화 가능.
- **하이브리드 sim+real 파이프라인**: 시뮬에서 사전 학습, 소량 실제 데이터로 파인튜닝. 현재의 실용적 합의.
- **지속적 도메인 적응**: Sim-to-real 전이 후 실세계 데이터를 사용하여 시뮬 학습 능력을 보존하면서 정책을 지속 적응.

### MolmoBot의 주장과 근거 수준

MolmoBot은 순수 시뮬 데이터로 79.2% 실세계 pick-and-place 성공률을 보고합니다 (Pi-0.5의 39.2% 대비).

**관찰된 사실:** MolmoBot은 대규모 환경 다양성 + 시각적 DR을 사용하며, 물리 충실도 강화 없이도 pick-and-place에서 높은 전이 성능을 달성했습니다.

**저자의 해석 (인과 주장):** "공격적 다양성"이 물리 충실도보다 중요하며, 충분한 환경 변이가 있으면 실세계도 "또 하나의 변이"처럼 보인다는 것.

**근거 수준: 미분리.** 이 성공이 다양성, 데이터 규모, CuRobo 궤적 품질, 또는 이들의 조합 중 어디에 기인하는지 분리하는 ablation이 공개되지 않았습니다. 접촉이 중요한 태스크에서 동일 전략이 유효한지도 검증되지 않았습니다.

---

## 6. 규모 vs 품질: 170만 에피소드는 충분한가?

### 데이터 스케일링 법칙 (ICLR 2025 Oral, CoRL 2024 Best Paper)

Lin et al.의 핵심 연구 결과:

1. **거듭제곱 법칙**: 정책 일반화는 학습 환경, 객체, 환경-객체 쌍의 수와 거듭제곱 관계를 따름.

2. **다양성이 양을 지배**: 환경/객체당 시연 수가 임계값 (~50)에 도달하면 추가의 효과 미미. 더 많은 *환경*과 *객체*를 추가하면 성능이 지속 향상.

3. **실용적 레시피**: 32 환경 x 고유 객체 1개 x 50 시연 = 총 1,600 시연으로 새로운 환경과 객체에 대한 일반화와 함께 단일 태스크 90% 성공 달성 가능.

4. **규모**: 40,000+ 시연과 15,000+ 실세계 롤아웃 수집.

### MolmoBot과의 비교

| 지표 | 스케일링 법칙 논문 | MolmoBot |
|------|-------------------|----------|
| 에피소드 | ~40,000 시연 | 1,700,000 에피소드 |
| 환경 | 32 | 94,200 |
| 객체 | 태스크당 32 | 11,400+ |
| 태스크 | 2 (단일 태스크) | 8개 태스크 유형 |
| 로봇 | UMI (핸드헬드) | Franka, RBY1 |
| 데이터 출처 | 실제 시연 | 완전 합성 |

MolmoBot의 170만 에피소드는 **원시 데이터량** 측면에서는 매우 크며, 94,200개 환경과 11,400+ 객체의 다양성 역시 인상적입니다.

### 중요한 한계: 스케일링 법칙의 적용 범위

위 스케일링 법칙 연구는 **실제 로봇 데모 데이터**를 대상으로 수행되었습니다. 합성(시뮬레이션) 데이터에 동일한 스케일링 관계가 성립하는지는 **검증되지 않았습니다.** 구체적으로:

- 합성 에피소드의 정보량이 실제 에피소드와 다를 수 있음 (sim-to-real 갭으로 인해)
- 환경 다양성 vs 물리 충실도의 트레이드오프가 합성 데이터에서는 다르게 작용할 수 있음
- MolmoBot의 79.2% 실세계 성공률이 다양성 때문인지, 데이터 규모 때문인지, 다른 요인 때문인지 분리되지 않음

**따라서 "다양성이 핵심"이라는 주장은 실제 데이터 연구에서의 관찰이며, 합성 데이터에서는 아직 가설 단계입니다.** 이 가설을 우리 환경에서 검증하려면 다양성/규모/충실도를 통제한 ablation 실험이 필요합니다.

### 타 데이터셋과의 비교

| 데이터셋 | 규모 | 출처 | 비고 |
|---------|------|------|------|
| Open X-Embodiment | 1M+ 에피소드 | 실제, 22개 로봇 | 다기관; 이질적 품질 |
| RoboCasa365 | 2,200+ 시간 | 합성 + 실제 | 365개 태스크, 주방 도메인 |
| MolmoBot-Data | 170만 에피소드 (5,700+ 시간) | 완전 합성 | 8개 태스크 유형, 2개 로봇 |
| DROID | 76K 에피소드 | 실제 텔레오퍼레이션 | 단일 팔 테이블탑 |

---

## 7. 대안 시뮬레이션 플랫폼

### 플랫폼 비교

| 플랫폼 | 엔진 | GPU 병렬 | 렌더링 | 강점 | 약점 |
|--------|------|---------|--------|------|------|
| **MuJoCo** (MolmoBot) | CPU | 없음 (CPU 클러스터 필요) | 기본 / MJX로 GPU 가능 | 빠른 CPU 시뮬; 정확한 접촉; 성숙한 API; 오픈소스 (Apache 2.0) | CPU 전용; 네이티브 GPU 병렬 없음; 변형 지원 제한 |
| **Isaac Lab** (구 Orbit) | PhysX (GPU) | 예 (4,096+ 환경) | RTX 레이트레이싱 | 8 GPU에서 160만 FPS; 포토리얼리스틱; NVIDIA 생태계 | 비공개 (Isaac Sim); NVIDIA GPU 필수; 복잡한 설정 |
| **ManiSkill3** (SAPIEN) | SAPIEN (GPU) | 예 (최대 30,000+ FPS) | GPU 래스터화 | 오픈소스; CPU 대비 10-1000배 빠름; 12개 태스크 도메인; 대안 대비 GPU 메모리 2-3배 적음 | 베타 상태; MuJoCo/Isaac보다 작은 커뮤니티 |
| **RoboCasa / RoboCasa365** | MuJoCo (robosuite) | CPU 전용 | 기본 | 365개 태스크; 2,500 주방 장면; 2,200+ 시간 데모; 강력한 벤치마크 | CPU 시뮬만 (느림); 주방 중심 |
| **Genesis** | 커스텀 (GPU) | 예 (43만배 실시간 주장) | GPU | 미분 가능; 변형/유체/입자 지원; Isaac Gym 대비 10-80배 빠름 (주장) | 매우 신규; 실세계 검증 제한; 커뮤니티 형성 중 |
| **Habitat 3.0** (Meta) | 커스텀 | 제한적 | GPU | 인간-로봇 협업; 소셜 내비게이션/재배치; ICLR 2024 | 내비게이션 중심; 조작 충실도 제한 |
| **RoboVerse** | MetaSim (8+ 엔진) | 백엔드 의존 | 백엔드 의존 | MuJoCo, Isaac, SAPIEN, Genesis 등 통합 API; 크로스 플랫폼 재작성 불필요 | 추상화 오버헤드; 최신 (2025) |

### 핵심 트레이드오프

**MuJoCo (MolmoBot의 선택):**
- 장점: 가장 정확한 CPU 기반 접촉 시뮬레이션; 대규모 연구 커뮤니티; MolmoBot 인프라 전체가 이 위에 구축
- 단점: CPU 전용이므로 데이터 생성에 GPU 팜이 아닌 CPU 클러스터 필요. 170만 에피소드에서 실현 가능했으나 비용 큼

**Isaac Lab (가장 강력한 대안):**
- 장점: GPU 병렬로 대규모 처리량 (160만 FPS); 포토리얼리스틱 렌더링; IndustReal이 접촉 중심 조립 성공 시연
- 단점: NVIDIA 생태계에 고정; MuJoCo와 시뮬레이션 충실도 차이 (PhysX vs MuJoCo 접촉 모델)

**ManiSkill3 (부상하는 경쟁자):**
- 장점: 오픈소스 GPU 병렬; 경쟁력 있는 성능; 성장하는 태스크 라이브러리
- 단점: 아직 베타; 검증된 결과 기반이 작음

**Genesis (주시 대상):**
- 장점: 미분 가능; 다중 물리 (강체 + 변형 + 유체); 속도 기록 주장
- 단점: 프로덕션 사용에 너무 신규; 벤치마크 독립 검증 안됨

### Path A/B/C 관련 — 미해결 질문

`00_interim_results.md`의 Path A/B/C 결정에 대해, 각 경로의 트레이드오프는 **아직 검증되지 않은 가설**입니다. 현재 알 수 있는 사실과 미확인 사항을 분리합니다:

**Path A (MolmoBot 전체 스택, MuJoCo):**
- 사실: CPU 기반 데이터 생성, 94,200개 환경 파이프라인 상속
- 미확인: 이 다양성이 우리 로봇/태스크에서 핵심 요인인지

**Path B (학습 코드 + Isaac Sim 데이터):**
- 사실: GPU 병렬 데이터 생성 가능, MolmoBot 환경 다양성 미포함
- 미확인: 다양성 손실이 GPU 처리량 향상을 상쇄하는지, Isaac Sim에서 유사한 다양성을 구축할 수 있는지

**Path C (Isaac Sim으로 재구현):**
- 사실: 가장 높은 초기 비용, GPU 병렬성 + 다양성 재구축 가능성
- 미확인: 재구축 비용 대비 성능 향상이 정당화되는지, RoboVerse MetaSim이 비용을 실제로 줄이는지

**결정 전 필요한 실험:**
1. 다양성 ablation: 환경 수를 줄였을 때 성능 저하 정도 측정
2. 시뮬레이터 비교: 동일 태스크를 MuJoCo vs Isaac Sim에서 소규모로 생성하여 downstream 성능 비교
3. 처리량 측정: 우리 인프라에서 각 경로의 실제 데이터 생성 속도 벤치마크

---

## 8. 요약: MolmoBot의 위치

### MolmoBot이 SOTA 또는 SOTA 근접인 영역

1. **데이터 규모와 다양성**: 94,200개 환경에서 170만 에피소드 — 공개된 완전 합성 조작 데이터셋 중 최대 규모
2. **모션 플래닝 백본**: CuRobo + IK + 재시도가 표준 고처리량 접근법
3. **도메인 랜덤화 범위**: 전방향 카메라 랜덤화가 일반 파이프라인보다 공격적
4. **제로샷 sim-to-real**: 순수 시뮬 데이터에서 실세계 pick-and-place 79.2%는 강력한 결과
5. **오픈소스 완전성**: 전체 파이프라인 (데이터 생성 + 학습 + 평가) 공개는 이 분야에서 이례적으로 완전

### MolmoBot이 SOTA 대비 갭이 있는 영역

1. **접촉 중심 태스크**: 정밀 공차 조립, 삽입, 도구 사용 없음 — IndustReal이 이를 처리
2. **변형 가능 물체**: 미대응 — Genesis/DiffTactile이 프론티어
3. **GPU 병렬 생성**: MuJoCo는 CPU 기반; Isaac Lab과 ManiSkill3이 10-1000배 처리량 제공
4. **학습된 그래스프 품질**: 샘플링 + IK 필터링 사용, GraspGen 같은 학습된 판별기 미사용
5. **물리 랜덤화**: 접촉 물리 (마찰, 반발 계수, 강성) 랜덤화 깊이가 불명확
6. **SysID 통합**: 실제 로봇 측정에서 시뮬레이션 파라미터를 교정하는 메커니즘 없음
7. **촉각 센싱**: 촉각 모달리티 없음 — 접촉 중심 태스크에서 점차 중요해지는 영역

### 미해결 질문

- 합성 데이터에서도 다양성이 실제 데이터와 동일하게 핵심 요인인가?
- 다양성 vs 물리 충실도 트레이드오프가 우리 태스크에서 어떻게 작용하는가?
- 시뮬레이터 선택(MuJoCo vs Isaac Sim)이 downstream 정책 성능에 얼마나 영향을 미치는가?
- 우리 타겟 로봇/태스크에서 MolmoBot의 결과가 재현되는가?
- Path A/B/C 중 어느 것이 최적인지는 위 질문들에 대한 답이 나오기 전까지 알 수 없음

---

## 출처

### 모션 플래닝
- [Industrial Robot Motion Planning with GPUs: Integration of cuRobo](https://arxiv.org/html/2508.04146v2)
- [cuRobo Official](https://curobo.org/)
- [Comparative Benchmark of Sampling-Based and DRL Motion Planning Methods](https://www.mdpi.com/1424-8220/25/17/5282)

### 그래스프 생성
- [GraspGen: A Diffusion-based Framework for 6-DOF Grasping (2025)](https://arxiv.org/html/2507.13097v1)
- [Contact-GraspNet: Efficient 6-DoF Grasp Generation](https://arxiv.org/abs/2103.14127)
- [AnyDexGrasp](https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf)

### 도메인 랜덤화
- [Continual Domain Randomization (IROS 2024)](https://arxiv.org/abs/2403.12193)
- [DROPO: Sim-to-real transfer with offline domain randomization](https://www.sciencedirect.com/science/article/pii/S0921889023000714)
- [Domain Randomization via Entropy Maximization (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/56adf9cb91aedfa41ce24398782a012f-Paper-Conference.pdf)

### 접촉 중심 조작
- [IndustReal: Transferring Contact-Rich Assembly (NVIDIA)](https://arxiv.org/abs/2305.17110)
- [Bridging the Sim-to-Real Gap for Industrial Assembly (NVIDIA Isaac Lab)](https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/)
- [Survey on Imitation Learning for Contact-Rich Tasks](https://arxiv.org/html/2506.13498v1)
- [Contact-Rich Whole-Body Manipulation (Science Robotics 2025)](https://www.science.org/doi/10.1126/scirobotics.ads6790)
- [DiffTactile: Differentiable Tactile Simulator (ICLR 2024)](https://github.com/Genesis-Embodied-AI/DiffTactile)

### Sim-to-Real 갭
- [The Reality Gap in Robotics: Challenges, Solutions, and Best Practices (2025)](https://arxiv.org/html/2510.20808v1)
- [Real-is-Sim: Dynamic Digital Twin for Policy Evaluation](https://arxiv.org/html/2504.03597v1)
- [Safe Continual Domain Adaptation after Sim2Real Transfer](https://arxiv.org/html/2503.10949)

### 데이터 스케일링 법칙
- [Data Scaling Laws in Imitation Learning for Robotic Manipulation (ICLR 2025 Oral)](https://arxiv.org/abs/2410.18647)
- [Is Diversity All You Need for Scalable Robotic Manipulation? (2025)](https://arxiv.org/html/2507.06219v1)

### 시뮬레이션 플랫폼
- [MolmoBot (Allen AI)](https://allenai.org/blog/molmobot-robot-manipulation)
- [Isaac Lab / Orbit](https://isaac-orbit.github.io/)
- [ManiSkill3 (RSS 2025)](https://arxiv.org/abs/2410.00425)
- [RoboCasa365](https://arxiv.org/abs/2603.04356)
- [Genesis Simulator](https://genesis-embodied-ai.github.io/)
- [Habitat 3.0 (ICLR 2024)](https://arxiv.org/abs/2310.13724)
- [RoboVerse: Unified Platform (2025)](https://arxiv.org/html/2504.18904v1)

### VLA 기반 모델
- [Pi-0: Vision-Language-Action Flow Model](https://arxiv.org/html/2410.24164v1)
- [Pi-0.5: Open-World Generalization](https://www.physicalintelligence.company/blog/pi05)
