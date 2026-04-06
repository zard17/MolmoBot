# MolmoBot 기술 분석 — 가상 데이터 파이프라인 구조와 우리 로봇 적용 경로

> 시뮬레이션 전용 데이터로 로봇 조작을 학습하는 MolmoBot의 내부 구조, 데이터 생성 방식, 학습 인프라, 그리고 우리 로봇에 적용할 경우의 구체적 경로를 분석한다.

---

## 1. 개요

**MolmoBot**은 Allen Institute for AI가 개발한 Vision-Language-Action (VLA) 모델로, 시뮬레이션 데이터만으로 학습하여 실물 로봇에서 제로샷 조작을 수행한다.

| 항목 | 내용 |
|------|------|
| 논문 | [arXiv:2603.16861](https://arxiv.org/abs/2603.16861) "MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation" |
| 라이선스 | Apache 2.0 (상업적 사용 가능) |
| 기반 모델 | Molmo2-4B (4B 파라미터 VLM, Qwen3 LLM 백본) |
| 지원 로봇 | Franka (DROID), RBY1 (모바일 매니퓰레이터) |
| 코드 | https://github.com/allenai/MolmoBot |
| 모델 | https://huggingface.co/collections/allenai/molmobot-models |
| 데이터 | https://huggingface.co/collections/allenai/molmobot-data |

### 핵심 주장
- 시뮬레이션 데이터만으로 학습 (실물 로봇 데이터 불필요)
- 다양한 로봇 형태에 적용 가능 (8-DOF ~ 29-DOF)
- 랜덤 카메라 배치로 학습하여 임의 시점에서 동작

---

## 2. 모델 아키텍처

### 전체 구조

```
이미지 (멀티뷰) + 텍스트 (태스크 설명)
        ↓
┌──────────────────────────────┐
│  Vision Backbone (ViT)       │  이미지 → 비전 토큰
├──────────────────────────────┤
│  Connector (Linear)          │  비전 토큰 → LLM 임베딩 공간
├──────────────────────────────┤
│  LLM Decoder (Molmo2-4B)     │  32 레이어, d_model=2048
│  → per-layer hidden states   │  각 레이어 출력을 ActionExpert에 전달
├──────────────────────────────┤
│  ActionExpert (Flow Matching)│  DiT 스타일 Transformer
│  → 액션 시퀀스 생성          │  노이즈 → 클린 액션 궤적
└──────────────────────────────┘
        ↓
  액션 청크 (16 steps 예측, 8 steps 실행)
```

### 주요 컴포넌트별 파일

| 컴포넌트 | 파일 | 역할 |
|----------|------|------|
| 코어 모델 | `olmo/models/molmobot/molmobot.py` | VLM 백본 + flow-matching 액션 헤드 |
| 액션 전문가 | `olmo/nn/action_expert.py` | 액션 궤적 디노이징 Transformer |
| 추론 래퍼 | `olmo/models/molmobot/inference_wrapper.py` | 체크포인트 로드, 추론 실행 |
| 데이터셋 | `olmo/data/synthmanip_dataset.py` | HDF5 궤적 로더 |
| 평가 설정 | `olmo/eval/configure_molmo_spaces.py` | 시뮬레이션 평가 구성 |
| 프리셋 | `olmo/data/synthmanip_presets.py` | 로봇별 카메라/액션 프리셋 |

### ActionExpert 상세

Flow Matching 기반 diffusion 모델로, LLM의 각 레이어 출력을 조건으로 받아 액션을 생성한다.

```
ActionExpert 구조:
├── Timestep Embedding: Sinusoidal + 2×Linear
├── Action Embedding: nn.Embedding(action_dim → 768)
├── Position Embedding: Learnable (1, max_horizon, 768)
├── State Encoder: Linear + LayerNorm (로봇 상태 입력)
└── Transformer Block ×32 (LLM 레이어 수와 동일):
    ├── Self-Attention (액션 시퀀스 내)
    ├── Cross-Attention (LLM 컨텍스트 참조)
    ├── MLP (4× expansion)
    └── AdaLN-Zero (timestep 조건부 modulation)
```

**기본 설정:**
- hidden_size: 768, num_heads: 8, num_layers: 32
- action_horizon: 16 (예측), n_action_steps: 8 (실행)
- flow_matching_num_steps: 10 (추론 시 ODE 적분 스텝)

### 학습 손실 함수 (Flow Matching)

```
1. t ~ Beta(1.0, 1.5) 샘플링 (0 ~ 0.999)
2. x_t = (1-t) × noise + t × target_action  (보간)
3. v_target = target_action - noise  (목표 velocity)
4. v_pred = ActionExpert(x_t, t, llm_states, robot_state)
5. loss = MSE(v_pred, v_target)
```

### 추론 (ODE Integration)

```
x_0 ~ N(0, I)  # 랜덤 노이즈에서 시작
for i in range(10):
    t = i / 10
    v = ActionExpert(x_t, t, llm_states, state)
    x_{t+dt} = x_t + (1/10) × v
→ x_1 = 최종 액션 시퀀스
```

---

## 3. 지원 로봇 및 태스크

### 로봇 비교

| 항목 | Franka (DROID) | RBY1 |
|------|---------------|------|
| 타입 | 정적 테이블탑 매니퓰레이터 | 모바일 듀얼암 휴머노이드 |
| 액션 차원 | 8 (arm 7 + gripper 1) | 19~29 (base 3 + arms 14 + grippers 2 + torso) |
| 카메라 | exo + wrist (2대) | wrist_r + head + wrist_l (3대) |
| 정책 주기 | 66ms (200ms with random cam) | 100ms |
| 모델 | allenai/MolmoBot-DROID | allenai/MolmoBot-RBY1Multitask |
| 태스크 | pick & place, color sorting | door opening, pick & place, multitask |

### 태스크 구성 방식

태스크는 `synthmanip_presets.py`에 프리셋으로 정의된다:

```python
# 예: Franka joint control
ACTION_SPECS = {
    "franka_joint": {"arm": 7, "gripper": 1},      # 8-DOF
    "RBY1_door_opening": {                           # 19-DOF
        "base": 3, "left_arm": 7, "left_gripper": 1,
        "right_arm": 7, "right_gripper": 1
    },
}

CAMERA_PRESETS = {
    "franka_droid": ["wrist_camera_zed_mini", "droid_shoulder_light_randomization"],
    "RBY1_right_arm": ["wrist_camera_r", "head_camera"],
}
```

**타당성 포인트:** 새로운 로봇 형태를 추가하려면 이 프리셋에 액션 스펙과 카메라 설정을 정의하면 된다.

---

## 4. 학습 인프라

### 하드웨어 요구사항

| 항목 | 권장 | 최소 |
|------|------|------|
| GPU | 8×H100 (80GB) | 가능: 소규모 GPU (A100 등) + 마이크로배칭 |
| 분산 학습 | torchrun + FSDP2 | 단일 노드부터 멀티노드까지 |
| 정밀도 | bfloat16 mixed precision | bfloat16 지원 GPU 필수 |
| 파일시스템 | 공유 파일시스템 (NFS 등) | `OLMO_SHARED_FS=1` 필요 |

### 학습 명령 예시

```bash
torchrun --nnodes=1 --nproc-per-node=8 --master_port=29401 \
  launch_scripts/train_molmobot.py <checkpoint> \
  --data_paths /data/franka_pick_place \
  --device_batch_size=32 \
  --global_batch_size=1024 \
  --seq_len=528 \
  --action_preset franka_joint \
  --camera_preset franka_droid
```

### 학습률 설정

| 컴포넌트 | 학습률 | Warmup |
|----------|--------|--------|
| LLM Backbone | 1e-5 | 2000 steps |
| ActionExpert | 1e-4 | 200 steps |
| Vision Backbone | 5e-6 | 200 steps |
| Connector | 5e-6 | 200 steps |

- Optimizer: AdamW (betas=[0.9, 0.95], weight_decay=0.0)
- Gradient clipping: max_norm=1.0
- 저장 주기: 2000 steps

### 배치 크기 계산

```
gradient_accumulation = global_batch / (device_batch × num_gpus)
예: 1024 / (32 × 8) = 4 accumulation steps
```

소규모 GPU에서는 `device_batch_size`를 줄이고 `gradient_accumulation`을 늘려서 동일한 `global_batch_size`를 유지할 수 있다.

### 분산 학습 스택

```
torchrun (launcher)
  └── torch.distributed (NCCL backend)
      └── FSDP2 (FullyShardedDataParallel v2)
          ├── 파라미터 분할 (FULL_SHARD)
          ├── 그래디언트 분할
          ├── 옵티마이저 상태 분할
          └── Activation Checkpointing
```

---

## 5. 시뮬레이션 환경

### 구성 요소

| 컴포넌트 | 출처 | 역할 |
|----------|------|------|
| MuJoCo | DeepMind | 물리 시뮬레이션 엔진 |
| molmo_spaces | Allen AI (GitHub) | 로봇/카메라 설정, 평가 프레임워크 |
| molmospaces-resources | Allen AI (GitHub) | 벤치마크 정의, 씬 에셋 |
| ProcTHOR 10K | Allen AI | 절차적 생성 주택 환경 (10,000개) |
| Objaverse | Allen AI | 3D 오브젝트 (그릇, 용기 등) |

### 평가 구성

```python
# olmo/eval/configure_molmo_spaces.py에서 정의
SynthVLAFrankaBenchmarkOriginalEvalConfig  # Franka 벤치마크
MolmoBotRBY1DoorEvalConfig                 # RBY1 문열기
MolmoBotRBY1PickPnPEvalConfig              # RBY1 픽앤플레이스

# 평가 실행
python launch_scripts/run_eval.py \
  --checkpoint_path <path> \
  --benchmark_path <benchmark_dir> \
  --eval_config_cls "olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig"
```

### 시뮬레이션 타이밍

| 로봇 | policy_dt | ctrl_dt | sim_dt |
|------|-----------|---------|--------|
| Franka | 66ms | - | MuJoCo default |
| RBY1 | 100ms | 20ms | 4ms |

---

## 6. 데이터 파이프라인

### HDF5 데이터 구조

```
{data_path}/{split}/house_*/*.h5
├── traj_0/
│   ├── obs/
│   │   ├── sensor_data/
│   │   │   ├── exo_camera_1 → "path/to/video.mp4"
│   │   │   └── wrist_camera → "path/to/video.mp4"
│   │   ├── extra/
│   │   │   └── object_image_points/  (선택)
│   │   └── state: (T, state_dim)
│   ├── actions/
│   │   └── {action_key}: JSON 인코딩
│   └── success: 궤적 성공 플래그
├── valid_traj_mask: boolean
└── stats/: 정규화 통계
```

### 데이터 처리 특징

| 기능 | 설명 |
|------|------|
| 정규화 | quantile 기반 (q01/q99), min-max, mean-std 지원 |
| 샘플링 | Grasp-aware 가중 샘플링 (잡기 동작 주변 과샘플링) |
| 비디오 디코딩 | decord (ffmpeg 기반) |
| 프롬프트 | 태스크 설명 텍스트 + 선택적 point prompt |
| 증강 | 카메라 랜덤화, fisheye 왜곡 시뮬레이션 |

### 데이터 다운로드

```bash
# HuggingFace에서 학습 데이터 다운로드
python bulk_download.py --config FrankaPickAndPlaceOmniCamConfig --part 1 --split all <data_root>
```

---

## 7. Sim-to-Real 전이

### 배포 방식

WebSocket 서버를 통해 실물 로봇에 정책을 서빙한다:

```bash
# 서버 시작
python launch_scripts/serve_molmo.py --hf-repo allenai/MolmoBot-DROID

# 또는 로컬 체크포인트
python launch_scripts/serve_molmo.py --local-path /path/to/checkpoint
```

### 안전 장치

| 장치 | 설명 |
|------|------|
| Joint delta clamping | 스텝당 최대 0.2 rad 제한 |
| Gripper 이진 제어 | threshold 기반 open/close (연속 제어 아님) |
| Action chunking | 16 steps 예측 중 8 steps만 실행 후 재추론 |
| Fisheye 보정 | GoPro 카메라 왜곡 보정 내장 |

### 실물 로봇 정책 흐름

```
실물 로봇 관측 (이미지 + qpos + 태스크 텍스트)
  → WebSocket → MolmoBot 추론 서버
  → 전처리 (정규화, 이미지 리사이즈)
  → 모델 추론 (flow matching ODE)
  → 후처리 (역정규화, delta clamping)
  → WebSocket → 로봇 컨트롤러
```

---

## 8. 커스터마이징 가이드 (새 로봇/태스크 적용)

### 새 로봇 추가 시 필요 작업

**Step 1: 프리셋 정의** (`olmo/data/synthmanip_presets.py`)

```python
# 액션 스펙 추가
ACTION_SPECS["my_robot"] = {"arm": 6, "gripper": 1}  # 7-DOF

# 카메라 프리셋 추가
CAMERA_PRESETS["my_robot_cams"] = ["wrist_cam", "overhead_cam"]

# 데이터 키 매핑
ACTION_DATASET_KEYS["my_robot"] = {"arm": "joint_pos", "gripper": "joint_pos"}
```

**Step 2: 시뮬레이션 데이터 생성**
- MuJoCo MJCF로 로봇 모델 정의
- molmo_spaces에 로봇 설정 클래스 추가
- 궤적 수집 → HDF5 포맷으로 저장

**Step 3: 학습 설정**
```bash
torchrun ... train_molmobot.py <molmo2_checkpoint> \
  --data_paths /data/my_robot_tasks \
  --action_move_groups arm gripper \
  --camera_names wrist_cam overhead_cam \
  --action_dim 7
```

**Step 4: 평가 설정** (`olmo/eval/configure_molmo_spaces.py`)
- 새 EvalConfig 클래스 정의
- 정책 래퍼 구현

### 수정이 필요한 파일 목록

| 파일 | 수정 내용 |
|------|----------|
| `olmo/data/synthmanip_presets.py` | 로봇 프리셋 추가 |
| `olmo/eval/configure_molmo_spaces.py` | 평가 설정 추가 |
| `olmo/eval/configure_real_robot.py` | 실물 배포 설정 (필요 시) |
| molmo_spaces (외부) | 로봇 MJCF, 카메라 설정, 씬 구성 |

---

## 9. 라이선스 및 의존성

### 라이선스

| 항목 | 라이선스 | 상업적 사용 |
|------|---------|-----------|
| MolmoBot 코드 | Apache 2.0 | 가능 |
| 모델 가중치 | HuggingFace 모델 카드 확인 필요 | 확인 필요 |
| MuJoCo | Apache 2.0 | 가능 |
| molmo_spaces | Allen AI 관리 (확인 필요) | 확인 필요 |

### 외부 의존성 리스크

| 의존성 | 리스크 수준 | 비고 |
|--------|-----------|------|
| PyTorch / torchrun | 낮음 | 업계 표준 |
| MuJoCo | 낮음 | DeepMind 오픈소스, 활발히 유지보수 |
| molmo_spaces | **중간** | Allen AI 내부 관리, 특정 커밋 고정 |
| molmospaces-resources | **중간** | Allen AI 내부 관리, 에셋 포함 |
| Molmo2-4B 기반 모델 | **중간** | Allen AI 생태계에 종속 |
| HuggingFace Hub | 낮음 | 모델/데이터 호스팅 |

---

## 10. 우리 로봇에 가상 데이터 파이프라인을 구축할 경우

> 강점/리스크 요약은 [00_interim_results.md](./00_interim_results.md)의 4절 참조. 여기서는 투자 규모만 정리한다.

### 투자 규모

| 단계 | 목표 | 예상 인프라 |
|------|------|------------|
| 현재: 검증 | 릴리즈 모델로 가상 데이터 학습의 sim 성능 확인 | GPU 1× (추론) |
| 다음: PoC | 우리 로봇 MJCF 구축 → 소규모 가상 데이터 생성 (1,000 궤적) → 파인튜닝 | GPU 4-8× A100/H100, 스토리지 500GB |
| 이후: 스케일업 | 대규모 가상 데이터 생성 → 풀 학습 → sim-to-real 전이 테스트 | GPU 8-16× H100, 스토리지 수 TB, 분산 데이터 생성 클러스터 |

> **참고:** 구체적 로드맵과 의사결정 기준은 [00_interim_results.md](./00_interim_results.md)의 Next Steps 참조.

---

## 부록: 로컬 데모 실행 결과

macOS M3 Pro CPU에서 동작 확인. 추론은 실시간 대비 ~190배 느림 (GPU 필수).
성능 수치 및 트러블슈팅은 [03_cpu_demo_guide.md](./03_cpu_demo_guide.md) 참조.

**체크포인트에서 확인된 특이점:**
- 토크나이저가 `Qwen/Qwen3-4B-Instruct-2507`에서 로드됨 (Molmo2의 LLM이 Qwen3 계열)
- `n_obs_steps=2` — 노트북 기본값(1)을 체크포인트 config가 override. MolmoBot-DROID는 2-frame 관측 모델

---

*본 문서는 MolmoBot 코드베이스 (commit 96fcc2b) 분석 및 로컬 데모 실행 기반으로 2026년 4월 6일 작성되었습니다.*
