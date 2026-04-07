# ACT 환경 통합 분석: MolmoBot RBY1 제로샷 테스트

> **날짜:** 2026-04-07
> **배경:** ACT (제조 공정 시뮬레이션, Isaac Sim) 환경에 MolmoBot의 RBY1 로봇을 로드하여 제로샷 pick-and-place 성능을 확인한다.
> **현재 상태:** ACT 환경에서 OpenPI (Pi 모델)로 추론 중. ROS 2 인터페이스 사용.

---

## 1. 개요

### 목표
MolmoBot이 MuJoCo에서 학습한 RBY1 pick-and-place 정책이, Isaac Sim 기반 ACT 환경에서 제로샷으로 작동하는지 확인.

### 핵심 구조
```
MolmoBot RBY1 로봇 (동일 로봇)
  학습: MuJoCo 시뮬레이터
  평가: Isaac Sim (ACT 환경)
  → sim-to-sim 전이 테스트
```

### 호환성 요약

| 항목 | MolmoBot 학습 환경 | ACT 평가 환경 | 호환성 |
|------|-------------------|-------------|--------|
| 로봇 | RBY1 | RBY1 (동일 로봇 로드) | **동일** |
| 관절 구성 | 20 DOF (multitask) | 20 DOF (동일) | **1:1 매핑** |
| 물리 엔진 | MuJoCo | PhysX (Isaac Sim) | **sim-to-sim 갭** |
| 렌더링 | PyOpenGL | RTX 레이트레이싱 | **시각 갭 있음** |
| 태스크 | pick-and-place | 핸드폰 → 캐비닛 슬롯 배치 | **동일 유형** |
| 인터페이스 | WebSocket (msgpack) | ROS 2 | **브릿지 필요** |

---

## 2. MolmoBot RBY1 사양

### 관절 구성 (RBY1_multitask, 20 DOF)

| 부위 | DOF | 제어 방식 |
|------|-----|----------|
| base (x, y, yaw) | 3 | delta |
| left_arm (7-DOF) | 7 | delta |
| left_gripper | 1 | absolute |
| right_arm (7-DOF) | 7 | delta |
| right_gripper | 1 | absolute |
| torso | 1 | absolute |

### 카메라 구성

| MolmoBot 카메라 | 위치 |
|----------------|------|
| `wrist_camera_r` | 우측 손목 |
| `wrist_camera_l` | 좌측 손목 |
| `head_camera` | 헤드 (GoPro) |

### 공개 체크포인트

| 체크포인트 | 태스크 | DOF |
|-----------|--------|-----|
| `allenai/MolmoBot-RBY1Multitask` | pick-and-place + 문 열기 | 20 |
| `allenai/MolmoBot-RBY1DoorOpening` | 문 열기 전용 | 19 |

pick-and-place 테스트에는 **`MolmoBot-RBY1Multitask`** 사용.

---

## 3. ACT 환경 카메라 매핑

| ACT 카메라 (Isaac Sim) | → MolmoBot 카메라 | 비고 |
|------------------------|-------------------|------|
| `right_wrist_camera_rgb` | `wrist_camera_r` | 직접 매핑 |
| `left_wrist_camera_rgb` | `wrist_camera_l` | 직접 매핑 |
| `lower_head_right_camera` 또는 `lower_head_left_camera` | `head_camera` | 둘 중 하나 선택 |

ACT 환경에 4개 카메라가 있으므로, 3개만 선택하여 MolmoBot에 전달.

---

## 4. Sim-to-Sim 갭 분석

MuJoCo에서 학습 → Isaac Sim에서 평가 시 발생할 수 있는 차이:

| 영역 | MuJoCo | Isaac Sim (PhysX) | 영향도 |
|------|--------|-------------------|--------|
| 접촉 모델 | 소프트 접촉, 임플리시트 | 리지드 접촉, PhysX solver | 중간 — 그래스프 성공률에 영향 가능 |
| 마찰 | MuJoCo 마찰 모델 | PhysX 마찰 모델 | 중간 — 미끄러짐 차이 |
| 렌더링 | PyOpenGL (기본) | RTX 레이트레이싱 | 낮음 — MolmoBot이 공격적 시각 DR로 학습되어 내성 있음 |
| 관절 제어 | MuJoCo actuator | Isaac Sim articulation controller | 낮음 — position 명령은 유사 |
| 중력/관성 | MuJoCo 기본값 | PhysX 기본값 | 낮음 — 큰 차이 없음 |
| 타이밍 | 100ms 정책 주기 | ROS 2 기반 (설정 가능) | 낮음 — 매칭 가능 |

**예상:** MolmoBot이 94,200개 환경에서 공격적 DR로 학습되었으므로, 시각 갭에 대한 내성이 높음. **주요 리스크는 접촉/마찰 차이로 인한 그래스프 실패.**

---

## 5. 구현 계획

### 5.1 Isaac Sim에 RBY1 로드

MolmoBot 레포에서 RBY1 URDF/모델 파일을 확보하여 Isaac Sim USD로 변환:

```
MolmoBot/.venv/lib/.../molmo_spaces/
  └── assets/robots/rby1m/  ← RBY1 모델 파일 위치 (확인 필요)
```

Isaac Sim에서:
1. URDF Importer로 RBY1 로드
2. Articulation Controller 설정
3. 카메라 3개 배치 (wrist_r, wrist_l, head 위치에)
4. ACT 태스크 장면 (핸드폰 + 캐비닛) 배치

### 5.2 ROS 2 ↔ MolmoBot 브릿지

```
Isaac Sim (ROS 2)                         MolmoBot 서버
  /joint_states (20 DOF) ──┐              ┌── ws://localhost:8000
  /right_wrist_camera_rgb ─┤              │
  /left_wrist_camera_rgb  ─┤→ [브릿지] →─┤── obs (msgpack)
  /lower_head_*_camera    ─┤              │
                           │← [브릿지] ←─┤── action (msgpack)
  /joint_command ←─────────┘              └──
```

**브릿지 노드 역할:**

1. **관측 수집** (ROS 2 → MolmoBot obs):
   - 3개 카메라 이미지 구독 → RGB numpy 배열로 변환
   - `/joint_states`에서 20 DOF 관절 값 추출 → move group별로 분리
   - 태스크 텍스트 설정 (예: "pick the phone and place it in the slot")

2. **액션 전달** (MolmoBot action → ROS 2):
   - delta 액션을 현재 관절값에 더하여 absolute 명령으로 변환
   - 그리퍼 임계값 처리 (±100 → Isaac Sim 그리퍼 범위)
   - `/joint_command`로 퍼블리시

### 5.3 관절 매핑 (1:1)

```python
# Isaac Sim joint_states → MolmoBot obs["qpos"]
obs["qpos"] = {
    "base":          joint_states[base_indices],          # (3,)
    "left_arm":      joint_states[left_arm_indices],      # (7,)
    "left_gripper":  joint_states[left_gripper_indices],  # (1,)
    "right_arm":     joint_states[right_arm_indices],     # (7,)
    "right_gripper": joint_states[right_gripper_indices], # (1,)
}

# MolmoBot action → Isaac Sim joint commands
joint_cmd = current_joint_states.copy()
joint_cmd[left_arm_indices]  += action["left_arm"]       # delta
joint_cmd[right_arm_indices] += action["right_arm"]      # delta
joint_cmd[base_indices]      += action["base"]           # delta
joint_cmd[left_gripper_indices]  = action["left_gripper"]   # absolute
joint_cmd[right_gripper_indices] = action["right_gripper"]  # absolute
joint_cmd[torso_indices]         = action["torso"]          # absolute
```

관절 이름 → 인덱스 매핑은 Isaac Sim에 RBY1을 로드한 후 확인하여 설정.

### 5.4 MolmoBot 서버 실행

```bash
# GPU 서버에서 실행
python launch_scripts/serve_molmobot_rby1_multitask.py \
    --hf-repo allenai/MolmoBot-RBY1Multitask \
    --task_type pick_pnp \
    --port 8000
```

---

## 6. 예상 레이턴시

| 구간 | 시간 |
|------|------|
| Isaac Sim 렌더링 + ROS 2 퍼블리시 | ~16ms (60Hz) |
| ROS 2 전송 (로컬) | ~1-5ms |
| 브릿지 변환 | ~1ms |
| MolmoBot 추론 (GPU) | **~100-200ms** (병목) |
| 명령 적용 | ~1-5ms |
| **전체 루프** | **~120-220ms (~5-8Hz)** |

MolmoBot RBY1은 100ms 정책 주기로 설계되었으므로 호환.

---

## 7. 제로샷 성공 가능성 평가

### 유리한 요인
- **동일 로봇, 동일 DOF** — 체형 매핑 문제 없음
- **동일 태스크 유형** — pick-and-place
- **공격적 DR로 학습** — 94,200개 환경, 시각 변이에 대한 내성
- **카메라 구성 호환** — 3개 카메라 위치 매칭 가능

### 리스크 요인
- **물리 엔진 차이** — MuJoCo vs PhysX 접촉/마찰 모델 차이로 그래스프 실패 가능
- **객체 차이** — MolmoBot은 Objaverse 객체로 학습, ACT는 특정 핸드폰 모델
- **장면 레이아웃** — 캐비닛 슬롯 배치가 학습 데이터와 다를 수 있음
- **렌더링 차이** — RTX vs PyOpenGL (DR로 완화 가능)

### 예상 결과
- **낙관적:** 50-70% 성공률 — 팔 동작은 전이되나 일부 그래스프 실패
- **현실적:** 20-50% — 물리 차이로 인한 그래스프/배치 정밀도 저하
- **비관적:** <20% — 접촉 모델 차이가 너무 커서 대부분 실패

어느 경우든, **MolmoBot의 sim-to-sim 전이 능력에 대한 데이터 포인트**를 얻을 수 있어 가치 있음.

---

## 8. 다음 단계

| 순서 | 작업 | 담당 | 예상 기간 |
|------|------|------|----------|
| 1 | RBY1 URDF를 Isaac Sim USD로 변환 및 로드 | 팀 | 1-2일 |
| 2 | 카메라 3개 배치 (wrist_r, wrist_l, head) | 팀 | 0.5일 |
| 3 | ACT 장면 구성 (핸드폰 + 캐비닛) | 팀 | 1일 |
| 4 | ROS 2 ↔ MolmoBot WebSocket 브릿지 노드 작성 | 구현 | 1-2일 |
| 5 | MolmoBot 서버 실행 + 연결 테스트 | 구현 | 0.5일 |
| 6 | 제로샷 테스트 실행 및 결과 수집 | 공동 | 1일 |
| **합계** | | | **~5-7일** |

### 팀에서 확인 필요
- RBY1 URDF 파일 위치 (MolmoBot 레포 내 또는 molmo_spaces 패키지 내)
- Isaac Sim에 RBY1을 로드한 후 관절 이름 → 인덱스 매핑
- ACT 환경의 현재 ROS 2 토픽 이름 및 메시지 타입

---

## 9. MolmoBot 서빙 참고

```bash
# RBY1 멀티태스크 (pick-and-place)
python launch_scripts/serve_molmobot_rby1_multitask.py \
    --hf-repo allenai/MolmoBot-RBY1Multitask \
    --task_type pick_pnp

# 서버 상태 확인
curl http://localhost:8000/healthz
```

### WebSocket 프로토콜 요약

**관측 전송 (클라이언트 → 서버):**
```python
obs = {
    "wrist_camera_r": np.ndarray(H, W, 3, uint8),
    "head_camera": np.ndarray(H, W, 3, uint8),
    "wrist_camera_l": np.ndarray(H, W, 3, uint8),
    "qpos": {
        "base": np.ndarray(3,),
        "left_arm": np.ndarray(7,),
        "left_gripper": np.ndarray(1,),
        "right_arm": np.ndarray(7,),
        "right_gripper": np.ndarray(1,),
        "torso": np.ndarray(6,),
    },
    "task": "pick the phone and place it in the slot",
}
ws.send(msgpack_numpy.packb(obs))
```

**액션 수신 (서버 → 클라이언트):**
```python
action = msgpack_numpy.unpackb(ws.recv())
# action = {
#     "base": np.ndarray(3,),          # delta
#     "left_arm": np.ndarray(7,),      # delta
#     "right_arm": np.ndarray(7,),     # delta
#     "left_gripper": np.ndarray(1,),  # absolute
#     "right_gripper": np.ndarray(1,), # absolute
#     "torso": np.ndarray(1,),         # absolute
# }
```
