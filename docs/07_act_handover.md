# ACT Integration Handover: MolmoBot RBY1 Zero-Shot Test

> **날짜:** 2026-04-07
> **목적:** ACT 팀이 Isaac Sim 환경에서 MolmoBot RBY1 정책의 제로샷 테스트를 수행할 수 있도록 필요한 정보와 코드를 전달.

---

## 1. MolmoBot 측 준비 완료 사항

### 1a. WebSocket 클라이언트 (`olmo/eval/websocket_client.py`)

MolmoBot 정책 서버에 연결하는 동기식 Python 클라이언트:

```python
from olmo.eval.websocket_client import MolmoBotClient, build_rby1_obs

with MolmoBotClient(host="<gpu-server>", port=8000) as client:
    print(client.metadata)  # {"model_name": "...", ...}

    obs = build_rby1_obs(
        head_camera=img_head,           # (H, W, 3) uint8
        wrist_camera_l=img_left,        # (H, W, 3) uint8
        wrist_camera_r=img_right,       # (H, W, 3) uint8
        base=np.zeros(3),               # (x, y, yaw)
        left_arm=np.zeros(7),           # 7-DOF
        left_gripper=np.zeros(1),       # 1-DOF
        right_arm=np.zeros(7),          # 7-DOF
        right_gripper=np.zeros(1),      # 1-DOF
        torso=np.zeros(6),             # 6-DOF raw (서버가 [1,2,3] 인덱스 추출)
        task="pick the phone and place it in the slot",
    )
    action = client.get_action(obs)
    # action["base"]          → (3,) delta
    # action["left_arm"]      → (7,) delta
    # action["right_arm"]     → (7,) delta
    # action["left_gripper"]  → (1,) absolute ±100
    # action["right_gripper"] → (1,) absolute ±100
    # action["torso"]         → (1,) absolute
```

- **프로토콜:** WebSocket + msgpack-numpy (바이너리, 고효율)
- **리셋:** `client.reset()` → 연결 재수립, 서버 정책 상태 초기화
- **테스트:** `python -m olmo.eval.test_websocket_client --mock` (GPU/서버 없이 로컬 검증)
- **16개 유닛 테스트 통과:** `pytest olmo/eval/tests/test_websocket_client.py`

### 1b. 정책 서버 실행 방법

```bash
# GPU 서버에서:
cd MolmoBot
python launch_scripts/serve_molmobot_rby1_multitask.py \
    --hf-repo allenai/MolmoBot-RBY1Multitask \
    --task_type pick_pnp \
    --port 8000

# 헬스 체크:
curl http://<gpu-server>:8000/healthz
```

### 1c. ACT 브릿지 스켈레톤 (`act_bridge/`)

| 파일 | 용도 |
|------|------|
| `config.py` | 카메라 토픽, 관절 이름, 그리퍼 범위 매핑 |
| `bridge_node.py` | ROS 2 노드 스켈레톤 (subscriber, publisher, timer) |
| `isaac_sim_helpers.py` | Isaac Sim 씬 설정 스텁 함수 |
| `README.md` | 아키텍처 다이어그램 및 퀵스타트 |

---

## 2. ACT 팀 작업 목록

### 2a. RBY1 URDF 확보 및 로드

`molmo_spaces` 에셋 번들에 **Isaac Sim 전용 URDF + OBJ 메쉬**가 이미 포함되어 있음.

**다운로드 방법** (MolmoBot venv에서):

```python
# molmospaces_resources가 자동으로 다운로드 + 캐시
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

# 또는 직접 캐시 경로 확인:
#   ~/.cache/molmo-spaces-resources/robots/rby1m/20251224/
```

**Isaac Sim 전용 URDF 경로:**
```
~/.cache/molmo-spaces-resources/robots/rby1m/20251224/curobo_config/urdf/model_holobase_isaac/
├── model_holobase_isaac.urdf   # Isaac Sim용 URDF (38KB)
├── base_spheres.yaml           # CuRobo 충돌 구
└── meshes/                     # 31개 .obj 메쉬 파일
```

**일반 URDF와 Isaac 버전 차이점:**
- 메쉬 포맷: `.dae` (Collada) → `.obj` (Isaac Sim 호환)
- world 링크: MuJoCo용 5mm 높이 오프셋 제거
- 관절 이름/구조/관성값: **동일**

이 URDF를 Isaac Sim URDF Importer로 바로 로드하면 됨. 관절 이름이 MolmoBot 학습 환경과 동일하므로 `config.py` 수정 불필요.

### 2b. MuJoCo 관절 이름 참조 (확인 완료)

`molmo_spaces/robots/robot_views/rby1_view.py` 기반:

| 그룹 | MuJoCo 관절 이름 | DOF |
|------|-----------------|-----|
| base | `base_x`, `base_y`, `base_theta` | 3 |
| left_arm | `left_arm_0` ~ `left_arm_6` | 7 |
| left_gripper | `gripper_finger_l1`, `gripper_finger_l2` (커플드) | 1 (제어용) |
| right_arm | `right_arm_0` ~ `right_arm_6` | 7 |
| right_gripper | `gripper_finger_r1`, `gripper_finger_r2` (커플드) | 1 (제어용) |
| torso | `torso_0` ~ `torso_5` | 6 (raw), 3 (정책이 [1,2,3] 추출) |
| head | `head_0`, `head_1` | 2 (정책 미사용) |

### 2c. 카메라 설정

MolmoBot이 기대하는 3개 카메라:

| MolmoBot 키 | 위치 | ROS 2 토픽 (제안) |
|------------|------|------------------|
| `wrist_camera_r` | 우측 손목 | `/rby1/right_wrist_camera/image_raw` |
| `wrist_camera_l` | 좌측 손목 | `/rby1/left_wrist_camera/image_raw` |
| `head_camera` | 머리 | `/rby1/head_camera/image_raw` |

- 해상도: 제약 없음 (서버가 내부적으로 리사이즈)
- 포맷: RGB uint8
- 위치: 실제 RBY1 카메라 마운트 위치와 가능한 한 동일하게

### 2d. 브릿지 노드 완성

`act_bridge/bridge_node.py`의 TODO를 채워야 함:
1. ROS 2 import 활성화 (`rclpy`, `sensor_msgs`, `cv_bridge`)
2. `Node` 상속
3. Subscriber/Publisher/Timer 생성
4. `_camera_callback`: `cv_bridge.imgmsg_to_cv2(msg, "rgb8")`
5. `_joint_state_callback`: `msg.name` → `msg.position` 매핑
6. `_publish_joint_command`: JointState 메시지 구성 및 퍼블리시

### 2e. 액션 변환 로직 (이미 구현됨)

`bridge_node.py`의 `_action_to_joint_command()`과 `_remap_gripper()`는 이미 구현되어 있음:
- **delta 그룹** (base, left_arm, right_arm): `target = current + delta`
- **absolute 그룹** (gripper): MolmoBot ±100 → Isaac Sim 미터 단위로 리매핑
- **torso**: absolute 값 그대로 전달

---

## 3. 통합 체크리스트

```
[ ] molmo_spaces 캐시에서 Isaac URDF 복사 (섹션 2a 참조)
[ ] Isaac Sim URDF Importer로 로드 → USD 변환
[ ] 관절 이름 확인 (config.py와 일치하는지 — 변경 불필요할 것으로 예상)
[ ] 카메라 3개 배치 (wrist_r, wrist_l, head)
[ ] ROS 2 카메라 퍼블리셔 설정
[ ] ROS 2 JointState 퍼블리셔 설정
[ ] bridge_node.py의 ROS 2 코드 활성화
[ ] MolmoBot 서버 실행 (GPU 서버)
[ ] 클라이언트 연결 테스트 (test_websocket_client --host ...)
[ ] bridge_node 실행 → 데이터 플로우 확인 (rqt, rviz)
[ ] 제로샷 pick-and-place 테스트 실행
[ ] 결과 수집 (성공률, 실패 모드 분류)
```

---

## 4. 예상 이슈 및 대응

| 이슈 | 대응 |
|------|------|
| Isaac Sim 관절 이름이 MuJoCo와 다름 | `config.py`의 `joint_groups` 업데이트 |
| 그래스프 실패 (물리 엔진 차이) | PhysX 마찰/접촉 파라미터 튜닝 |
| 카메라 시점 차이 | DR로 학습되어 어느 정도 내성 있음, 필요시 위치 조정 |
| 네트워크 레이턴시 | 같은 네트워크 내 배치, 추론 ~100-200ms가 지배적 |
| 그리퍼 범위 불일치 | `config.py`의 `gripper_sim_open/close` 값 조정 |

---

## 5. 연락처

MolmoBot 측 질문은 이 레포의 이슈로 등록하거나 직접 연락 부탁드립니다.
