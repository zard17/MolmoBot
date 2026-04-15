# RBY1 Freeze Benchmark

## Benchmark 구조

`generate_rby1_pickpnp_benchmark.py`로 salt shaker pick-and-place 벤치마크를 생성한다.

- 5개 에피소드, 각각 다른 위치에 salt shaker 배치
- 커스텀 책상 씬 (`benchmarks/rby1_pickpnp_benchmark/custom_scene.xml`)
- 물체는 팔 범위 내에 배치 (frozen base로도 도달 가능한 위치)

생성된 벤치마크: `benchmarks/rby1_pickpnp_benchmark/benchmark.json`

## 정책 수정사항

### 1. `clamp_gripper` 버그 수정

`SynthVLAPolicy.__init__`에서 `self.clamp_gripper`를 설정하지 않았으나, 하위 클래스(`MolmoBotRBY1DoorOpeningPolicy`, `MolmoBotRBY1MultitaskPolicy`)에서 `self.clamp_gripper`를 참조하여 `AttributeError` 발생.

```python
# 수정: getattr로 안전하게 읽기
self.clamp_gripper = getattr(config.policy_config, "clamp_gripper", True)
self.gripper_representation_count = getattr(config.policy_config, "gripper_representation_count", 1)
```

Pick+pnp에서는 `clamp_gripper=False`로 설정하여 gripper 값을 이진화하지 않고 raw 값을 그대로 전달한다.

### 2. `relative_max_joint_delta` null guard

기존 코드는 `relative_max_joint_delta`가 `None`일 때 delta scaling 로직에서 크래시 발생. `None` 체크 추가:

```python
if self.relative_max_joint_delta is not None and "arm" in action:
    # delta scaling 로직
```

### 3. `SynthVLARBY1PolicyConfig` 필드 추가

`states_mode`, `relative_max_joint_delta` 필드가 누락되어 있어 추가:

```python
states_mode: str = "cross_attn"
relative_max_joint_delta: list[float] | None = None
```

## Freeze 메커니즘

`MolmoBotRBY1PickPnPFrozenBasePolicy`는 모델 inference 후 base action을 0으로 설정한다. 모델은 여전히 base action을 예측하지만, 실행 전에 제거된다.

```python
def _populate_action_buffer(self, observation):
    super()._populate_action_buffer(observation)
    if self.freeze_base:
        for action in self.action_buffer:
            if "base" in action:
                action["base"] = np.zeros_like(action["base"])
```

두 조건(default vs frozen)을 같은 벤치마크에서 비교하여 mobile base의 필요성을 평가한다.

## 실행 방법

### Eval (molmo_spaces 시뮬레이션)

```bash
bash scripts/run_rby1_freeze_test.sh
```

### Serving (실제 로봇 / WebSocket)

```bash
PYTHONPATH=. python launch_scripts/serve_molmobot_rby1_multitask.py \
    --task_type pick_pnp \
    --freeze-base \
    --hf_repo allenai/MolmoBot-RBY1Multitask
```

`--freeze-base` 플래그로 base action을 비활성화한다.
