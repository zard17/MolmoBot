# GPU 벤치마크 로그 및 평가 워크플로우

> Ubuntu / A100 환경에서 가상 데이터 학습 정책(MolmoBot)의 시뮬레이션 성능을 측정한 기록

---

## 평가 워크플로우

가상 데이터 학습 정책의 유효성 검증은 4단계로 구성된다.

### Phase 0. 노트북 sanity check (선택)

시각적 확인용. [`demo_policy.ipynb`](../MolmoBot/demo_policy.ipynb)를 열어 체크포인트 로드, 렌더링, 액션 생성이 정상인지 확인한다. 성능 판단 시그널로 사용하지 않는다.

### Phase 1. Sim 벤치마크 smoke test

```bash
cd MolmoBot/MolmoBot
uv sync --extra eval
. .venv/bin/activate

python launch_scripts/run_feasibility.py benchmark-smoke \
  --benchmark-path /path/to/benchmark_slice \
  --output-dir /tmp/molmobot_franka_smoke
```

기본값: checkpoint `allenai/MolmoBot-DROID`, eval config `FrankaState8ClampAbsPosConfig`.
선택 플래그: `--local-path`, `--use-filament`, `--manifest-path`.

### Phase 2. Real policy 서버 구동

> **주의:** 아래는 Franka/DROID 기준 예시. 타겟 로봇(RBY1)에서는 `--hf-repo allenai/MolmoBot-RBY1Multitask` 및 해당 카메라/액션 설정 사용.

```bash
PYTHONPATH=. python launch_scripts/serve_molmo.py \
  --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
```

카메라: `exo_camera_1` + `wrist_camera`. Franka joint-position 실행.

### Phase 3. Real-robot 태스크 실행

```bash
cd MolmoBot/robot_eval
conda activate molmobot
python scripts/droid/run_feasibility_trials.py \
  --robot-host <nuc_ip> \
  --wrist-camera-id <wrist_id> \
  --exo-camera-id <exo_id> \
  --tasks-file config/feasibility_tasks_franka.txt \
  --output-dir outputs/feasibility_run_01
```

`--dry-run`으로 명령어 미리 확인 가능. `suite_manifest.json` + `summary.json` 자동 생성.

> RBY1의 경우 카메라 ID와 tasks-file을 RBY1 구성에 맞게 변경 필요.

### 결과 해석 기준

- Sim smoke → 환경이 클린하게 동작하는가? 가상 데이터로 학습한 정책이 유의미한 rollout을 생성하는가?
- Real suite → 가상 데이터 학습 정책이 실물에서 zero-shot 성공을 보이는가? 실패 시 camera/action/timing/task 중 어느 gap이 지배적인가?

---

## 실행 기록 (2026-04-06, Ubuntu A100)

### 환경 설정

```bash
# 의존성 설치
python -m pip install uv
cd /MolmoBot/MolmoBot
uv sync --extra eval

# 렌더링 수정 (EGL)
apt-get update
apt-get install -y libegl1 libgl1 libopengl0 libglvnd0 libosmesa6 libosmesa6-dev
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### 노트북 sanity check

`demo_policy.ipynb` 동등 스크립트 실행:
- Franka scene 빌드 → exo + wrist 카메라 렌더 → `RealRobotVLAPolicy` 로드 → 다수 step 액션 생성
- 결과: `sanity_ok`

### Smoke 벤치마크 (1 에피소드)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path ckpts/molmobot/MolmoBot-DROID \
  --benchmark-path .venv/lib/python3.11/site-packages/assets/benchmarks/molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_1ep_benchmark \
  --output-dir /tmp/molmobot_franka_smoke_1ep \
  --task-horizon 600 \
  --notes '1-episode built-in smoke benchmark'
```

| 항목 | 결과 |
|------|------|
| 성공 | 1 |
| 전체 | 1 |
| 성공률 | **100%** |
| 태스크 | Pick up the yellow handheld gps with antenna and place it in or on the shallow round wooden bowl with grain |

### Pick-only DROID mini (10 에피소드)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path ckpts/molmobot/MolmoBot-DROID \
  --benchmark-path /tmp/molmobot_bench_subsets/franka_pick_droidmini_10ep \
  --output-dir /tmp/molmobot_franka_pick10 \
  --task-horizon 200 \
  --notes '10-episode DROID mini pick subset; shortened horizon for sim-only throughput'
```

| 항목 | 결과 |
|------|------|
| 완료 house | 7 |
| 스킵 house | 0 |
| 성공 | 7 |
| 전체 | 10 |
| 성공률 | **70%** |

해석: 현재 최강 sim 시그널. horizon 200 단축으로 논문 재현 수치는 아님.

### Pick-and-place DROID mini (10 에피소드)

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path ckpts/molmobot/MolmoBot-DROID \
  --benchmark-path /tmp/molmobot_bench_subsets/franka_pnp_droidmini_10ep \
  --output-dir /tmp/molmobot_franka_pnp10 \
  --task-horizon 300 \
  --notes '10-episode DROID mini pick-and-place subset; shortened horizon for sim-only throughput'
```

| 항목 | 결과 |
|------|------|
| 완료 house | 0 |
| 스킵 house | 10 |
| 성공률 | **0% (미실행)** |
| 실패 원인 | `Expected max_place_receptacle_pos_displacement=0.15, got 0.05` |

해석: 벤치마크/config 호환성 이슈. MolmoBot 정책 성능과 무관. 해결 전까지 pick-only 결과를 기준으로 사용.

### RBY1 door smoke (normalized 1 에피소드)

released RBY1 door benchmark는 legacy task class path와 missing `task_type` 때문에 그대로는 실행되지 않았다. 아래 순서로 normalize 후 `MolmoBot-RBY1Multitask` smoke를 수행했다.

```bash
cd /workspace/MolmoBot/MolmoBot

./.venv/bin/python launch_scripts/normalize_rby1_benchmark.py \
  --input-dir /root/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260325_1/ithor/rby1_bennchmarks/door_opening_benchmark \
  --output-dir /tmp/rby1_door_smoke_norm_1ep \
  --episode-idx 0

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu \
./.venv/bin/python -m molmo_spaces.evaluation.eval_main \
  olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorEvalConfig \
  --benchmark_dir /tmp/rby1_door_smoke_norm_1ep \
  --checkpoint_path /workspace/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-RBY1Multitask \
  --task_horizon_steps 300 \
  --output_dir /tmp/molmobot_rby1_door_smoke_20260407_impl1 \
  --num_workers 1 \
  --no_wandb
```

| 항목 | 결과 |
|------|------|
| 성공 | 1 |
| 전체 | 1 |
| 성공률 | **100%** |
| 태스크 | Pull the door open |
| 비고 | 기존 `19D` door alias 대신 multitask `20D action / 22D state` contract로 정렬, RBY1 grouped action path 정상 통과 |

해석: RBY1 path는 더 이상 startup/config 수준에서 막히지 않는다. normalized benchmark 기준으로 first rollout부터 episode save까지 end-to-end 완료했다. 다음 검증은 1ep smoke가 아니라 small-slice door/open 및 pick-pnp coverage 확대다.

### RBY1 door small slice (reusable workflow, 3 에피소드)

`run_feasibility.py`에 RBY1 benchmark normalize + deterministic slice export를 붙인 뒤, 같은 checkpoint로 3ep door slice를 반복 가능한 방식으로 실행했다.

```bash
cd /workspace/MolmoBot/MolmoBot

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu \
./.venv/bin/python launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path /workspace/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-RBY1Multitask \
  --benchmark-path /root/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260325_1/ithor/rby1_bennchmarks/door_opening_benchmark \
  --eval-config-cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorPlusOpenEvalConfig \
  --normalize-rby1-benchmark \
  --slice-count 3 \
  --output-dir /tmp/molmobot_rby1_doorplusopen_small_20260407 \
  --num-workers 1
```

| 항목 | 결과 |
|------|------|
| 성공 | 1 |
| 전체 | 3 |
| 성공률 | **33.3%** |
| 태스크 | door opening only (released door benchmark first-3) |
| 비고 | normalized benchmark + manifest + eval output이 한 workflow로 재현됨 |

해석: RBY1 door path는 이제 ad hoc command가 아니라 reusable workflow로 재현 가능하다. 성능 시그널은 `1/3`로 non-zero이지만 아직 sample 수가 작다.

### RBY1 pnp small slice (reusable workflow, blocker 확인)

같은 reusable workflow를 `MolmoBotRBY1PickPnPEvalConfig`와 released `pnp_benchmark` first-3 slice에 적용했다.

```bash
cd /workspace/MolmoBot/MolmoBot

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cpu \
./.venv/bin/python launch_scripts/run_feasibility.py benchmark-smoke \
  --local-path /workspace/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-RBY1Multitask \
  --benchmark-path /root/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260325_1/procthor-objaverse/rby1_bennchmarks/pnp_benchmark \
  --eval-config-cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1PickPnPEvalConfig \
  --normalize-rby1-benchmark \
  --slice-count 3 \
  --task-horizon 400 \
  --output-dir /tmp/molmobot_rby1_pnp_small_20260407 \
  --num-workers 1
```

| 항목 | 결과 |
|------|------|
| 성공 | 0 |
| 전체 | 0 |
| 성공률 | **0.0% (미실행)** |
| 실패 단계 | task sampling |
| 실패 원인 | `Expected max_place_receptacle_pos_displacement=0.15, got 0.1` |

해석: reusable workflow 자체는 정상 동작했다. pnp는 policy rollout 전 단계에서 released benchmark와 current config 간 parameter mismatch가 재현성 있게 드러난 상태다. 다음 수정 대상은 workflow가 아니라 pnp task/config compatibility다.

---

## 출력 경로

| 항목 | 경로 |
|------|------|
| Smoke manifest | `/tmp/molmobot_franka_smoke_1ep/feasibility_manifest.json` |
| Smoke log | `/tmp/molmobot_franka_smoke_1ep/FrankaState8ClampAbsPosConfig/20260406_162001/running_log.log` |
| Pick-only manifest | `/tmp/molmobot_franka_pick10/feasibility_manifest.json` |
| Pick-only log | `/tmp/molmobot_franka_pick10/FrankaState8ClampAbsPosConfig/20260406_164652/running_log.log` |
| PnP manifest | `/tmp/molmobot_franka_pnp10/feasibility_manifest.json` |
| PnP log | `/tmp/molmobot_franka_pnp10/FrankaState8ClampAbsPosConfig/20260406_164116/running_log.log` |
| RBY1 smoke log | `/tmp/molmobot_rby1_door_smoke_20260407_impl1/MolmoBotRBY1DoorEvalConfig/20260407_002159/running_log.log` |
| RBY1 smoke artifacts (repo copy) | `docs/artifacts/molmobot_feasibility/rby1_door_smoke_20260407/` |
| RBY1 door small-slice manifest | `/tmp/molmobot_rby1_doorplusopen_small_20260407/feasibility_manifest.json` |
| RBY1 door small-slice log | `/tmp/molmobot_rby1_doorplusopen_small_20260407/MolmoBotRBY1DoorPlusOpenEvalConfig/20260407_011148/running_log.log` |
| RBY1 door small-slice artifacts (repo copy) | `docs/artifacts/molmobot_feasibility/rby1_doorplusopen_small_20260407/` |
| RBY1 pnp blocker manifest | `/tmp/molmobot_rby1_pnp_small_20260407/feasibility_manifest.json` |
| RBY1 pnp blocker log | `/tmp/molmobot_rby1_pnp_small_20260407/MolmoBotRBY1PickPnPEvalConfig/20260407_013852/running_log.log` |
| RBY1 pnp blocker artifacts (repo copy) | `docs/artifacts/molmobot_feasibility/rby1_pnp_blocker_20260407/` |

---

## 주의사항

- Objaverse 에셋 버전 불일치 경고 발생 (기능 영향 없음, 정식 재현 시 주의)
- JAX는 현재 CPU-only. `jax-cuda12-plugin[with-cuda]==0.6.2` 설치로 CUDA 전환 가능 (dry-run 확인됨)
- Pick-only 10ep 결과는 horizon 200 단축 → throughput 우선 feasibility 시그널로 해석

---

*기록일: 2026-04-07 / 환경: Ubuntu 22.04, NVIDIA A100 SXM 80GB, Python 3.11*
