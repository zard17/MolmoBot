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

---

## 주의사항

- Objaverse 에셋 버전 불일치 경고 발생 (기능 영향 없음, 정식 재현 시 주의)
- JAX는 현재 CPU-only. `jax-cuda12-plugin[with-cuda]==0.6.2` 설치로 CUDA 전환 가능 (dry-run 확인됨)
- Pick-only 10ep 결과는 horizon 200 단축 → throughput 우선 feasibility 시그널로 해석

---

*기록일: 2026-04-06 / 환경: Ubuntu, NVIDIA A100, Python 3.11*
