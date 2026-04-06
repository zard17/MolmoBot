# Minimal Benchmark 실행 가이드

이 가이드는 MolmoBot의 minimal_benchmark를 로컬 환경에서 실행하는 방법을 설명합니다.

## 빠른 시작

```bash
# 1. 가상환경 활성화
cd /home/youngsun/simul/MolmoBot/MolmoBot
source .venv/bin/activate

# 2. CPU 실행 설정 (16GB GPU인 경우)
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

# 3. 벤치마크 실행 (첫 실행 시 에셋 자동 다운로드)
python run_eval_with_glfw.py \
    --checkpoint_path /home/youngsun/simul/MolmoBot-DROID/ \
    --benchmark_path ./benchmarks/minimal_benchmark \
    --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \
    --task_horizon 600 \
    --output_dir ./minimal_eval_results

# 4. 결과 확인
ls -la minimal_eval_results/FrankaState8ClampAbsPosConfig/
```

---

## 목차

1. [사전 준비](#사전-준비)
2. [리소스 다운로드](#리소스-다운로드)
3. [벤치마크 실행](#벤치마크-실행)
4. [결과 확인](#결과-확인)
5. [문제 해결](#문제-해결)
6. [참고 정보](#참고-정보)

---

## 사전 준비

### 환경 요구사항

- **Python**: 3.8+
- **GPU 메모리**: 
  - **16GB GPU**: ❌ 부족함 (CUDA OOM 오류 발생)
  - **20GB+ GPU**: ✅ 권장 (GPU 실행 시)
- **CPU 메모리**: 32GB 이상 (CPU 실행 시)
- **디스크 공간**: 10GB 이상 (리소스 캐시용)

**⚠️ 중요**: MolmoBot-DROID 모델은 약 14.67GB의 GPU 메모리를 필요로 합니다. 16GB GPU에서는 OOM(Out of Memory) 오류가 발생하므로 **CPU 실행을 권장합니다.**

### 의존성 설치

```bash
cd MolmoBot/MolmoBot
uv sync --extra eval
source .venv/bin/activate
```

### 모델 체크포인트 준비

```bash
# HuggingFace에서 다운로드
pip install huggingface-cli
huggingface-cli download allenai/MolmoBot-DROID --local-dir /path/to/MolmoBot-DROID

# 또는 로컬에 있는 체크포인트 경로 사용
# 예: /home/youngsun/simul/MolmoBot-DROID
```

---

## 리소스 다운로드

MolmoSpaces는 필요한 에셋(iTHOR 장면, THOR 객체)을 **자동으로 다운로드**합니다.

### 자동 다운로드 방식

벤치마크를 처음 실행할 때, 필요한 에셋이 자동으로 다운로드됩니다:

```bash
python run_eval_with_glfw.py \
    --checkpoint_path /home/youngsun/simul/MolmoBot-DROID/ \
    --benchmark_path ./benchmarks/minimal_benchmark \
    --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \
    --task_horizon 600 \
    --output_dir ./minimal_eval_results
```

첫 실행 시 다음이 표시됩니다:
```
Extracting:  35%|███▌      | 6/17 [00:05<00:08,  1.27it/s]
```

**중요**: 별도의 환경 변수(`MLSPACES_FORCE_INSTALL`, `MLSPACES_DOWNLOAD_EXTRACT_ALL_SCENES_OBJECTS_GRASPS`) 설정이 필요하지 않습니다.

### 캐시 위치

```
~/.cache/molmo-spaces-resources/
├── robots/          # 로봇 에셋
├── objects/          # 객체 에셋
└── scenes/           # 장면 에셋
```

### 다운로드 크기 및 시간

- **크기**: 약 2-3GB
- **시간**: 네트워크 속도에 따라 10-30분

---

## 벤치마크 실행

### 환경 변수 설정

**CPU 실행 (권장, 16GB GPU인 경우):**
```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
```

**GPU 실행 (20GB+ GPU인 경우):**
위 환경 변수 설정 불필요

### 벤치마크 구조

`benchmarks/minimal_benchmark/benchmark.json`에 **1개의 에피소드** 포함:
- 장면: iTHOR house 321
- 작업: 책 집기 (Pick up the book)
- 카메라: exo_camera_1 (외부), wrist_camera (그리퍼)
- 최대 스텝: 50

### 실행 방법

**glfw 래퍼가 OpenGL 컨텍스트를 자동으로 설정합니다.**

```bash
python run_eval_with_glfw.py \
    --checkpoint_path /home/youngsun/simul/MolmoBot-DROID/ \
    --benchmark_path ./benchmarks/minimal_benchmark \
    --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \
    --task_horizon 600 \
    --output_dir ./minimal_eval_results
```

### 실행 중 로그

각 스텝마다 로그가 출력됩니다. CPU 실행 시 약 30-60분 소요.

### 추가 옵션

```bash
python run_eval_with_glfw.py \
    --checkpoint_path /home/youngsun/simul/MolmoBot-DROID/ \
    --benchmark_path ./benchmarks/minimal_benchmark \
    --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaState8ClampAbsPosConfig \
    --task_horizon 600 \
    --output_dir ./minimal_eval_results \
    --use_filament \
    --environment_light_intensity 15000
```

---

## 결과 확인

### 성공 여부

```
Success rate: 100.0%
321/ep0: pass
```

### 출력 디렉토리

```
minimal_eval_results/
└── FrankaState8ClampAbsPosConfig/
    └── <timestamp>/                    # 타임스탬프 (예: 20260403_175845)
        ├── running_log.log             # 실행 로그
        ├── experiment_config_*.pkl     # 실험 설정 데이터
        └── house_321/
            ├── episode_00000000_exo_camera_1_batch_1_of_1.mp4
            ├── episode_00000000_wrist_camera_batch_1_of_1.mp4
            └── trajectories_batch_1_of_1.h5
```

### 결과 파일 확인

```bash
# 최신 결과 확인
ls -lt minimal_eval_results/FrankaState8ClampAbsPosConfig/

# 비디오 재생
vlc minimal_eval_results/FrankaState8ClampAbsPosConfig/*/house_321/*.mp4
```

---

## 문제 해결

### CUDA Out of Memory (16GB GPU)

**문제**: MolmoBot-DROID 모델이 14.67GB 필요하여 16GB GPU에서는 OOM 발생

**해결**: CPU 실행 사용
```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
```

### 캐시 손상 오류

다음 오류가 발생하면 손상된 캐시를 삭제하세요:

```
RuntimeError: Directory path exists on disk but is not recorded in the cache manifest
```

```bash
rm -rf ~/.cache/molmo-spaces-resources/objects/thor/20251117
rm -rf ~/.cache/molmo-spaces-resources/objects/objathor_metadata/20260129
rm -rf ~/.cache/molmo-spaces-resources/scenes/thor/20251117
```

### 캐시 매니페스트 오류

다운로드 후에도 manifest 오류가 발생하면 다음 Python 코드로 매니페스트를 업데이트하세요:

```python
import json

manifest = {
  "robots": {
    "rby1": ["20251224"],
    "rby1m": ["20251224"],
    "franka_droid": ["20260127"],
    "franka_cap": ["20260213"],
    "floating_rum": ["20251110"],
    "floating_robotiq": ["20260208_retry4"],
    "franka_fr3": ["20260303"]
  },
  "scenes": {
    "ithor": ["20251217"],
    "refs": ["20250923"],
    "procthor-10k-train": ["20251122"],
    "procthor-10k-val": ["20251217"],
    "procthor-10k-test": ["20251121"],
    "holodeck-objaverse-train": ["20251217"],
    "holodeck-objaverse-val": ["20251217"],
    "procthor-objaverse-train": ["20251205"],
    "procthor-objaverse-val": ["20251205"]
  },
  "objects": {
    "thor": ["20251117"],
    "objathor_metadata": ["20260129"]
  }
}

with open('/home/youngsun/.cache/molmo-spaces-resources/mjthor_data_type_to_source_to_versions.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print('Manifest updated')
```

### 모델 로딩 실패

```bash
# 체크포인트 경로 확인
ls -la /home/youngsun/simul/MolmoBot-DROID/
```

### warp 모듈 import 실패 (무시 가능)

```
Failed to import warp: No module named 'warp'
Failed to import mujoco_warp: No module named 'warp'
```

이 경고는 무시해도 됩니다. warp는 선택적 최적화 모듈입니다.

---

## 참고 정보

### 커스텀 벤치마크 생성

```bash
python generate_minimal_benchmark.py \
    --output_path benchmarks/custom_benchmark/benchmark.json \
    --num_episodes 1
```

### 여러 에피소드 실행

벤치마크 JSON 파일에 여러 에피소드 추가 가능

### 에셋 미리 다운로드 (선택 사항)

필요한 경우 벤치마크 실행 전에 에셋을 미리 다운로드할 수 있습니다:

```python
from molmo_spaces import molmo_spaces_constants

# 리소스 매니저 초기화 (모든 에셋 다운로드)
resource_manager = molmo_spaces_constants.get_resource_manager()
print('All assets downloaded')
```

---

## 변경 사항

- **2026-04-06**: 초기 가이드 작성