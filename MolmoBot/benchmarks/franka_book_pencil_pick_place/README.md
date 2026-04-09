# Franka Pick-and-Place Benchmark

책상, 책장, Franka 로봇으로 구성된 커스텀 씬 벤치마크.

## 태스크

| 에피소드 | 태스크 | 집는 물체 | 목표 위치 |
|----------|--------|-----------|-----------|
| 0 | 티슈박스 -> 책장 | 책상 위 Tissue_Box_1 | 책장 선반 |
| 1 | 연필 -> 컵 | 책상 위 Pencil_1 | 책상 위 Cup_5 |

## 설치

```bash
# 클론 및 설치 (MolmoBot/ 디렉토리에서)
pip install -e .

# 체크포인트 다운로드
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/MolmoBot-DROID')"
```

## 빠른 시작: 실시간 뷰어로 실행

```bash
# Linux (X11/EGL) — MuJoCo 뷰어 창이 뜸
MUJOCO_GL=egl python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/

# macOS — 뷰어 사용시 mjpython 필요
mjpython scripts/run_benchmark_with_viewer.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/

# 뷰어 없이 (영상만 저장)
python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ \
  --no-viewer
```

옵션:
- `--task_horizon 200` — 에피소드당 최대 스텝 수 (기본값: 200)
- `--episode 0` — 실행할 태스크: 0=티슈박스, 1=연필 (기본값: 0)
- `--output_dir <path>` — 영상 저장 경로

## 전체 평가 (run_eval.py)

두 에피소드 모두 실행하고 성공률을 보고함.

```bash
# 벤치마크 JSON + 커스텀 씬 생성
python scripts/create_book_pencil_benchmark.py

# 평가 실행
python launch_scripts/run_eval.py \
  --checkpoint_path ~/.cache/huggingface/hub/models--allenai--MolmoBot-DROID/snapshots/*/ \
  --benchmark_path benchmarks/franka_book_pencil_pick_place \
  --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaCustomSceneEvalConfig \
  --task_horizon 600
```

결과는 `eval_output/FrankaCustomSceneEvalConfig/<timestamp>/house_0/`에 저장됨:
- `episode_*_exo_camera_1_*.mp4` — 외부 카메라 영상
- `episode_*_wrist_camera_*.mp4` — 손목 카메라 영상
- `trajectories_*.h5` — 궤적 데이터

## GPU 설정 (빠른 추론)

### 24GB GPU에서 OOM 해결

OOM 에러 발생 시, `olmo/models/molmobot/inference_wrapper.py`에서 `to_empty` 라인을 제거:

```python
# 수정 전 (128-136줄):
with torch.device("meta"):
    self.model = self.model_config.build_model()
if self.use_bfloat16:
    self.model.to(torch.bfloat16)
self.model.to_empty(device=self.device)   # <-- 이 줄 삭제
load_model_state(self.checkpoint_path, self.model)
self.model.to(self.device)

# 수정 후:
with torch.device("meta"):
    self.model = self.model_config.build_model()
if self.use_bfloat16:
    self.model.to(torch.bfloat16)
load_model_state(self.checkpoint_path, self.model)
self.model.to(self.device)
```

모델을 GPU 메모리에 두 번 할당하는 것(빈 텐서 + 로드된 가중치)을 방지함.

### GPU 평가

```bash
# 뷰어와 함께
python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path <path> \
  --task_horizon 600

# 전체 평가 (두 에피소드 모두)
python launch_scripts/run_eval.py \
  --checkpoint_path <path> \
  --benchmark_path benchmarks/franka_book_pencil_pick_place \
  --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaCustomSceneEvalConfig \
  --task_horizon 600
```

bfloat16은 `SynthManipMolmoInferenceWrapper`에서 기본으로 활성화됨. 별도 플래그 불필요.

## 씬 확인 (정책 없이)

```bash
# 미리보기 이미지 렌더링
python scripts/view_benchmark_scene.py --preview

# 대화형 뷰어 (Linux)
python scripts/view_benchmark_scene.py

# 대화형 뷰어 (macOS)
python scripts/view_benchmark_scene.py  # mujoco.viewer.launch 사용 (블로킹)
```

## 원본 ProcTHOR 데모 (학습 씬)

학습 데이터와 동일한 ProcTHOR 씬에서 정책을 실행. 학습 씬 성능과 새 벤치마크 씬 비교에 유용.

```bash
# Linux — MUJOCO_GL=egl 필요
MUJOCO_GL=egl python scripts/run_original_demo_with_viewer.py --checkpoint_path <path>

# macOS
mjpython scripts/run_original_demo_with_viewer.py --checkpoint_path <path>

# 문 열기 (18초)
python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --task door_open

# 시간 변경
python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --duration_s 30

# 뷰어 없이
python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --no-viewer
```

## 파일 구성

| 파일 | 설명 |
|------|------|
| `scripts/create_book_pencil_benchmark.py` | 벤치마크 JSON + 커스텀 씬 XML 생성 |
| `scripts/view_benchmark_scene.py` | 씬 미리보기 및 대화형 뷰어 |
| `scripts/run_benchmark_with_viewer.py` | 실시간 MuJoCo 뷰어와 함께 평가 실행 |
| `benchmarks/.../benchmark.json` | 에피소드 명세 |
| `benchmarks/.../custom_scene.xml` | 씬 XML (바닥 + 책상 + 책장) |
| `benchmarks/.../desk.xml` | 책상 프리미티브 XML |
| `benchmarks/.../bookcase.xml` | 책장 프리미티브 XML |
| `scripts/run_original_demo_with_viewer.py` | 원본 ProcTHOR 데모 (학습 씬) + 뷰어 |
| `olmo/eval/configure_molmo_spaces.py` | `FrankaCustomSceneEvalConfig` 평가 설정 |
