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

### RunPod GPU 환경 설치

```bash
cd /workspace
git clone https://github.com/zard17/MolmoBot.git
cd MolmoBot/MolmoBot
git checkout feat/interactive-object-swap
bash scripts/setup_runpod.sh
```

## 빠른 시작: 실시간 뷰어로 실행 (커스텀 벤치마크 씬)

커스텀 씬에서 단일 에피소드를 실행. 뷰어로 실시간 확인 가능.

```bash
# 에피소드 0: "pick up the tissue box and place it in the bookcase"
MUJOCO_GL=egl python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path <path> --episode 0

# 에피소드 1: "pick up the pencil and put it in the cup"
MUJOCO_GL=egl python scripts/run_benchmark_with_viewer.py \
  --checkpoint_path <path> --episode 1

# macOS — 뷰어 사용시 mjpython 필요
mjpython scripts/run_benchmark_with_viewer.py --checkpoint_path <path>

# 뷰어 없이 (영상만 저장)
python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> --no-viewer
```

옵션:
- `--task_horizon 200` — 에피소드당 최대 스텝 수 (기본값: 200)
- `--episode 0` — 실행할 태스크: 0=티슈박스→책장, 1=연필→컵 (기본값: 0)
- `--output_dir <path>` — 영상 저장 경로

## 인터랙티브 오브젝트 교체

CLI 인자로 pickup/receptacle 오브젝트를 자유롭게 교체 가능. 학습 분포(Thor) vs 새로운 오브젝트(Objaverse) 일반화 테스트에 유용.

```bash
# 사용 가능한 오브젝트 목록 확인
python scripts/run_benchmark_with_viewer.py --list

# 커스텀 오브젝트 조합
python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \
    --pickup Candle_1 --receptacle Bowl_3 \
    --prompt "put the candle in the bowl"

# 책장에 넣기
python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \
    --pickup Apple_1 --receptacle bookcase \
    --prompt "put the apple in the bookcase"

# Objaverse 오브젝트 (처음 사용 시 자동 다운로드)
python scripts/run_benchmark_with_viewer.py --checkpoint_path <path> \
    --pickup Egg_1 \
    --receptacle objaverse:45bb173c0384450487421b687bf3bf5b \
    --prompt "put the egg in the bowl"
```

Thor 오브젝트 18종 + Objaverse 오브젝트 (해시로 지정) 사용 가능.

## 배치 평가 (Thor vs Objaverse 비교)

여러 오브젝트 조합을 순차 실행하고 성공률 레포트 생성. 모델은 한 번만 로드.

```bash
# 1. 기본 설정 생성 (30 조합 × 3 반복 = 90 에피소드)
python scripts/run_batch_eval.py --generate-config

# 2. 배치 실행 (GPU 권장, A6000에서 ~2-3시간)
python scripts/run_batch_eval.py \
  --checkpoint_path <path> \
  --config benchmarks/franka_book_pencil_pick_place/batch_config.json

# 3. 파이프라인 테스트 (짧게)
python scripts/run_batch_eval.py \
  --checkpoint_path <path> \
  --config benchmarks/franka_book_pencil_pick_place/batch_config.json \
  --task_horizon_override 10
```

### 배치 설정 (batch_config.json)

```json
{
  "task_horizon": 600,
  "repeats": 3,
  "tasks": [
    {"pickup": "Mug_1", "receptacle": "Bowl_3"},
    {"pickup": "Apple_1", "receptacle": "objaverse:45bb173c..."},
    {"pickup": "Egg_1", "receptacle": "bookcase"}
  ]
}
```

### 출력물

- `report_<timestamp>.md` — Thor vs Objaverse 성공률 비교 테이블
- `results_<timestamp>.csv` — 에피소드별 데이터 (분석용)
- `results.json` — 전체 결과
- `results_partial.json` — 중간 저장 (중단 시 진행 상황 확인용)
- `ep*_*.mp4` — 에피소드별 영상

### 레포트 예시

```
| Group             | Success | Total | Rate  |
|-------------------|---------|-------|-------|
| Thor (training)   | 18      | 45    | 40.0% |
| Objaverse (novel) | 9       | 45    | 20.0% |
| Total             | 27      | 90    | 30.0% |
```

## 전체 평가 (run_eval.py)

두 에피소드 모두 실행하고 성공률을 보고함.

```bash
# 벤치마크 JSON + 커스텀 씬 생성
python scripts/create_book_pencil_benchmark.py

# 평가 실행
python launch_scripts/run_eval.py \
  --checkpoint_path <path> \
  --benchmark_path benchmarks/franka_book_pencil_pick_place \
  --eval_config_cls olmo.eval.configure_molmo_spaces:FrankaCustomSceneEvalConfig \
  --task_horizon 600
```

결과는 `eval_output/FrankaCustomSceneEvalConfig/<timestamp>/house_0/`에 저장됨:
- `episode_*_exo_camera_1_*.mp4` — 외부 카메라 영상
- `episode_*_wrist_camera_*.mp4` — 손목 카메라 영상
- `trajectories_*.h5` — 궤적 데이터

## GPU 설정

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

### GPU 예상 실행 시간

| GPU | VRAM | 배치 90ep (600 steps) |
|-----|------|----------------------|
| A100 80GB | 충분 | ~2-3시간 |
| A6000 48GB | 충분 | ~2-3시간 |
| A40 48GB | 충분 | ~4-5시간 |

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
# pick_and_place: "put the salt shaker in the bowl" (기본, ProcTHOR 주방, 24초)
MUJOCO_GL=egl python scripts/run_original_demo_with_viewer.py --checkpoint_path <path>

# door_open: "open the door" (ProcTHOR 문, 18초)
MUJOCO_GL=egl python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --task door_open

# macOS
mjpython scripts/run_original_demo_with_viewer.py --checkpoint_path <path>

# 시간 변경
python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --duration_s 30

# 뷰어 없이
python scripts/run_original_demo_with_viewer.py --checkpoint_path <path> --no-viewer
```

## 파일 구성

| 파일 | 설명 |
|------|------|
| `scripts/run_benchmark_with_viewer.py` | 인터랙티브 오브젝트 교체 + 뷰어 평가 |
| `scripts/run_batch_eval.py` | 배치 평가 + Thor vs Objaverse 레포트 |
| `scripts/run_original_demo_with_viewer.py` | 원본 ProcTHOR 데모 (학습 씬) + 뷰어 |
| `scripts/create_book_pencil_benchmark.py` | 벤치마크 JSON + 커스텀 씬 XML 생성 |
| `scripts/view_benchmark_scene.py` | 씬 미리보기 및 대화형 뷰어 |
| `scripts/setup_runpod.sh` | RunPod GPU 환경 자동 설치 |
| `benchmarks/.../batch_config.json` | 배치 평가 설정 (오브젝트 조합) |
| `benchmarks/.../benchmark.json` | 에피소드 명세 |
| `benchmarks/.../custom_scene.xml` | 씬 XML (바닥 + 책상 + 책장) |
| `olmo/eval/configure_molmo_spaces.py` | `FrankaCustomSceneEvalConfig` 평가 설정 |
