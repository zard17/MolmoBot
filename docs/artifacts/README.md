# Experiment Artifacts Index

모든 실험 결과와 데모 영상의 위치를 정리한 인덱스.

## 현재 벤치마크 (feat/interactive-object-swap)

**위치:** `MolmoBot/benchmarks/franka_book_pencil_pick_place/`

| 디렉토리/파일 | 내용 |
|--------------|------|
| `REPORT_KR.md` | 전체 벤치마크 레포트 (한국어) |
| `results_main/` | A/B/C/D 그룹 평가 결과 (44 영상, CSV, 레포트) |
| `results_position/` | 위치 변화 robustness 테스트 (10 영상) |
| `results_early_exploratory/` | 초기 탐색 실험 — mug→cup, bookcase, apple 실패 분석 (12 영상) |
| `results_camera_position/` | 카메라 위치 변화 테스트 (6 영상, 100%) |
| `results_camera_angle/` | 카메라 각도 변화 테스트 (4 영상, 100%) |
| `results_camera_fov/` | 카메라 FOV 변화 테스트 (3 영상, 100%) |
| `report_screenshots/` | 레포트용 프레임 스크린샷 |
| `benchmark.json` | 벤치마크 에피소드 정의 |
| `batch_config.json` | 배치 실행 설정 |
| `custom_scene.xml` | 커스텀 씬 (바닥 + 책상 + 책장) |

### 주요 결과 요약

| 그룹 | 씬 | 오브젝트 | 성공률 |
|------|------|----------|--------|
| A: 베이스라인 | ProcTHOR | Thor | 66.7% |
| B: 오브젝트 일반화 | ProcTHOR | Objaverse | 100% |
| C: 씬 일반화 | Custom | Thor | 85.7% |
| D: 씬+오브젝트 일반화 | Custom | Objaverse | 100% |
| E: 위치 변화 | Both | Thor | 87.5% |

## 데모 영상

**위치:** `MolmoBot/demo_outputs/`

| 디렉토리 | 내용 |
|----------|------|
| `door_pick_place_shared_scene/` | 원본 노트북 데모 — 문 열기 + salt shaker→bowl (ProcTHOR) |
| `original_demo_viewer/` | 뷰어 스크립트 데모 — pick_and_place (ProcTHOR) |

## 사전 Feasibility 연구

**위치:** `docs/artifacts/molmobot_feasibility/`

| 디렉토리 | 내용 |
|----------|------|
| `rby1_door_smoke_20260407/` | RBY1 로봇 문 열기 smoke test |
| `rby1_doorplusopen_small_20260407/` | RBY1 문+열기 성공/실패 영상 |
| `rby1_pnp_blocker_20260407/` | RBY1 pick-and-place blocker 분석 |
| `smoke_success_exo.mp4` | Franka pick smoke test 성공 |
| `pick_subset_success/failure_exo.mp4` | Franka pick subset 결과 |
| `rollout_macos_cpu.mp4` | macOS CPU rollout 데모 |

## 평가 출력 (임시)

**위치:** `MolmoBot/eval_output/` (gitignored)

`run_eval.py` 실행 시 생성되는 임시 결과. Git에 포함되지 않음.
공식 결과는 `benchmarks/` 디렉토리에 저장.

## 실행 스크립트 참조

| 스크립트 | 용도 |
|----------|------|
| `scripts/run_batch_eval.py` | 배치 평가 (Thor vs Objaverse 비교) |
| `scripts/run_benchmark_with_viewer.py` | 단일 에피소드 + 뷰어 |
| `scripts/run_original_demo_with_viewer.py` | 원본 ProcTHOR 데모 + 뷰어 |
| `scripts/view_benchmark_scene.py` | 씬 미리보기 |
| `scripts/run_batch.sh` | RunPod 배치 실행 |
