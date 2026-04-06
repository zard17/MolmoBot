# MolmoBot 데모 실행 가이드 (macOS + CPU)

> Apple Silicon Mac에서 MolmoBot demo_policy.ipynb를 로컬 실행하는 방법
>
> 검증 환경: macOS 15 (Darwin 24.6.0), Apple M3 Pro, 18GB RAM, Python 3.11

---

## 사전 요구사항

- macOS (Apple Silicon)
- Python 3.11 (`brew install python@3.11`)
- Git

---

## 1. 환경 설치

```bash
# uv 패키지 매니저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론 (이미 있으면 스킵)
git clone https://github.com/allenai/MolmoBot.git
cd MolmoBot/MolmoBot

# 의존성 설치 (eval extras 포함, .venv 자동 생성)
uv sync --extra eval
```

설치 완료 후 검증:
```bash
.venv/bin/python -c "import molmo_spaces; import mujoco; import torch; print('OK')"
```

## 2. Jupyter 커널 등록

```bash
source .venv/bin/activate
python -m ipykernel install --user --name molmobot --display-name "MolmoBot"
```

## 3. 노트북 실행

```bash
# Jupyter로 실행
jupyter notebook demo_policy.ipynb

# 또는 VS Code에서 열고 커널을 "MolmoBot"으로 선택
```

### 첫 실행 시 자동 다운로드

| 항목 | 크기 | 소요 시간 |
|------|------|----------|
| 시뮬레이션 에셋 (ProcTHOR, Objaverse) | ~수백 MB | ~5분 |
| MolmoBot-DROID 모델 (HuggingFace) | ~8 GB | ~16분 |
| Qwen3 토크나이저 | ~수 MB | ~10초 |

두 번째 실행부터는 캐시되어 스킵된다.

## 4. 에피소드 길이 조절

CPU에서는 추론이 느리므로 `episode_dur` 값을 조절하여 실행 시간을 관리한다.

| episode_dur | steps | 예상 소요 시간 |
|-------------|-------|--------------|
| 3.0 (테스트) | ~45 | ~9분 |
| 13.2 (중간) | ~200 | ~40분 |
| 30.0 (전체) | ~454 | ~1.5시간 |

노트북 마지막 셀에서 변경:
```python
episode_dur = 3.0  # 기본 30.0에서 줄이기
```

---

## 추론 성능 벤치마크

### Action Chunk 패턴

MolmoBot은 action chunking을 사용한다: 16 steps를 한 번에 예측하고, 8 steps를 실행한 뒤 재추론.

```
Step 1:  ~80초 (모델 추론 — VLM forward + flow matching ODE 10 steps)
Step 2-8: ~2초/step (버퍼에서 꺼내서 실행만)
Step 9:  ~80초 (재추론)
Step 10-16: ~2초/step
...반복
```

### 측정 결과 (M3 Pro, CPU, bfloat16)

| 항목 | 값 |
|------|-----|
| 모델 로드 | 10.8초 |
| Action chunk 추론 (8 steps 단위) | ~80초 |
| 버퍼 step 실행 | ~2초 |
| 평균 step 시간 | 12.45초 |
| 200 steps 총 시간 | ~40분 |
| 목표 실시간 (Franka) | 66ms/step |
| CPU 대비 실시간 비율 | ~190배 느림 |

---

## 알려진 이슈 및 트러블슈팅

### macOS 라이브러리 충돌 경고

```
objc[...]: Class AVFFrameReceiver is implemented in both
  cv2/.dylibs/libavdevice.61.3.100.dylib and
  av/.dylibs/libavdevice.62.1.100.dylib
```

cv2, decord, av 패키지 간 libavdevice/libSDL2 중복 로드 경고. **무시해도 된다** — 실제 크래시는 발생하지 않았다.

### warp 모듈 미설치

```
Failed to import warp: No module named 'warp'
Failed to import mujoco_warp: No module named 'warp'
```

NVIDIA Warp (GPU 물리 가속)이 없어서 발생. macOS/CPU에서는 불필요하므로 **무시 가능**.

### CUDA autocast 경고

```
UserWarning: User provided device_type of 'cuda', but CUDA is not available. Disabling
```

추론 코드가 `torch.autocast(device_type='cuda')`를 호출하지만 CPU fallback됨. **정상 동작**.

### Objaverse 버전 불일치

```
UserWarning: Using objaverse data version 20260131.
This is a newer version that may contain different assets than the original benchmark version (20251016_from_20250610).
```

설치된 에셋 버전과 벤치마크 기준 버전이 다름. 데모 실행에는 영향 없으나, **정식 벤치마크 평가 시 재현성 확인 필요**.

### HuggingFace 인증

```
Warning: You are sending unauthenticated requests to the HF Hub.
```

`HF_TOKEN`을 설정하면 다운로드 속도가 빨라진다:
```bash
export HF_TOKEN=hf_your_token_here
```

---

## 모델 상세 (체크포인트에서 확인)

실행 로그에서 확인된 실제 모델 설정:

```
Model config:
  action_horizon: 16
  action_dim: 8
  n_obs_steps: 2
  flow_steps: 10
  states_mode: cross_attn
  action_norm_mode: quantiles
  state dims: 8 (q01/q99 각 8차원)
  action dims: 8 (q01/q99 각 8차원)
```

**참고**: 노트북 코드의 `RealRobotVLAPolicyConfig` 기본값은 `n_obs_steps=1`이지만, 체크포인트의 config.yaml이 `n_obs_steps=2`로 override한다. 즉 MolmoBot-DROID는 2-frame 관측 모델이다.

---

## 출력 파일

| 파일 | 설명 |
|------|------|
| `sample_render.png` | 시뮬레이션 렌더링 (exo + wrist 카메라 나란히) |
| `rollout.mp4` / `rollout_200.mp4` | 정책 실행 비디오 |

---

*작성일: 2026-04-06 / 검증 환경: macOS 15, M3 Pro 18GB, Python 3.11.14, PyTorch 2.7.1*
