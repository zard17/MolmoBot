# 원격 프레임 뷰어 가이드 (GPU 서버 → 브라우저)

> 헤드리스 GPU 서버에서 벤치마크를 실행하면서 로컬 브라우저에서 시뮬레이션 화면을 실시간으로 확인하는 방법

---

## 개요

MuJoCo passive viewer는 디스플레이가 필요하므로 헤드리스 서버에서는 사용할 수 없다.
이 기능은 평가 루프에서 카메라 프레임(numpy 배열)을 JPEG로 인코딩하여 WebSocket으로 스트리밍하고,
브라우저에서 실시간으로 렌더링한다.

```
GPU 서버                              로컬 머신
─────────────────                    ───────────
run_feasibility.py                   브라우저
  └─ 시뮬레이션 루프                    └─ http://localhost:9090/
       └─ observation에서 프레임 추출        └─ JPEG → canvas 렌더링
       └─ WebSocket 전송 ──────────────────►
```

---

## 사전 요구사항

| 항목 | 조건 |
|------|------|
| GPU 서버 | MolmoBot eval 환경 설치 완료 (`uv sync --extra eval`) |
| 로컬 머신 | 모던 브라우저 (Chrome, Firefox, Safari) |
| 네트워크 | 서버 → 로컬 간 SSH 접속 가능 |

추가 패키지 설치는 필요 없다. `websockets`, `Pillow`은 eval 의존성에 이미 포함되어 있다.

---

## Step 1: GPU 서버에서 벤치마크 + 스트리머 실행

```bash
ssh your-gpu-server
cd MolmoBot/MolmoBot
. .venv/bin/activate

python launch_scripts/run_feasibility.py benchmark-smoke \
  --benchmark-path /path/to/benchmark \
  --output-dir /tmp/molmobot_stream_test \
  --num-workers 1 \
  --stream-port 9090 \
  --stream-quality 70
```

| 플래그 | 설명 |
|-------|------|
| `--stream-port 9090` | 이 포트에서 WebSocket 프레임 스트리머를 시작한다 |
| `--stream-quality 70` | JPEG 품질 (1-100). 낮을수록 대역폭 절약, 기본값 70 |
| `--num-workers 1` | 스트리밍은 단일 워커에서만 동작한다 (필수) |

실행하면 다음 메시지가 출력된다:

```
Frame viewer: open http://0.0.0.0:9090/ in your browser
```

---

## Step 2: 포트 포워딩 (회사 네트워크)

GPU 서버의 포트가 외부에서 직접 접근 불가능한 경우, SSH 터널을 사용한다.

### 방법 A: SSH 로컬 포트 포워딩 (가장 간단)

**로컬 머신에서 실행:**

```bash
ssh -N -L 9090:localhost:9090 your-gpu-server
```

이제 로컬에서 `http://localhost:9090/` 으로 접속하면 GPU 서버의 스트리머에 연결된다.

> `-N`: 원격 명령 실행 없이 터널만 유지
> `-L 9090:localhost:9090`: 로컬 9090 → 서버 localhost:9090 포워딩

### 방법 B: SSH 접속과 동시에 포워딩

Step 1의 SSH 접속을 처음부터 포트 포워딩과 함께 실행하면 터미널 하나로 해결된다:

```bash
ssh -L 9090:localhost:9090 your-gpu-server

# 서버에 접속된 상태에서 바로 벤치마크 실행
cd MolmoBot/MolmoBot && . .venv/bin/activate
python launch_scripts/run_feasibility.py benchmark-smoke \
  --benchmark-path /path/to/benchmark \
  --output-dir /tmp/molmobot_stream_test \
  --num-workers 1 \
  --stream-port 9090
```

### 방법 C: VS Code Remote SSH 사용 시

VS Code로 이미 서버에 접속 중이라면 포트 포워딩을 자동으로 처리할 수 있다:

1. `Ctrl+Shift+P` → **Ports: Forward a Port** 선택
2. 포트 `9090` 입력
3. 브라우저에서 `http://localhost:9090/` 접속

---

## Step 3: 브라우저에서 뷰어 열기

```
http://localhost:9090/
```

화면 상단 HUD에 다음 정보가 표시된다:

| 항목 | 설명 |
|------|------|
| Status | WebSocket 연결 상태 (Connected / Disconnected) |
| Episode | 현재 에피소드 번호 |
| Step | 현재 스텝 번호 |
| FPS | 초당 수신 프레임 수 |

연결이 끊어지면 자동으로 재접속을 시도한다 (1초 → 2초 → 4초 → 8초 백오프).

---

## 커스텀 설정

### 다른 포트 사용

서버 측:
```bash
python launch_scripts/run_feasibility.py benchmark-smoke \
  --stream-port 8888 ...
```

SSH 터널:
```bash
ssh -L 8888:localhost:8888 your-gpu-server
```

브라우저:
```
http://localhost:8888/
```

### JPEG 품질 조정

- 네트워크가 느린 경우: `--stream-quality 40` (대역폭 절약)
- 고화질이 필요한 경우: `--stream-quality 95`

### 별도 호스트에서 뷰어 접속

뷰어 HTML을 직접 열거나, WebSocket 주소를 쿼리 파라미터로 지정할 수 있다:

```
http://localhost:9090/?host=other-server:9090
```

---

## 제한사항

- `--num-workers 1` 필수: 멀티워커 모드에서는 프레임 스트리밍이 동작하지 않는다
- 클라이언트 미접속 시: 오버헤드 없음 (프레임 인코딩 자체를 건너뜀)
- 느린 클라이언트: 최신 프레임만 유지하므로 중간 프레임은 건너뛴다 (백로그 없음)

---

## 트러블슈팅

| 증상 | 원인 및 해결 |
|------|-------------|
| 브라우저에서 "Reconnecting..." 반복 | SSH 터널이 끊어짐 → `ssh -L` 명령 재실행 |
| `Address already in use` 에러 | 포트 충돌 → 다른 포트 사용 (`--stream-port 9091`) |
| 프레임이 검게 나옴 | 시뮬레이션이 아직 시작되지 않았거나 카메라 설정 문제 |
| `Frame streaming requires --num-workers 1` 에러 | `--num-workers 1`을 명시적으로 추가 |
| 뷰어 페이지가 로드되지 않음 | `http://` 확인 (https 아님), 포트 번호 확인 |
