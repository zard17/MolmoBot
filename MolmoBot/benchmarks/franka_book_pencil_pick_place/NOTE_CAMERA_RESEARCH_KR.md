# 카메라 시스템 분석 노트

**날짜:** 2026-04-13
**관련 레포트:** REPORT_KR.md 섹션 6 (카메라 Robustness)

---

## 1. 2-카메라 구조

MolmoBot-DROID는 매 스텝마다 **두 대의 카메라 이미지**를 입력으로 받는다.

```mermaid
graph LR
    subgraph Franka FR3
        BASE["로봇 베이스<br/>(fr3_link0)"]
        ARM["7-DOF 팔"]
        GRIP["그리퍼"]
        BASE --> ARM --> GRIP
    end

    EXO["Exo Camera<br/>640x360<br/>고정"]
    WRIST["Wrist Camera<br/>640x360<br/>이동"]

    BASE -.- EXO
    GRIP -.- WRIST

    style EXO fill:#4a90d9,color:#fff
    style WRIST fill:#d94a4a,color:#fff
    style GRIP fill:#555,color:#fff
```

| | Exo Camera (외부) | Wrist Camera (손목) |
|---|---|---|
| **장착 위치** | 로봇 베이스 (`fr3_link0`) — 고정 | 그리퍼 (`gripper/wrist_camera`) — 이동 |
| **역할** | 씬 전체 파악, 오브젝트 위치 인식 | 근접 파지 시 정밀 가이드 |
| **움직임** | 고정 (로봇 동작과 무관) | 엔드이펙터와 함께 이동 |
| **해상도** | 640×360 | 640×360 |

## 2. 이미지 → 로봇 동작 파이프라인

```mermaid
flowchart TD
    subgraph 입력["입력 (매 스텝)"]
        EXO_IMG["Exo 이미지<br/>640x360"]
        WRIST_IMG["Wrist 이미지<br/>640x360"]
        LANG["언어 지시<br/>'pick up the salt shaker'"]
        STATE["관절 상태 (qpos)<br/>7D joint positions"]
    end

    subgraph VIT["① Vision Encoder (ViT)"]
        VIT_E["Exo 패치 임베딩"]
        VIT_W["Wrist 패치 임베딩"]
    end

    subgraph LLM["② Molmo LLM Backbone (Transformer)"]
        TOKENS["이미지 토큰 + 텍스트 토큰 + 상태"]
        LAYERS["다중 레이어 히든 스테이트 수집<br/>(모든 레이어 출력 보존)"]
        TOKENS --> LAYERS
    end

    subgraph ACTION["③ Action Expert (DiT, 32 블록)"]
        NOISE["랜덤 노이즈<br/>(16 x 7D)"]
        FM["Flow-Matching<br/>10회 적분 스텝"]
        CROSS["각 블록 ← LLM 레이어 N<br/>(cross-attention)"]
        NOISE --> FM
        CROSS -.-> FM
    end

    subgraph 출력["출력"]
        ACTIONS["16 x 7D 행동 궤적<br/>[x, y, z, rx, ry, rz, gripper]"]
        UNNORM["역정규화"]
        ROBOT["로봇 관절 전송<br/>~15Hz"]
        ACTIONS --> UNNORM --> ROBOT
    end

    EXO_IMG --> VIT_E
    WRIST_IMG --> VIT_W
    VIT_E --> TOKENS
    VIT_W --> TOKENS
    LANG --> TOKENS
    STATE --> TOKENS
    LAYERS --> CROSS
    FM --> ACTIONS

    style EXO_IMG fill:#4a90d9,color:#fff
    style WRIST_IMG fill:#d94a4a,color:#fff
    style LANG fill:#4a9,color:#fff
    style STATE fill:#a84,color:#fff
    style FM fill:#764ba2,color:#fff
```

- **7D 행동**: `[x, y, z, rx, ry, rz, gripper]` — 위치 델타 + 그리퍼 개폐
- **Flow-matching**: 랜덤 노이즈에서 시작, 10회 적분 스텝으로 유효한 행동 궤적 생성
- **Action Expert**의 각 블록은 LLM의 **서로 다른 레이어** 출력을 cross-attention으로 수신 → 다중 스케일 씬 이해

## 3. Exo 카메라 Robustness 100%의 이유

카메라 robustness 테스트 (13/13, 100%) 결과의 원인 분석:

```mermaid
graph TD
    R["Exo 카메라 Robustness<br/>13/13 (100%)"]

    A["카메라 메타데이터 미사용<br/>픽셀만 입력 → overfitting 불가"]
    B["VLM 의미론적 특징 추출<br/>공간적 관계 학습<br/>(좌표가 아닌 관계)"]
    C["DROID 학습 데이터<br/>다양한 카메라 배치에서 수집<br/>→ viewpoint-invariant"]
    D["Wrist 카메라 보완<br/>Exo: 대략적 위치<br/>Wrist: 정밀 파지 가이드"]

    R --- A
    R --- B
    R --- C
    R --- D

    style R fill:#2d8,color:#fff,stroke:#2d8
    style A fill:#369,color:#fff
    style B fill:#369,color:#fff
    style C fill:#369,color:#fff
    style D fill:#369,color:#fff
```

1. **모델에 카메라 외부 파라미터가 입력되지 않음** — 픽셀만 입력하므로 특정 카메라 위치에 overfitting 불가
2. **VLM의 의미론적 특징 추출** — ViT가 픽셀 좌표가 아닌 공간적 관계("소금통이 그릇 왼쪽에 있다")를 학습
3. **학습 데이터(DROID)의 다양한 카메라 배치** — 다양한 실제 환경에서 수집되어 viewpoint-invariant 특징 학습
4. **Wrist 카메라의 보완 역할** — Exo가 대략적 위치를 잡고, 최종 파지는 wrist 카메라가 정밀 가이드. Exo 시점이 약간 바뀌어도 wrist 카메라가 일관된 근접 뷰 제공

## 4. 향후 실험 제안

- **Wrist 카메라 robustness 테스트**: Exo보다 영향이 클 가능성 높음 (파지 정밀도에 직접 관여)
- **단일 카메라 ablation**: Exo only vs Wrist only 성능 비교
- **카메라 극한 테스트**: ±20cm, ±20° 이상에서 성능 경계 확인
