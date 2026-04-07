# State-of-the-Art: Simulation Data Generation for Robot Manipulation Training

> Literature survey as of April 2026 | Context: comparison against MolmoBot's approach
> MolmoBot reference: 1.7M episodes, MuJoCo, CuRobo planner, 6-DoF grasp sampling, aggressive domain randomization

---

## 1. Motion Planning for Data Generation

### Planner Landscape

| Planner | Backend | Speed | Strengths | Weaknesses |
|---------|---------|-------|-----------|------------|
| **CuRobo** (NVIDIA) | GPU (CUDA) | ~45 ms avg, 60x faster than CPU baselines | Sub-50 ms planning; parallel batch (512 solutions); lower jerk (2.1 vs 5.8 rad/s^3); mesh-based collision | Requires NVIDIA GPU; limited to supported robot models; closed-source core |
| **OMPL** (via MoveIt) | CPU | ~1200 ms avg | Mature ecosystem; many sampling-based algorithms (RRT*, PRM, etc.); well-documented | CPU-only; variable cycle times (4-16 s); needs trajectory post-processing (CHOMP/STOMP) |
| **MoveIt 2** | CPU (OMPL/CHOMP/STOMP) | 4-16 s cycle time | ROS2 integration; industry standard; large community | Slow for large-scale data generation; self-collision model less reliable than mesh-based |
| **cuMotion** (NVIDIA, 2024+) | GPU | Real-time capable | Successor integration path for CuRobo in Isaac; MoveIt2 plugin available | Newer, less battle-tested |

### Best Practice for Data Generation

CuRobo is the clear winner for **high-throughput trajectory generation** at scale. Its 60x speed advantage over CPU planners is what enables pipelines like MolmoBot to generate 1.7M episodes. MoveIt/OMPL remains relevant for real-time deployment and ROS integration but is impractical as a data generation backbone at million-episode scale.

**MolmoBot's approach (CuRobo + IK + retry logic) is aligned with current best practice.** The retry logic compensates for CuRobo's occasional planning failures, and the combination with IK-based grasp sampling provides a robust pipeline.

### Gap in MolmoBot

CuRobo currently supports a limited set of robot models natively. Extending to custom robots (e.g., a modified RBY1) requires building CuRobo configuration files (robot YAML, collision spheres), which is non-trivial.

---

## 2. Grasp Generation

### Method Comparison

| Method | Year | Approach | Training Data | Key Result | Limitation |
|--------|------|----------|---------------|------------|------------|
| **Contact-GraspNet** | 2021 | 4-DoF conditioned on point cloud | 17M sim grasps | 90%+ on unseen objects in clutter | Parallel-jaw only; tabletop bias |
| **AnyGrasp** | 2023 | Scene-centric, real point clouds | Real-world data | Fast inference; robust in clutter | Restrictive license (machine-locked); poor generalization beyond tabletop |
| **GraspGen** (NVIDIA) | 2025 | DiffusionTransformer + discriminator | 53M sim grasps | 81.3% real-world (28% over M2T2); 17% over Contact-GraspNet in sim | Requires large training dataset; compute-heavy |
| **6-DoF GraspNet** | 2019 | VAE-based | Sim grasps | Foundational work | Superseded by newer methods |
| **AnyDexGrasp** | 2024 | Extension to dexterous hands | Real + sim | Multi-finger grasp synthesis | Dexterous-specific; not for parallel-jaw |

### Current Best Practice

For **data generation pipelines** (not real-time deployment), the dominant approach is:
1. Sample candidate grasp poses on object surfaces (contact-point or surface-normal based)
2. Filter by collision/reachability with IK
3. Score with a learned discriminator or analytical metric
4. Plan approach trajectory with CuRobo

**MolmoBot uses 6-DoF grasp sampling + IK filtering**, which is a solid baseline. GraspGen (2025) represents the new SOTA for learned grasp generation but is primarily designed for runtime use, not batch data generation. For offline batch generation, MolmoBot's approach of sampling + filtering is more practical.

### Gap

MolmoBot does not use a learned grasp quality predictor (like GraspGen's discriminator). Adding one could improve the quality of generated grasps and reduce the retry rate.

---

## 3. Domain Randomization Best Practices

### What to Randomize (2024-2025 Consensus)

| Category | Parameters | Importance | Notes |
|----------|-----------|------------|-------|
| **Visual** | Textures, lighting (color, intensity, direction), shadows, camera pose, camera intrinsics (FoV, distortion) | Critical | Most mature area; MolmoBot does this extensively including 360-degree camera randomization |
| **Object diversity** | Shape, size, color, material appearance, object count, placement | Critical | ICLR 2025 scaling laws paper shows this is the single most impactful factor |
| **Physics/Dynamics** | Friction coefficients, mass, inertia, damping, joint stiffness | Important | System identification (SysID) for measurable params is better than blind randomization |
| **Actuation** | Motor torque limits, control latency, action noise, gripper force | Important | Continual Domain Randomization (CDR) shows ordering matters: torque first, then noise |
| **Scene layout** | Clutter, background objects, table height, distractor objects | Important | RoboTwin 2.0 identifies this as one of five key dimensions |
| **Language** | Diverse task descriptions, paraphrasing | Moderate | Helps with VLA model generalization |

### Advanced Techniques

- **Continual Domain Randomization (CDR, IROS 2024)**: Sequentially introduce randomization parameters instead of all at once. Start with torque, then add noise. Uses continual learning to retain effects of previous randomizations. Matches or outperforms combined randomization with better training stability.

- **DROPO (2023)**: Offline domain randomization that estimates distributions from a small real-world dataset using likelihood-based optimization. Explicitly models parameter uncertainty.

- **Automatic Domain Randomization (ADR)**: Entropy-regularized reward maximization using normalizing flows to discover optimal randomization distributions automatically.

### MolmoBot's Approach

MolmoBot applies aggressive randomization across visual (lighting, textures, camera pose including omnidirectional), physics (action noise), and scene (94,200 procedural environments, 11,400+ objects). This is **more aggressive than most published pipelines**, particularly the omnidirectional camera randomization which is unusual.

### Gap in Typical Pipelines (Including MolmoBot)

1. **Contact physics randomization** is underexplored -- friction, restitution, and contact stiffness are rarely randomized at the level needed for contact-rich tasks
2. **Actuator dynamics** (motor curves, backlash, cable stretch for tendon-driven robots) are usually simplified
3. **Sensor noise models** beyond Gaussian (structured noise, occlusion patterns, depth sensor artifacts) are rarely modeled
4. **SysID vs DR tradeoff**: Recent work shows that for parameters you *can* measure (mass, inertia), system identification outperforms domain randomization. Blind randomization of measurable parameters is counterproductive

---

## 4. Contact-Rich Manipulation

### Current State

| Task Type | Sim Maturity | Best Approach | Key Reference |
|-----------|-------------|---------------|---------------|
| Pick-and-place | High | Scripted + DR + IL | MolmoBot, RoboCasa, many others |
| Peg insertion (<0.5mm clearance) | Medium-High | RL + SDF rewards + curriculum | IndustReal (NVIDIA), 83-99% real success |
| Gear meshing, nut-bolt | Medium | RL + force-based rewards | Factory/IndustReal (Isaac Lab) |
| Tool use | Low-Medium | RL + demonstration | Limited published sim-to-real |
| Deformable objects | Low | MPM/FEM + tactile sim | DiffTactile (ICLR 2024), Genesis |
| Multi-arm assembly | Low-Medium | Bimanual IL + force/torque | BIP policies with multi-modal sensing |

### Key Findings

- **IndustReal (NVIDIA)** demonstrated 83-99% success on real industrial assembly tasks with <0.6mm clearance, trained entirely in Isaac Sim using RL with signed-distance-field rewards and sampling-based curricula.

- **Deformable object manipulation** remains the hardest frontier. MLS-MPM (Material Point Method) is the leading simulation approach. DiffTactile provides differentiable tactile simulation for contact-rich tasks. Genesis supports deformable simulation with differentiable MPM solver.

- **Tactile sensing** is increasingly recognized as essential for contact-rich tasks. Vision-Based Tactile Sensors (VBTSs) in simulation produce transferable observations (contact area, contour).

### MolmoBot's Position

MolmoBot focuses on pick-and-place and articulated object manipulation (door opening). It does **not** address tight-tolerance assembly, deformable objects, or tool use. This is consistent with the field -- no single pipeline handles all contact-rich tasks well. Pick-and-place at MolmoBot's scale (1.7M episodes, 8 task types) is the most mature category.

---

## 5. Sim-to-Real Gaps

### Known Remaining Gaps (2025-2026)

| Gap | Severity | Description |
|-----|----------|-------------|
| **Contact dynamics** | High | Simplified collision detection (convex decomposition, sphere approximations); static friction hysteresis not modeled; nonlinear material deformation absent |
| **Deformable bodies** | High | Most simulators assume rigid bodies; real robots have flexible links, compliant joints; objects deform under load |
| **Sensor fidelity** | Medium-High | Depth sensor noise models are approximate; real cameras have rolling shutter, auto-exposure, white balance shifts |
| **Actuator dynamics** | Medium | Real motors have nonlinear torque curves, thermal effects, backlash; cable-driven robots have stretch and hysteresis |
| **Environmental stochasticity** | Medium | Air currents, table vibration, lighting changes, electromagnetic interference -- hard to model systematically |
| **Object physics** | Medium | Material properties (friction, weight distribution) vary between instances of the "same" object |
| **Visual gap** | Low-Medium | Largely solved by aggressive DR + real-image augmentation; remaining gap in transparent/reflective objects |

### Emerging Solutions

- **Real-is-Sim / Digital Twins**: Build high-fidelity digital twins from real-world scans, then randomize around the twin. Bridges the gap by starting from reality.
- **Differentiable Simulation**: Genesis, DiffTactile -- enable gradient-based parameter optimization to match real behavior.
- **Hybrid sim+real pipelines**: Pre-train in sim, fine-tune with small real dataset. This is the pragmatic consensus.
- **Continual Domain Adaptation**: After sim-to-real transfer, continue adapting the policy using real-world data while preserving sim-learned skills.

### MolmoBot의 주장과 근거 수준

MolmoBot은 순수 시뮬 데이터로 79.2% 실세계 pick-and-place 성공률을 보고합니다 (Pi-0.5의 39.2% 대비).

**관찰된 사실:** MolmoBot은 대규모 환경 다양성 + 시각적 DR을 사용하며, 물리 충실도 강화 없이도 pick-and-place에서 높은 전이 성능을 달성했습니다.

**저자의 해석 (인과 주장):** "공격적 다양성"이 물리 충실도보다 중요하며, 충분한 환경 변이가 있으면 실세계도 "또 하나의 변이"처럼 보인다는 것.

**근거 수준: 미분리.** 이 성공이 다양성, 데이터 규모, CuRobo 궤적 품질, 또는 이들의 조합 중 어디에 기인하는지 분리하는 ablation이 공개되지 않았습니다. 접촉이 중요한 태스크에서 동일 전략이 유효한지도 검증되지 않았습니다.

---

## 6. Scale vs Quality: Is 1.7M Episodes Enough?

### Data Scaling Laws (ICLR 2025 Oral, Best Paper at CoRL 2024)

The landmark study by Lin et al. establishes:

1. **Power-law relationship**: Policy generalization follows a power law with the number of training environments, objects, and environment-object pairs.

2. **Diversity dominates quantity**: Once demonstrations per environment/object reach a threshold (~50), adding more has minimal effect. Adding more *environments* and *objects* continues to improve performance.

3. **Practical recipe**: 32 environments x 1 unique object each x 50 demos = 1,600 demos total can achieve 90% success for a single task with generalization to new environments and objects.

4. **Scale**: Study collected 40,000+ demonstrations and 15,000+ real-world rollouts.

### How MolmoBot Compares

| Metric | Scaling Laws Paper | MolmoBot |
|--------|-------------------|----------|
| Episodes | ~40,000 demos | 1,700,000 episodes |
| Environments | 32 | 94,200 |
| Objects | 32 per task | 11,400+ |
| Tasks | 2 (single-task) | 8 task types |
| Robot | UMI (handheld) | Franka, RBY1 |
| Data source | Real demonstrations | Fully synthetic |

MolmoBot의 170만 에피소드는 **원시 데이터량** 측면에서는 매우 크며, 94,200개 환경과 11,400+ 객체의 다양성 역시 인상적입니다.

### 중요한 한계: 스케일링 법칙의 적용 범위

위 스케일링 법칙 연구는 **실제 로봇 데모 데이터**를 대상으로 수행되었습니다. 합성(시뮬레이션) 데이터에 동일한 스케일링 관계가 성립하는지는 **검증되지 않았습니다.** 구체적으로:

- 합성 에피소드의 정보량이 실제 에피소드와 다를 수 있음 (sim-to-real 갭으로 인해)
- 환경 다양성 vs 물리 충실도의 트레이드오프가 합성 데이터에서는 다르게 작용할 수 있음
- MolmoBot의 79.2% 실세계 성공률이 다양성 때문인지, 데이터 규모 때문인지, 다른 요인 때문인지 분리되지 않음

**따라서 "다양성이 핵심"이라는 주장은 실제 데이터 연구에서의 관찰이며, 합성 데이터에서는 아직 가설 단계입니다.** 이 가설을 우리 환경에서 검증하려면 다양성/규모/충실도를 통제한 ablation 실험이 필요합니다.

### Comparison with Other Datasets

| Dataset | Scale | Source | Notes |
|---------|-------|--------|-------|
| Open X-Embodiment | 1M+ episodes | Real, 22 embodiments | Multi-institution; heterogeneous quality |
| RoboCasa365 | 2,200+ hours | Synthetic + real | 365 tasks, kitchen domain |
| MolmoBot-Data | 1.7M episodes (5,700+ hours) | Fully synthetic | 8 task types, 2 robots |
| DROID | 76K episodes | Real teleoperation | Single-arm tabletop |

---

## 7. Alternative Simulation Platforms

### Platform Comparison

| Platform | Engine | GPU Parallel | Rendering | Strengths | Weaknesses |
|----------|--------|-------------|-----------|-----------|------------|
| **MuJoCo** (MolmoBot) | CPU (MuJoCo) | No (needs CPU cluster) | Basic / MJX for GPU | Fast CPU sim; accurate contact; mature API; open-source (Apache 2.0) | CPU-only for main sim; no native GPU parallelism; limited deformable support |
| **Isaac Lab** (was Orbit) | PhysX (GPU) | Yes (4,096+ envs) | RTX ray tracing | 1.6M FPS across 8 GPUs; photorealistic; NVIDIA ecosystem | Closed-source (Isaac Sim); NVIDIA GPU required; complex setup |
| **ManiSkill3** (SAPIEN) | SAPIEN (GPU) | Yes (up to 30,000+ FPS) | GPU rasterization | Open-source; 10-1000x faster than CPU baselines; 12 task domains; 2-3x less GPU memory than alternatives | Beta status; smaller community than MuJoCo/Isaac |
| **RoboCasa / RoboCasa365** | MuJoCo (robosuite) | CPU only | Basic | 365 tasks; 2,500 kitchen scenes; 2,200+ hours demos; strong benchmark | CPU sim only (slow); kitchen-focused |
| **Genesis** | Custom (GPU) | Yes (claimed 430,000x real-time) | GPU | Differentiable; supports deformable/fluid/granular; 10-80x faster than Isaac Gym (claimed) | Very new; limited real-world validation; community still forming |
| **Habitat 3.0** (Meta) | Custom | Limited | GPU | Human-robot collaboration; social navigation/rearrangement; ICLR 2024 | Navigation-focused; limited manipulation fidelity |
| **RoboVerse** | MetaSim (8+ engines) | Depends on backend | Depends on backend | Unified API across MuJoCo, Isaac, SAPIEN, Genesis, etc.; eliminates cross-platform rewrites | Abstraction overhead; newest entry (2025) |

### Key Tradeoffs

**MuJoCo (MolmoBot's choice):**
- Pro: Most accurate CPU-based contact simulation; huge research community; all of MolmoBot's infrastructure built on it
- Con: CPU-only means data generation requires CPU clusters, not GPU farms. At 1.7M episodes, this was clearly feasible but expensive.

**Isaac Lab (strongest alternative):**
- Pro: GPU parallelism enables massive throughput (1.6M FPS); photorealistic rendering; IndustReal showed contact-rich assembly success
- Con: Locked to NVIDIA ecosystem; simulation fidelity differs from MuJoCo (PhysX vs MuJoCo contact model)

**ManiSkill3 (rising contender):**
- Pro: Open-source GPU parallelism; competitive performance; growing task library
- Con: Still in beta; smaller validated result base

**Genesis (watch):**
- Pro: Differentiable; multi-physics (rigid + deformable + fluid); claimed speed records
- Con: Too new for production use; benchmarks not independently verified

### Path A/B/C 관련 — 미해결 질문 (가설 단계)

`00_interim_results.md`의 Path A/B/C 결정에 대해, 각 경로의 트레이드오프는 **아직 검증되지 않은 가설**입니다. 현재 알 수 있는 사실과 미확인 사항을 분리합니다:

**Path A (MolmoBot 전체 스택, MuJoCo):**
- 사실: CPU 기반 데이터 생성, 94,200개 환경 파이프라인 상속
- 미확인: 이 다양성이 우리 로봇/태스크에서 핵심 요인인지

**Path B (학습 코드 + Isaac Sim 데이터):**
- 사실: GPU 병렬 데이터 생성 가능, MolmoBot 환경 다양성 미포함
- 미확인: 다양성 손실이 GPU 처리량 향상을 상쇄하는지, Isaac Sim에서 유사한 다양성을 구축할 수 있는지

**Path C (Isaac Sim으로 재구현):**
- 사실: 가장 높은 초기 비용, GPU 병렬성 + 다양성 재구축 가능성
- 미확인: 재구축 비용 대비 성능 향상이 정당화되는지, RoboVerse MetaSim이 비용을 실제로 줄이는지

**결정 전 필요한 실험:**
1. 다양성 ablation: 환경 수를 줄였을 때 성능 저하 정도 측정
2. 시뮬레이터 비교: 동일 태스크를 MuJoCo vs Isaac Sim에서 소규모로 생성하여 downstream 성능 비교
3. 처리량 측정: 우리 인프라에서 각 경로의 실제 데이터 생성 속도 벤치마크

---

## 8. Summary: MolmoBot in Context

### Where MolmoBot is SOTA or near-SOTA

1. **Data scale and diversity**: 1.7M episodes across 94,200 environments is the largest published fully-synthetic manipulation dataset
2. **Motion planning backbone**: CuRobo + IK + retry is the standard high-throughput approach
3. **Domain randomization breadth**: Omnidirectional camera randomization is more aggressive than typical pipelines
4. **Zero-shot sim-to-real**: 79.2% on real pick-and-place from pure sim data is a strong result
5. **Open-source completeness**: Full pipeline (datagen + training + eval) is unusually complete for this space

### Where MolmoBot has gaps relative to SOTA

1. **Contact-rich tasks**: No tight-tolerance assembly, insertion, or tool use -- IndustReal handles these
2. **Deformable objects**: Not addressed -- Genesis/DiffTactile are the frontier
3. **GPU-parallel generation**: MuJoCo is CPU-bound; Isaac Lab and ManiSkill3 offer 10-1000x throughput
4. **Learned grasp quality**: Uses sampling + IK filtering, not learned discriminators like GraspGen
5. **Physics randomization**: Contact physics (friction, restitution, stiffness) randomization depth is unclear
6. **SysID integration**: No mechanism to calibrate simulation parameters from real robot measurements
7. **Tactile sensing**: No tactile modality, which is increasingly important for contact-rich tasks

### 미해결 질문

- 합성 데이터에서도 다양성이 실제 데이터와 동일하게 핵심 요인인가?
- 다양성 vs 물리 충실도 트레이드오프가 우리 태스크에서 어떻게 작용하는가?
- 시뮬레이터 선택(MuJoCo vs Isaac Sim)이 downstream 정책 성능에 얼마나 영향을 미치는가?
- 우리 타겟 로봇/태스크에서 MolmoBot의 결과가 재현되는가?
- Path A/B/C 중 어느 것이 최적인지는 위 질문들에 대한 답이 나오기 전까지 알 수 없음

---

## Sources

### Motion Planning
- [Industrial Robot Motion Planning with GPUs: Integration of cuRobo](https://arxiv.org/html/2508.04146v2)
- [cuRobo Official](https://curobo.org/)
- [Comparative Benchmark of Sampling-Based and DRL Motion Planning Methods](https://www.mdpi.com/1424-8220/25/17/5282)

### Grasp Generation
- [GraspGen: A Diffusion-based Framework for 6-DOF Grasping (2025)](https://arxiv.org/html/2507.13097v1)
- [Contact-GraspNet: Efficient 6-DoF Grasp Generation](https://arxiv.org/abs/2103.14127)
- [AnyDexGrasp](https://graspnet.net/anydexgrasp/assets/files/AnyDexGrasp.pdf)

### Domain Randomization
- [Continual Domain Randomization (IROS 2024)](https://arxiv.org/abs/2403.12193)
- [DROPO: Sim-to-real transfer with offline domain randomization](https://www.sciencedirect.com/science/article/pii/S0921889023000714)
- [Domain Randomization via Entropy Maximization (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/56adf9cb91aedfa41ce24398782a012f-Paper-Conference.pdf)

### Contact-Rich Manipulation
- [IndustReal: Transferring Contact-Rich Assembly (NVIDIA)](https://arxiv.org/abs/2305.17110)
- [Bridging the Sim-to-Real Gap for Industrial Assembly (NVIDIA Isaac Lab)](https://developer.nvidia.com/blog/bridging-the-sim-to-real-gap-for-industrial-robotic-assembly-applications-using-nvidia-isaac-lab/)
- [Survey on Imitation Learning for Contact-Rich Tasks](https://arxiv.org/html/2506.13498v1)
- [Contact-Rich Whole-Body Manipulation (Science Robotics 2025)](https://www.science.org/doi/10.1126/scirobotics.ads6790)
- [DiffTactile: Differentiable Tactile Simulator (ICLR 2024)](https://github.com/Genesis-Embodied-AI/DiffTactile)

### Sim-to-Real Gap
- [The Reality Gap in Robotics: Challenges, Solutions, and Best Practices (2025)](https://arxiv.org/html/2510.20808v1)
- [Real-is-Sim: Dynamic Digital Twin for Policy Evaluation](https://arxiv.org/html/2504.03597v1)
- [Safe Continual Domain Adaptation after Sim2Real Transfer](https://arxiv.org/html/2503.10949)

### Data Scaling Laws
- [Data Scaling Laws in Imitation Learning for Robotic Manipulation (ICLR 2025 Oral)](https://arxiv.org/abs/2410.18647)
- [Is Diversity All You Need for Scalable Robotic Manipulation? (2025)](https://arxiv.org/html/2507.06219v1)

### Simulation Platforms
- [MolmoBot (Allen AI)](https://allenai.org/blog/molmobot-robot-manipulation)
- [Isaac Lab / Orbit](https://isaac-orbit.github.io/)
- [ManiSkill3 (RSS 2025)](https://arxiv.org/abs/2410.00425)
- [RoboCasa365](https://arxiv.org/abs/2603.04356)
- [Genesis Simulator](https://genesis-embodied-ai.github.io/)
- [Habitat 3.0 (ICLR 2024)](https://arxiv.org/abs/2310.13724)
- [RoboVerse: Unified Platform (2025)](https://arxiv.org/html/2504.18904v1)

### VLA Foundation Models
- [Pi-0: Vision-Language-Action Flow Model](https://arxiv.org/html/2410.24164v1)
- [Pi-0.5: Open-World Generalization](https://www.physicalintelligence.company/blog/pi05)
