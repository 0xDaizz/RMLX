# RMLX — Apple Silicon용 Rust ML 프레임워크

> **Apple Silicon에 최적화된 Rust 기반 Metal GPU ML 프레임워크**
>
> 상태: 전 Phase 완료 (0-9B-opt + S1-S5 + Phase KO + Audit Remediation + Phase 10 + Phase 11) (1,356 tests, 0 failures) | 라이선스: MIT | Rust 1.80+ | macOS (Apple Silicon)

---

## 🔍 RMLX란 무엇인가요?

RMLX는 Apple MLX 프레임워크의 핵심 Metal GPU 추론 파이프라인을 **Rust로 재구현**하는 프로젝트입니다. Mac Studio M3 Ultra 클러스터에서 Expert Parallelism(EP) 기반 분산 ML 추론의 이론적 성능 한계에 도달하는 것을 목표로 합니다.

MLX의 C++/Python 아키텍처에서 확인된 구조적 병목을 Rust의 언어적 강점으로 근본적으로 해결합니다.

---

## 💡 왜 RMLX가 필요한가요?

MLX는 훌륭한 프레임워크이지만, 분산 추론 시나리오에서 다음과 같은 소프트웨어 오버헤드가 존재합니다.

| 병목 지점 | MLX 현상 | RMLX 해결 방식 |
|-----------|----------|----------------|
| **Per-op 오버헤드** | 94μs/op 중 84μs가 소프트웨어 오버헤드 | Eager 실행 + Rust zero-cost 추상화 |
| **동기화 비효율** | `waitUntilCompleted` 블로킹 (424.9μs) | `MTLSharedEvent` spin-wait (263.9μs, 1.61x 개선) |
| **메모리 복사** | `std::copy` 기반 RDMA 전송 | Zero-copy: 동일 물리 주소에 Metal + RDMA 이중 등록 |
| **파이프라인 부재** | Compute → Transfer 순차 실행 | 듀얼 `MTLCommandQueue`로 GPU 레벨 오버랩 |
| **CB 오버헤드** | 65 CBs/layer, per-op CPU 동기화 | ExecGraph: 5 CBs/layer (92.3% 감소, 17.4x 속도 향상) |
| **Lazy evaluation** | 단일 토큰 디코드에도 그래프 빌드 오버헤드 | Eager-first + 선택적 tracing 컴파일 |

---

## 🎯 핵심 목표

```
2-node EP decode: 64ms/step → 33ms/step (~30 tok/s)
단일 노드 32ms/step 대비 near-parity 달성

Phase KO 결과: ~109ms/layer → ~0.71ms/layer (64x 속도 향상, MLX 대비 6.34x 빠름 (60L))
Phase 10 결과: 커널 융합 (9→7 디스패치), 703.4 us/layer (MLX 대비 6.43x 빠름)
Phase 11 결과: 열-주도 GEMV 12개 커널 추가 (실험적, 핫 패스 미사용), 성능 Phase 10 수준 유지 (703.4 us/layer, MLX 대비 6.43x 빠름)
92.3% CB 감소, 98.5% CPU-GPU 동기화 감소
```

두 대의 Mac Studio M3 Ultra를 Thunderbolt 5 RDMA로 연결하여, 단일 노드와 거의 동일한 추론 성능을 달성하는 것이 최종 목표입니다.

---

## ⚡ MLX 대비 핵심 차별점

1. **Zero-copy RDMA 데이터 경로**
   `posix_memalign` → `newBufferWithBytesNoCopy` → `ibv_reg_mr` — 3개 뷰가 동일 물리 주소를 공유하여 RDMA hot path에서 `memcpy`를 완전 제거합니다.

2. **MTLSharedEvent 기반 동기화**
   `waitUntilCompleted` 대신 event signal/wait 방식으로 1.61x 동기화 성능 개선을 달성합니다.

3. **듀얼 큐 파이프라인**
   Compute 큐와 Transfer 큐를 분리하여 GPU 레벨에서 연산과 통신을 오버랩합니다.

4. **Eager-first 실행 모델**
   단일 토큰 디코드에서 lazy evaluation 그래프 빌드 오버헤드를 제거합니다. Prefill 등 배치 연산에는 선택적 tracing을 적용합니다.

5. **통합 버퍼 풀**
   Metal + RDMA 이중 등록 버퍼를 사전 할당하여 런타임 등록 오버헤드를 제거합니다.

6. **ExecGraph CB 배칭 + 커널 융합 + 열-주도 GEMV**
   Phase KO는 단일 CB에서 레이어당 9개 디스패치로 추가 축소하여
   MLX 대비 6.34x 빠름 (64x 속도 향상, ~109ms → ~0.71ms)을 달성합니다.
   Phase 10은 `fused_rms_gemv`와 `fused_swiglu_down` 커널 융합으로 9→7 디스패치로 줄여
   703.4 us/layer (MLX 대비 6.43x 빠름)를 달성합니다.
   Phase 11은 열-주도(column-major) GEMV 12개 Metal 커널을 추가하였으나, stride-M 접근 패턴이 대형 M(M>=4096)에서 성능을 저하시켜 디코드 핫 패스에서는 미사용합니다. 성능은 Phase 10 수준을 유지합니다 (703.4 us/layer, MLX 대비 6.43x 빠름).

7. **Expert Parallelism (EP)**
   MLX에는 없는 완전한 EP 스택을 제공합니다. 3-zone 자동 백엔드 정책(CPU/Metal/RDMA)이 데이터 크기에 따라 최적 경로를 자동 선택하고, 7개 전용 MoE Metal 커널, SparseGuard 오버플로우 모니터링, compute-RDMA 파이프라인 오버랩을 통해 Mixtral/DeepSeek-V3 같은 MoE 모델의 분산 추론을 지원합니다. 감사 완료 후 EP-1~EP-6 최적화 단계를 통해 GPU-native top-k 라우팅(topk_route), 그룹형 expert GEMM(ExpertGroup + GatherMM), 가변 길이 v3 프로토콜, TBO/SBO 오버랩(MoePipeline), FP8 와이어 양자화(fp8_exchange), ICB sparse 실행 + slab 링 전송(icb_sparse + slab_ring)을 추가로 구현하였습니다.

---

## 🛠️ 기술 스택

| 영역 | 기술 |
|------|------|
| 언어 | Rust 1.80+ (edition 2021) |
| GPU | metal-rs 0.31 (Apple Metal API) |
| RDMA | ibverbs FFI (Thunderbolt 5 UC QP) |
| 하드웨어 | Apple Silicon UMA (M3 Ultra, 80-core GPU, 512GB) |
| 빌드 | Cargo workspace (7 crates) |

---

## 📚 다음 단계

- [아키텍처 개요](architecture/overview.md) — 전체 시스템 레이어 다이어그램과 설계 철학
- [크레이트 구조](architecture/crate-structure.md) — 워크스페이스 레이아웃과 각 크레이트 역할
- [설계 결정](architecture/design-decisions.md) — 주요 기술 결정의 근거
- [GPU Pipeline](gpu-pipeline.md) — ExecGraph 아키텍처와 벤치마크 결과
- [RMLX vs MLX vs CUDA](comparison.md) — 솔직한 아키텍처 비교

---

## 🔧 분산 RDMA 런북 (2-node 최소)

RMLX 내장 헬퍼(`mlx.distributed_config`, `mlx.launch` 벤치마킹)를 사용합니다.

```bash
# 1) 호스트파일 생성 + 기본 셋업
rmlx config \
  --hosts node1,node2 \
  --backend rdma \
  --over thunderbolt \
  --control-iface en0 \
  --auto-setup \
  --output rmlx-hosts.json \
  --verbose

# 2) 각 노드 RDMA 디바이스 가시성 검증
rmlx launch \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- ibv_devices
```

상세 전제 조건(SSH + passwordless sudo)은
[시작하기: 시스템 요구사항](getting-started/prerequisites_ko.md)을 참고하세요.

---

## 📁 프로젝트 구조

```
rmlx/
├── crates/
│   ├── rmlx-metal/          # Metal GPU 추상화
│   ├── rmlx-alloc/          # Zero-copy 메모리 할당자
│   ├── rmlx-rdma/           # RDMA 통신 (ibverbs)
│   ├── rmlx-core/           # 연산 엔진 (Op registry, VJP autodiff, LoRA)
│   ├── rmlx-distributed/    # 분산 프리미티브 (EP, AllReduce, MoE)
│   ├── rmlx-nn/             # 신경망 레이어 (Transformer, MoE)
│   └── rmlx-cli/            # 네이티브 CLI 도구 (rmlx launch, rmlx config)
├── shaders/                 # Metal 셰이더 소스
├── tests/                   # 통합 테스트
├── benches/                 # 벤치마크
└── examples/                # 사용 예제
```
