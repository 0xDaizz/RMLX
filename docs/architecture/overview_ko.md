# 아키텍처 개요

RMLX는 4개의 레이어로 구성된 계층형 아키텍처이며, 전 Phase(0~9B-opt)가 완료되어 완전히 구현된 상태입니다. 각 레이어는 명확한 책임 경계를 가지며, Cargo 크레이트 단위로 분리되어 있습니다. Phase 7에서 추가된 VJP autodiff, LoRA fine-tuning, 프로덕션 하드닝(structured logging, metrics, precision guard, graceful shutdown)이 포함됩니다. Phase 9에서는 ExecGraph (5 CBs/layer, 92.3% 감소), CommandBatcher, Indirect Command Buffer, 가중치 사전 캐싱이 추가되어 16.15x 속도 향상을 달성했습니다.

---

## 전체 레이어 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ~/rmlx/ (이 리포지토리 — ML 프레임워크)                                      │
│                        rmlx-core (연산 엔진)                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Compute Graph / Op Registry (14 op modules)     │   │
│  │  matmul · softmax · rms_norm · rope · quantized_matmul · moe_gate   │   │
│  │  sdpa · silu · binary · reduce · copy · indexing · ...              │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                          │
│  ┌──────────────────────────────┼───────────────────────────────────────┐   │
│  │                ExecGraph / CommandBatcher / ICB Layer                │   │
│  │  ┌────────────┐  ┌──────────────────┐  ┌─────────────────────┐     │   │
│  │  │ ExecGraph   │  │ CommandBatcher   │  │ IcbBuilder/         │     │   │
│  │  │ (5 CBs/     │  │ (encoder         │  │  IcbReplay/         │     │   │
│  │  │  layer)     │  │  grouping)       │  │  IcbCache           │     │   │
│  │  └────────────┘  └──────────────────┘  └─────────────────────┘     │   │
│  └──────────────────────────────┼───────────────────────────────────────┘   │
│                                 │                                          │
│  ┌──────────────────────────────┼───────────────────────────────────────┐   │
│  │                        Metal Pipeline Layer                          │   │
│  │  ┌────────────┐  ┌──────────┴──────────┐  ┌─────────────────────┐   │   │
│  │  │ Kernel     │  │ CommandEncoder       │  │ Pipeline            │   │   │
│  │  │ Manager    │  │ (barrier, fence,     │  │ Scheduler           │   │   │
│  │  │ (JIT/AOT)  │  │  concurrent ctx)     │  │ (dual-queue)        │   │   │
│  │  └─────┬──────┘  └──────────┬───────────┘  └─────────┬───────────┘   │   │
│  └────────┼─────────────────────┼──────────────────────────┼────────────┘   │
│           │                     │                          │                │
│  ┌────────┼─────────────────────┼──────────────────────────┼────────────┐   │
│  │        ▼                     ▼                          ▼            │   │
│  │                       Sync & Memory Layer                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐   │   │
│  │  │ SharedEvent      │  │ Zero-Copy       │  │ Buffer Pool        │   │   │
│  │  │ Manager          │  │ Allocator       │  │ (Metal + ibv_mr    │   │   │
│  │  │ (signal/wait)    │  │ (posix_memalign │  │  dual-registered)  │   │   │
│  │  │                  │  │  + NoCopy)       │  │                    │   │   │
│  │  └─────────────────┘  └─────────────────┘  └────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
          │                                           │
┌─────────┼───────────────────────────────────────────┼───────────────────────┐
│         ▼                                           ▼                       │
│                      rmlx-rdma (통신 레이어)                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────────┐   │
│  │ ibverbs FFI     │  │ Connection      │  │ EP Collective              │   │
│  │ (UC QP,         │  │ Manager         │  │ (all-to-all, ring          │   │
│  │  send/recv,     │  │ (hosts.json,    │  │  allreduce, send/recv)     │   │
│  │  CQ polling)    │  │  GID, warmup)   │  │                            │   │
│  └─────────────────┘  └─────────────────┘  └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Hardware: M3 Ultra (80-core GPU, 512GB UMA) × 2, TB5 RDMA (16MB max_mr)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 레이어별 상세 설명

### 1. Compute Engine — rmlx-core

연산 그래프와 커널 디스패치를 담당하는 핵심 엔진입니다.

| 컴포넌트 | 역할 |
|----------|------|
| **Op Registry** | matmul, softmax, rms_norm, rope, quantized_matmul, moe_gate, sdpa, silu 등 14개 op 모듈 등록 |
| **Compute Graph** | 선택적 tracing 기반 연산 그래프 (eager-first, prefill 시 tracing) |
| **Kernel Dispatch** | Op → Metal 커널 매핑 및 실행, dtype/shape 기반 최적 커널 선택 |

rmlx-core는 rmlx-metal과 rmlx-alloc에 의존하며, 상위 레이어(rmlx-nn, rmlx-distributed)에 연산 API를 제공합니다.

---

### ExecGraph / CommandBatcher Layer

다수의 GPU 연산을 최소한의 command buffer로 배칭하여 per-op CPU 오버헤드를 줄입니다.

| 컴포넌트 | 역할 |
|----------|------|
| **ExecGraph** | 결정론적 op 시퀀스를 5개 CB/layer로 재생하는 사전 빌드 실행 그래프 (65개에서 92.3% 감소) |
| **CommandBatcher** | 인코더 작업을 공유 command buffer로 그룹화하여 per-op CB 생성 제거 |
| **ICB** | 사전 인코딩된 커맨드 시퀀스를 CPU 오버헤드 없이 재생하는 Indirect Command Buffer |

---

### 2. Metal Pipeline Layer

Metal GPU 실행 파이프라인을 관리합니다. rmlx-metal 크레이트가 이 레이어를 구현합니다.

| 컴포넌트 | 역할 |
|----------|------|
| **Kernel Manager** | `.metallib` AOT 로드 및 소스 문자열 JIT 컴파일, ComputePipelineState 캐싱 |
| **CommandEncoder** | CommandBuffer/Encoder 수명 관리, barrier/fence 삽입, concurrent dispatch context |
| **Pipeline Scheduler** | 듀얼 `MTLCommandQueue` 관리 — compute 큐와 transfer 큐를 분리하여 GPU 레벨 오버랩 |

듀얼 큐 아키텍처는 PoC Phase 3.6에서 검증되었습니다. Compute 연산과 RDMA 데이터 전송 준비를 동시에 실행하여 파이프라인 효율을 극대화합니다.

---

### 3. Sync & Memory Layer

동기화와 메모리 관리를 담당합니다. rmlx-metal(이벤트)과 rmlx-alloc(할당자, 버퍼 풀)이 이 레이어를 구현합니다.

| 컴포넌트 | 역할 |
|----------|------|
| **SharedEvent Manager** | `MTLSharedEvent` signal/wait 기반 비블로킹 동기화 (263.9μs, `waitUntilCompleted` 대비 1.61x 개선) |
| **Zero-Copy Allocator** | `posix_memalign` → `newBufferWithBytesNoCopy` — 동일 물리 메모리에 Metal 뷰를 생성하여 복사 없는 GPU 접근 |
| **Buffer Pool** | Metal + `ibv_mr` 이중 등록 버퍼를 사전 할당, 런타임 등록 오버헤드 제거, size-binned 캐싱 |

Zero-copy 경로는 RDMA 통신 hot path에 한정됩니다. 모델 가중치 로딩 등 1회성 초기화에서의 복사는 최적화 대상이 아닙니다.

---

### 4. Communication Layer — rmlx-rdma

Thunderbolt 5 RDMA를 통한 노드 간 통신을 담당합니다.

| 컴포넌트 | 역할 |
|----------|------|
| **ibverbs FFI** | ibverbs C 라이브러리 FFI 바인딩, UC QP 생성/관리, CQ polling |
| **Connection Manager** | `hosts.json` 기반 피어 관리, GID 교환, 연결 수립 및 warmup |
| **EP Collective** | all-to-all, ring allreduce, send/recv 등 분산 통신 프리미티브 |
| **Multi-port Striping** | 듀얼 TB5 포트 대역폭 활용을 위한 스트라이핑 |

---

## 크레이트 의존성 그래프

```
          rmlx-nn  rmlx-distributed
              │       │       │
              └───┬───┘       │
                  ▼           │
              rmlx-core       │
               │    │         │
               ▼    ▼         ▼
          rmlx-metal  rmlx-alloc
                        │
                        ▼
                    rmlx-rdma
```

정확한 의존 관계를 정리하면 다음과 같습니다.

| 크레이트 | 의존하는 크레이트 |
|----------|-------------------|
| `rmlx-metal` | (외부: metal-rs 0.31, objc2, block2) |
| `rmlx-alloc` | `rmlx-metal`, libc |
| `rmlx-rdma` | `rmlx-alloc`, libc (ibverbs FFI) |
| `rmlx-core` | `rmlx-metal`, `rmlx-alloc` |
| `rmlx-distributed` | `rmlx-core`, `rmlx-rdma` |
| `rmlx-nn` | `rmlx-core` |

**의존성 원칙**: 하위 레이어(rmlx-metal, rmlx-alloc)는 상위 레이어(rmlx-core, rmlx-nn)에 대해 무지합니다. 모든 의존성은 단방향이며 순환 의존은 허용되지 않습니다.

---

## 하드웨어 타겟

| 항목 | 사양 |
|------|------|
| GPU | Apple M3 Ultra, 80-core GPU |
| 메모리 | 512GB Unified Memory Architecture (UMA) |
| 노드 수 | 2대 (Mac Studio) |
| 인터커넥트 | Thunderbolt 5 RDMA (16MB max_mr) |
| RDMA 방식 | UC QP (Unreliable Connection) — TB5에서 RC 미지원 |
