# RMLX — Rust Metal LLM Inference Engine

> **Apple Silicon에 최적화된 Rust 기반 Metal GPU 추론 엔진**
>
> 상태: Phase 0-8 완료 (339 tests) | **Phase 9: GPU Pipeline 진행 중** (16.15x 속도 향상, CB 92% 감소) | 라이선스: MIT OR Apache-2.0 | Rust 1.80+ | macOS (Apple Silicon)

---

## RMLX란 무엇인가요?

RMLX는 Apple MLX 프레임워크의 핵심 Metal GPU 추론 파이프라인을 **Rust로 재구현**하는 프로젝트입니다. Mac Studio M3 Ultra 클러스터에서 Expert Parallelism(EP) 기반 분산 LLM 추론의 이론적 성능 한계에 도달하는 것을 목표로 합니다.

MLX의 C++/Python 아키텍처에서 확인된 구조적 병목을 Rust의 언어적 강점으로 근본적으로 해결합니다.

---

## 왜 RMLX가 필요한가요?

MLX는 훌륭한 프레임워크이지만, 분산 추론 시나리오에서 다음과 같은 소프트웨어 오버헤드가 존재합니다.

| 병목 지점 | MLX 현상 | RMLX 해결 방식 |
|-----------|----------|----------------|
| **Per-op 오버헤드** | 94μs/op 중 84μs가 소프트웨어 오버헤드 | Eager 실행 + Rust zero-cost 추상화 |
| **동기화 비효율** | `waitUntilCompleted` 블로킹 (424.9μs) | `MTLSharedEvent` spin-wait (263.9μs, 1.61x 개선) |
| **메모리 복사** | `std::copy` 기반 RDMA 전송 | Zero-copy: 동일 물리 주소에 Metal + RDMA 이중 등록 |
| **파이프라인 부재** | Compute → Transfer 순차 실행 | 듀얼 `MTLCommandQueue`로 GPU 레벨 오버랩 |
| **Lazy evaluation** | 단일 토큰 디코드에도 그래프 빌드 오버헤드 | Eager-first + 선택적 tracing 컴파일 |

---

## 핵심 목표

```
2-node EP decode: 64ms/step → 33ms/step (~30 tok/s)
단일 노드 32ms/step 대비 near-parity 달성
```

두 대의 Mac Studio M3 Ultra를 Thunderbolt 5 RDMA로 연결하여, 단일 노드와 거의 동일한 추론 성능을 달성하는 것이 최종 목표입니다.

---

## MLX 대비 핵심 차별점

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

6. **CPU-minimal 실행 (GPU Pipeline)**
   RMLX 재작성의 핵심 동기. `CommandBatcher`로 디스패치를 통합하고, `ExecGraph` 이벤트 DAG로 command buffer를 체이닝하며, 전치 가중치를 사전 캐싱합니다. Command buffer를 토큰당 65개에서 5개로 줄이고, GPU를 포화 상태로 유지하면서 CPU는 hot path에서 거의 유휴 상태를 달성합니다. **16.15x 속도 향상** (110.4ms -> 6.8ms, 93.8% 지연 시간 감소).

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| 언어 | Rust 1.80+ (edition 2021) |
| GPU | metal-rs 0.31 (Apple Metal API) |
| RDMA | ibverbs FFI (Thunderbolt 5 UC QP) |
| 하드웨어 | Apple Silicon UMA (M3 Ultra, 80-core GPU, 512GB) |
| 빌드 | Cargo workspace (6 crates) |

---

## 다음 단계

- [아키텍처 개요](architecture/overview.md) — 전체 시스템 레이어 다이어그램과 설계 철학
- [크레이트 구조](architecture/crate-structure.md) — 워크스페이스 레이아웃과 각 크레이트 역할
- [설계 결정](architecture/design-decisions.md) — 주요 기술 결정의 근거

---

## 프로젝트 구조

이 리포지토리(`~/rmlx/`)는 **프레임워크 전용**입니다. 모델 서빙 레이어(`rmlx-lm`)는 [별도 리포지토리](https://github.com/rmlx-lm)에서 관리됩니다.

```
rmlx/
├── crates/
│   ├── rmlx-metal/          # Metal GPU 추상화
│   ├── rmlx-alloc/          # Zero-copy 메모리 할당자
│   ├── rmlx-rdma/           # RDMA 통신 (ibverbs)
│   ├── rmlx-core/           # 연산 엔진 (Op registry, VJP autodiff, LoRA)
│   ├── rmlx-distributed/    # 분산 프리미티브 (EP, AllReduce, MoE)
│   └── rmlx-nn/             # 신경망 레이어 (Transformer, MoE)
├── shaders/                 # Metal 셰이더 소스
├── tests/                   # 통합 테스트
├── benches/                 # 벤치마크
└── examples/                # 사용 예제
```
