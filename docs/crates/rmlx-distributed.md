# rmlx-distributed — 분산 프리미티브

## 개요

`rmlx-distributed`는 분산 추론을 위한 통신 그룹, MoE (Mixture of Experts) 디스패치/결합 교환, 3-zone 백엔드 정책, compute↔RDMA 파이프라인 오버랩, 오버플로우 감시(SparseGuard), 워밍업 프로토콜, MoE 메트릭을 제공하는 크레이트입니다.

> **상태:** 모든 모듈이 구현되어 있습니다. group, moe_exchange, moe_policy, pipeline, sparse_guard, warmup, metrics.

---

## 모듈 구조

```
rmlx-distributed/src/
├── lib.rs           # 모듈 선언
├── group.rs         # 분산 통신 그룹
├── moe_exchange.rs  # MoE 디스패치/결합 교환
├── moe_policy.rs    # 3-zone 백엔드 정책
├── pipeline.rs      # compute↔RDMA 파이프라인 오버랩
├── sparse_guard.rs  # Expert 오버플로우 감시
├── warmup.rs        # RDMA + JIT 사전 워밍업
└── metrics.rs       # 원자적 MoE 메트릭
```

---

## group.rs — 분산 통신 그룹

통신 그룹을 추상화하여 rank 식별과 피어 관리를 제공합니다.

```rust
pub struct Group {
    ranks: Vec<u32>,      // 정렬된 고유 rank 목록
    local_rank: u32,      // 현재 노드 rank
    world_size: u32,      // 전체 노드 수
}
```

| 메서드 | 설명 |
|--------|------|
| `Group::new(ranks, local_rank, world_size)` | rank 목록에서 그룹 생성 (자동 정렬/중복 제거) |
| `Group::world(world_size, local_rank)` | 전체 rank [0, world_size) 그룹 |
| `ranks()` | 그룹 내 rank 목록 |
| `local_rank()` | 현재 노드 rank |
| `size()` | 그룹 내 rank 수 |
| `world_size()` | 전체 월드 크기 |
| `peers()` | 자신을 제외한 피어 rank 목록 |
| `contains(rank)` | rank가 그룹에 속하는지 확인 |

---

## moe_exchange.rs — MoE 디스패치/결합 교환

### MoeDispatchExchange

토큰을 expert에 라우팅하는 디스패치 교환입니다.

```rust
pub struct MoeDispatchConfig {
    pub num_experts: usize,
    pub top_k: usize,
    pub capacity_factor: f32,   // 1.0 = 정확, >1.0 = 오버프로비저닝
    pub group: Group,
}

pub struct MoeDispatchExchange {
    config: MoeDispatchConfig,
    policy: MoePolicy,
    metrics: MoeMetrics,        // moe_exchange 내부 메트릭
}
```

| 메서드 | 설명 |
|--------|------|
| `new(config, policy)` | 디스패치 교환 생성 |
| `dispatch(batch_size, expert_indices, expert_weights)` | 토큰 디스패치 → `DispatchResult` |
| `metrics()` | 내부 메트릭 조회 |
| `policy()` / `policy_mut()` | 정책 참조/변경 |

### DispatchResult

```rust
pub struct DispatchResult {
    pub backend: MoeBackend,
    pub tokens_per_expert: usize,         // Expert당 최대 토큰 수 (capacity)
    pub expert_counts: Vec<usize>,        // Expert별 실제 토큰 수
    pub overflow_count: u64,              // 오버플로우 토큰 수
    pub local_expert_range: (usize, usize),  // 로컬 expert 인덱스 범위 [start, end)
}
```

### MoeCombineExchange

Expert 출력을 원래 토큰 순서로 결합합니다.

```rust
pub struct MoeCombineExchange {
    group: Group,
}
```

| 메서드 | 설명 |
|--------|------|
| `combine_cpu(expert_outputs, weights, indices, batch_size, top_k, hidden_dim)` | CPU 폴백 결합 |
| `group()` | 그룹 참조 |

### MoeMetrics (moe_exchange 내부)

```rust
#[derive(Debug, Clone, Default)]
pub struct MoeMetrics {
    pub tokens_dispatched: u64,
    pub overflow_count: u64,
    pub cpu_dispatches: u64,
    pub metal_dispatches: u64,
    pub rdma_dispatches: u64,
}
```

---

## moe_policy.rs — 3-zone 백엔드 정책

데이터 크기에 따라 CPU/Metal/RDMA 백엔드를 자동 선택합니다. 쿨다운으로 진동을 방지합니다.

### MoeBackend

```rust
pub enum MoeBackend {
    Cpu,
    Metal,
    Rdma,
}
```

### MoePolicy

```rust
pub struct MoePolicy {
    cpu_max: u32,                         // 기본값: 64
    gpu_min: u32,                         // 기본값: 320
    byte_threshold: usize,               // 기본값: 4096 (4KB)
    cooldown_steps: u32,                 // 기본값: 32
    current_backend: MoeBackend,
    cooldown_remaining: AtomicU32,
    step_count: AtomicU32,
}
```

**선택 로직:**

```
N <= cpu_max       → Cpu
N >= gpu_min       → Metal
cpu_max < N < gpu_min:
  byte_size < byte_threshold → Cpu
  byte_size >= byte_threshold → Metal
쿨다운 중 → 현재 백엔드 유지
```

| 메서드 | 설명 |
|--------|------|
| `MoePolicy::new()` | 기본 임계값으로 생성 |
| `with_thresholds(cpu_max, gpu_min, byte_threshold)` | 커스텀 임계값 |
| `select(n_elements, byte_size)` | 백엔드 선택 |
| `switch_backend(new_backend)` | 백엔드 전환 (쿨다운 활성화) |
| `step()` | 스텝 카운터 증가 |

---

## pipeline.rs — Compute↔RDMA 파이프라인

레이어 단위의 compute↔RDMA 파이프라인 오버랩을 관리합니다.

### PipelineStage

```rust
pub enum PipelineStage {
    WaitingForInput,
    Computing,
    Transferring,
    Complete,
}
```

### PipelineConfig

```rust
pub struct PipelineConfig {
    pub num_layers: usize,
    pub enable_overlap: bool,        // 기본값: true
    pub sync_timeout: Duration,      // 기본값: 5초
}
```

### LayerPipeline

```rust
pub struct LayerPipeline {
    config: PipelineConfig,
    stages: Vec<PipelineStage>,
}
```

| 메서드 | 설명 |
|--------|------|
| `new(config)` | 파이프라인 생성 (모든 레이어 WaitingForInput) |
| `begin_compute(layer)` | 레이어 컴퓨트 시작 표시 |
| `begin_transfer(layer)` | 레이어 전송 시작 표시 |
| `complete(layer)` | 레이어 완료 표시 |
| `stage(layer)` | 레이어 현재 단계 조회 |
| `all_complete()` | 모든 레이어 완료 여부 |
| `reset()` | 모든 단계 WaitingForInput으로 초기화 |
| `measure_overlap(compute_fn, transfer_fn)` | 직렬 vs 파이프라인 실행 시간 측정 |

### PipelineStats

```rust
pub struct PipelineStats {
    pub serial_time: Duration,
    pub pipeline_time: Duration,
    pub overlap_gain: f64,          // (serial - pipeline) / serial
    pub compute_time: Duration,
    pub transfer_time: Duration,
    pub sync_overhead: Duration,
}
```

---

## sparse_guard.rs — Expert 오버플로우 감시

오버플로우 비율을 EMA로 추적하고, 용량 증가 또는 Dense 폴백을 권고합니다.

### GuardAction

```rust
pub enum GuardAction {
    None,
    IncreaseCapacity(f64),   // 용량 증가 팩터
    DenseFallback,           // Dense 연산 폴백
    Reset,                   // 정상 복귀
}
```

### SparseGuard

```rust
pub struct SparseGuard {
    overflow_ema: f64,         // EMA 값
    ema_alpha: f64,            // 기본값: 0.1
    capacity_factor: f64,      // 기본값: 1.0
    dense_fallback: bool,
    window_size: usize,        // 기본값: 100
    step_count: usize,
    overflow_count_window: usize,
    total_count_window: usize,
}
```

| 메서드 | 설명 |
|--------|------|
| `record_step(overflow_count, total_count)` | 스텝 기록 |
| `evaluate()` | 윈도우 종료 시 EMA 갱신 → `GuardAction` 반환 |
| `should_increase_capacity()` | EMA > 0.05 |
| `should_dense_fallback()` | EMA > 0.20 |
| `capacity_factor()` | 현재 용량 팩터 |
| `is_dense_fallback()` | Dense 폴백 활성 여부 |
| `overflow_ema()` | 현재 EMA 값 |

**정책:**
- EMA > 0.05 → 용량 1.25배 증가 (최대 2.0)
- EMA > 0.20 → Dense 폴백 전환
- Dense 중 EMA <= 0.05 → 정상 복귀 (Reset)

---

## warmup.rs — RDMA + JIT 사전 워밍업

추론 시작 전에 RDMA 연결 워밍업과 Metal JIT 커널 컴파일을 수행합니다.

### WarmupConfig

```rust
pub struct WarmupConfig {
    pub rdma_rounds: usize,      // 기본값: 10
    pub jit_precompile: bool,    // 기본값: true
}
```

### WarmupState

```rust
pub struct WarmupState {
    rdma_warmed: bool,
    jit_warmed: bool,
    last_result: Option<WarmupResult>,
}
```

| 메서드 | 설명 |
|--------|------|
| `set_rdma_warmed()` | RDMA 워밍업 완료 표시 |
| `set_jit_warmed()` | JIT 워밍업 완료 표시 |
| `is_ready()` | 둘 다 완료 여부 |
| `set_result(result)` | 워밍업 결과 저장 |
| `last_result()` | 마지막 워밍업 결과 |

### WarmupResult

```rust
pub struct WarmupResult {
    pub rdma_warmup: Duration,
    pub jit_warmup: Duration,
    pub total: Duration,
}
```

---

## metrics.rs — 원자적 MoE 메트릭

`AtomicU64` 기반의 lock-free MoE 운영 카운터입니다.

### MoeMetrics (metrics 모듈)

```rust
pub struct MoeMetrics {
    pub dispatch_count: AtomicU64,
    pub combine_count: AtomicU64,
    pub cpu_dispatches: AtomicU64,
    pub metal_dispatches: AtomicU64,
    pub rdma_dispatches: AtomicU64,
    pub overflow_events: AtomicU64,
    pub zone_switches: AtomicU64,
    pub total_tokens_routed: AtomicU64,
    pub dense_fallback_count: AtomicU64,
}
```

| 메서드 | 설명 |
|--------|------|
| `record_dispatch(tokens)` | 디스패치 횟수 + 토큰 수 기록 |
| `record_combine()` | 결합 횟수 기록 |
| `record_cpu_dispatch()` | CPU 디스패치 기록 |
| `record_metal_dispatch()` | Metal 디스패치 기록 |
| `record_rdma_dispatch()` | RDMA 디스패치 기록 |
| `record_overflow()` | 오버플로우 이벤트 기록 |
| `record_zone_switch()` | Zone 전환 기록 |
| `record_dense_fallback()` | Dense 폴백 기록 |
| `snapshot()` | 시점 스냅샷 → `MoeMetricsSnapshot` |

### MoeMetricsSnapshot

```rust
#[derive(Debug, Clone)]
pub struct MoeMetricsSnapshot {
    pub dispatch_count: u64,
    pub combine_count: u64,
    pub cpu_dispatches: u64,
    pub metal_dispatches: u64,
    pub rdma_dispatches: u64,
    pub overflow_events: u64,
    pub zone_switches: u64,
    pub total_tokens_routed: u64,
    pub dense_fallback_count: u64,
}
```

---

## 의존성

```mermaid
graph BT
    A[rmlx-distributed] --> B[rmlx-core]
    A --> C[rmlx-rdma]
    B --> D[rmlx-metal]
    B --> E[rmlx-alloc]
    C --> E
```

```toml
[dependencies]
rmlx-core = { path = "../rmlx-core" }
rmlx-rdma = { path = "../rmlx-rdma" }
```
