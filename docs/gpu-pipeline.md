# GPU Pipeline Architecture: ExecGraph

## 개요

TransformerBlock의 forward pass에서 command buffer 수를 65개 -> 5개로 줄여 (92% 감소) GPU 활용률을 극대화하는 아키텍처. **16.15x 속도 향상** 달성 (110.4ms -> 6.8ms).

Metal의 `MTLSharedEvent`를 활용한 GPU-side event chaining으로 CPU-GPU sync를 layer당 1회로 최소화.

## 기존 문제

### Baseline (`forward()`)
- 각 op마다 개별 command buffer 생성 → commit → waitUntilCompleted
- TransformerBlock 1개당 **~65개 CB**, 각각 CPU-GPU round-trip 발생
- CPU가 GPU 완료를 매번 기다리므로 GPU idle time 과다

### Pipelined (legacy, `forward_pipelined()`)
- ExecGraph 미사용, fused SwiGLU만 적용
- 성능 차이 거의 없음 (1.00x)

## ExecGraph 아키텍처

### 6-CB Pipeline per TransformerBlock

```
CB1: norm1 + Q/K/V projections          (4 encoders)  → signal(1)
CB2: wait(1) + 3D copy + RoPE + cache   (5-6 encoders) → signal(2)
CB3: wait(2) + batched SDPA             (N_heads encoders) → signal(3)
CB4: wait(3) + head concat + O_proj + residual  (3 encoder groups) → signal(4)
CB5: wait(4) + norm2 + gate + up + fused_silu_mul  (4 encoders) → signal(5)
CB6: wait(5) + down_proj + residual     (2 encoders)  → signal(6)

CPU sync: ONCE per layer (graph.sync_and_reset())
```

### 핵심 기법

1. **`_into_cb()` 패턴**: 모든 GPU op에 기존 CommandBuffer에 인코딩하는 변형 추가
   - `copy_into_cb`, `rms_norm_into_cb`, `rope_ext_into_cb`, `add_into_cb`, `sdpa_batched_into_cb`, `fused_silu_mul_into_cb`
   - 자체 CB 생성/commit/wait 없이 caller의 CB에 encoder 추가

2. **3D Batched RoPE**: Q/K를 `[seq, n_heads*d]` → `[n_heads, seq, d]`로 reshape 후 단일 3D dispatch
   - 기존: head당 copy+RoPE → ~80 CB
   - 개선: 2 copy + 2 RoPE = 4 encoders in 1 CB

3. **Interleave Heads 커널**: head concat을 head당 1 encoder로 처리
   - 기존: head * seq_len개 개별 copy encoder
   - 개선: head개 encoder (32 encoders for 32 heads)

4. **MTLSharedEvent GPU-side chaining**: `signal(N)` → `wait(N)`으로 CB 간 의존성 표현
   - CPU는 전혀 개입하지 않음 (GPU가 자체적으로 순서 관리)
   - layer 끝에서 한 번만 `cpu_wait` 호출

## 벤치마크 결과 (Apple M3 Ultra, 512GB)

### 설정
- hidden=4096, heads=32/8, head_dim=128, seq_len=1
- Llama-style SwiGLU FFN (intermediate=11008)
- 50 iterations, 5 warmup

### 수치 정확도
```
baseline forward() vs forward_graph() output:
max_diff=6.44e-6  mean_diff=9.64e-7  ✅ OK (f32 precision)
```

### 성능 (weight 사전 캐싱 미적용)

| 지표 | Baseline | Pipelined | ExecGraph |
|------|----------|-----------|-----------|
| Command Buffers | 65 | 64 | 5 (92% ↓) |
| CPU-GPU Sync | 65 | 64 | 1 (98% ↓) |
| Latency (mean) | 111.5ms | 111.9ms | 37.1ms |
| **Speedup** | 1.00x | 1.00x | **3.00x** |
| **Latency 감소** | - | -0.3% | **66.7%** |

### 성능 (weight 사전 캐싱 적용 — `prepare_weights_for_graph`)

| 지표 | Baseline | Pipelined | ExecGraph (캐싱 미적용) | ExecGraph + weight 캐싱 |
|------|----------|-----------|------------------------|------------------------|
| Latency (mean) | 110.4ms | 110.8ms | 37.1ms | **6.8ms** |
| Latency (p50) | - | - | - | **6.5ms** |
| **Speedup** | 1.00x | 1.00x | 3.00x | **16.15x** |
| **Latency 감소** | - | - | 66.7% | **93.8%** |
| Command Buffers | 65 | 64 | 5 | **5 (92.3% ↓)** |
| CPU-GPU Sync | 65 | 64 | 1 | **1 (98.5% ↓)** |
| 수치 정확도 | - | - | max_diff=6.44e-6 | max_diff=6.44e-6 ✅ |

## 최적화 아이디어

### 1. Weight 사전 캐싱 (구현 완료)

**문제**: `forward_into_cb`와 `batched_qkv_proj_into`가 매 forward pass마다 non-contiguous transposed weight를 contiguous로 복사. 7개 linear layer x 대형 weight = **~676MB/pass** 복사.

```
Q:     4096x4096 = 64MB
K:     4096x1024 = 16MB
V:     4096x1024 = 16MB
O:     4096x4096 = 64MB
Gate:  4096x11008 = 172MB
Up:    4096x11008 = 172MB
Down:  11008x4096 = 172MB
합계: ~676MB per forward pass
```

**해결**: `Linear::prepare_weight_t()` 호출하여 contiguous transposed weight를 한 번만 생성 후 캐싱. `Model::prepare_weights_for_graph()`로 모든 Linear layer에 전파. `forward_into_cb`에서 캐시된 weight 직접 사용.

**측정 결과**: ~30ms 복사 비용 제거, 37ms -> **6.8ms** (**16.15x** speedup, 93.8% 지연 시간 감소)

**관련 버그 수정**: `forward_into_cb`와 `batched_qkv_proj_into`가 non-contiguous transposed weight view를 GEMM 커널에 직접 전달하던 버그 수정 완료. contiguity 보장 로직 추가.

### 2. GEMV 특화 경로 (미구현)

**관찰**: baseline `matmul`은 M=1일 때 GEMV (matrix-vector) 경로 사용. ExecGraph의 `encode_gemm`은 항상 tiled GEMM 사용.

**아이디어**: `forward_into_cb`에서 M=1 감지 시 GEMV 커널 dispatch. Single-token decode (추론의 99% 시간)에서 GEMV가 GEMM보다 훨씬 효율적.

**구현**: `gemv_into_cb` 추가, `forward_into_cb`에서 M=1 분기

### 3. Weight Layout 최적화 (미구현)

**아이디어**: 모델 로드 시 weight를 처음부터 `[in, out]` contiguous로 저장. Transpose view + copy 과정 자체를 제거.

**트레이드오프**:
- 기존 `forward()` (baseline) 경로도 영향받음
- safetensors 파일의 weight layout과 불일치 가능
- 모델 로딩 코드 수정 필요

### 4. 다중 Layer Pipelining (미구현)

**현재**: layer당 sync_and_reset() → layer 간 GPU idle time 존재

**아이디어**: layer N의 CB5-6 (FFN)과 layer N+1의 CB1 (norm+QKV)을 중첩 실행. layer 간 데이터 의존성은 residual output뿐이므로 CB4 완료 후 바로 다음 layer 시작 가능.

**난이도**: 높음 (메모리 관리, 이벤트 카운터 관리 복잡)

### 5. Dual-Queue Compute/Transfer Overlap (미구현)

**아이디어**: KV cache append를 별도 transfer queue에서 실행. Compute queue의 SDPA와 병렬로 cache 쓰기 가능.

**제약**: Metal의 blit command encoder 필요, 현재 compute encoder만 사용 중

## Metal vs CUDA 비교

| Feature | CUDA | Metal | 대안 |
|---------|------|-------|------|
| Graph capture-replay | ✅ CUDA Graphs | ❌ 없음 | 매 pass 6-CB 재인코딩 |
| Persistent kernels | ✅ grid-level loop | ❌ 불가 | MTLSharedEvent chaining |
| Auto DAG scheduling | ✅ driver 관리 | ❌ 없음 | 수동 ExecGraph |
| ICB (indirect) | ✅ 유용 | ⚠️ 16K cmd 제한 | 직접 CB batching |
| GEMV for M=1 | ✅ cuBLAS auto | ❌ 수동 분기 | forward_into_cb에서 분기 |

## 파일 구조

| 파일 | 역할 |
|------|------|
| `crates/rmlx-metal/src/exec_graph.rs` | ExecGraph, EventToken, ExecGraphStats |
| `crates/rmlx-metal/src/event.rs` | GpuEvent (MTLSharedEvent wrapper) |
| `crates/rmlx-metal/src/batcher.rs` | CommandBatcher (multi-encoder CB) |
| `crates/rmlx-core/src/ops/*.rs` | `_into_cb()` variants for all ops |
| `crates/rmlx-nn/src/attention.rs` | `Attention::forward_graph()` |
| `crates/rmlx-nn/src/transformer.rs` | `TransformerBlock/Model::forward_graph()` |
| `benches/pipeline_bench.rs` | 3-way benchmark (baseline/pipelined/graph) |
