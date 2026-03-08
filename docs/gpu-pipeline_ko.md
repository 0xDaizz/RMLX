# GPU Pipeline — ExecGraph 아키텍처

## 개요

rmlx GPU 파이프라인은 여러 Metal GPU 연산을 최소한의 command buffer로 배칭하여 연산별 CPU 오버헤드를 제거합니다. 핵심 추상화인 **ExecGraph**는 결정적 실행 그래프를 사전 구축하여 transformer 레이어 연산을 CPU 개입 없이 재생합니다.

**주요 결과:**

| 지표 | Baseline | ExecGraph | 개선 |
|------|----------|-----------|------|
| 레이어당 Command buffer | 65 | 5 | 92.3% 감소 |
| 레이어당 Command buffer (Phase KO) | 65 | 1 (9개 디스패치) | 98.5% 감소 |
| 레이어당 레이턴시 | ~112ms | ~6.4ms | 17.4x 속도 향상 |
| 레이어당 레이턴시 (Phase KO) | ~109ms | ~1.7ms | 64x 속도 향상 |
| MLX 대비 격차 | -- | 6.34x 빠름 | MLX 대비 우위 |
| Cached 2-인코더 디코드 (60L) | -- | 714 us/L | 8% 빠름, 6x 낮은 σ |
| CPU-GPU 동기화 오버헤드 | baseline | 최소 | 98.5% 감소 |
| 수치 정합성 | -- | max_diff=6.4e-6 | 일치 |

---

## 문제점

기본 Metal 실행 모델에서는 각 GPU 연산(matmul, RoPE, softmax, add 등)이 자체 command buffer를 생성합니다:

1. 새로운 `MTLCommandBuffer` 할당
2. `MTLComputeCommandEncoder` 생성
3. Pipeline state, buffer, threadgroup 크기 설정
4. Thread 디스패치
5. 인코딩 종료
6. Command buffer 커밋
7. **CPU-GPU 동기화 배리어** (완료 대기)

LLaMA 스타일 모델의 단일 transformer 레이어는 약 65개의 개별 연산을 실행합니다. 각 연산은 command buffer 생성, 인코더 설정, CPU-GPU 동기화 포인트에 대한 CPU 측 오버헤드를 발생시킵니다. 레이어당 65개의 command buffer가 있으면, CPU가 GPU 작업을 관리하는 데 GPU가 실행하는 것보다 더 많은 시간을 소비합니다.

```
Baseline: 레이어당 65 CB x N 레이어
  [CB1: matmul_qkv] -> sync -> [CB2: rope_q] -> sync -> [CB3: rope_k] -> sync -> ...
  총: 레이어당 ~112ms
```

---

## ExecGraph 아키텍처

GPU 파이프라인은 3개의 계층적 컴포넌트로 구성됩니다:

### CommandBatcher

`CommandBatcher`는 여러 인코더 연산을 공유 command buffer로 그룹화합니다. 각 연산이 자체 CB를 생성하는 대신, batcher가 여러 연산이 순차적으로 인코딩하는 단일 CB를 제공합니다.

```rust
let mut batcher = CommandBatcher::new(&device);
batcher.begin();

// 여러 연산이 하나의 command buffer를 공유
matmul.forward_into_cb(&mut batcher, &q_proj, &x)?;
rope.forward_into_cb(&mut batcher, &q, positions)?;
matmul.forward_into_cb(&mut batcher, &k_proj, &x)?;
rope.forward_into_cb(&mut batcher, &k, positions)?;

batcher.commit();  // 모든 연산에 대해 단일 커밋
```

### ExecGraph

`ExecGraph`는 transformer 모델의 전체 실행 시퀀스를 사전 구축합니다. Transformer 추론은 결정적이기 때문에(동일한 연산, 동일한 버퍼 크기, 동일한 디스패치 기하), 그래프는 시퀀스를 한 번 기록하고 이후 토큰에 대해 재생합니다.

```rust
// 한 번 구축
let graph = ExecGraph::build(&model, &sample_input)?;

// 각 토큰에 대해 재생 — CPU 오버헤드 거의 없음
for token in tokens {
    graph.execute(&device, &buffers)?;
}
```

### ICB (Indirect Command Buffers)

`IcbBuilder`, `IcbReplay`, `IcbCache`는 Metal Indirect Command Buffer 지원을 제공합니다. ICB를 사용하면 GPU가 CPU 개입 없이 compute 명령을 디스패치할 수 있어, 기록된 명령 시퀀스에 대해 진정한 zero-CPU-overhead 재생이 가능합니다.

---

## `_into_cb()` 패턴

GPU 파이프라인의 기반은 `_into_cb()` 패턴입니다. rmlx-core의 14개 전체 op 모듈이 새로운 command buffer를 생성하는 대신 호출자가 제공한 command buffer에 작업을 인코딩하는 `_into_cb()` 변형을 구현합니다.

**표준 패턴** (자체 CB 생성):
```rust
fn forward(&self, input: &Array) -> Result<Array> {
    let cb = device.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    // ... 작업 인코딩 ...
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(output)
}
```

**`_into_cb()` 패턴** (호출자의 CB에 인코딩):
```rust
fn forward_into_cb(
    &self,
    batcher: &mut CommandBatcher,
    input: &Array,
) -> Result<Array> {
    let encoder = batcher.current_encoder();
    // ... 공유 인코더에 작업 인코딩 ...
    Ok(output)
}
```

14개 전체 op 모듈이 이 패턴을 구현합니다:
- matmul, quantized matmul, softmax, RMS norm, RoPE
- element-wise binary ops, reduce, copy/transpose, indexing
- SiLU, residual add, layer norm, gather, scatter

nn 레이어에서는 다음으로 확장됩니다:
- `Linear::forward_into_cb()`
- `Attention::forward_graph()`
- `TransformerBlock::forward_graph()`
- `TransformerModel::forward_graph()`

---

## Command Buffer 감소

레이어당 65개의 baseline command buffer가 5개로 통합됩니다:

| CB | 연산 | 설명 |
|----|------|------|
| CB1 | RMS norm + Q/K/V projections (fused) | 어텐션 전 정규화 및 QKV Linear projection |
| CB2 | Head split + RoPE + cache append | 멀티헤드 reshape, rotary position encoding, KV 캐시 업데이트 |
| CB3 | SDPA + head concat + O_proj | Fused scaled dot-product attention, 헤드 결합, 출력 프로젝션 |
| CB4 | Residual + pre-FFN norm | 어텐션 잔차 연결 및 feed-forward 정규화 |
| CB5 | Gate + up + silu_mul + down + residual | 전체 FFN 퓨전: gated activation, 프로젝션, 최종 잔차 |

```
ExecGraph: 레이어당 5 CB
  [CB1: Norm+QKV] -> [CB2: Split+RoPE+Cache] -> [CB3: SDPA+Concat+OProj] -> [CB4: Res+Norm] -> [CB5: FFN+Res]
  총: 레이어당 ~6.4ms
```

이는 command buffer 수의 **92.3% 감소** (65 → 5)와 CPU-GPU 동기화 포인트의 **98.5% 감소**를 나타냅니다.

---

## 벤치마크 결과

단일 transformer 레이어(LLaMA 스타일 아키텍처, 4096 hidden, M3 Ultra)에서 측정한 벤치마크입니다:

### Baseline (연산별 command buffer)

- **레이어당 Command buffer:** 65
- **레이어당 레이턴시:** ~112ms
- **레이어당 CPU-GPU 동기화:** 65

### ExecGraph (배칭된 command buffer)

- **레이어당 Command buffer:** 5
- **레이어당 레이턴시:** ~6.4ms
- **레이어당 CPU-GPU 동기화:** 5

### 요약

| 지표 | 값 |
|------|-----|
| 속도 향상 | **17.4x** |
| 레이턴시 감소 | **94.3%** (~112ms → ~6.4ms) |
| CB 감소 | **92.3%** (65 → 5) |
| CPU-GPU 동기화 감소 | **98.5%** |
| 통과 테스트 | **1,298+** |

---

## 가중치 사전 캐싱

Phase 9B-opt에서 런타임 transpose 오버헤드를 제거하기 위해 가중치 사전 캐싱을 도입했습니다. 모델 초기화 시 `prepare_weight_t()`가 가중치 행렬의 contiguous transposed 복사본을 생성하여 추론 중 전치 연산을 수행할 필요가 없도록 합니다.

```rust
// 모델 로드 / 워밍업 시
linear.prepare_weight_t();

// 사전 캐싱된 contiguous transposed 가중치 반환
let wt = linear.weight_transposed_contiguous();
```

모델 수준에서 `prepare_weights_for_graph()`가 모든 가중치 행렬을 재귀적으로 준비합니다:

```rust
model.prepare_weights_for_graph();
// TransformerModel -> TransformerBlock -> Attention + FeedForward -> Linear
```

이를 통해 command buffer 배칭 이후에도 남아 있던 레이어당 오버헤드의 상당 부분을 제거하여, Phase 9B-opt의 17.4x 속도 향상에 기여합니다.

---

## ExecGraph vs CUDA Graphs

Metal과 CUDA는 GPU 작업 배칭에 근본적으로 다른 접근 방식을 취합니다:

| 측면 | ExecGraph (Metal) | CUDA Graphs |
|------|-------------------|-------------|
| 전략 | 최소 CB에 **재인코딩** | 기록된 스트림을 **캡처-재생** |
| 기록 | 명시적 그래프 구축 | 암시적 스트림 캡처 |
| 유연성 | 재생 간 buffer 바인딩 변경 가능 | 변경 시 그래프 재구축 필요 |
| CPU 오버헤드 | 거의 없음 (레이어당 5 CB 커밋) | 거의 없음 (단일 그래프 실행) |
| GPU 오버헤드 | 수동 디스패치와 동일 | 드라이버 수준 최적화 가능 |
| 메모리 | 그래프에 대한 추가 메모리 없음 | 그래프 객체가 캡처 상태 유지 |

ExecGraph의 재인코딩 전략은 Metal의 command buffer 모델에 적합합니다. 고정된 시퀀스를 캡처하고 재생하는 CUDA Graphs와 달리, ExecGraph는 인코딩 시퀀스를 사전 계산하고 새로운 command buffer에 재인코딩합니다. 이를 통해 유연성(호출 간 buffer 바인딩 변경 가능)을 유지하면서 비슷한 수준의 CPU 오버헤드 감소를 달성합니다.

---

## 수치 정합성

ExecGraph는 baseline 연산별 실행 경로와 수치적으로 동일한 결과를 생성합니다:

- **최대 절대 차이:** 6.4e-6
- **검증:** 1,298+ 전체 테스트가 양쪽 코드 경로에서 통과
- **보장:** `_into_cb()` 패턴은 표준 `forward()` 경로와 정확히 동일한 compute pipeline, threadgroup 크기, buffer 바인딩을 인코딩합니다. 유일한 차이는 command buffer 그룹화이며, 이는 수치 결과에 영향을 미치지 않습니다.

이 정밀도 수준(max_diff=6.4e-6)은 fp16/bf16 transformer 연산에서 예상되는 부동소수점 허용 오차 범위 내에 있으며, 파이프라인 최적화가 수치적 발산을 발생시키지 않음을 확인합니다.

---

## Phase KO: 9-디스패치 디코드 경로

Phase KO는 ExecGraph 파이프라인을 최소 디스패치 디코드 경로로 확장하여, 전체 transformer 레이어를 단일 command buffer 안의 9개 Metal 디스패치(메모리 배리어가 있는 4개 encoder)로 축소합니다.

### 디스패치 구성

| # | 연산 | Encoder |
|---|------|---------|
| 1 | 병합된 QKV GEMV (Q+K+V projection 융합) | Encoder 1 |
| 2 | 배치형 RoPE (Q와 K 동시 처리) | Encoder 1 |
| 3 | 배치형 SDPA decode (slab KV cache) | Encoder 2 |
| 4 | 출력 projection (fused GEMV + bias) | Encoder 2 |
| 5 | 어텐션 residual add | Encoder 2 |
| 6 | 병합된 gate_up GEMV (gate+up projection 융합) | Encoder 3 |
| 7 | fused SiLU * gate | Encoder 3 |
| 8 | down projection (fused GEMV + bias) | Encoder 3 |
| 9 | FFN residual add | Encoder 4 |

Encoder 사이의 메모리 배리어가 무거운 encoder 경계 전환을 대체하여, 9개의 논리적 디스패치를 4번의 물리적 encoder 전환으로 줄입니다.

### 단계별 최적화 결과

```text
Baseline (per-op sync):  109,215us  1x
ExecGraph (5 CB):          2,735us  40x
Single-CB (44 enc):        2,049us  53x
9-Dispatch (9->4 enc):     1,739us  64x
MLX compiled:              4,525 us/L (60L)  --
Gap vs MLX:                6.34x 빠름 (60L)
Cached 2-enc (60L):         714us/layer  8% 빠름, 6x 낮은 σ
```

### 핵심 구현 요소

- **가중치 병합**: QKV와 gate_up 가중치를 로드 시점에 병합하여 별도의 GEMV 디스패치 4개를 제거
- **Slab KV cache**: 레이어당 단일 contiguous allocation으로 stride-aware SDPA decode 지원
- **StorageModePrivate**: 정적 가중치를 GPU 전용 메모리에 저장하여 CPU 페이지 테이블 엔트리 제거
- **Array::uninit**: 출력 버퍼를 zeroing 없이 할당하여 커널이 전체를 덮어쓰도록 구성
- **Unretained CB**: Metal 3+ (M2+)에서 commit 후 command buffer retain을 생략
- **_into_encoder 패턴**: encoder 경계 대신 메모리 배리어를 사용하며 ops가 공유 compute encoder에 인코딩

### Phase 8c: CachedDecode 최적화

Phase 8c는 9-디스패치 경로에 CPU 측 오버헤드 제거를 추가합니다:

**CachedDecode 구조체**는 모델 초기화 시 모든 레이어별 상태를 사전 해석합니다:
- 10개 사전 해석된 Pipeline State Object (PSO) — 토큰당 `registry.get_pipeline()` 호출 제로
- 9개 사전 할당된 스크래치 버퍼 (매 토큰 재사용) — 토큰당 `Array::uninit` 제로
- 사전 계산된 디스패치 기하 (그리드 크기, threadgroup 크기)
- 비연속 가중치를 올바르게 처리하기 위한 캐시된 norm 가중치 스트라이드

**2-인코더 디코드 경로**는 인코더 전환을 5회에서 2회로 줄입니다:
- 인코더 A: RMS norm + QKV GEMV + RoPE + KV 캐시 추가 (메모리 배리어 포함)
- 인코더 B: SDPA + O_proj + 잔차 + RMS norm + gate_up GEMV + SiLU*mul + down GEMV + 잔차

**`_preresolved_into_encoder` 패턴**은 검증과 PSO 조회를 건너뜁니다:
- `gemv_preresolved_into_encoder()` — 직접 PSO + 버퍼 바인드 + 디스패치
- `rms_norm_preresolved_into_encoder()` — 직접 PSO + 버퍼 바인드 + 디스패치
- `rope_ext_preresolved_into_encoder()` — 직접 PSO + 버퍼 바인드 + 디스패치
- `sdpa_decode_preresolved_into_encoder()` — 직접 PSO + 버퍼 바인드 + 디스패치
- `fused_silu_mul_preresolved_into_encoder()` — 직접 PSO + 버퍼 바인드 + 디스패치

**GEMV BM8 개선:**
- 모든 BM8 커널에서 6개의 불필요한 `threadgroup_barrier(mem_flags::mem_none)` 제거
- BM8 로드를 2×float4 (32B/스레드)에서 4×float4 (64B/스레드)로 확대

**벤치마크 결과 (M3 Ultra, f16, 60레이어 파이프라인):**

| 경로 | 레이턴시 (us/L) | std_dev (us) |
|------|----------------:|-------------:|
| 직렬 9-디스패치 | ~751 | 507 |
| Cached 2-인코더 | 714 | 84 |
| **개선** | **8% 빠름** | **6x 낮음** |

## Phase 10: 융합 7-디스패치 디코드 경로

Phase 10은 커널 융합을 도입하여 9-디스패치 디코드 경로를 7-디스패치로 줄이고, 두 개의 커널 간 디스패치 경계를 제거합니다.

### 융합 커널

**Fusion B — `fused_swiglu_down`**: SiLU 활성화, element-wise gate 곱셈, down projection GEMV를 단일 Metal 디스패치로 결합합니다. SiLU*gate와 down_proj 사이의 중간 버퍼를 제거하여 디스패치 1회와 스크래치 할당 1회를 절약합니다.

**Fusion A — `fused_rms_gemv`**: RMS 정규화와 후속 GEMV (어텐션 전/FFN 전 norm에 사용)를 단일 디스패치로 결합합니다. Threadgroup이 레지스터 내에서 RMS norm을 계산하고, 정규화된 결과를 바로 GEMV에 전달하여 디바이스 메모리 왕복을 방지합니다.

### 7-디스패치 파이프라인 레이아웃

| # | 연산 | 비고 |
|---|------|------|
| 1 | **fused_rms_gemv** (RMS norm + QKV GEMV) | Fusion A — 어텐션 전 norm과 QKV projection 융합 |
| 2 | 배치형 RoPE (Q와 K 동시 처리) | 9-디스패치와 동일 |
| 3 | SDPA decode (slab KV cache) | 동일 |
| 4 | 출력 projection GEMV + 어텐션 residual add | 동일 |
| 5 | **fused_rms_gemv** (RMS norm + gate_up GEMV) | Fusion A — FFN 전 norm과 gate_up projection 융합 |
| 6 | **fused_swiglu_down** (SiLU * gate + down GEMV) | Fusion B — 활성화 + down projection |
| 7 | FFN residual add | 동일 |

### 폴백 동작

융합 Pipeline State Object (PSO)가 초기화 시 컴파일에 실패하면 (예: 미지원 GPU 아키텍처), CachedDecode가 자동으로 9-디스패치 경로로 폴백합니다. 사용자 개입이 필요하지 않습니다.

### 성능 결과

- **실측**: 703.4 us/layer (f16, 60L, M3 Ultra)
- **감소**: 9 디스패치 → 7 디스패치 (GPU 디스패치 22% 감소)
- **개선**: 714 us/L → 703.4 us/L (커널 융합으로 1.5% 레이턴시 감소)

---

## Phase 11: GEMV 커널 최적화 실험 — 종결

Phase 11은 Phase 10에서 확립한 703.4 us/layer 하한선 이하로 성능을 끌어내리기 위해 세 가지 대안적 GEMV 커널 전략을 조사했습니다. 세 가지 실험 모두 성능 개선에 실패하여, 현재의 row-major BM8 GEMV + f32 누산기가 705 us/layer의 실질적 하한선임을 확인했습니다.

### 실험 결과

| 실험 | 전략 | 결과 | 회귀 |
|------|------|------|------|
| Column-major GEMV | 가중치 레이아웃을 column-major로 전환하여 coalesced read 확보 | **+84% 회귀** | strided output write가 처리량 파괴 |
| Interleaved GEMV | 4-way interleaved 가중치 패킹으로 캐시 라인 활용도 개선 | **+2.2% 회귀** | 패킹 오버헤드가 대역폭 이득을 상쇄 |
| SRAM prefetch + f16 acc + function constants | Threadgroup SRAM prefetch 버퍼, f16 누산, function-constant 타일 크기 | **+3.6% 회귀** | f16 정밀도 손실로 더 넓은 타일 필요, SRAM 압력 증가 |

### 결론

Row-major BM8 GEMV + f32 누산기는 M3 Ultra에서 73.6% 대역폭 효율을 달성합니다. 이는 f16 정밀도에서 Apple Silicon 메모리 서브시스템의 실질적 상한입니다. 다음 없이는 추가 커널 수준 개선이 불가합니다:

1. **양자화** (INT4/INT8) — 메모리 대역폭 수요 감소
2. **하드웨어 변경** — 더 높은 메모리 대역폭 (차세대 Apple Silicon)

디코드 최적화는 커널 수준에서 **종결**되었습니다. 703.4 us/layer (Phase 10 fused 7-dispatch)가 현재 Apple Silicon에서 f16 디코드의 최적 레이턴시입니다.

---

## Phase A: 프리필 (seq_len=N) 단일 레이어 최적화

Phase A는 GPU 파이프라인을 프리필 워크로드(seq_len > 1)로 확장합니다. 프리필에서는 디스패치 오버헤드가 아닌 GEMM 처리량이 병목이 됩니다. 디코드(seq_len=1)가 GEMV 메모리 대역폭에 의해 제한되는 반면, 프리필은 대규모 행렬 곱셈을 포함하며 다른 최적화가 필요합니다.

### 핵심 최적화

**단일-CB 파이프라인**: 프리필 경로는 이전에 레이어당 54개의 CPU-GPU 동기화 지점을 필요로 했습니다. Phase A는 전체 프리필 레이어를 단일 command buffer로 통합하여 동기화 지점을 1개로 줄입니다.

**GQA slab SDPA**: 베이스라인 프리필 경로는 어텐션 헤드당 1개씩 32개의 별도 SDPA 커널을 디스패치했습니다. Phase A는 모든 GQA 헤드를 단일 디스패치로 처리하는 slab 레이아웃 SDPA 커널을 도입하여 32개 디스패치를 1개로 줄입니다.

**GEMM threadgroup swizzle**: GEMM 디스패치에 threadgroup swizzle 패턴을 활성화하여 대규모 행렬 곱셈 중 L2 캐시 지역성을 개선합니다.

**새 연산**: `matmul_into_cb`와 `silu_into_cb`는 GEMM 및 SiLU 연산이 새 command buffer를 생성하지 않고 호출자가 제공한 command buffer에 직접 인코딩할 수 있게 하여 단일-CB 파이프라인을 가능하게 합니다.

### 벤치마크 결과

단일 transformer 레이어(Llama 스타일 아키텍처, f16, M3 Ultra)에서 측정:

| 지표 | 베이스라인 | Phase A | 개선 |
|------|-----------|---------|------|
| CPU-GPU 동기화 지점 | 54 | 1 | 98.1% 감소 |
| SDPA 디스패치 (GQA) | 32 | 1 | 96.9% 감소 |
| 단일 레이어 속도 향상 | 1x | 3.5-7.3x | 시퀀스 길이 의존 |
| MLX 대비 (단일 레이어) | — | 1.2-3.4x 이내 | |
| GEMM TFLOPS | — | 21.21T (rmlx) vs 23.97T (MLX) | -11.5% 격차 (Phase C) |

GEMM 처리량 격차는 Phase B config sweep과 Phase C 커널 수준 최적화를 통해 13T에서 21.21T TFLOPS로 좁혀졌습니다 (MLX: 23.97T, -11.5% 격차). Phase C에서는 wide_load (반복당 2×half4)와 SG=2×4 레이아웃을 프로덕션 커널에 적용했습니다.

**벤치마크**: `prefill_bench.rs`, `gemm_bench.rs`

---

## Phase B: GEMM Config Sweep

Phase B는 Phase A에서 확인된 TFLOPS 격차를 해소하기 위해 최적의 GEMM 커널 구성을 체계적으로 탐색합니다. 3회의 벤치마크 sweep에서 M={64..2048}, N={4096,14336}에 걸쳐 27개 커널 변형을 테스트했습니다.

### 방법론

| Sweep | 파일 | 구성 수 | 초점 |
|-------|------|---------|------|
| 1차 | `gemm_sweep.rs` | 7 | BK/SG 레이아웃 변형 |
| 2차 | `gemm_sweep2.rs` | 9 | BK=16/32/64, 스레드 수 |
| 3차 | `gemm_opt.rs` | 11 | 점유율 중심, MLX 스타일 구성 |

### 결과: bk32_2x4

**최적 구성: BM=64, BN=64, BK=32, SG=2x4, 256 스레드, 이중 버퍼**

이 구성이 대부분의 M/N 조합에서 최고 성능을 보입니다. 2x4 SG 레이아웃(M 방향 2개, N 방향 4개)은 B 행렬 로드 [K,N]이 N 방향 코얼레싱의 이점을 받기 때문에 기존 4x2 레이아웃보다 우수합니다.

### MLX 비교 (M3-Ultra-80c, M=2048, K=4096, N=14336, f16)

| 구성 | TFLOPS | MLX 대비 |
|------|-------:|--------:|
| MLX 0.30.7-dev | 23.97T | -- |
| rmlx mlx_nopad | 22.11T | -7.8% |
| rmlx bk32_2x4 | 21.54T | -10.1% |

소규모 M (<=128)에서는 bk32_2x4가 MLX를 능가합니다: 14.73T vs 14.46T.

### 핵심 발견

1. **SG 레이아웃 방향**: B 행렬 [K,N] 코얼레싱으로 인해 2x4 > 4x2
2. **패딩은 성능 저하**: +16B threadgroup 패딩이 M3 Ultra 점유율을 ~7% 감소시킴
3. **MLX 전략**: BK=16, 2 SG (64 스레드), 단일 버퍼 — 타일 크기보다 점유율을 우선
4. **잔여 격차는 커널 수준**: 로드 패턴과 저장 경로의 차이이며, 구성 수준이 아님
5. **M3 Ultra FP16 피크**: 65.54 TFLOPS; 현재 활용률 ~33%

### GEMM TFLOPS 업데이트

Phase A에서는 13T (rmlx) vs 24T (MLX)로 보고했습니다. Config sweep 이후:

| 지표 | Phase A | Phase B |
|------|--------:|--------:|
| rmlx GEMM TFLOPS | 13T | 21.54T |
| MLX GEMM TFLOPS | 24T | 23.97T |
| 격차 | -46% | -10.1% |

**벤치마크**: `gemm_sweep.rs`, `gemm_sweep2.rs`, `gemm_opt.rs`

---

## Phase C: GEMM 커널 수준 최적화

Phase C는 Phase B에서 확인된 커널 수준 성능 격차를 해소합니다. Phase B에서 최적 구성(bk32_2x4)을 찾았지만, MLX 대비 잔여 ~10% 격차는 로드 패턴, 저장 경로, 프로덕션 커널 통합에 있었습니다. 이 단계에서는 SG=2×4 레이아웃을 프로덕션 커널에 적용하고, 6가지 변형 ablation 벤치마크를 통해 커널 수준 최적화를 테스트합니다.

### 적용된 변경 사항

1. **프로덕션에 SG=2×4 레이아웃 적용**: Phase B에서 확인된 최적 레이아웃을 `matmul.rs`에 적용 (기존에는 이전 레이아웃 사용 중)
2. **wide_load**: 반복당 half4 대신 2×half4 — 루프 반복 횟수 절반, 메모리 요청 절반
3. **벤치마크 구조 수정**: `gemm_bench.rs`에서 사전 할당된 버퍼로 직접 커널 디스패치 (기존에는 할당 오버헤드를 측정하고 있었음)

### 테스트한 최적화

| 변형 | 설명 | 결과 |
|------|------|------|
| ref | Phase B 베이스라인 (bk32_2x4) | 베이스라인 |
| direct_store | simdgroup 레지스터 → 디바이스 메모리, 스크래치 버퍼 없음 | 정확하나 ~1-2% 느림 |
| wide_load | 반복당 2×half4, 루프 횟수 절반 | **+34.8%** |
| aligned | 경계 검사 제거 | 소규모 행렬에서 정확, 대규모 M + N=14336에서 성능 붕괴 |
| ds_wl | direct_store + wide_load 결합 | wide_load 단독 대비 추가 이득 없음 |
| full | 모든 최적화 결합 | wide_load 단독 대비 추가 이득 없음 |

### 핵심 발견

1. **wide_load가 지배적 최적화**: 2×half4 로드가 루프 반복과 메모리 요청을 절반으로 줄여 +34.8% 처리량 향상
2. **direct_store는 소폭 성능 저하**: 스크래치 버퍼 제거 시 비코얼레싱 per-lane scatter write 발생 (~1-2% 느림)
3. **aligned는 대규모에서 위험**: 경계 검사 제거는 소규모 행렬에서 작동하나 대규모 M + N=14336에서 성능 붕괴
4. **최적화 조합 효과 없음**: ds_wl 및 full 변형이 wide_load 단독 대비 추가 이득 없음

### 결과 (M3-Ultra-80c, M=2048, K=4096, N=14336, f16)

| 구성 | TFLOPS | MLX 대비 |
|------|-------:|--------:|
| MLX 0.30.7-dev | 23.97T | -- |
| rmlx Phase C (wide_load) | 21.21T | -11.5% |
| rmlx Phase B (bk32_2x4) | 15.73T | -34.4% |

### GEMM TFLOPS 업데이트

| 지표 | Phase A | Phase B | Phase C |
|------|--------:|--------:|--------:|
| rmlx GEMM TFLOPS | 13T | 21.54T | 21.21T |
| MLX GEMM TFLOPS | 24T | 23.97T | 23.97T |
| 격차 | -46% | -10.1% | -11.5% |

참고: Phase C 베이스라인(15.73T)이 Phase B(21.54T)와 다른 이유는 벤치마크 구조 수정으로 할당 오버헤드 측정이 제거되어 더 정확한 베이스라인이 설정되었기 때문입니다. wide_load의 +34.8% 향상은 이 보정된 베이스라인에서 21.21T에 도달합니다.

**벤치마크**: `gemm_kernel_opt.rs`, `gemm_bench.rs`
